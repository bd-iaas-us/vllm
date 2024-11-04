# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.utils import (is_first_decode_pass, is_second_decode_pass,
                                                is_prefill_run)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

import time
from vllm.logger import init_logger
logger = init_logger(__name__)

kv_cache_decode_data = {}

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config, config.ffn_dim)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(config.hidden_size,
                                                config.word_embed_proj_dim,
                                                bias=False,
                                                quant_config=quant_config)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                               config.hidden_size,
                                               bias=False,
                                               quant_config=quant_config)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OPTDecoderLayer(config, cache_config, quant_config),
            prefix=f"{prefix}.layers")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        global kv_cache_decode_data
        first_decode_pass = is_first_decode_pass(input_ids, attn_metadata)
        second_decode_pass = is_second_decode_pass(input_ids, attn_metadata,
                                                      kv_caches[0])
        prefill_pass = is_prefill_run(input_ids)

        if first_decode_pass or second_decode_pass or prefill_pass:
            if 'kv_cache_transporter' not in kwargs:
                raise ValueError(
                    "Missing 'kv_cache_transporter' in keyword arguments.")
            kv_cache_transporter = kwargs['kv_cache_transporter']
        
        if second_decode_pass or first_decode_pass:
            if "request_ids" not in kwargs:
                raise ValueError(
                    "Missing 'request_ids' in keyword arguments.")
            
            request_ids = kwargs["request_ids"]

            if first_decode_pass and len(request_ids) != len(attn_metadata.seq_lens):
                raise ValueError(
                    "The number of request_ids should be equal to the number of sequences.")


        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if first_decode_pass:

            start = time.time()

            kv_cache_transporter.read_hidden_states(input_ids, attn_metadata,
                                                    hidden_states)
            
            
            seq_lens = attn_metadata.seq_lens
            seq_index = 0
            for i, request_id in enumerate(request_ids):
                seq_len = seq_lens[i]
                kv_cache_decode_data[request_id] = {
                    "prompt_token_ids": input_ids[seq_index:seq_index+seq_len],
                    "slot_mapping": attn_metadata.slot_mapping[seq_index:seq_index+seq_len],
                }
                seq_index += seq_len

            kv_cache_transporter.synchronize()

            # logger.info(f"Qian ~~~~~~~ Decode hidden state: Synchronize: {time.time() - start}")

            return hidden_states


        if second_decode_pass:

            if missing_ids := [req for req in request_ids if req not in kv_cache_decode_data]:
                raise ValueError(f"Missing kv cache data for request_ids {missing_ids}")

            prompt_token_ids_tensor = torch.cat([torch.tensor(kv_cache_decode_data[req]["prompt_token_ids"]) for req in request_ids], dim=0)
            lengths_tensor = torch.cat([torch.tensor([len(kv_cache_decode_data[req]["prompt_token_ids"])]) for req in request_ids], dim=0)
            slot_mapping_tensor = torch.cat([torch.tensor(kv_cache_decode_data[req]["slot_mapping"]) for req in request_ids], dim=0)

            [kv_cache_decode_data.pop(req, None) for req in request_ids]
            
            kv_cache_transporter.read_kv_cache(prompt_token_ids_tensor, lengths_tensor, 
                                               slot_mapping_tensor, 0,
                                                   kv_caches[0])
            
            for i in range(self.start_layer, self.end_layer):
                start = time.time()
                kv_cache_transporter.synchronize()
                logger.info(f"Qian ~~~~~~~ second pass Decode compute: Synchronize: {time.time() - start}")
                if i < self.end_layer - 1:
                    kv_cache_transporter.read_kv_cache(prompt_token_ids_tensor, lengths_tensor, 
                                               slot_mapping_tensor, i+1,
                                                    kv_caches[i+1])
                logger.info(f"Qian ~~~~~~~ second pass Decode compute: compute 1: {time.time() - start}")
                
                layer = self.layers[i]
                hidden_states = layer(hidden_states,
                                    kv_caches[i - self.start_layer],
                                    attn_metadata)
                logger.info(f"Qian ~~~~~~~ second pass Decode compute: compute 2: {time.time() - start}")
        else:
            for i in range(self.start_layer, self.end_layer):
                start = time.time()
                layer = self.layers[i]
                hidden_states = layer(hidden_states,
                                    kv_caches[i - self.start_layer],
                                    attn_metadata)
                # logger.info(f"Qian ~~~~~~~ Decode compute: Layer {i} {time.time() - start}")
                
                if prefill_pass:
                    start = time.time()
                    kv_cache_transporter.save_kv_cache(
                        input_ids, attn_metadata.seq_lens, attn_metadata.slot_mapping, i,
                        kv_caches[i - self.start_layer])
                    
                    logger.info(f"Qian ~~~~~~~ prefill: Save kv cache: layer {i} {time.time() - start}")

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)

        if prefill_pass:
            start = time.time()
            kv_cache_transporter.save_hidden_states(input_ids, attn_metadata,
                                                    hidden_states)
            
            # logger.info(f"Qian ~~~~~~~ prefill: Save hidden states: {time.time() - start}")
            start = time.time()
            kv_cache_transporter.synchronize()
            # logger.info(f"Qian ~~~~~~~ prefill: Synchronize: {time.time() - start}")

        return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, cache_config, quant_config)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.decoder(input_ids,
                            positions,
                            kv_caches,
                            attn_metadata,
                            intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **kwargs)


class OPTForCausalLM(nn.Module, SupportsPP):

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
    }
    default_bitsandbytes_target_modules = [
        ".q_proj.", ".k_proj.", ".v_proj.", ".out_proj.", ".fc1.", ".fc2."
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".out_proj.", ".fc2."]

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(config, cache_config, quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.word_embed_proj_dim)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue
            if name.startswith("decoder."):
                name = "model." + name

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
