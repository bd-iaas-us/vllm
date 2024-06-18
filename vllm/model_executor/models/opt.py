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
from typing import Iterable, List, Optional, Tuple

import math
import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


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
                              scale=self.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        # print("Forward")
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # if kv_cache is not None:
        #     print("kv_cache shape")
        #     print(kv_cache.shape)
        # print(sparse_condition.shape)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata, 1.0, sparse_condition) #?? kv_scale, sparse_condition
        # print("WTF here")
        # print(sparse_condition)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
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
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata,
                                       sparse_condition=sparse_condition)
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
        quant_config: Optional[QuantizationConfig] = None,
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

        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.n_times = -2

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # print("OPTDecoder forward start")
        # print(attn_metadata.slot_mapping.shape)
        # print(input_ids.size(0))
        # print(input_ids)
        # print(positions)
        # print(positions.shape)
        # print(attn_metadata.num_decode_tokens)
        # print(attn_metadata.num_prefills)
        # print(attn_metadata.slot_mapping)
        # print(attn_metadata.slot_mapping // 16)
        # print(attn_metadata.slot_mapping % 16)
        # key_cache.size(3)
        # tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], device='cuda:0') 
        # tensor([6, 8, 6, 6], device='cuda:0')
        # if positions.size(0) == 26:
        #     positions = positions[1::2]
        #     input_ids = input_ids[1::2]
        #     print(positions)
        #     print(input_ids)
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds, _ = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds
        # print("OPTDecoder forward in the middle")
        # if sparse_condition is not None:
        #     print(sparse_condition.shape)
        #     print(sparse_condition.size(2))
        # if kv_caches[0] is not None:
        #     print(kv_caches[0].shape)
        # print(self.config)
        # print(self.config.num_hidden_layers)
        # print(self.config.num_attention_heads)
        # print(hidden_states[0])
        # print(hidden_states.shape)


        # block_size = 16
        # num_blocks = sparse_condition.size(2) // block_size # 3 
        block_dimensions = 0
        if sparse_condition is not None:
            block_dimensions = sparse_condition.size(2)
        seq_size = 0  # 4
        if attn_metadata:
            seq_size = attn_metadata.num_decode_tokens + attn_metadata.num_prefills
        # ??
        if seq_size == 256:
            seq_size = 4

        head_size = self.config.num_attention_heads # 12 ?
        percentage = 1.0 # 0.5 # ??

        # sparse_condition_size = len(self.layers) * seq_size * block_size
        # if sparse_condition is None: # ??
        #     sparse_condition = torch.zeros((len(self.layers), seq_size, block_dimensions), dtype=torch.int64)
        # else:
        #     sparse_condition_size = sparse_condition.size(0)
        # print("WWWWWWWWWWWW")
        if sparse_condition is not None:
            sparse_condition.zero_()
            # print(sparse_condition.shape)
            # print(sparse_condition)

        def find_increasing_subsequences_lengths(tensor: torch.Tensor):
            lengths = []
            start_index = 0
            for i in range(1, len(tensor)):
                if tensor[i] == 0:
                    lengths.append(i - start_index)
                    start_index = i
            lengths.append(len(tensor) - start_index)
            return lengths
        
        # temp_sparse_condition = torch.zeros(sparse_condition_size, dtype=torch.float16)
        # print(sparse_condition)
        #print("QUTAMADE")
        for i in range(len(self.layers)):
            #print("QUTAMA")
            # print(sparse_condition)
            layer = self.layers[i]
            # print("OPT Layers" + str(i))
            # Here we change the sparse_condition from 1d to 3d.
            layer_sparse_condition = torch.zeros(seq_size * block_dimensions * head_size, dtype=torch.float32)
            # split layers to do
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata, layer_sparse_condition)
            #print(layer_sparse_condition)
            layer_sparse_condition_2d = layer_sparse_condition.view(seq_size, block_dimensions, head_size).mean(dim=2) # 4 * 16
            #print(layer_sparse_condition_2d)
            # temp_sparse_condition = (temp_sparse_condition * i + layer_sparse_condition) / (i + 1)
            # print(layer_sparse_condition_2d)
            #print("QUTA")
            # print(i)
            # print(positions)
            # print(sparse_condition)
            for j in range(seq_size):
                num_to_select = math.ceil(block_dimensions * percentage) # ????
                # 1. sparse condition continue for multiple rounds
                n_times = self.n_times
                # print("NTNTNT" + str(n_times))
                # print(sparse_condition)
                if n_times >= 0:
                    percentage = 1.0 # 0.5 # ??
                    step = 20 # ??
                    times = n_times // step
                    temp = positions[j] + 1 - n_times # after inserting one token
                    temp = math.floor(temp * percentage)
                    for _ in range(times):
                        temp += step
                        temp = math.floor(temp * percentage)
                    num_to_select = temp + n_times % step
                # print("NUMBERNUMBER" + str(num_to_select))
                # print(sparse_condition)
                # print("RRRRNUMBERNUMBER")
                if num_to_select <= 0 or layer_sparse_condition_2d[j].size(0) == 0:
                    continue
                # print(layer_sparse_condition_2d[j].shape)
                #print(layer_sparse_condition_2d[j])
                # print(num_to_select-1)
                _, top_indices = torch.topk(torch.abs(layer_sparse_condition_2d[j][1:]), num_to_select-1) #?? absolute value
                # print(sparse_condition)
                # print(top_indices)
                is_all_zero = torch.all(layer_sparse_condition_2d[j] == 0)
                # print("ALL zero")
                if is_all_zero: # prefill case ??
                    temp_length = find_increasing_subsequences_lengths(positions)
                    # print("TTTTTTTT")
                    # print(temp_length)
                    # print("zzzzzero")
                    # print(j)
                    # print(len(temp_length))
                    #num_to_select = math.floor(temp_length[j] * percentage)
                    num_to_select = temp_length[j]
                    print(num_to_select)
                    top_indices = [i for i in range(num_to_select)]
                # print(num_to_select)
                # print(top_indices)
                # print("IIIIIIIIIIIIIIIIIIIIIIIIIII:" + str(i))
                # print(top_indices)
                # print(block_dimensions)
                # print(sparse_condition)
                for k in range(block_dimensions):
                    if k == 0: # starting token
                        # print(i)
                        # print(j)
                        # print(k)
                        sparse_condition[i, j, k] = 1
                    if k in top_indices:
                        # print("KKKK " + str(k))
                        sparse_condition[i, j, k] = 1
                    # else:
                    #     sparse_condition[i, j, k] = 0
                
        #print("WTF man")
        # print(sparse_condition)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        self.n_times += 1
        return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # print("OPT Model")
        # print(sparse_condition)
        # if sparse_condition is None: # ??
        #     print("YYYYYYYYYYYYYY")
        #     sparse_condition = torch.zeros((12, 4, 16 * 3), dtype=torch.int64)
        #     #sparse_condition = torch.zeros((12, 4, 16), dtype=torch.int64)
        t = self.decoder(input_ids, positions, kv_caches, attn_metadata, sparse_condition)
        # print("OPT Model done")
        # print(sparse_condition)
        return t


class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(config, quant_config)
        self.lm_head_weight = self.model.decoder.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # if sparse_condition is None:
        #     print("XXXXXXXXXXXXXXXXXXX")
        #     #sparse_condition = torch.zeros((12, 4, 16), dtype=torch.int64)
        #     sparse_condition = torch.zeros((12, 4, 16 * 3), dtype=torch.int64) #??
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, sparse_condition)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
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
            if "lm_head.weight" in name:
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
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
