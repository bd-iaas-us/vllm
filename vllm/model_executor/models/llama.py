# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip

from .interfaces import SupportsLoRA
from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers

import os
import hashlib
import math

from infinity import InfinityConnection
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
import torch


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def save_kv_caches(self, input_ids: torch.Tensor,
                   attn_metadata: AttentionMetadata, layer_idx: int,
                   kv_cache: torch.Tensor, conn: InfinityConnection):
    
        seq_index = 0
        tokens_per_page = 16  # Adjust this value as needed
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel()  # Size of key or value cache per token

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            prev_hash = ""

            seq_length = seq_length_tensor.item()
            sequence = input_ids[seq_index:seq_index + seq_length]
            seq_tokens = sequence.cpu().numpy()

            num_pages = math.ceil(seq_length / tokens_per_page)

            block_offsets = []

            for page_num in range(num_pages):
                # Calculate the token indices for the current page
                start_token = page_num * tokens_per_page
                end_token = min((page_num + 1) * tokens_per_page, seq_length)
                tokens_in_page = seq_tokens[start_token:end_token]

                # Compute the hash for the current page
                tokens_bytes = tokens_in_page.tobytes()
                hash_input = prev_hash.encode('utf-8') + tokens_bytes
                current_hash = hashlib.sha256(hash_input).hexdigest()

                # Generate cache keys using the current hash
                k_cache_key = f"{current_hash}_layer_{layer_idx}_k"
                v_cache_key = f"{current_hash}_layer_{layer_idx}_v"

                # Calculate the offset in the kv_cache for the current page
                page_offset = attn_metadata.slot_mapping[page_num*tokens_per_page].item() // tokens_per_page * page_size

                block_offsets.append((k_cache_key, page_offset))
                block_offsets.append((v_cache_key, page_offset + k_or_v_cache_size))

                # Update the previous hash for the next page
                prev_hash = current_hash

                print(f"write: layer {layer_idx} page {page_num} k_cache_key {k_cache_key} v_cache_key {v_cache_key}")

            # write to cache
            conn.write_kvcache(kv_cache, block_offsets, page_size)
                               
            seq_index += seq_length  # Update sequence index for the next sequence

            # print(f"Saved kv_cache for layer {layer_idx}")

    def load_kv_caches(self, input_ids: torch.Tensor,
                   attn_metadata: AttentionMetadata, layer_idx: int,
                   kv_cache: torch.Tensor, conn: InfinityConnection):

        seq_index = 0
        tokens_per_page = 16  # Should match the value used in save_kv_caches
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel()  # Size of key or value cache per token

        block_offsets = []
        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            prev_hash = ""  # Initialize the previous hash as an empty string

            seq_length = seq_length_tensor.item()
            sequence = input_ids[seq_index:seq_index + seq_length]
            seq_tokens = sequence.cpu().numpy()

            num_pages = math.ceil(seq_length / tokens_per_page)

            for page_num in range(num_pages):
                # Calculate the token indices for the current page
                start_token = page_num * tokens_per_page
                end_token = min((page_num + 1) * tokens_per_page, seq_length)
                tokens_in_page = seq_tokens[start_token:end_token]

                # Compute the hash for the current page
                tokens_bytes = tokens_in_page.tobytes()
                hash_input = prev_hash.encode('utf-8') + tokens_bytes
                current_hash = hashlib.sha256(hash_input).hexdigest()

                # Generate cache keys using the current hash
                k_cache_key = f"{current_hash}_layer_{layer_idx}_k"
                v_cache_key = f"{current_hash}_layer_{layer_idx}_v"

               # Calculate the offset in the kv_cache for the current page
                page_offset = attn_metadata.slot_mapping[page_num*tokens_per_page].item() // tokens_per_page * page_size

                block_offsets.append((k_cache_key, page_offset))
                block_offsets.append((v_cache_key, page_offset + k_or_v_cache_size))

                # Update the previous hash for the next page
                prev_hash = current_hash

                print(f"read: layer {layer_idx} page {page_num} k_cache_key {k_cache_key} v_cache_key {v_cache_key}")

            conn.read_kvcache(kv_cache, block_offsets, page_size)
            seq_index += seq_length  # Update sequence index for the next sequence

            print(f"Loaded kv_cache for layer {layer_idx}")


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        print("~~~~~~~~~~Running input_ids", input_ids)

        # profile_run will send in an all-zero input_ids tensor
        execute_run = torch.any(input_ids != 0).item()
        pd_stage = os.environ.get("pd_separate_stage", "").lower()

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        first_decode_pass = False
        if pd_stage == "decode" and execute_run:
            original_input_ids = input_ids.clone()
            if kv_caches is not None and len(
                    kv_caches) > 0 and kv_caches[0] is not None:
                contain_nonzero = torch.any(kv_caches[0]).item()
                first_decode_pass = not contain_nonzero

        prefill_execute_run = False
        if pd_stage == "prefill" and execute_run:
            prefill_execute_run = True


        if first_decode_pass or prefill_execute_run:
            conn = InfinityConnection()
            conn.connect()

        if first_decode_pass:
            prefill_input_ids = original_input_ids[0:len(original_input_ids) - 1]
            # make it return the block_table
            for i in range(self.start_layer, self.end_layer):
                self.load_kv_caches(prefill_input_ids, attn_metadata, i,
                                    kv_caches[i - self.start_layer], conn)
            
            input_ids = original_input_ids[-1:]
            positions = positions[-1:]
            hidden_states = hidden_states[-1:]

            page_size =16
            indices = torch.arange(0, attn_metadata.slot_mapping.size(0), page_size)
            block_tables =  attn_metadata.slot_mapping[indices] //page_size
            block_tables = block_tables.unsqueeze(0)
            block_tables = block_tables.to(dtype=torch.int32, device=attn_metadata.block_tables.device)
            # last_block_index = attn_metadata.slot_mapping[-1]//page_size
            # block_tables = torch.tensor([[last_block_index]], dtype=torch.int32, device = attn_metadata.block_tables.device)

            new_metadata = FlashAttentionMetadata(
                seq_lens=attn_metadata.seq_lens,
                seq_lens_tensor=attn_metadata.seq_lens_tensor,
                max_query_len=1,  
                max_prefill_seq_len=0,  
                max_decode_seq_len=attn_metadata.max_prefill_seq_len,    
                query_start_loc=torch.tensor([0, 1], device=hidden_states.device, dtype=torch.int32), 
                seq_start_loc=attn_metadata.seq_start_loc, 
                context_lens_tensor=torch.tensor([attn_metadata.max_prefill_seq_len-1], dtype=attn_metadata.context_lens_tensor.dtype, device = attn_metadata.context_lens_tensor.device),  
                block_tables=block_tables,   # need to change
                use_cuda_graph  =attn_metadata.use_cuda_graph,  
                num_decode_tokens=1,
                num_prefill_tokens=0,
                num_prefills=0,
                slot_mapping=attn_metadata.slot_mapping[-1:],
            )

            attn_metadata = new_metadata

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

            if prefill_execute_run:
                self.save_kv_caches(input_ids, attn_metadata, i,
                                    kv_caches[i - self.start_layer], conn)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        if first_decode_pass or prefill_execute_run:
            conn.sync_local()

        return hidden_states


class LlamaForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output

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

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name, loaded_weight = self.maybe_remap_mistral(name, loaded_weight)

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
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
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
            self, name: str,
            loaded_weight: torch.Tensor) -> Tuple[str, torch.Tensor]:

        def permute(w, n_heads):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
