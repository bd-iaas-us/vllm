import os
import torch
from typing import List, Optional

from vllm.attention import AttentionMetadata


def _get_pd_sep_stage():
    return os.environ.get("PD_SEPARATE_STAGE", "").lower()


def _is_profile_run(input_ids: torch.Tensor):
    # profile_run will send in an all-zero input_ids tensor
    return torch.any(input_ids == 0).item()


def _is_first_pass(attn_metadata: AttentionMetadata):

    return (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None)

def _is_kv_cache_zero(kv_cache: torch.Tensor, attn_metadata: AttentionMetadata):

    if kv_cache is None:
        return False

    first_kv_block_index = attn_metadata.block_tables[0][0].item()
    first_k_block = kv_cache[0][0][first_kv_block_index]
    all_zero = torch.all(first_k_block == 0).item()

    return all_zero

def is_decode_run(input_ids: torch.Tensor):
    if _get_pd_sep_stage() != "decode":
        return False

    return not _is_profile_run(input_ids)

def is_prefill_run(input_ids: torch.Tensor):
    if _get_pd_sep_stage() != "prefill":
        return False

    return not _is_profile_run(input_ids)

def is_first_decode_pass(input_ids: torch.Tensor, attn_metadata: AttentionMetadata):
    if not is_decode_run(input_ids):
        return False

    return _is_first_pass(attn_metadata)

def is_second_decode_pass(input_ids: torch.Tensor, attn_metadata: AttentionMetadata, kv_caches: Optional[List[torch.Tensor]]):
    if not is_decode_run(input_ids):
        return False

    if _is_first_pass(attn_metadata):
        return False
    
    return _is_kv_cache_zero(kv_caches, attn_metadata)


