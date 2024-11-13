import math
import os
import torch
from typing import List
import hashlib

from vllm.attention import AttentionMetadata

PAGE_SIZE = 16


def _get_pd_sep_stage():
    return os.environ.get("PD_SEPARATE_STAGE", "").lower()


def _is_profile_run(input_ids: torch.Tensor):
    # profile_run will send in an all-zero input_ids tensor
    return torch.any(input_ids == 0).item()


def _is_first_pass(attn_metadata: AttentionMetadata):

    return (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None)


def is_decode_run(input_ids: torch.Tensor):
    if _get_pd_sep_stage() != "decode":
        return False

    return not _is_profile_run(input_ids)


def is_prefill_run(input_ids: torch.Tensor):
    if _get_pd_sep_stage() != "prefill":
        return False

    return not _is_profile_run(input_ids)


def is_first_decode_pass(input_ids: torch.Tensor,
                         attn_metadata: AttentionMetadata):
    if not is_decode_run(input_ids):
        return False

    return _is_first_pass(attn_metadata)


def compute_token_page_hashes(prompt_token_ids: torch.Tensor,
                              prompt_seq_lengths: List[int],
                              tokens_per_page=PAGE_SIZE) -> List[str]:

    hashes = []
    seq_index = 0

    for seq_len in prompt_seq_lengths:
        seq_tokens = prompt_token_ids[seq_index:seq_index +
                                      seq_len].cpu().numpy()
        num_pages = math.ceil(seq_len / tokens_per_page)
        prev_hash = ""

        # Loop over each page within the current sequence
        for page_num in range(num_pages):
            start_token = page_num * tokens_per_page
            end_token = min((page_num + 1) * tokens_per_page, seq_len)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            hashes.append(current_hash)

        seq_index += seq_len

    return hashes
