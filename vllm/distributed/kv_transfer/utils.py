from enum import Enum
import math
import os
import torch
from typing import List
import hashlib

from vllm.attention import AttentionMetadata

PAGE_SIZE = 16

class pd_separate_stage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ForwardPassType(Enum):
    PREFILL = "prefill_pass"
    FIRST_DECODE = "first_decode_pass"
    REGULAR = "regular_pass"


def get_forward_pass_type(input_ids: torch.Tensor, attn_metadata: AttentionMetadata):
    pd_stage = os.environ.get("PD_SEPARATE_STAGE", "").lower()
    is_profile_run = torch.any(input_ids == 0).item()
    if pd_stage not in pd_separate_stage._value2member_map_ or is_profile_run:
        return ForwardPassType.REGULAR

    if pd_stage == "prefill":
        return ForwardPassType.PREFILL
    else:
        if (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None):
            return ForwardPassType.FIRST_DECODE

    return ForwardPassType.REGULAR


def prepare_kv_cache_transport(input_ids, attn_metadata, cache_config, kwargs):

    fp_type = get_forward_pass_type(input_ids, attn_metadata)

    input_token_hashes = []
    if fp_type in (ForwardPassType.PREFILL, ForwardPassType.FIRST_DECODE):
        input_token_hashes = compute_token_page_hashes(input_ids,
                                                       attn_metadata.seq_lens)
        
        # print("Qian_____________________________________________")
        # covered_pages = 0
        # start = 0
        # for seqlen in attn_metadata.seq_lens:
        #     if seqlen == 1456:
        #         print("Qian ----- hit the 1456 seq_len")
        #     print("Qian check hash-----seq_len: ", seqlen)
        #     print("Qian check hash-----input_ids: ", input_ids[start:start + seqlen])
        #     start += seqlen
        #     page_len = math.ceil(seqlen / PAGE_SIZE)
        #     print("Qian check hash-----input_token_hashes: ", input_token_hashes[covered_pages: covered_pages + page_len])
        #     covered_pages += page_len

    return fp_type, cache_config.kv_cache_transporter, input_token_hashes


def finalize_kv_cache_transport(fp_type, kv_cache_transporter,
                                input_token_hashes, attn_metadata,
                                hidden_states):
    if fp_type == ForwardPassType.PREFILL:
        kv_cache_transporter.save_hidden_states(input_token_hashes,
                                                attn_metadata.seq_lens,
                                                hidden_states)

        kv_cache_transporter.synchronize()
        kv_cache_transporter.publish_hidden_states_ready(input_token_hashes, attn_metadata.seq_lens)

    return

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
