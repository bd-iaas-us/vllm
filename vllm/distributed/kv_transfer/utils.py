from enum import Enum
import math
import os
import torch
from typing import List
import hashlib

from vllm.attention import AttentionMetadata

PAGE_SIZE = 16

kv_cache_decode_data = {}


class pd_separate_stage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ForwardPassCategory(Enum):
    PREFILL = "prefill_pass"
    DECODE = "decode_pass"
    REGULAR = "regular_pass"


class DecodeForwardClassType(Enum):
    FIRST_DECODE = "first_decode_pass"
    SECOND_DECODE = "second_decode_pass"
    FOLLOWING_DECODE = "following_decode_pass"


class ForwardPassType(Enum):
    PREFILL = "prefill_pass"
    FIRST_DECODE = "first_decode_pass"
    SECOND_DECODE = "second_decode_pass"
    FOLLOWING_DECODE = "following_decode_pass"
    REGULAR = "regular_pass"


def map_to_forward_pass_type(input_type: Enum) -> ForwardPassType:
    # Check if input is a DecodeForwardClassType
    if isinstance(input_type, DecodeForwardClassType):
        # Map DecodeForwardClassType to ForwardPassType by value
        for pass_type in ForwardPassType:
            if pass_type.value == input_type.value:
                return pass_type
    # Check if input is a ForwardPassCategory
    elif isinstance(input_type, ForwardPassCategory):
        # Handle the DECODE category specifically, mapping to FIRST_DECODE
        if input_type == ForwardPassCategory.DECODE:
            return ForwardPassType.FIRST_DECODE  # or any other default decode type
        else:
            # Map other ForwardPassCategory values by matching string value
            for pass_type in ForwardPassType:
                if pass_type.value == input_type.value:
                    return pass_type
    # If no match is found, raise an error
    raise ValueError(f"No matching ForwardPassType found for {input_type}")


def get_forward_pass_type(input_ids: torch.Tensor):
    pd_stage = os.environ.get("PD_SEPARATE_STAGE", "").lower()
    is_profile_run = torch.any(input_ids == 0).item()
    if pd_stage not in pd_separate_stage._value2member_map_ or is_profile_run:
        return ForwardPassCategory.REGULAR

    if pd_stage == "prefill":
        return ForwardPassCategory.PREFILL

    return ForwardPassCategory.DECODE


def is_decode_run(input_ids: torch.Tensor):
    return get_forward_pass_type(input_ids) == ForwardPassCategory.DECODE


def get_decode_forward_pass_type(attn_metadata: AttentionMetadata,
                                 request_ids: List[str]):
    global kv_cache_decode_data
    if (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None):
        return DecodeForwardClassType.FIRST_DECODE

    if request_ids[0] in kv_cache_decode_data:
        return DecodeForwardClassType.SECOND_DECODE

    return DecodeForwardClassType.FOLLOWING_DECODE


def prepare_kv_cache_transport(input_ids, attn_metadata, cache_config, kwargs):
    global kv_cache_decode_data

    fp_type = get_forward_pass_type(input_ids)
    if fp_type == ForwardPassCategory.REGULAR:
        return ForwardPassType.REGULAR, None, [], [], None, []

    if cache_config is None or cache_config.kv_cache_transporter is None:
        raise ValueError("kv_cache_transporter is None")

    request_ids = kwargs.get("request_ids", [])
    if fp_type == ForwardPassCategory.DECODE:
        if not request_ids:
            raise ValueError("Missing 'request_ids' in keyword arguments.")
        fp_type = get_decode_forward_pass_type(attn_metadata, request_ids)

    fp_type = map_to_forward_pass_type(fp_type)

    if fp_type == ForwardPassType.FIRST_DECODE and len(request_ids) != len(
            attn_metadata.seq_lens):
        raise ValueError(
            "The number of request_ids should be equal to the number of sequences."
        )

    input_token_hashes = []
    if fp_type in (ForwardPassType.PREFILL, ForwardPassType.FIRST_DECODE):
        input_token_hashes = compute_token_page_hashes(input_ids,
                                                       attn_metadata.seq_lens)

    slot_mapping_tensor = None
    seq_lens = None
    if fp_type == ForwardPassType.SECOND_DECODE:
        if missing_ids := [
                req for req in request_ids if req not in kv_cache_decode_data
        ]:
            raise ValueError(
                f"Missing kv cache data for request_ids {missing_ids}")

        prompt_token_ids_tensor = torch.cat([
            torch.tensor(kv_cache_decode_data[req]["prompt_token_ids"])
            for req in request_ids
        ],
                                            dim=0)
        seq_lens = [
            len(kv_cache_decode_data[req]["prompt_token_ids"])
            for req in request_ids
        ]

        input_token_hashes = compute_token_page_hashes(prompt_token_ids_tensor,
                                                       seq_lens)

        slot_mapping_tensor = torch.cat([
            torch.tensor(kv_cache_decode_data[req]["slot_mapping"])
            for req in request_ids
        ],
                                        dim=0)

    return fp_type, cache_config.kv_cache_transporter, request_ids, input_token_hashes, slot_mapping_tensor, seq_lens


def finalize_kv_cache_transport(fp_type, request_ids, kv_cache_transporter,
                                input_token_hashes, attn_metadata,
                                hidden_states):
    if fp_type == ForwardPassType.PREFILL:
        kv_cache_transporter.save_hidden_states(input_token_hashes,
                                                attn_metadata.seq_lens,
                                                hidden_states)

        kv_cache_transporter.synchronize()

    if fp_type == ForwardPassType.SECOND_DECODE:
        [kv_cache_decode_data.pop(req, None) for req in request_ids]

    return


def retrieve_hidden_state(kv_cache_transporter, request_ids, input_ids,
                          input_token_hashes, attn_metadata, hidden_states):
    global kv_cache_decode_data
    kv_cache_transporter.read_hidden_states(input_token_hashes,
                                            attn_metadata.seq_lens,
                                            hidden_states)

    seq_lens = attn_metadata.seq_lens
    seq_index = 0
    for i, request_id in enumerate(request_ids):
        seq_len = seq_lens[i]
        kv_cache_decode_data[request_id] = {
            "prompt_token_ids":
            input_ids[seq_index:seq_index + seq_len],
            "slot_mapping":
            attn_metadata.slot_mapping[seq_index:seq_index + seq_len],
        }
        seq_index += seq_len

    kv_cache_transporter.synchronize()

    # return hidden_states
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
