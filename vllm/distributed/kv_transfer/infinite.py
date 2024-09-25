import torch
import math
import hashlib
import logging
from typing import Dict, List, Tuple
import os

from infinity import InfinityConnection

from vllm.distributed.kv_transfer.base import KVCacheTransporterBase
from vllm.attention import AttentionMetadata

logger = logging.getLogger(__name__)

class InfiniStoreKVCacheTransporter(KVCacheTransporterBase):
    def __init__(self, model_name: str, tokens_per_page: int = 16):
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be greater than 0.") 

        self.model_name = model_name
        self.tokens_per_page = tokens_per_page
        self.conn = InfinityConnection()
        self.conn.connect()

    def _compute_kv_cache_block_offsets(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        seq_index: int,
        seq_length: int,
        layer_idx: int,
        kv_cache: torch.Tensor
    ) -> Tuple[List[Tuple[str, int]], int]:
        """Compute block offsets and cache keys for kv_cache."""
        seq_tokens = input_ids[seq_index : seq_index + seq_length].cpu().numpy()
        num_pages = math.ceil(seq_length / self.tokens_per_page)
        block_offsets = []
        prev_hash = ""
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel()  # Size of key or value cache per token

        for page_num in range(num_pages):
            # Calculate token indices for the current page
            start_token = page_num * self.tokens_per_page
            end_token = min((page_num + 1) * self.tokens_per_page, seq_length)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute the hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            # Generate cache keys using the current hash
            k_cache_key = f"{self.model_name}_{current_hash}_layer_{layer_idx}_k"
            v_cache_key = f"{self.model_name}_{current_hash}_layer_{layer_idx}_v"

            # Calculate the offset in the kv_cache for the current page
            try:
                slot_index = page_num * self.tokens_per_page
                slot_mapping_value = attn_metadata.slot_mapping[seq_index + slot_index].item()
                page_offset = (slot_mapping_value // self.tokens_per_page) * page_size
            except IndexError as e:
                logger.error(f"Invalid slot mapping index {slot_index}: {e}")
                raise

            block_offsets.append((k_cache_key, page_offset))
            block_offsets.append((v_cache_key, page_offset + k_or_v_cache_size))

            # Update the previous hash for the next page
            prev_hash = current_hash

            logger.debug(
                f"Computed kv_cache block offsets: layer {layer_idx}, page {page_num}, "
                f"k_cache_key {k_cache_key}, v_cache_key {v_cache_key}"
            )

        return block_offsets, page_size

    def _compute_hidden_states_block_offsets(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        seq_index: int,
        seq_length: int,
        hidden_states: torch.Tensor
    ) -> Dict[int, List[Tuple[str, int]]]:
        """Compute block offsets and cache keys for hidden_states."""
        seq_tokens = input_ids[seq_index : seq_index + seq_length].cpu().numpy()
        num_pages = math.ceil(seq_length / self.tokens_per_page)
        block_offsets = {}
        prev_hash = ""
        hidden_size = hidden_states.size(-1)

        for page_num in range(num_pages):
            # Calculate token indices for the current page
            start_token = page_num * self.tokens_per_page
            end_token = min((page_num + 1) * self.tokens_per_page, seq_length)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute the hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            # Generate cache key using the current hash
            cache_key = f"{self.model_name}_{current_hash}_hidden_states"

            # Calculate cache size and offset
            cache_size = hidden_size * (end_token - start_token)
            offset = (seq_index + start_token) * hidden_size

            if cache_size in block_offsets:
                block_offsets[cache_size].append((cache_key, offset))
            else:
                block_offsets[cache_size] = [(cache_key, offset)]

            # Update the previous hash for the next page
            prev_hash = current_hash

            logger.debug(
                f"Computed hidden_states block offsets: page {page_num}, cache_key {cache_key}"
            )

        return block_offsets

    def save_kv_cache(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_idx: int,
        kv_cache: torch.Tensor,
    ):
        """Save kv_cache to Infinity store."""
        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets, page_size = self._compute_kv_cache_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, layer_idx, kv_cache
            )

            # Write to cache
            try:
                self.conn.write_kvcache(kv_cache, block_offsets, page_size)
            except Exception as e:
                logger.error(f"Failed to write kv_cache: {e}")
                raise

            seq_index += seq_length  # Update sequence index for the next sequence

        logger.debug(f"Saved kv_cache for layer {layer_idx}")

    def read_kv_cache(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_idx: int,
        kv_cache: torch.Tensor,
    ):
        """Read kv_cache from Infinity service."""
        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets, page_size = self._compute_kv_cache_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, layer_idx, kv_cache
            )

            # Read from cache
            try:
                self.conn.read_kvcache(kv_cache, block_offsets, page_size)
            except Exception as e:
                logger.error(f"Failed to read kv_cache: {e}")
                raise

            seq_index += seq_length  # Update sequence index for the next sequence

        logger.debug(f"Loaded kv_cache for layer {layer_idx}")

    # def save_hidden_states(
    #     self,
    #     input_ids: torch.Tensor,
    #     attn_metadata: AttentionMetadata,
    #     hidden_states: torch.Tensor,
    # ):
    #     """Save hidden_states to Infinity service."""
    #     seq_index = 0

    #     for seq_length_tensor in attn_metadata.seq_lens_tensor:
    #         seq_length = seq_length_tensor.item()
    #         block_offsets = self._compute_hidden_states_block_offsets(
    #             input_ids, attn_metadata, seq_index, seq_length, hidden_states
    #         )

    #         # Write to cache
    #         try:
    #             for cache_size, offsets in block_offsets.items():
    #                 self.conn.write_kvcache(hidden_states, offsets, cache_size)
    #         except Exception as e:
    #             logger.error(f"Failed to write hidden states: {e}")
    #             raise

    #         seq_index += seq_length  # Update sequence index for the next sequence

    #     logger.debug("Saved hidden states")

    # def read_hidden_states(
    #     self,
    #     input_ids: torch.Tensor,
    #     attn_metadata: AttentionMetadata,
    #     hidden_states: torch.Tensor,
    # ):
    #     """Read hidden_states from Infinity service."""
    #     seq_index = 0

    #     for seq_length_tensor in attn_metadata.seq_lens_tensor:
    #         seq_length = seq_length_tensor.item()
    #         block_offsets = self._compute_hidden_states_block_offsets(
    #             input_ids, attn_metadata, seq_index, seq_length, hidden_states
    #         )

    #         # Read from cache
    #         try:
    #             for cache_size, offsets in block_offsets.items():
    #                 self.conn.read_kvcache(hidden_states, offsets, cache_size)
    #         except Exception as e:
    #             logger.error(f"Failed to read hidden states: {e}")
    #             raise

    #         seq_index += seq_length  # Update sequence index for the next sequence

    #     logger.debug("Loaded hidden states")

    def save_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """Save hidden_states to disk files."""
        seq_index = 0

        for seq_length in attn_metadata.seq_lens_tensor:
            sequence = input_ids[seq_index:seq_index + seq_length.item()]
            seq_bytes = sequence.cpu().numpy().tobytes()
            seq_hash = hashlib.sha256(seq_bytes).hexdigest()

            folder_path = "/tmp/vllm_vllm_hidden_states"
            os.makedirs(folder_path, exist_ok=True)

            file_name = f"{seq_hash}.hidden_states"
            torch.save(hidden_states[seq_index:seq_index + seq_length, ...],
                       os.path.join(folder_path, file_name))
            seq_index += seq_length.item()


    def read_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """Read hidden_states from disk files."""
        seq_index = 0

        folder_path = "/tmp/vllm_vllm_hidden_states"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"cache folder does not exist: {folder_path}")

        for seq_length in attn_metadata.seq_lens_tensor:
            sequence = input_ids[seq_index:seq_index + seq_length.item()]
            seq_bytes = sequence.cpu().numpy().tobytes()
            seq_hash = hashlib.sha256(seq_bytes).hexdigest()

            file_name = f"{seq_hash}.hidden_states"
            hidden_states_path = os.path.join(folder_path, file_name)
            if not os.path.exists(hidden_states_path):
                raise FileNotFoundError(
                    f"Hidden states file does not exist: {hidden_states_path}")

            loaded_hidden_states = torch.load(hidden_states_path)
            hidden_states[seq_index:seq_index + seq_length,
                          ...] = loaded_hidden_states
            
            seq_index += seq_length.item()

    def synchronize(self):
        """Synchronize with Infinity service."""
        try:
            self.conn.sync_local()
            logger.debug("Synchronized with Infinity service")
        except Exception as e:
            logger.error(f"Failed to synchronize: {e}")
            raise