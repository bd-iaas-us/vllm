import math
import hashlib
import logging
from typing import Dict, List, Tuple
import torch
import os

import infinistore

from vllm.attention import AttentionMetadata
from vllm.distributed.kv_transfer.base import KVCacheTransporterBase

logger = logging.getLogger(__name__)

Default_Infinite_Server = "127.0.0.1"

class InfiniStoreKVCacheTransporter(KVCacheTransporterBase):
    #Class-level singleton connection instance
    _singleton_conn = None

    def __init__(self, model: str, tokens_per_page=16) -> None:
        if not model:
            raise ValueError("model cannot be empty.")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be greater than 0.")
        
        self.model = model
        self.tokens_per_page = tokens_per_page

        # TODO: when server is local, use connection_type=infinistore.TYPE_GPU, otherwise RDMA

        infinite_server = os.environ.get("INFINITE_STORE_SERVER", Default_Infinite_Server)
        infinite_server = infinite_server.strip('"')
        if InfiniStoreKVCacheTransporter._singleton_conn is None:
            infinte_config = infinistore.ClientConfig(
                host_addr=infinite_server,
                service_port=22345,
                log_level="info",
                connection_type=infinistore.TYPE_RDMA,
                ib_port=1,
                link_type="IB",
                dev_name="mlx5_1",
            )
            InfiniStoreKVCacheTransporter._singleton_conn = infinistore.InfinityConnection(infinte_config)
            logger.info("Connecting to infinite store server: %s", infinite_server)

            InfiniStoreKVCacheTransporter._singleton_conn.connect()

        # Assign the singleton connection to the instance attribute
        self.conn = InfiniStoreKVCacheTransporter._singleton_conn


    def _compute_kv_cache_block_offsets(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_seq_lengths: List[int],
        slot_mapping: torch.Tensor,
        layer_idx: int,
        kv_cache: torch.Tensor
    ) -> Tuple[List[Tuple[str, int]], int]:
        """
        Compute the block offsets in the kv_cache for multiple sequences.

        Args:
            prompt_token_ids (torch.Tensor): Token IDs of all sequences concatenated.
            prompt_seq_lengths (List[int]): List of sequence lengths for each prompt.
            slot_mapping (torch.Tensor): Slot mapping for each token.
            layer_idx (int): The index of the layer for the key-value cache.
            kv_cache (torch.Tensor): The key-value cache tensor.

        Returns:
            Tuple[List[Tuple[str, int]], int]: A list of tuples with cache keys and offsets, and the page size.
        """
        block_offsets: List[Tuple[str, int]] = []
        seq_index = 0
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel()  # Size of key or value cache per token

        # Loop over each sequence length
        for seq_length in prompt_seq_lengths:
            seq_tokens = prompt_token_ids[seq_index:seq_index + seq_length].cpu().numpy()
            seq_len = len(seq_tokens)
            num_pages = math.ceil(seq_len / self.tokens_per_page)
            prev_hash = ""

            # Loop over each page within the current sequence
            for page_num in range(num_pages):
                start_token = page_num * self.tokens_per_page
                end_token = min((page_num + 1) * self.tokens_per_page, seq_len)
                tokens_in_page = seq_tokens[start_token:end_token]

                # Compute hash for the current page
                tokens_bytes = tokens_in_page.tobytes()
                hash_input = prev_hash.encode('utf-8') + tokens_bytes
                current_hash = hashlib.sha256(hash_input).hexdigest()

                # Generate cache keys for the current page
                k_cache_key = f"{self.model}_{current_hash}_layer_{layer_idx}_k"
                v_cache_key = f"{self.model}_{current_hash}_layer_{layer_idx}_v"

                # Calculate offset in the kv_cache
                try:
                    slot_mapping_value = slot_mapping[seq_index + start_token].item()
                    page_offset = (slot_mapping_value // self.tokens_per_page) * page_size
                except IndexError as e:
                    logger.error("Invalid slot mapping index %s: %s", seq_index + start_token, e)
                    raise

                block_offsets.append((k_cache_key, page_offset))
                block_offsets.append((v_cache_key, page_offset + k_or_v_cache_size))

                # Update previous hash for the next page
                prev_hash = current_hash

                logger.debug(
                    "Computed kv_cache block offsets: layer %s, page %s, "
                    "k_cache_key %s, v_cache_key %s", layer_idx, page_num,
                    k_cache_key, v_cache_key
                )

            # Update seq_index to move to the next sequence in prompt_seq_lengths
            seq_index += seq_length

        return block_offsets, page_size


    def _compute_hidden_states_block_offsets(
            self, input_ids: torch.Tensor, attn_metadata: AttentionMetadata,
            seq_index: int, seq_length: int,
            hidden_states: torch.Tensor) -> Dict[int, List[Tuple[str, int]]]:

        seq_tokens = input_ids[seq_index:seq_index + seq_length].cpu().numpy()
        num_pages = math.ceil(seq_length / self.tokens_per_page)
        block_offsets: Dict[int, List[Tuple[str, int]]] = {}
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
            cache_key = f"{self.model}_{current_hash}_hidden_states"

            # Calculate cache size and offset
            cache_size = hidden_size * (end_token - start_token)
            offset = (seq_index + start_token) * hidden_size

            if cache_size not in block_offsets:
                block_offsets[cache_size] = []
            block_offsets[cache_size].append((cache_key, offset))

            # Update the previous hash for the next page
            prev_hash = current_hash

            logger.debug(
                "Computed hidden_states block offsets: page %s, cache_key %s",
                page_num, cache_key)

        return block_offsets

    def save_kv_cache(self, prompt_token_ids: torch.Tensor,
                      prompt_seq_lengths: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_ids,
            prompt_seq_lengths,
            slot_mapping,
            layer_idx,
            kv_cache)
        
        try:
            import time
            start = time.time()
            self.conn.write_cache(kv_cache, block_offsets, page_size)
            logger.info("tocheck: ~~~~~~~~ write_cache time: %s", time.time() - start)
            
        except Exception as e:
            logger.error("Failed to write kv_cache: %s", e)
            raise

        logger.debug("Saved kv_cache for layer %s", layer_idx)

    def read_kv_cache(self, prompt_token_ids: torch.Tensor,
                      prompt_seq_lengths: torch.Tensor, slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_ids,
            prompt_seq_lengths,
            slot_mapping,
            layer_idx,
            kv_cache)

        try:
            import time
            start = time.time()
            self.conn.read_cache(kv_cache, block_offsets, page_size)
            logger.info("tocheck: ~~~~~~~~ read_cache time: %s", time.time() - start)
            
        except Exception as e:
            logger.error("Failed to read kv_cache: %s", e)
            raise

        logger.debug("Loaded kv_cache for layer %s", layer_idx)

    def save_hidden_states(self, input_ids: torch.Tensor,
                           attn_metadata: AttentionMetadata,
                           hidden_states: torch.Tensor) -> None:

        seq_index = 0
        self.conn.register_mr(hidden_states)

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets = self._compute_hidden_states_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, hidden_states)

            # Write to cache
            try:
                for cache_size, offsets in block_offsets.items():
                    self.conn.write_cache(hidden_states, offsets, cache_size)
            except Exception as e:
                logger.error("Failed to write hidden_states: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Saved hidden_states")

    def read_hidden_states(self, input_ids: torch.Tensor,
                           attn_metadata: AttentionMetadata,
                           hidden_states: torch.Tensor) -> None:

        seq_index = 0
        self.conn.register_mr(hidden_states)

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets = self._compute_hidden_states_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, hidden_states)

            # Read from cache
            try:
                for cache_size, offsets in block_offsets.items():
                    self.conn.read_cache(hidden_states, offsets, cache_size)
            except Exception as e:
                logger.error("Failed to read hidden_states: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Loaded hidden_states")

    def synchronize(self) -> None:
        try:
            self.conn.sync()
            logger.debug("Synchronized with Infinity service")
        except Exception as e:
            logger.error("Failed to synchronize: %s", e)
            raise
