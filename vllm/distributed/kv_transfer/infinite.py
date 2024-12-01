import math
import logging
from typing import Dict, List, Tuple
import torch
import os

import infinistore

from vllm.distributed.kv_transfer.base import KVCacheTransporterBase
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)

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

        #TODO: when server is local, use connection_type=infinistore.TYPE_GPU,
        # otherwise RDMA

        infinite_server = os.environ.get("INFINITE_STORE_SERVER",
                                         Default_Infinite_Server)
        infinite_server = infinite_server.strip('"')
        if InfiniStoreKVCacheTransporter._singleton_conn is None:
            infinte_config = infinistore.ClientConfig(
                host_addr=infinite_server,
                service_port=22345,
                log_level="info",
                connection_type=infinistore.TYPE_LOCAL_GPU,
                ib_port=1,
                link_type="Ethernet",
                dev_name="mlx5_0",
            )
            InfiniStoreKVCacheTransporter._singleton_conn = infinistore.InfinityConnection(
                infinte_config)
            logger.info("Connecting to infinite store server: %s",
                        infinite_server)

            InfiniStoreKVCacheTransporter._singleton_conn.connect()

        # Assign the singleton connection to the instance attribute
        self.conn = InfiniStoreKVCacheTransporter._singleton_conn

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def _compute_kv_cache_block_offsets(
            self, prompt_token_page_hashes: List[str], seq_lens: List[int],
            slot_mapping: torch.Tensor, layer_idx: int,
            kv_cache: torch.Tensor) -> Tuple[List[Tuple[str, int]], int]:

        block_offsets: List[Tuple[str, int]] = []
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel(
        )  # Size of key or value cache per token

        seq_start_index = 0
        page_start_index = 0
        for seq_length in seq_lens:
            num_pages = math.ceil(seq_length / self.tokens_per_page)

            for page_num in range(num_pages):
                start_token_idx = page_num * self.tokens_per_page
                page_idx = page_num + page_start_index
                current_hash = prompt_token_page_hashes[page_idx]

                # Generate cache keys for the current page
                cache_key = f"{self.model}_{current_hash}_layer_{layer_idx}_tp{self.tp_rank}/{self.tp_size}"
                k_cache_key = cache_key + "_k"
                v_cache_key = cache_key + "_v"

                # Calculate offset in the kv_cache
                try:
                    slot_mapping_value = slot_mapping[seq_start_index +
                                                      start_token_idx].item()
                    page_offset = (slot_mapping_value //
                                   self.tokens_per_page) * page_size
                except IndexError as e:
                    logger.error("Invalid slot mapping index %s: %s", page_idx,
                                 e)
                    raise

                block_offsets.append((k_cache_key, page_offset))
                block_offsets.append(
                    (v_cache_key, page_offset + k_or_v_cache_size))

            seq_start_index += seq_length
            page_start_index += num_pages

        return block_offsets, page_size

    def _compute_hidden_states_block_offsets(
            self, prompt_token_page_hashes: List[str], seq_lens: List[int],
            hidden_states: torch.Tensor) -> Dict[int, List[Tuple[str, int]]]:

        block_offsets: Dict[int, List[Tuple[str, int]]] = {}
        hidden_size = hidden_states.size(-1)

        seq_start_index = 0
        page_start_index = 0
        for seq_length in seq_lens:
            num_pages = math.ceil(seq_length / self.tokens_per_page)

            for page_num in range(num_pages):
                start_token_idx = page_num * self.tokens_per_page
                end_token_idx = min((page_num + 1) * self.tokens_per_page,
                                    seq_length)
                current_hash = prompt_token_page_hashes[page_start_index +
                                                        page_num]

                cache_key = f"{self.model}_{current_hash}_tp{self.tp_rank}/{self.tp_size}_hs"

                cache_size = hidden_size * (end_token_idx - start_token_idx)
                offset = (seq_start_index + start_token_idx) * hidden_size

                if cache_size not in block_offsets:
                    block_offsets[cache_size] = []
                block_offsets[cache_size].append((cache_key, offset))

            seq_start_index += seq_length
            page_start_index += num_pages

        return block_offsets

    def save_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int],
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, slot_mapping,
            layer_idx, kv_cache)

        try:
            self.conn.write_cache(kv_cache, block_offsets, page_size)

        except Exception as e:
            logger.error("Failed to write kv_cache: %s", e)
            raise

        logger.debug("Saved kv_cache for layer %s", layer_idx)

    def read_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int],
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, slot_mapping,
            layer_idx, kv_cache)

        try:
            self.conn.read_cache(kv_cache, block_offsets, page_size)

        except Exception as e:
            logger.error("Failed to read kv_cache: %s", e)
            raise

        logger.debug("Loaded kv_cache for layer %s", layer_idx)

    def save_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        if self.conn.rdma_connected:
            self.conn.register_mr(hidden_states)
        print("1------- register mr done")
        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)
        print("2------- compute offsets")

        try:
            print("2.5 -----------", block_offsets.keys())
            for cache_size, offsets in block_offsets.items():
                print(f"3------- write cache {cache_size}")
                self.conn.write_cache(hidden_states, offsets, cache_size)
                print(f"4------- write cache {cache_size}")
        except Exception as e:
            logger.error("Failed to read hidden_states: %s", e)
            raise

        logger.debug("Saved hidden_states")

    def read_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:
        if self.conn.rdma_connected:
            self.conn.register_mr(hidden_states)
        print("1------- register mr done")
        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)
        print("2------- compute offsets")

        try:
            print("2.5 -----------", block_offsets.keys())
            for cache_size, offsets in block_offsets.items():
                print(f"3------- read cache {cache_size}")
                self.conn.read_cache(hidden_states, offsets, cache_size)
                print(f"4------- read cache {cache_size}")
        except Exception as e:
            logger.error("Failed to read hidden_states: %s", e)
            raise

        logger.debug("Loaded hidden_states")


    def key_exists(self, key: str) -> bool:
        return self.conn.check_exist(key)

    def get_match_last_index(self, keys: List[str]) -> int:
        return self.conn.get_match_last_index(keys)

    def synchronize(self) -> None:
        try:
            self.conn.sync()
            logger.debug("Synchronized with Infinity service")
        except Exception as e:
            logger.error("Failed to synchronize: %s", e)
            raise
