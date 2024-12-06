import math
import logging
from typing import Dict, List, Tuple
import torch
import os
import time

import infinistore

from vllm.distributed.kv_transfer.base import KVCacheTransporterBase
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)


logger = logging.getLogger(__name__)

Default_Infinite_Server = "127.0.0.1"
interval = 0.01
count = 0
shared_signal_folder = "/tmp/infinistore"

class InfiniStoreKVCacheTransporter(KVCacheTransporterBase):
    #Class-level singleton connection instance
    _singleton_conn = None

    def __init__(self, model: str, tokens_per_page=16) -> None:
        if not model:
            raise ValueError("model cannot be empty.")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be greater than 0.")

        # escape the slash in the model name
        self.model = model.replace("/", "_")
        self.tokens_per_page = tokens_per_page

        #TODO: when server is local, use connection_type=infinistore.TYPE_LOCAL_GPU,
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

        os.makedirs(shared_signal_folder, exist_ok=True)

    def get_hidden_states_cache_key(self, page_hash: str) -> str:
        return f"{self.model}_{page_hash}_tp_{self.tp_rank}_{self.tp_size}_hs"
    
    def get_kv_cache_key(self, page_hash: str, layer_idx: int) -> Tuple[str, str]:
        initial = f"{self.model}_{page_hash}_layer_{layer_idx}_tp_{self.tp_rank}_{self.tp_size}"
        # initial = f"{page_hash}_{layer_idx}_{self.tp_rank}_{self.tp_size}"
        k_cache_key = f"{initial}_k"
        v_cache_key = f"{initial}_v"
        return k_cache_key, v_cache_key

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
                k_cache_key, v_cache_key = self.get_kv_cache_key(
                    current_hash, layer_idx)

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

                cache_key = self.get_hidden_states_cache_key(current_hash)

                cache_size = hidden_size * (end_token_idx - start_token_idx)
                offset = (seq_start_index + start_token_idx) * hidden_size

                if cache_size not in block_offsets:
                    block_offsets[cache_size] = []
                block_offsets[cache_size].append((cache_key, offset))

            seq_start_index += seq_length
            page_start_index += num_pages

        return block_offsets
    
    def _publish_write_completion(self, key: str) -> None:
        import time
        start = time.time()
        open(os.path.join(shared_signal_folder, key), mode="w").close()

    def publish_kv_cache_prefill_ready(self, input_token_hashes: List[str], seq_lens: List[int], layer_idx: int) -> None:

        covered_pages = 0
        for seq_len in seq_lens:
            covered_pages += math.ceil(seq_len / self.tokens_per_page)
            current_hash = input_token_hashes[covered_pages-1]
            _, v_cache_key = self.get_kv_cache_key(
                    current_hash, layer_idx)
            
            # only need to publish V cache key, as V cache is always written after K cache
            self._publish_write_completion(v_cache_key)

    def verify_kv_cache_prefill_ready(self, input_token_hashes: List[str], seq_lens: List[int], layer_idx: int) :
        covered_pages = 0
        for seq_len in seq_lens:
            covered_pages += math.ceil(seq_len / self.tokens_per_page)
            current_hash = input_token_hashes[covered_pages-1]
            _, v_cache_key = self.get_kv_cache_key(
                    current_hash, layer_idx)
            if os.path.exists(os.path.join(shared_signal_folder, v_cache_key)):
                continue
            
            wt = 0
            while not os.path.exists(os.path.join(shared_signal_folder, v_cache_key)):
                time.sleep(interval)
                wt += 1
                if wt % 100 == 0:
                    print(f"Qian wait for kv cache prefill done {wt} times {v_cache_key}")

    def publish_hidden_states_ready(self, input_token_hashes: List[str], seq_lens: List[int]) -> None:

        covered_pages = 0
        for seq_len in seq_lens:
            covered_pages += math.ceil(seq_len / self.tokens_per_page)
            current_hash = input_token_hashes[covered_pages-1]
            hs_cache_key = self.get_hidden_states_cache_key(current_hash)
            
            self._publish_write_completion(hs_cache_key)

    def verify_hidden_states_ready(self, input_token_hashes: List[str], seq_lens: List[int]) :
        covered_pages = 0
        for seq_len in seq_lens:
            covered_pages += math.ceil(seq_len / self.tokens_per_page)
            current_hash = input_token_hashes[covered_pages-1]
            hs_cache_key = self.get_hidden_states_cache_key(current_hash)
            if os.path.exists(os.path.join(shared_signal_folder, hs_cache_key)):
                continue
            
            wt = 0
            while not os.path.exists(os.path.join(shared_signal_folder, hs_cache_key)):
                time.sleep(interval)
                wt += 1
                if wt % 100 == 0:
                    print(f"Qian wait for hidden states ready {wt} times {hs_cache_key}")

    def save_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int],
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, slot_mapping,
            layer_idx, kv_cache)
        
        try:            
            if self.conn.rdma_connected:
                self.conn.rdma_write_cache(kv_cache, block_offsets, page_size)
            else:
                self.conn.local_gpu_write_cache(kv_cache, block_offsets, page_size)

            if layer_idx == 0:
                global count
                count += len(prompt_seq_lengths)
                last_hash = prompt_token_page_hashes[-1]

        except Exception as e:
            logger.error("Failed to write kv_cache: %s", e)
            raise

        logger.debug("Saved kv_cache for layer %s", layer_idx)

    def read_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int],
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:
        
        self.verify_kv_cache_prefill_ready(prompt_token_page_hashes, prompt_seq_lengths, layer_idx)

        block_offsets, page_size = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, slot_mapping,
            layer_idx, kv_cache)
        
        RETRY_LIMIT = 10

        for attempt in range(RETRY_LIMIT):
            try:
                self.conn.read_cache(kv_cache, block_offsets, page_size)
                break  # Exit the loop if successful
            except Exception as e:
                logger.error("Attempt %d: Failed to read kv_cache: %s", attempt + 1, e)
                
                if attempt > RETRY_LIMIT - 1:
                    logger.error("All retry attempts failed.")
                    raise

        logger.debug("Loaded kv_cache for layer %s", layer_idx)

    def save_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        if self.conn.rdma_connected:
            self.conn.register_mr(hidden_states)
        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)

        try:
            for cache_size, offsets in block_offsets.items():
                if self.conn.rdma_connected:
                    self.conn.rdma_write_cache(hidden_states, offsets, cache_size)
                else:
                    self.conn.local_gpu_write_cache(hidden_states, offsets, cache_size)
        except Exception as e:
            logger.error("Failed to read hidden_states: %s", e)
            raise

        logger.debug("Saved hidden_states")

    def read_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        self.verify_hidden_states_ready(prompt_token_page_hashes, prompt_seq_lengths)

        if self.conn.rdma_connected:
            self.conn.register_mr(hidden_states)
        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)

        try:
            for cache_size, offsets in block_offsets.items():
                self.conn.read_cache(hidden_states, offsets, cache_size)
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
