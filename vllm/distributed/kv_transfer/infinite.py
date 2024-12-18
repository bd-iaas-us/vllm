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
    _singleton_rdma_conn = None

    def __init__(self, model: str, kv_cache_list: List[torch.Tensor], tokens_per_page: int =16) -> None:
        if not model:
            raise ValueError("model cannot be empty.")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be greater than 0.")

        # escape the slash in the model name
        self.model = model.replace("/", "_")
        self.kv_cache_list = kv_cache_list
        self.tokens_per_page = tokens_per_page
        self.page_size = kv_cache_list[0][0][0].numel() 
        self.k_or_v_total_size = kv_cache_list[0][0].numel()

        
        # TODO: grab the config values dynamically instead of hardcoding
        infinite_server = os.environ.get("INFINITE_STORE_SERVER",
                                         Default_Infinite_Server)
        infinite_server = infinite_server.strip('"')

        # TODO: change to use the config file
        infinistore_log_level = "info"
        infinistore_service_port = 22345
        infinistore_ib_port = 1
        infinistore_link_type = infinistore.LINK_ETHERNET
        infinistore_dev_name = "mlx5_0"

        if InfiniStoreKVCacheTransporter._singleton_conn is None:
            infinte_config = infinistore.ClientConfig(
                host_addr=infinite_server,
                service_port=infinistore_service_port,
                log_level=infinistore_log_level,
                connection_type=infinistore.TYPE_LOCAL_GPU,
                ib_port=infinistore_ib_port,
                link_type=infinistore_link_type,
                dev_name=infinistore_dev_name,
            )
            InfiniStoreKVCacheTransporter._singleton_conn = infinistore.InfinityConnection(
                infinte_config)
            logger.info("Connecting to infinite store server: %s",
                        infinite_server)

            InfiniStoreKVCacheTransporter._singleton_conn.connect()

        # Assign the singleton connection to the instance attribute
        self.conn = InfiniStoreKVCacheTransporter._singleton_conn

        if self.conn.rdma_connected:
            self.rdma_conn = self.conn
        else:
            if InfiniStoreKVCacheTransporter._singleton_rdma_conn is None:
                infinite_rdma_config = infinistore.ClientConfig(
                    host_addr=infinite_server,
                    service_port=infinistore_service_port,
                    log_level=infinistore_log_level,
                    connection_type=infinistore.TYPE_RDMA,
                    ib_port=infinistore_ib_port,
                    link_type=infinistore_link_type,
                    dev_name=infinistore_dev_name,
                )
                InfiniStoreKVCacheTransporter._singleton_rdma_conn = infinistore.InfinityConnection(infinite_rdma_config)
                InfiniStoreKVCacheTransporter._singleton_rdma_conn.connect()
                logger.info("RDMA Connecting to infinite store server: %s",
                            infinite_server)
            self.rdma_conn = InfiniStoreKVCacheTransporter._singleton_rdma_conn

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.hs_key_initial = f"hs_{self.model}_tp_{self.tp_rank}_{self.tp_size}_"
        self.kv_key_initial = f"{self.model}_tp_{self.tp_rank}_{self.tp_size}_"


    def get_hidden_states_cache_key(self, page_hash: str) -> str:
        return self.hs_key_initial + page_hash
  
    def get_kv_cache_key(self, page_hash: str, layer_idx: int) -> Tuple[str, str]:
        k_cache_key = self.kv_key_initial + f"{layer_idx}_{page_hash}_k"
        v_cache_key = self.kv_key_initial + f"{layer_idx}_{page_hash}_v"
        return k_cache_key, v_cache_key

    def _compute_kv_cache_block_offsets(
            self, prompt_token_page_hashes: List[str], offsets: List[Tuple[int, int]],
            layer_idx: int) -> List[Tuple[str, int]]:
        
        block_offsets: List[Tuple[str, int]] = []

        for current_hash, offset in zip(prompt_token_page_hashes, offsets):
            k_cache_key, v_cache_key = self.get_kv_cache_key(current_hash, layer_idx)
            block_offsets.append((k_cache_key, offset[0]))
            block_offsets.append((v_cache_key, offset[1]))

        return block_offsets

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
        open(os.path.join(shared_signal_folder, key), mode="w").close()

    def publish_kv_cache_prefill_done(self, input_token_hashes: List[str], seq_lens: List[int], layer_idx: int) -> None:

        covered_pages = 0
        for seq_len in seq_lens:
            covered_pages += math.ceil(seq_len / self.tokens_per_page)
            current_hash = input_token_hashes[covered_pages-1]
            _, v_cache_key = self.get_kv_cache_key(
                    current_hash, layer_idx)
            
            # only need to publish V cache key, as V cache is always written after K cache
            self._publish_write_completion(v_cache_key)

    def verify_kv_cache_prefill_done(self, input_token_hashes: List[str], seq_lens: List[int], layer_idx: int) :
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
                    logger.warning(f"Wait for kv cache key {v_cache_key} for {wt} times")

    def save_kv_cache(self, prompt_token_page_hashes: List[str],
                      offsets: List[Tuple[int, int]], layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        block_offsets = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, offsets, layer_idx)
        
        try:            
            if self.conn.rdma_connected:
                keys, key_offsets = zip(*block_offsets)
                remote_addrs = self.conn.allocate_rdma(keys, self.page_size * kv_cache.element_size())
                self.conn.rdma_write_cache(kv_cache, key_offsets, self.page_size, remote_addrs)
            else:
                self.conn.local_gpu_write_cache(kv_cache, block_offsets, self.page_size)

        except Exception as e:
            logger.error("Failed to write kv_cache: %s", e)
            raise

        logger.debug("Saved kv_cache for layer %s", layer_idx)

    def read_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int],
                      offsets: List[Tuple[int, int]], layer_idx: int,
                      kv_cache: torch.Tensor) -> None:
        
        self.verify_kv_cache_prefill_done(prompt_token_page_hashes, prompt_seq_lengths, layer_idx)

        block_offsets = self._compute_kv_cache_block_offsets(
            prompt_token_page_hashes, offsets, layer_idx)

        try:
            self.conn.read_cache(kv_cache, block_offsets, self.page_size)
        except Exception as e:
            logger.error("Failed to read kv_cache: %s", e)

        logger.debug("Loaded kv_cache for layer %s", layer_idx)

    def save_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        self.rdma_conn.register_mr(hidden_states)

        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)

        try:
            for cache_size, offsets in block_offsets.items():
                keys, key_offsets = zip(*offsets)
                remote_addrs = self.rdma_conn.allocate_rdma(keys, cache_size * hidden_states.element_size())
                self.rdma_conn.rdma_write_cache(hidden_states, key_offsets, cache_size, remote_addrs)
        except Exception as e:
            logger.error("Failed to read hidden_states: %s", e)
            raise

        logger.debug("Saved hidden_states")

    def read_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        hs_cache_key = self.get_hidden_states_cache_key(prompt_token_page_hashes[-1])
        self.rdma_conn.register_mr(hidden_states)
        wt = 0
        while not self.conn.check_exist(hs_cache_key):
            time.sleep(interval)
            wt += 1
            if wt % 100 == 0:
                logger.warning(f"Wait for hidden states cache key {hs_cache_key} for {wt} times")


        # self.rdma_conn.register_mr(hidden_states)
        block_offsets = self._compute_hidden_states_block_offsets(
            prompt_token_page_hashes, prompt_seq_lengths, hidden_states)

        try:
            for cache_size, offsets in block_offsets.items():
                self.rdma_conn.read_cache(hidden_states, offsets, cache_size)
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
        except Exception as e:
            logger.error("Failed to synchronize: %s", e)
            raise
