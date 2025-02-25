"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches. Only (2) is supported for now.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class LMCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        from lmcache.integration.vllm.vllm_adapter import (
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_store, lmcache_store_kv)

        logger.info("Initializing LMCacheConfig under kv_transfer_config %s",
                    self.transfer_config)

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_lmcache_engine(config.model_config,
                                          config.parallel_config,
                                          config.cache_config)

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # TODO(Jiayi): This shouldn't be none for disagg prefill
        # hidden_or_intermediate_states = None

        # TODO (Jiayi): Only normal prefill is supported for now
        retrieve_status = [self.retrieve_status.PREFILL]

        model_input, hidden_or_intermediate_states, bypass_model_exec = self.lmcache_retrieve_kv(
            model_executable, model_input, self.cache_config, kv_caches,
            retrieve_status)
        
        print("~~~~~~~~ retrieve hidden states ", hidden_or_intermediate_states)
        # print("~~~~~~~~~ retrieve kv_caches [0]", kv_caches[0][0][0])

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        num_reqs = 0
        seq_group_list = model_input.sampling_metadata.seq_groups
        #assert seq_group_list is not None
        if seq_group_list is None:
            return

        for seq_group in seq_group_list:
            seq_ids = seq_group.seq_ids
            for seq_id in seq_ids:
                num_reqs += 1

        # TODO (Jiayi): Only normal prefill is supported for now
        #store_status = [self.store_status.PREFILL] * num_reqs
        store_status = self.lmcache_should_store(model_input)
        self.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
            hidden_or_intermediate_states,
        )
        print("-----after store kv")
        print("-----hidden states store", hidden_or_intermediate_states)
        # print("~~~~~~~~~ save kv_caches [0]", kv_caches[0][0][0])

    def close(self):
        self.engine.close()
