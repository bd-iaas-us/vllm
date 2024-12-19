from collections import deque
from typing import Deque, Dict
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

class GlobalCache:
    def __init__(self):
        self.cachedBlockNum: int = 0
        self.blockHashDict_k: Dict[int, Dict[int, torch.Tensor]] = {}
        self.blockHashDict_v: Dict[int, Dict[int, torch.Tensor]] = {}
        self.cachedBlockHashQ: Deque[int] = deque()
    def setGlabalCacheBlockNum(self, num_global_cache_blocks: int):
        if self.cachedBlockNum > 0 or num_global_cache_blocks <= 0:
            logger.warning("can not enable global kv cache")
            return
        self.cachedBlockNum = num_global_cache_blocks     
        logger.info("global kv cache enabled")   
    def writeCache(self, block_hash: int, layer_idx: int, k_block_tensor: torch.Tensor, v_block_tensor: torch.Tensor):
        if self.cachedBlockNum == 0:
            return
        if len(self.cachedBlockHashQ) == self.cachedBlockNum:
            poped_block_hash = self.cachedBlockHashQ.popleft()
            del self.blockHashDict_k[poped_block_hash]
            del self.blockHashDict_v[poped_block_hash]
        if block_hash not in self.blockHashDict_k or block_hash not in self.blockHashDict_v:
            self.blockHashDict_k[block_hash] = {}
            self.blockHashDict_v[block_hash] = {}
        else:
            self.cachedBlockHashQ.remove(block_hash)
        self.blockHashDict_k[block_hash][layer_idx] = k_block_tensor.to(device="cpu", non_blocking=True)
        self.blockHashDict_v[block_hash][layer_idx] = v_block_tensor.to(device="cpu", non_blocking=True)
        self.cachedBlockHashQ.append(block_hash)
    def readCache(self, block_hash: int, layer_idx: int, device: torch.device):
        if self.cachedBlockNum == 0:
            return
        if not self.checkExist(block_hash):
            return
        self.cachedBlockHashQ.remove(block_hash)
        self.cachedBlockHashQ.append(block_hash)
        return self.blockHashDict_k[block_hash][layer_idx].to(torch.device(device), non_blocking=True), self.blockHashDict_v[block_hash][layer_idx].to(torch.device(device), non_blocking=True)
    def checkExist(self, block_hash: int):
        return block_hash in self.blockHashDict_k and block_hash in self.blockHashDict_v

global_cache_instance = GlobalCache()