from abc import ABC, abstractmethod
import torch
from typing import List


class KVCacheTransporterBase(ABC):

    @abstractmethod
    def save_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: torch.Tensor,
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        raise NotImplementedError

    @abstractmethod
    def read_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: torch.Tensor,
                      slot_mapping: torch.Tensor, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        raise NotImplementedError

    @abstractmethod
    def save_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: torch.Tensor,
                           hidden_states: torch.Tensor):

        raise NotImplementedError

    @abstractmethod
    def read_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: torch.Tensor,
                           hidden_states: torch.Tensor):

        raise NotImplementedError

    @abstractmethod
    def synchronize(self):

        raise NotImplementedError
