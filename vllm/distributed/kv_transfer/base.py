from abc import ABC, abstractmethod
import torch

from vllm.attention import AttentionMetadata


class KVCacheTransporterBase(ABC):

    @abstractmethod
    def save_kv_cache(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_seq_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
        kv_cache: torch.Tensor
    ) -> None:
        """
        Save the key-value cache for a specific layer.

        Args:
            prompt_token_ids (torch.Tensor): Tensor of token IDs from the prompt sequence.
            prompt_seq_lengths (torch.Tensor): Tensor indicating the sequence lengths of each prompt.
            slot_mapping (torch.Tensor): Tensor mapping slots to their respective positions in the cache.
            layer_idx (int): The index of the layer for which the key-value cache is being saved.
            kv_cache (torch.Tensor): The key-value cache tensor to be updated.
        """
        raise NotImplementedError


    @abstractmethod
    def read_kv_cache(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_seq_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int,
        kv_cache: torch.Tensor
    ) -> None:
        """
        Read the key-value cache for a specific layer.

        Args:
            prompt_token_ids (torch.Tensor): Tensor of token IDs from the prompt sequence.
            prompt_seq_lengths (torch.Tensor): Tensor indicating the sequence lengths of each prompt.
            slot_mapping (torch.Tensor): Tensor mapping slots to their respective positions in the cache.
            layer_idx (int): Index of the layer for which the key-value cache is being read.
            kv_cache (torch.Tensor): The KV cache tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def save_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """
        Save the hidden states.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            hidden_states (torch.Tensor): The hidden states tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def read_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """
        read the hidden states.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            hidden_states (torch.Tensor): The hidden states tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self):
        """Synchronize any asynchronous operations."""
        raise NotImplementedError
