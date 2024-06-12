from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.attention.ops.prefix_prefill import context_attention_fwd

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch.
    max_seq_len: Optional[int]
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size() # ??
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        kv_scale: float,
    ) -> None:
        # torch.set_printoptions(threshold=float('inf'))
        print("write_to_paged_cache 11")
        print(key.shape)
        print(value.shape)
        print(key_cache.shape)
        print(value_cache.shape)
        #print(key_cache[-1,  :100])
        # slot_mapping = slot_mapping - 1
        print(slot_mapping)
        t = slot_mapping % 16
        print(t)
        # if t[0] == 6 or t[0] == 7:
        #     print(value_cache[-5, -1, -1, :])
        print("slot mapping")
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            kv_scale,
        )
        print("write_to_paged_cache 2")
        #print(key_cache.shape)
        print(value_cache.shape)
        #print(key_cache[-9:, :100])
        # if t[0] == 6 or t[0] == 7 or t[0] == 4 or t[0] == 5 or t[0] == 3: # ??
        #     print(value_cache[-5, -1, -1, :])

    @staticmethod
    def copy_to_paged_cache(
        src_key_cache: torch.Tensor,
        src_value_cache: torch.Tensor,
        tgt_key_cache: torch.Tensor,
        tgt_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        atten_score: torch.Tensor,
    ) -> None:
        ops.copy_to_cache(
            src_key_cache,
            src_value_cache,
            tgt_key_cache,
            tgt_value_cache,
            slot_mapping.flatten(),
            atten_score
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        kv_scale: float,
        sparse_cache_type: str,
        sparse_condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(query)

        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = (max_seq_len <= 8192
                  and (max_num_partitions == 1 or num_seqs * num_heads > 512))
        if use_v1:
            print("EEEEEEEEEEEEEEEEEEEEE")
            if sparse_condition is None: # ??
                # print("FFFFFFFFFFFFFFFFFFFFF")
                sparse_condition = torch.zeros(768, dtype=torch.float32)
                # sparse_condition = torch.zeros(768 * 3, dtype=torch.float32)
            else:
                print(sparse_condition.shape)
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                kv_scale,
                sparse_cache_type,
                sparse_condition,
            )
            # print("After updated tensor")
            # print(sparse_condition)
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                kv_scale,
                sparse_cache_type,
                sparse_condition,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        subquery_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            key_cache,
            value_cache,
            block_tables,
            # subquery_start_loc is (batch_size + 1,)
            subquery_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)

    @staticmethod
    def sparse_cache_copy(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
        sparse_condition: torch.Tensor,
        num_heads: int, 
        head_size: int, 
        block_size: int,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        num_seq = 4 # ??
        print("PPPPPSPRASE")
        print(src_to_dists)
        print(src_to_dists[:, 0])
        print(src_to_dists[:, 1])
        print(src_to_dists.size(0))
        print(len(key_caches))
        print(len(value_caches))
        print(key_caches[0].shape)
        print(value_caches[0].shape)
        # print("PPPPPSPRASE")
        torch.set_printoptions(threshold=float('inf'))
        #print(key_caches[0][-5:, :100])
        #print("WWWWTTTFFFFF")
        #print(value_caches[0][-5:, :100])
        print(sparse_condition)
        print("PPPPSPRASE end")
        num_blocks = src_to_dists.size(2)
        print("NNNNNNNUMBLOCKS " + str(num_blocks))
        selection_index_src_tensor = torch.full((12, src_to_dists.size(0), block_size * num_blocks), -1, dtype=torch.int64) # src_to_dists.size(0) = 5 ??
        selection_index_dst_tensor = torch.full((12, src_to_dists.size(0), block_size * num_blocks), -1, dtype=torch.int64) # src_to_dists.size(0) = 5
        print(selection_index_src_tensor.shape)
        print(selection_index_dst_tensor.shape)
        for i, row in enumerate(sparse_condition): # 0-3
            for j, value in enumerate(row):
                count = 0
                for k, num in enumerate(value):
                    if num == 1:
                        selection_index_src_tensor[i, j, k] = k + i * num_seq * block_size * num_blocks + j * block_size * num_blocks
                        selection_index_dst_tensor[i, j, k] = count + i * num_seq * block_size * num_blocks + j * block_size * num_blocks
                        count += 1
        # print(selection_index_src_tensor)
        # print(selection_index_dst_tensor)
        src_flatten = selection_index_src_tensor.flatten()
        dst_flatten = selection_index_dst_tensor.flatten()
        print("sssselection_index_src_tensor:", src_flatten)
        print("sssselection_index_dst_tensor:", dst_flatten)
        block_mapping_src = src_to_dists[:, 0].to(torch.int64)
        block_mapping_dst = src_to_dists[:, 1].to(torch.int64)
        print(block_mapping_src)
        print(block_mapping_dst)
        print("debug")
        print(block_mapping_src.shape)
        print(block_mapping_dst.shape)
        ops.sparse_cache_copy(key_caches, value_caches, block_mapping_src, block_mapping_dst, src_flatten, dst_flatten, num_heads, head_size, block_size)
        #print(key_caches[0][-9:, :100])
        #print("WTFWTFWTFWTFWTF end")
        #print(value_caches[0][-9:, :100])
