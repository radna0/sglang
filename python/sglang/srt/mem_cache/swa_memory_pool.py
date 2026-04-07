import logging
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        token_to_kv_pool_class: Type[KVCache] = MHATokenToKVPool,
        *,
        # Optional overrides to use different KV pool implementations for full-attention
        # and SWA layers (e.g. NSA/DSA index-K buffers only for full-attn layers).
        swa_token_to_kv_pool_class: Optional[Type[KVCache]] = None,
        full_token_to_kv_pool_class: Optional[Type[KVCache]] = None,
        swa_token_to_kv_pool_kwargs: Optional[Dict[str, Any]] = None,
        full_token_to_kv_pool_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.start_layer = 0
        self.page_size = page_size
        self.swa_loc = None

        base_kwargs: Dict[str, Any] = dict(kwargs)
        base_kwargs["page_size"] = page_size
        base_kwargs["enable_memory_saver"] = False
        base_kwargs["head_num"] = head_num
        base_kwargs["head_dim"] = head_dim
        base_kwargs["device"] = device
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        swa_pool_cls = swa_token_to_kv_pool_class or token_to_kv_pool_class
        full_pool_cls = full_token_to_kv_pool_class or token_to_kv_pool_class

        swa_kwargs = dict(base_kwargs)
        if swa_token_to_kv_pool_kwargs:
            swa_kwargs.update(swa_token_to_kv_pool_kwargs)

        full_kwargs = dict(base_kwargs)
        # SWA-only kwargs should never flow into the full-attn pool constructor.
        for k in ("swa_head_num", "swa_head_dim", "swa_v_head_dim"):
            full_kwargs.pop(k, None)
        if full_token_to_kv_pool_kwargs:
            full_kwargs.update(full_token_to_kv_pool_kwargs)

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        self.swa_kv_pool = swa_pool_cls(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **swa_kwargs,
        )
        self.full_kv_pool = full_pool_cls(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **full_kwargs,
        )
        # {layer_id: (index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool mem usage: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_state_buf_infos(self):
        # For hybrid SWA, we treat the SWA pool as "state" because only the full pool is
        # transferred through get_contiguous_buf_infos().
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        # If the full pool has additional state (e.g. NSA/DSA index-K buffers), include it
        # in the state payload so disaggregation/transfer can preserve it.
        if hasattr(self.full_kv_pool, "get_state_buf_infos"):
            full_state_ptrs, full_state_lens, full_state_item_lens = (
                self.full_kv_pool.get_state_buf_infos()
            )
            return (
                swa_kv_data_ptrs + full_state_ptrs,
                swa_kv_data_lens + full_state_lens,
                swa_kv_item_lens + full_state_item_lens,
            )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def _map_full_layer(self, layer_id: int) -> int:
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            raise RuntimeError(
                "NSA/DSA index-K buffers are only defined for full-attention layers; "
                f"got SWA layer_id={layer_id}."
            )
        return layer_id_pool

    # NSA/DSA index-K buffer APIs (delegated to the full-attention KV pool).
    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.full_kv_pool.get_index_k_with_scale_buffer(self._map_full_layer(layer_id))

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return self.full_kv_pool.get_index_k_continuous(
            self._map_full_layer(layer_id), seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return self.full_kv_pool.get_index_k_scale_continuous(
            self._map_full_layer(layer_id), seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        return self.full_kv_pool.get_index_k_scale_buffer(
            self._map_full_layer(layer_id), seq_len=seq_len, page_indices=page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        return self.full_kv_pool.set_index_k_scale_buffer(
            self._map_full_layer(layer_id),
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def set_swa_loc(self, loc: torch.Tensor):
        self.swa_loc = loc

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None

        # Note: kv_indices could have -1 values (from alloc_extend), which will be mapped to -1
        # since the last item of full_to_swa_index_mapping is -1.
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            if self.swa_loc is not None:
                loc = self.swa_loc
            else:
                if self.full_to_swa_index_mapping is not None:
                    loc = self.translate_loc_from_full_to_swa(loc)

            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def get_cpu_copy(self, indices):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(indices)

        # Get SWA indices through the mapping
        # Note: SWA allocation always creates 1:1 mapping, so no need to filter
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices)
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu}

    def load_cpu_copy(self, kv_cache_cpu, indices):
        # Load KV cache back from CPU to both full and SWA pools
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # Load full KV cache to the new indices
        self.full_kv_pool.load_cpu_copy(full_kv_cpu, indices)

        # Load SWA KV cache if it exists
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
        need_sort: bool,
    ):
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

        if page_size == 1:
            self.full_attn_allocator = TokenToKVPoolAllocator(
                size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = TokenToKVPoolAllocator(
                size_swa,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        else:
            self.full_attn_allocator = PagedTokenToKVPoolAllocator(
                size,
                page_size,
                dtype,
                device,
                kvcache.full_kv_pool,
                need_sort,
            )
            self.swa_attn_allocator = PagedTokenToKVPoolAllocator(
                size_swa,
                page_size,
                dtype,
                device,
                kvcache.swa_kv_pool,
                need_sort,
            )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    size + self.page_size,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

        self.need_sort = need_sort
        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

        self.clear()
        self._kvcache = kvcache
        self._kvcache.register_mapping(weakref.proxy(self.full_to_swa_index_mapping))

    def available_size(self):
        return min(
            self.full_attn_allocator.available_size(),
            self.swa_attn_allocator.available_size(),
        )

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size(self):
        return min(self._size_full, self._size_swa)

    @property
    def size_swa(self):
        return self._size_swa

    @property
    def size_full(self):
        return self._size_full

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self._kvcache.full_to_swa_index_mapping is not None
        return self._kvcache.translate_loc_from_full_to_swa(kv_indices)

    def alloc(self, need_size: int):
        assert self.page_size == 1
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
        extend_num_tokens: int,
    ):
        assert self.page_size > 1
        num_tokens = extend_num_tokens + len(seq_lens) * self.page_size
        if num_tokens > self.full_attn_allocator.available_size():
            return None
        if num_tokens > self.swa_attn_allocator.available_size():
            return None

        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            swa_last_loc,
            extend_num_tokens,
        )
        assert alloc_full_indices is not None
        assert alloc_swa_indices is not None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,  # last_loc for full layers
    ):
        assert self.page_size > 1
        swa_last_loc = self.translate_loc_from_full_to_swa(last_loc)

        alloc_full_indices = self.full_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc
        )
        alloc_swa_indices = self.swa_attn_allocator.alloc_decode(
            seq_lens, seq_lens_cpu, swa_last_loc
        )

        if alloc_full_indices is None or alloc_swa_indices is None:
            return None

        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices

        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # NOTE: the API is not idempotent.
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        # Note: the last item is -1, we don't clear it, see the comment in __init__
        self.full_to_swa_index_mapping[:-1].fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
