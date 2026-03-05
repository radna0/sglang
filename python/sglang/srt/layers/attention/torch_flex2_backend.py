from __future__ import annotations

from typing import TYPE_CHECKING

import logging
import os

import torch
from torch.nn.attention.flex_attention import BlockMask

from sglang.srt.layers.attention.flex_utils import apply_attention_sinks
from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.layers.radix_attention import RadixAttention


logger = logging.getLogger(__name__)

_DISALLOW_FALLBACK_ENVS = {"1", "true", "True", "yes", "YES"}


def _load_flashattention_backend():
    # Import lazily so environments without triton (e.g. CPU-only dev) can still import
    # this module. The actual FA backend is only needed when force_flash is enabled.
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return FlashAttentionBackend


class TorchFlexAttnBackendV2(TorchFlexAttnBackend):
    """
    FlexAttention-II style backend: avoid per-request KV gather by using a BlockMask
    over the *global* KV cache pages.

    Requirements / assumptions:
      - Works best when `page_size` is reasonably large (e.g. 64/128). When
        page_size==1 this degenerates to token-blocking and is usually slower.
      - The KV cache memory pool uses page-aligned layout: token indices are
        `physical_page * page_size + offset`.

    Fallback:
      - If page_size <= 1 (or KV is not page-aligned), we fall back to the V1
        backend behavior (slice/gather per request).
    """

    def __init__(self, model_runner: "ModelRunner", *, kernel_options: dict | None = None):
        super().__init__(model_runner, kernel_options=kernel_options)
        self._page_size = int(getattr(model_runner.server_args, "page_size", 1) or 1)
        # "FlexFlash2" fastpath: when kernel_options contains {"force_flash": true},
        # bypass PyTorch FlexAttention entirely and delegate to the native FA backend.
        #
        # This is the highest-ROI path on Hopper (SM90): it preserves the ability to
        # select a "flex_*" backend while avoiding Flex dispatcher/mask overhead, and
        # it is CUDA-graph friendly because it uses the same graph hooks as FA3.
        self._force_flash: bool = bool(getattr(self, "flex_kernel_options", {}).get("force_flash", False))
        self._fa_delegate: object | None = None
        if self._force_flash:
            FlashAttentionBackend = _load_flashattention_backend()
            self._fa_delegate = FlashAttentionBackend(model_runner, fa_impl_ver=3)
            if (os.environ.get("SGLANG_FLEX2_LOG_FORCE_FLASH") or "1").strip() in {"1", "true", "True"}:
                try:
                    print("[flex2] force_flash=1 delegate=FlashAttentionBackend(fa_impl_ver=3)", flush=True)
                except Exception:
                    pass
        self._physical_to_logical: torch.Tensor | None = None
        self._physical_to_logical_batched: torch.Tensor | None = None
        self._kv_start_rel: torch.Tensor | None = None
        self._q_kv_offset: torch.Tensor | None = None
        self._kv_start_rel_batched: torch.Tensor | None = None
        self._q_kv_offset_batched: torch.Tensor | None = None
        self._debug_remaining: int = int(os.environ.get("SGLANG_FLEX_DEBUG_PRINTS", "0") or 0)
        self._logged_vectorized_decode: bool = False
        self._logged_vectorized_extend: bool = False
        self._logged_run_decode: bool = False
        self._logged_run_extend: bool = False
        self._logged_page_size1: bool = False
        self._page_col_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._logical_pages_row_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._pos_row_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        # Persistent (across decode steps) batched decode caches keyed by static shape.
        # This is required to eliminate per-step allocations for CUDA-graph / compile viability.
        self._flex2_decode_cache_batched_by_sig: dict[tuple, dict] = {}
        self._flex2_decode_cache_batched_key_order: list[tuple] = []
        self._flex2_decode_cache_batched_max_keys: int = int(
            os.environ.get("SGLANG_FLEX2_DECODE_CACHE_MAX_KEYS", "32") or 32
        )
        self._flex2_decode_cache_batched_alloc_ct: int = 0

        # Persistent batched extend caches keyed by static shape.
        # This avoids per-step BlockMask.from_kv_blocks allocations, which can trigger
        # torch.compile cache churn and destroy throughput.
        self._flex2_extend_cache_batched_by_sig: dict[tuple, dict] = {}
        self._flex2_extend_cache_batched_key_order: list[tuple] = []
        self._flex2_extend_cache_batched_max_keys: int = int(
            os.environ.get("SGLANG_FLEX2_EXTEND_CACHE_MAX_KEYS", "32") or 32
        )
        self._flex2_extend_cache_batched_alloc_ct: int = 0

        # page_size==1 token-block (dense KV blocks) cache for DFlash verify + prefill.
        # Keyed by static signature so we can reuse BlockMask/kv_indices buffers.
        self._flex2_tokenblock_cache_by_sig: dict[tuple, dict] = {}
        self._flex2_tokenblock_cache_key_order: list[tuple] = []
        self._flex2_tokenblock_cache_max_keys: int = int(
            os.environ.get("SGLANG_FLEX2_TOKENBLOCK_CACHE_MAX_KEYS", "16") or 16
        )
        self._flex2_tokenblock_cache_alloc_ct: int = 0
        self._flex2_fallback_decode_ct: int = 0
        self._flex2_fallback_extend_ct: int = 0

    # ------------------------
    # Delegate-to-FA3 fastpath
    # ------------------------
    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        if self._fa_delegate is not None:
            return self._fa_delegate.init_forward_metadata(forward_batch)
        # NOTE: do not call TorchFlexAttnBackend.init_forward_metadata() because it
        # unconditionally calls torch.cuda.empty_cache(), which is extremely costly
        # for long-context inference and breaks CUDA-graph stability.
        self._last_forward_mode_is_extend = forward_batch.forward_mode.is_extend()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        if self._fa_delegate is not None:
            return self._fa_delegate.init_cuda_graph_state(max_bs, max_num_tokens)
        return super().init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        if self._fa_delegate is not None:
            return self._fa_delegate.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        return super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        if self._fa_delegate is not None:
            return self._fa_delegate.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )
        return super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        if self._fa_delegate is not None:
            return self._fa_delegate.get_cuda_graph_seq_len_fill_value()
        return super().get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        if self._fa_delegate is not None:
            return self._fa_delegate.get_verify_buffers_to_fill_after_draft()
        return super().get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        if self._fa_delegate is not None:
            return self._fa_delegate.update_verify_buffers_to_fill_after_draft(
                spec_info, cuda_graph_bs
            )
        return super().update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

        # FlexAttention relies on torch inductor and (typically) Triton for kernel generation.
        # Ensure at least one GEMM backend is enabled for autotuning, otherwise inductor can
        # raise NoValidChoicesError during flex kernel compilation.
        #
        # IMPORTANT: avoid importing torch._inductor at module init (it can be slow and can
        # deadlock/hang under some containerized launchers). Prefer the env var.
        if not (os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS") or "").strip():
            os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "CUTLASS,TRITON,ATEN"

    def _extend_cache_put(self, key: tuple, cache: dict) -> None:
        self._flex2_extend_cache_batched_by_sig[key] = cache
        self._flex2_extend_cache_batched_key_order.append(key)
        max_keys = int(self._flex2_extend_cache_batched_max_keys)
        if max_keys > 0 and len(self._flex2_extend_cache_batched_key_order) > max_keys:
            evict = self._flex2_extend_cache_batched_key_order.pop(0)
            self._flex2_extend_cache_batched_by_sig.pop(evict, None)

    def _get_page_col(self, *, device: torch.device, total_pages: int) -> torch.Tensor:
        key = (device, int(total_pages))
        t = self._page_col_cache.get(key)
        if t is None:
            t = torch.arange(int(total_pages), device=device, dtype=torch.int64)
            self._page_col_cache[key] = t
        return t

    def _get_logical_pages_row(self, *, device: torch.device, total_pages: int) -> torch.Tensor:
        key = (device, int(total_pages))
        t = self._logical_pages_row_cache.get(key)
        if t is None:
            t = torch.arange(int(total_pages), device=device, dtype=torch.int32)
            self._logical_pages_row_cache[key] = t
        return t

    def _get_pos_row(self, *, device: torch.device, n: int) -> torch.Tensor:
        key = (device, int(n))
        t = self._pos_row_cache.get(key)
        if t is None:
            t = torch.arange(int(n), device=device, dtype=torch.int64)
            self._pos_row_cache[key] = t
        return t

    def _tokenblock_cache_put(self, key: tuple, cache: dict) -> None:
        self._flex2_tokenblock_cache_by_sig[key] = cache
        self._flex2_tokenblock_cache_key_order.append(key)
        max_keys = int(self._flex2_tokenblock_cache_max_keys)
        if max_keys > 0 and len(self._flex2_tokenblock_cache_key_order) > max_keys:
            evict = self._flex2_tokenblock_cache_key_order.pop(0)
            self._flex2_tokenblock_cache_by_sig.pop(evict, None)

    def _get_or_build_tokenblock_mask_batched(
        self,
        *,
        device: torch.device,
        batch_size: int,
        q_len: int,
        kv_total_len: int,
        q_block_size: int,
        kv_block_size: int,
    ) -> dict:
        """
        Build a dense token-block BlockMask for page_size==1.

        KV blocks are contiguous token blocks over the *global* KV pool:
          kv_block_id in [0..num_kv_blocks-1], kv_idx = kv_block_id * kv_block_size + offset.

        Correctness is enforced by `_paged_mask_mod`, which uses per-request
        `physical_to_logical_batched` to invalidate tokens not belonging to the request
        (logical=-1) and apply window constraints via kv_start_rel/q_kv_offset.
        """
        bsz = int(batch_size)
        q_len_i = int(q_len)
        kv_total_len_i = int(kv_total_len)
        q_block = int(q_block_size) if int(q_block_size) > 0 else 128
        kv_block = int(kv_block_size) if int(kv_block_size) > 0 else 128
        if q_len_i <= 0:
            raise ValueError(f"q_len must be positive, got {q_len_i}")
        if kv_total_len_i <= 0:
            raise ValueError(f"kv_total_len must be positive, got {kv_total_len_i}")
        if kv_block <= 0:
            raise ValueError(f"kv_block_size must be positive, got {kv_block}")

        static_sig = (device, bsz, q_len_i, kv_total_len_i, q_block, kv_block)
        cache = self._flex2_tokenblock_cache_by_sig.get(static_sig)
        if cache is not None:
            return cache

        q_num_blocks = (q_len_i + q_block - 1) // q_block
        num_kv_blocks = (kv_total_len_i + kv_block - 1) // kv_block
        max_cols = int(num_kv_blocks) + 1
        kv_indices = torch.full(
            (bsz, 1, int(q_num_blocks), int(max_cols)),
            fill_value=int(num_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        if num_kv_blocks > 0:
            kv_indices[:, :, :, : int(num_kv_blocks)] = torch.arange(
                int(num_kv_blocks), device=device, dtype=torch.int32
            ).view(1, 1, 1, -1)
        kv_num_blocks = torch.full(
            (bsz, 1, int(q_num_blocks)),
            fill_value=int(num_kv_blocks),
            dtype=torch.int32,
            device=device,
        )
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=(int(q_block), int(kv_block)),
            mask_mod=self._paged_mask_mod,
            seq_lengths=(int(q_len_i), int(kv_total_len_i)),
        )
        cache = {
            "sig": static_sig,
            "q_num_blocks": int(q_num_blocks),
            "num_kv_blocks": int(num_kv_blocks),
            "kv_indices": kv_indices,
            "kv_num_blocks": kv_num_blocks,
            "block_mask": block_mask,
        }
        self._tokenblock_cache_put(static_sig, cache)
        self._flex2_tokenblock_cache_alloc_ct += 1
        if (os.environ.get("SGLANG_FLEX2_LOG_CACHE") or "").strip() in {"1", "true", "True"}:
            logger.info(
                "Flex2 tokenblock cache alloc_ct=%s sig=%s",
                int(self._flex2_tokenblock_cache_alloc_ct),
                static_sig,
            )
        return cache

    def _ensure_paged_buffers(self, *, total_pages: int) -> None:
        device = self.device
        # +1 slot so invalid entries can safely point to index=total_pages without
        # clobbering valid mappings (and without clamping).
        total_pages_plus1 = int(total_pages) + 1
        if (
            self._physical_to_logical is None
            or int(self._physical_to_logical.numel()) != int(total_pages_plus1)
        ):
            self._physical_to_logical = torch.empty(
                (int(total_pages_plus1),), device=device, dtype=torch.int32
            )
        if self._kv_start_rel is None:
            self._kv_start_rel = torch.zeros((), device=device, dtype=torch.int32)
        if self._q_kv_offset is None:
            self._q_kv_offset = torch.zeros((), device=device, dtype=torch.int32)

    def _ensure_paged_buffers_batched(self, *, total_pages: int, batch_size: int) -> None:
        device = self.device
        total_pages_plus1 = int(total_pages) + 1
        bsz = int(batch_size)
        if (
            self._physical_to_logical_batched is None
            or tuple(self._physical_to_logical_batched.shape) != (bsz, total_pages_plus1)
        ):
            self._physical_to_logical_batched = torch.empty(
                (bsz, total_pages_plus1), device=device, dtype=torch.int32
            )
        if self._kv_start_rel_batched is None or int(self._kv_start_rel_batched.numel()) != bsz:
            self._kv_start_rel_batched = torch.empty((bsz,), device=device, dtype=torch.int32)
        if self._q_kv_offset_batched is None or int(self._q_kv_offset_batched.numel()) != bsz:
            self._q_kv_offset_batched = torch.empty((bsz,), device=device, dtype=torch.int32)

    def _paged_mask_mod(self, _b, _h, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        """
        Map physical kv indices -> logical (window-relative) indices, then apply:
          kv_start_rel <= logical_kv_idx <= q_kv_offset + q_idx
        """
        if self._physical_to_logical is None or self._kv_start_rel is None or self._q_kv_offset is None:
            raise RuntimeError("TorchFlexAttnBackendV2 buffers are not initialized")

        ps = int(self._page_size)
        # kv_idx is physical token index into the global KV pool.
        physical_page = torch.div(kv_idx, ps, rounding_mode="floor").to(torch.int64)
        offset_in_page = (kv_idx - physical_page.to(kv_idx.dtype) * ps).to(torch.int32)

        # Batched fastpath: [B, total_pages+1] lookup + per-request offsets.
        if (
            self._physical_to_logical_batched is not None
            and self._kv_start_rel_batched is not None
            and self._q_kv_offset_batched is not None
        ):
            b = _b
            if torch.is_tensor(b):
                b_i64 = b.to(torch.int64)
                logical_page = self._physical_to_logical_batched[b_i64, physical_page].to(torch.int32)
                lo = self._kv_start_rel_batched[b_i64]
                hi = self._q_kv_offset_batched[b_i64] + q_idx.to(torch.int32)
            else:
                b_int = int(b)
                logical_page = self._physical_to_logical_batched[b_int, physical_page].to(torch.int32)
                lo = self._kv_start_rel_batched[b_int]
                hi = self._q_kv_offset_batched[b_int] + q_idx.to(torch.int32)
        else:
            logical_page = self._physical_to_logical[physical_page].to(torch.int32)
            lo = self._kv_start_rel
            hi = self._q_kv_offset + q_idx.to(torch.int32)
        is_valid = logical_page >= 0
        logical_kv_idx = logical_page * ps + offset_in_page
        allowed = (logical_kv_idx >= lo) & (logical_kv_idx <= hi)
        return is_valid & allowed

    def _build_block_mask_from_pages(
        self,
        *,
        q_len: int,
        kv_total_len: int,
        total_pages: int,
        physical_pages: torch.Tensor,
        q_block_size: int = 128,
    ) -> BlockMask:
        """
        Construct a BlockMask with KV blocks as physical pages.

        Note: BlockMask's internal transpose path assumes the *max column index*
        is < kv_indices.shape[-1]. Therefore we allocate kv_indices with last dim
        == (total_pages + 1) so we can safely use `fill_value=total_pages`.
        """
        if q_len <= 0:
            raise ValueError(f"q_len must be positive, got {q_len}")
        if total_pages <= 0:
            raise ValueError(f"total_pages must be positive, got {total_pages}")

        q_block = int(q_block_size) if int(q_block_size) > 0 else 128
        q_num_blocks = (int(q_len) + q_block - 1) // q_block

        num_pages = int(physical_pages.numel())
        # kv_indices stores KV *block ids* in the global KV sequence.
        # Our KV tensor for flex_attention is the global KV cache pool, so these must be
        # physical page ids (0..total_pages-1), not compacted ids.
        max_cols = int(total_pages) + 1
        kv_indices = torch.full(
            (1, 1, int(q_num_blocks), int(max_cols)),
            fill_value=int(total_pages),
            dtype=torch.int32,
            device=physical_pages.device,
        )
        if num_pages > 0:
            kv_indices[:, :, :, :num_pages] = physical_pages.to(torch.int32).view(1, 1, 1, -1)

        kv_num_blocks = torch.full(
            (1, 1, int(q_num_blocks)),
            fill_value=int(num_pages),
            dtype=torch.int32,
            device=physical_pages.device,
        )

        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=(int(q_block), int(self._page_size)),
            mask_mod=self._paged_mask_mod,
            seq_lengths=(int(q_len), int(kv_total_len)),
        )

    def _build_block_mask_from_pages_batched(
        self,
        *,
        q_len: int,
        kv_total_len: int,
        total_pages: int,
        physical_pages_padded: torch.Tensor,
        kv_num_pages: torch.Tensor,
        q_block_size: int = 128,
    ) -> BlockMask:
        """
        Batched variant of `_build_block_mask_from_pages`.

        Args:
            physical_pages_padded: int32 tensor [B, total_pages] with unused entries filled
                with `total_pages`.
            kv_num_pages: int32 tensor [B] number of valid pages per request.
        """
        if q_len <= 0:
            raise ValueError(f"q_len must be positive, got {q_len}")
        if total_pages <= 0:
            raise ValueError(f"total_pages must be positive, got {total_pages}")
        if physical_pages_padded.ndim != 2:
            raise ValueError(
                f"physical_pages_padded must have shape [B, total_pages], got {tuple(physical_pages_padded.shape)}"
            )

        bsz = int(physical_pages_padded.shape[0])
        if int(physical_pages_padded.shape[1]) != int(total_pages):
            raise ValueError(
                "physical_pages_padded second dim must equal total_pages; "
                f"got {int(physical_pages_padded.shape[1])} vs {int(total_pages)}"
            )
        if kv_num_pages.ndim != 1 or int(kv_num_pages.shape[0]) != bsz:
            raise ValueError(f"kv_num_pages must have shape [B], got {tuple(kv_num_pages.shape)}")

        q_block = int(q_block_size) if int(q_block_size) > 0 else 128
        q_num_blocks = (int(q_len) + q_block - 1) // q_block

        max_cols = int(total_pages) + 1
        kv_indices = torch.full(
            (bsz, 1, int(q_num_blocks), int(max_cols)),
            fill_value=int(total_pages),
            dtype=torch.int32,
            device=physical_pages_padded.device,
        )
        kv_indices[:, :, :, : int(total_pages)] = physical_pages_padded.to(torch.int32).view(
            bsz, 1, 1, int(total_pages)
        )
        kv_num_blocks = kv_num_pages.to(torch.int32).view(bsz, 1, 1).expand(bsz, 1, int(q_num_blocks))

        return BlockMask.from_kv_blocks(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=None,
            full_kv_indices=None,
            BLOCK_SIZE=(int(q_block), int(self._page_size)),
            mask_mod=self._paged_mask_mod,
            seq_lengths=(int(q_len), int(kv_total_len)),
        )

    def _run_flex_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        if (
            not self._logged_run_extend
            and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
        ):
            self._logged_run_extend = True
            logger.info(
                "FlexAttention2 _run_flex_forward_extend entered. page_size=%s window_left=%s causal=%s",
                int(self._page_size),
                int(window_left),
                bool(causal),
            )
        if (os.environ.get("SGLANG_FLEX2_FORCE_FALLBACK") or "").strip() in {"1", "true", "True"}:
            return super()._run_flex_forward_extend(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )

        # FP8 KV-cache compatibility: fall back to V1 gather path for float8 KV.
        try:
            if k_cache.dtype == torch.float8_e5m2 or v_cache.dtype == torch.float8_e5m2:
                raise RuntimeError(
                    "FlexAttention2 does not support fp8_e5m2 KV-cache in this build (quality/perf). "
                    "Use fp8_e4m3 or bf16 KV-cache."
                )
            is_fp8_kv = k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn
        except Exception:
            is_fp8_kv = False
        if is_fp8_kv:
            return TorchFlexAttnBackend._run_flex_forward_extend(
                self,
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )
        # page_size==1 is slower (degenerates to token-blocking), but we allow it when
        # explicitly requested because DFLASH verify currently forces page_size=1.
        if int(self._page_size) <= 1:
            if (os.environ.get("SGLANG_FLEX2_ALLOW_PAGE_SIZE1") or "").strip() not in _DISALLOW_FALLBACK_ENVS:
                return super()._run_flex_forward_extend(
                    query,
                    output,
                    k_cache,
                    v_cache,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    extend_prefix_lens,
                    extend_seq_lens,
                    scaling=scaling,
                    enable_gqa=enable_gqa,
                    causal=causal,
                    sinks=sinks,
                    window_left=window_left,
                )
            if not self._logged_page_size1:
                self._logged_page_size1 = True
                logger.warning(
                    "FlexAttention2 _run_flex_forward_extend running with page_size=1 (requested via SGLANG_FLEX2_ALLOW_PAGE_SIZE1=1). "
                    "This can be significantly slower than page_size=64/128."
                )
            return self._run_flex_forward_extend_tokenblock(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )

        ps = int(self._page_size)
        if int(k_cache.shape[0]) % ps != 0:
            if (os.environ.get("SGLANG_FLEX2_DISALLOW_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS:
                raise RuntimeError(
                    "TorchFlexAttnBackendV2: KV cache is not page-aligned "
                    f"(k_cache.shape[0]={int(k_cache.shape[0])}, page_size={ps}); "
                    "fallback to gather is disallowed by SGLANG_FLEX2_DISALLOW_FALLBACK=1."
                )
            logger.warning(
                "TorchFlexAttnBackendV2: KV cache is not page-aligned (k_cache.shape[0]=%s, page_size=%s). "
                "Falling back to flex_attention (gather).",
                int(k_cache.shape[0]),
                ps,
            )
            return super()._run_flex_forward_extend(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        # Global KV in [H, KV, D] (B is added at call-site via unsqueeze(0)).
        key_global = k_cache.movedim(0, query.dim() - 2)
        value_global = v_cache.movedim(0, query.dim() - 2)

        kv_total_len = int(k_cache.shape[0])
        if kv_total_len % ps != 0:
            if (os.environ.get("SGLANG_FLEX2_DISALLOW_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS:
                raise RuntimeError(
                    "TorchFlexAttnBackendV2: KV cache is not page-aligned "
                    f"(k_cache.shape[0]={int(k_cache.shape[0])}, page_size={ps}); "
                    "fallback to gather is disallowed by SGLANG_FLEX2_DISALLOW_FALLBACK=1."
                )
            logger.warning(
                "TorchFlexAttnBackendV2: KV cache is not page-aligned (k_cache.shape[0]=%s, page_size=%s). "
                "Falling back to flex_attention (gather).",
                int(k_cache.shape[0]),
                int(ps),
            )
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        total_pages = kv_total_len // ps
        self._ensure_paged_buffers(total_pages=total_pages)

        # Optional tunable: q block size in BlockMask.
        q_block_size_env = (os.environ.get("SGLANG_FLEX2_Q_BLOCK") or "").strip()
        q_block_size = int(q_block_size_env) if q_block_size_env else 128
        debug = self._debug_remaining > 0
        if debug:
            num_pages_list: list[int] = []

        # High-ROI fastpath: vectorize extend when all requests share the same
        # extend length and prefix length (common in benchmarking / fixed-chunk prefill).
        if (os.environ.get("SGLANG_FLEX2_DISABLE_VECTORIZED_EXTEND") or "").strip() not in {"1", "true", "True"}:
            try:
                # Avoid synchronizing on GPU tensors just to decide whether the uniform-lens
                # fastpath applies. If lens tensors live on CUDA, skip this fastpath and
                # rely on the bucketed vectorized extend below.
                if getattr(extend_seq_lens, "is_cuda", False) or getattr(extend_prefix_lens, "is_cuda", False):
                    raise RuntimeError("skip uniform-lens fastpath for CUDA lens tensors")
                if (
                    int(extend_seq_lens.numel()) > 0
                    and bool(torch.all(extend_seq_lens == extend_seq_lens[0]).item())
                    and bool(torch.all(extend_prefix_lens == extend_prefix_lens[0]).item())
                    and int(extend_seq_lens[0].item()) > 0
                    and causal
                ):
                    if (
                        not self._logged_vectorized_extend
                        and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
                    ):
                        self._logged_vectorized_extend = True
                        logger.info(
                            "FlexAttention2 vectorized extend enabled (uniform lens). page_size=%s q_block=%s window_left=%s",
                            int(self._page_size),
                            int(q_block_size),
                            int(window_left),
                        )
                        print(
                            f"[flex2] vectorized_extend=1 page_size={int(self._page_size)} q_block={int(q_block_size)} "
                            f"window_left={int(window_left)}",
                            flush=True,
                        )
                    bsz = int(seq_lens.shape[0])
                    q_len = int(extend_seq_lens[0].item())
                    prefix_len = int(extend_prefix_lens[0].item())
                    if int(query.shape[1]) == bsz * q_len:
                        device = req_to_token.device
                        total_pages_i32 = int(total_pages)

                        seq_len_kv = seq_lens.to(torch.int64)
                        abs_q0 = torch.full_like(seq_len_kv, prefix_len, dtype=torch.int64)
                        abs_q_last = seq_len_kv - 1

                        wl = int(window_left)
                        kv_abs_start = (abs_q_last - wl).clamp_min(0) if wl >= 0 else torch.zeros_like(abs_q_last)
                        kv_abs_end = abs_q_last

                        kv_base = torch.div(kv_abs_start, ps, rounding_mode="floor") * ps
                        kv_start_rel = (kv_abs_start - kv_base).to(torch.int32)
                        q_kv_offset = (abs_q0 - kv_base).to(torch.int32)

                        page_start = torch.div(kv_base, ps, rounding_mode="floor")
                        page_end = torch.div(kv_abs_end, ps, rounding_mode="floor")
                        num_window_pages = (page_end - page_start + 1).to(torch.int32).clamp_min(0).clamp_max(total_pages_i32)

                        page_col = self._get_page_col(device=device, total_pages=total_pages_i32)
                        page_starts = kv_base.view(bsz, 1) + page_col.view(1, -1) * ps
                        valid = page_col.view(1, -1) < num_window_pages.to(torch.int64).view(bsz, 1)
                        page_starts_safe = torch.where(valid, page_starts, torch.zeros_like(page_starts))
                        req_tokens = req_to_token.index_select(0, req_pool_indices.to(torch.int64))
                        page_tokens = req_tokens.gather(1, page_starts_safe)
                        physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(torch.int32)
                        physical_pages = torch.where(
                            valid,
                            physical_pages,
                            torch.full_like(physical_pages, fill_value=total_pages_i32, dtype=torch.int32),
                        )

                        self._ensure_paged_buffers_batched(total_pages=total_pages_i32, batch_size=bsz)
                        assert self._physical_to_logical_batched is not None
                        assert self._kv_start_rel_batched is not None
                        assert self._q_kv_offset_batched is not None
                        self._physical_to_logical_batched.fill_(-1)
                        logical_pages = (
                            self._get_logical_pages_row(device=device, total_pages=total_pages_i32)
                            .view(1, -1)
                            .expand(bsz, -1)
                        )
                        self._physical_to_logical_batched.scatter_(1, physical_pages.to(torch.int64), logical_pages)
                        self._kv_start_rel_batched.copy_(kv_start_rel)
                        self._q_kv_offset_batched.copy_(q_kv_offset)

                        # Cache + reuse a single BlockMask instance for this static shape, updating
                        # its backing kv_indices / kv_num_blocks tensors in-place each step.
                        static_sig = (
                            device,
                            int(total_pages_i32),
                            int(bsz),
                            int(q_len),
                            int(q_block_size),
                        )
                        extend_cache = self._flex2_extend_cache_batched_by_sig.get(static_sig)
                        if extend_cache is None:
                            q_block = int(q_block_size) if int(q_block_size) > 0 else 128
                            q_num_blocks = (int(q_len) + q_block - 1) // q_block
                            max_cols = int(total_pages_i32) + 1
                            kv_indices = torch.full(
                                (bsz, 1, int(q_num_blocks), int(max_cols)),
                                fill_value=int(total_pages_i32),
                                dtype=torch.int32,
                                device=device,
                            )
                            kv_num_blocks = torch.empty(
                                (bsz, 1, int(q_num_blocks)), dtype=torch.int32, device=device
                            )
                            block_mask = BlockMask.from_kv_blocks(
                                kv_num_blocks=kv_num_blocks,
                                kv_indices=kv_indices,
                                full_kv_num_blocks=None,
                                full_kv_indices=None,
                                BLOCK_SIZE=(int(q_block), int(self._page_size)),
                                mask_mod=self._paged_mask_mod,
                                seq_lengths=(int(q_len), int(kv_total_len)),
                            )
                            extend_cache = {
                                "kv_indices": kv_indices,
                                "kv_num_blocks": kv_num_blocks,
                                "block_mask": block_mask,
                                "q_num_blocks": int(q_num_blocks),
                            }
                            self._extend_cache_put(static_sig, extend_cache)
                            self._flex2_extend_cache_batched_alloc_ct += 1
                            if (os.environ.get("SGLANG_FLEX2_LOG_CACHE") or "").strip() in {"1", "true", "True"}:
                                logger.info(
                                    "Flex2 extend cache alloc_ct=%s sig=%s",
                                    int(self._flex2_extend_cache_batched_alloc_ct),
                                    static_sig,
                                )

                        kv_indices = extend_cache["kv_indices"]
                        kv_num_blocks = extend_cache["kv_num_blocks"]
                        q_num_blocks = int(extend_cache["q_num_blocks"])
                        # Update per-step physical pages + counts.
                        kv_indices[:, :, :, : int(total_pages_i32)] = physical_pages.view(
                            bsz, 1, 1, int(total_pages_i32)
                        )
                        kv_num_blocks.copy_(
                            num_window_pages.to(torch.int32)
                            .view(bsz, 1, 1)
                            .expand(bsz, 1, int(q_num_blocks))
                        )
                        block_mask = extend_cache["block_mask"]

                        q_b = query.view(query.shape[0], bsz, q_len, query.shape[-1]).permute(1, 0, 2, 3)
                        k_b = key_global.unsqueeze(0).expand(bsz, -1, -1, -1)
                        v_b = value_global.unsqueeze(0).expand(bsz, -1, -1, -1)
                        need_sinks = sinks is not None and sinks.numel() > 0
                        if need_sinks:
                            out, lse = self._flex(
                                q_b,
                                k_b,
                                v_b,
                                block_mask=block_mask,
                                scale=scaling,
                                enable_gqa=enable_gqa,
                                return_lse=True,
                            )
                            out = apply_attention_sinks(out, lse, sinks)
                        else:
                            out = self._flex(
                                q_b,
                                k_b,
                                v_b,
                                block_mask=block_mask,
                                scale=scaling,
                                enable_gqa=enable_gqa,
                                return_lse=False,
                            )

                        out_tokens = out.permute(0, 2, 1, 3).reshape(bsz * q_len, out.shape[1], out.shape[-1])
                        output[: out_tokens.shape[0], :, :] = out_tokens
                        return output
            except Exception:
                # Best-effort fastpath; fall back to per-request loop below.
                pass

        # Bucketed vectorized extend: group requests by q_len so we can run a small
        # number of batched flex calls instead of per-request loops.
        #
        # prefix_len can vary across requests; we handle it via the batched paging
        # offsets (kv_start_rel/q_kv_offset) consulted by `_paged_mask_mod`.
        if (os.environ.get("SGLANG_FLEX2_DISABLE_BUCKETED_EXTEND") or "").strip() not in {"1", "true", "True"}:
            try:
                bsz = int(seq_lens.shape[0])
                if bsz > 0 and causal:
                    # Token offsets into the concatenated `query`/`output` token dimension.
                    # extend_seq_lens can be on GPU; use tensor ops to avoid `.item()` in hot paths.
                    extend_seq_lens_i64 = extend_seq_lens.to(torch.int64)
                    end_pos = torch.cumsum(extend_seq_lens_i64, dim=0)
                    start_pos = end_pos - extend_seq_lens_i64

                    q_lens_i64 = extend_seq_lens_i64
                    valid_mask = q_lens_i64 > 0
                    if bool(valid_mask.any()):
                        device = req_to_token.device
                        total_pages_i32 = int(total_pages)
                        page_col = self._get_page_col(device=device, total_pages=total_pages_i32)

                        if torch.cuda.is_current_stream_capturing():
                            # CUDA graphs require stable control flow and fixed-shape outputs.
                            # Avoid torch.unique/nonzero in capture; treat the batch as a single uniform-q_len group.
                            total_q = int(q.shape[0])
                            if total_q % bsz != 0:
                                raise RuntimeError(
                                    "FlexAttention2 extend during CUDA graph capture requires q.shape[0] divisible "
                                    f"by batch_size. Got total_q={total_q}, batch_size={bsz}."
                                )
                            q_lens_unique = [total_q // bsz]
                        else:
                            q_lens_unique = torch.unique(q_lens_i64.masked_select(valid_mask)).tolist()
                        if (
                            not self._logged_vectorized_extend
                            and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
                        ):
                            self._logged_vectorized_extend = True
                            logger.info(
                                "FlexAttention2 vectorized extend enabled (bucketed). page_size=%s q_block=%s window_left=%s groups=%s",
                                int(self._page_size),
                                int(q_block_size),
                                int(window_left),
                                int(q_lens_unique.numel()),
                            )
                            print(
                                f"[flex2] vectorized_extend=1 mode=bucketed page_size={int(self._page_size)} "
                                f"q_block={int(q_block_size)} window_left={int(window_left)} groups={int(q_lens_unique.numel())}",
                                flush=True,
                            )

                        for q_len in q_lens_unique:
                            q_len = int(q_len)
                            if q_len <= 0:
                                continue
                            if torch.cuda.is_current_stream_capturing():
                                idx_t = torch.arange(bsz, device=device, dtype=torch.int64)
                            else:
                                idx_t = torch.nonzero(q_lens_i64 == q_len, as_tuple=False).view(-1).to(torch.int64)
                            g_bsz = int(idx_t.numel())
                            if g_bsz <= 0:
                                continue

                            seq_len_kv = seq_lens.index_select(0, idx_t).to(torch.int64)
                            abs_q0 = extend_prefix_lens.index_select(0, idx_t).to(torch.int64)
                            abs_q_last = seq_len_kv - 1

                            wl = int(window_left)
                            kv_abs_start = (abs_q_last - wl).clamp_min(0) if wl >= 0 else torch.zeros_like(abs_q_last)
                            kv_abs_end = abs_q_last

                            kv_base = torch.div(kv_abs_start, ps, rounding_mode="floor") * ps
                            kv_start_rel = (kv_abs_start - kv_base).to(torch.int32)
                            q_kv_offset = (abs_q0 - kv_base).to(torch.int32)

                            page_start = torch.div(kv_base, ps, rounding_mode="floor")
                            page_end = torch.div(kv_abs_end, ps, rounding_mode="floor")
                            num_window_pages = (page_end - page_start + 1).to(torch.int32).clamp_min(0).clamp_max(total_pages_i32)

                            page_starts = kv_base.view(g_bsz, 1) + page_col.view(1, -1) * ps
                            valid = page_col.view(1, -1) < num_window_pages.to(torch.int64).view(g_bsz, 1)
                            page_starts_safe = torch.where(valid, page_starts, torch.zeros_like(page_starts))

                            req_pool_idx = req_pool_indices.index_select(0, idx_t).to(torch.int64)
                            req_tokens = req_to_token.index_select(0, req_pool_idx)
                            page_tokens = req_tokens.gather(1, page_starts_safe)
                            physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(torch.int64)
                            # Sentinel for padding (maps to total_pages which is out-of-range).
                            physical_pages = torch.where(
                                valid,
                                physical_pages,
                                torch.full_like(physical_pages, total_pages_i32, dtype=torch.int64),
                            )

                            self._ensure_paged_buffers_batched(total_pages=total_pages_i32, batch_size=g_bsz)
                            assert self._physical_to_logical_batched is not None
                            assert self._kv_start_rel_batched is not None
                            assert self._q_kv_offset_batched is not None
                            self._physical_to_logical_batched.fill_(-1)
                            # Scatter logical page indices for valid pages.
                            logical_pages = (
                                self._get_logical_pages_row(device=device, total_pages=total_pages_i32)
                                .view(1, -1)
                                .expand(g_bsz, -1)
                            )
                            scatter_vals = torch.where(valid, logical_pages, torch.full_like(logical_pages, -1))
                            self._physical_to_logical_batched.scatter_(
                                1, physical_pages.to(torch.int64).clamp_max(total_pages_i32), scatter_vals
                            )
                            self._kv_start_rel_batched.copy_(kv_start_rel)
                            self._q_kv_offset_batched.copy_(q_kv_offset)

                            # Reuse a single BlockMask instance for this static shape, updating
                            # its backing kv_indices/kv_num_blocks tensors in-place each step.
                            static_sig = (
                                device,
                                int(total_pages_i32),
                                int(g_bsz),
                                int(q_len),
                                int(q_block_size),
                            )
                            extend_cache = self._flex2_extend_cache_batched_by_sig.get(static_sig)
                            if extend_cache is None:
                                q_block = int(q_block_size) if int(q_block_size) > 0 else 128
                                q_num_blocks = (int(q_len) + q_block - 1) // q_block
                                max_cols = int(total_pages_i32) + 1
                                kv_indices = torch.full(
                                    (g_bsz, 1, int(q_num_blocks), int(max_cols)),
                                    fill_value=int(total_pages_i32),
                                    dtype=torch.int32,
                                    device=device,
                                )
                                kv_num_blocks = torch.empty((g_bsz, 1, int(q_num_blocks)), dtype=torch.int32, device=device)
                                block_mask = BlockMask.from_kv_blocks(
                                    kv_num_blocks=kv_num_blocks,
                                    kv_indices=kv_indices,
                                    full_kv_num_blocks=None,
                                    full_kv_indices=None,
                                    BLOCK_SIZE=(int(q_block), int(self._page_size)),
                                    mask_mod=self._paged_mask_mod,
                                    seq_lengths=(int(q_len), int(kv_total_len)),
                                )
                                extend_cache = {
                                    "kv_indices": kv_indices,
                                    "kv_num_blocks": kv_num_blocks,
                                    "block_mask": block_mask,
                                    "q_num_blocks": int(q_num_blocks),
                                }
                                self._extend_cache_put(static_sig, extend_cache)
                                self._flex2_extend_cache_batched_alloc_ct += 1
                                if (os.environ.get("SGLANG_FLEX2_LOG_CACHE") or "").strip() in {"1", "true", "True"}:
                                    logger.info(
                                        "Flex2 extend cache alloc_ct=%s sig=%s",
                                        int(self._flex2_extend_cache_batched_alloc_ct),
                                        static_sig,
                                    )

                            kv_indices = extend_cache["kv_indices"]
                            kv_num_blocks = extend_cache["kv_num_blocks"]
                            q_num_blocks = int(extend_cache["q_num_blocks"])
                            kv_indices[:, :, :, : int(total_pages_i32)] = physical_pages.to(torch.int32).view(
                                g_bsz, 1, 1, int(total_pages_i32)
                            )
                            kv_num_blocks.copy_(
                                num_window_pages.to(torch.int32)
                                .view(g_bsz, 1, 1)
                                .expand(g_bsz, 1, int(q_num_blocks))
                            )
                            block_mask = extend_cache["block_mask"]

                            # Gather query tokens for this group.
                            starts = start_pos.index_select(0, idx_t).to(torch.int64)
                            token_cols = torch.arange(int(q_len), device=device, dtype=torch.int64).view(1, -1)
                            token_idx = starts.view(g_bsz, 1) + token_cols
                            q_tokens = query[:, token_idx.reshape(-1), :]
                            q_b = q_tokens.view(query.shape[0], g_bsz, int(q_len), query.shape[-1]).permute(1, 0, 2, 3)

                            k_b = key_global.unsqueeze(0).expand(g_bsz, -1, -1, -1)
                            v_b = value_global.unsqueeze(0).expand(g_bsz, -1, -1, -1)
                            if need_sinks:
                                out, lse = self._flex(
                                    q_b,
                                    k_b,
                                    v_b,
                                    block_mask=block_mask,
                                    scale=scaling,
                                    enable_gqa=enable_gqa,
                                    return_lse=True,
                                )
                                out = apply_attention_sinks(out, lse, sinks)
                            else:
                                out = self._flex(
                                    q_b,
                                    k_b,
                                    v_b,
                                    block_mask=block_mask,
                                    scale=scaling,
                                    enable_gqa=enable_gqa,
                                    return_lse=False,
                                )
                            out_tokens = out.permute(0, 2, 1, 3).reshape(g_bsz * int(q_len), out.shape[1], out.shape[-1])
                            output[token_idx.reshape(-1), :, :] = out_tokens

                        return output
            except Exception:
                pass

        # CUDA graph capture cannot tolerate falling back to the per-request extend path
        # (it has Python loops and can allocate/cause compile cache churn).
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlexAttention2 extend fell back to per-request path during CUDA graph capture. "
                "CUDA graphs require the vectorized/batched extend path."
            )

        disallow_extend_fallback = (os.environ.get("SGLANG_FLEX2_DISALLOW_EXTEND_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS
        if disallow_extend_fallback:
            raise RuntimeError(
                "FlexAttention2 extend fell back to per-request path, but "
                "SGLANG_FLEX2_DISALLOW_EXTEND_FALLBACK=1 is set."
            )

        if (
            not self._logged_vectorized_extend
            and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
        ):
            self._logged_vectorized_extend = True
            logger.info(
                "FlexAttention2 vectorized extend disabled or not applicable; using per-request loop. "
                "page_size=%s window_left=%s",
                int(self._page_size),
                int(window_left),
            )
            print(
                f"[flex2] vectorized_extend=0 page_size={int(self._page_size)} window_left={int(window_left)}",
                flush=True,
            )

        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = int(extend_seq_lens[seq_idx])
            prefill_seq_len_q = int(extend_prefix_lens[seq_idx])
            seq_len_kv = int(seq_lens[seq_idx])

            if extend_seq_len_q == 0:
                continue

            if not causal:
                # Extend is only used for causal modes in SGLang right now.
                raise NotImplementedError("TorchFlexAttnBackendV2 only supports causal extend.")

            abs_q0 = prefill_seq_len_q
            abs_q_last = seq_len_kv - 1

            wl = int(window_left)
            if wl >= 0:
                kv_abs_start = max(0, abs_q_last - wl)
            else:
                kv_abs_start = 0
            kv_abs_end = abs_q_last

            # Align window start to page boundary so kv blocks map 1:1 to pages.
            kv_base = (kv_abs_start // ps) * ps
            kv_start_rel = kv_abs_start - kv_base
            q_kv_offset = abs_q0 - kv_base

            page_start = kv_base // ps
            page_end = kv_abs_end // ps
            num_window_pages = int(page_end - page_start + 1)
            if debug:
                num_pages_list.append(num_window_pages)
            page_starts = torch.arange(
                num_window_pages, device=req_to_token.device, dtype=torch.int64
            ) * ps + int(kv_base)

            req_pool_idx = req_pool_indices[seq_idx]
            page_tokens = req_to_token[req_pool_idx, page_starts]
            physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(torch.int64)

            # Update mapping buffers for mask_mod.
            assert self._physical_to_logical is not None
            assert self._kv_start_rel is not None
            assert self._q_kv_offset is not None
            self._physical_to_logical.fill_(-1)
            self._physical_to_logical[physical_pages] = torch.arange(
                num_window_pages, device=self._physical_to_logical.device, dtype=torch.int32
            )
            self._kv_start_rel.fill_(int(kv_start_rel))
            self._q_kv_offset.fill_(int(q_kv_offset))

            block_mask = self._build_block_mask_from_pages(
                q_len=int(extend_seq_len_q),
                kv_total_len=kv_total_len,
                total_pages=total_pages,
                physical_pages=physical_pages,
                q_block_size=q_block_size,
            )

            end_q = start_q + extend_seq_len_q
            per_req_query = query[:, start_q:end_q, :]

            need_sinks = sinks is not None and sinks.numel() > 0
            if need_sinks:
                out, lse = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=False,
                )

            per_req_out = out.squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        if debug:
            try:
                mean_pages = float(sum(num_pages_list)) / max(1, len(num_pages_list))
                print(
                    "[flex2-debug] decode "
                    f"page_size={ps} "
                    f"q_block={int(q_block_size)} "
                    f"pages_mean={mean_pages:.2f} "
                    f"pages_max={(max(num_pages_list) if num_pages_list else 0)} "
                    f"window_left={int(window_left)}",
                    flush=True,
                )
            except Exception:
                pass
            self._debug_remaining = max(0, int(self._debug_remaining) - 1)

        return output

    def _get_or_build_decode_cache(
        self,
        *,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        window_left: int,
        kv_total_len: int,
    ) -> list[tuple[torch.Tensor, int, int]]:
        """
        Cache per-request paging math for the *current* forward_batch so we don't
        recompute it for every layer.

        Returns a list of (physical_pages, kv_start_rel, q_kv_offset) per request.
        """
        cache = getattr(forward_batch, "_flex2_decode_cache", None)
        ps = int(self._page_size)
        kv_total_len = int(kv_total_len)
        # NOTE: Do NOT call `.item()` on CUDA tensors here; this function can be
        # executed under CUDA graph capture. Per-step `forward_batch` objects are
        # already unique in the non-cuda-graph path, so we only key on static
        # shapes/backends.
        batch_sig = (ps, kv_total_len, int(window_left))
        if cache is None or cache.get("sig") != batch_sig:
            by_req: list[tuple[torch.Tensor, int, int]] = []
            for seq_idx in range(seq_lens.shape[0]):
                seq_len_kv = int(seq_lens[seq_idx])
                abs_qpos = seq_len_kv - 1
                wl = int(window_left)
                kv_abs_start = max(0, abs_qpos - wl) if wl >= 0 else 0
                kv_abs_end = abs_qpos

                kv_base = (kv_abs_start // ps) * ps
                kv_start_rel = kv_abs_start - kv_base
                q_kv_offset = abs_qpos - kv_base

                page_start = kv_base // ps
                page_end = kv_abs_end // ps
                num_window_pages = int(page_end - page_start + 1)
                page_starts = (
                    torch.arange(
                        num_window_pages, device=req_to_token.device, dtype=torch.int64
                    )
                    * ps
                    + int(kv_base)
                )
                req_pool_idx = req_pool_indices[seq_idx]
                page_tokens = req_to_token[req_pool_idx, page_starts]
                physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(
                    torch.int64
                )
                by_req.append((physical_pages, int(kv_start_rel), int(q_kv_offset)))
            cache = {"sig": batch_sig, "by_req": by_req}
            setattr(forward_batch, "_flex2_decode_cache", cache)
        return cache["by_req"]

    def _get_or_build_decode_cache_batched(
        self,
        *,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        window_left: int,
        q_block_size: int,
        kv_total_len: int,
    ) -> dict:
        """
        Batched decode cache for the current forward_batch (reused across layers).

        This removes per-request Python loops in the hot decode path by building:
          - physical_pages [B, total_pages] (padded with sentinel=total_pages),
          - physical_to_logical [B, total_pages+1],
          - kv_indices/kv_num_blocks for q_len=1.
        """
        ps = int(self._page_size)
        kv_total_len = int(kv_total_len)
        bsz = int(seq_lens.shape[0])
        wl = int(window_left)
        total_pages = int(kv_total_len // ps)
        device = req_to_token.device
        static_sig = (ps, kv_total_len, wl, bsz, int(q_block_size))
        cache = self._flex2_decode_cache_batched_by_sig.get(static_sig)

        if cache is None:
            self._flex2_decode_cache_batched_alloc_ct += 1
            log_cache = (os.environ.get("SGLANG_FLEX2_LOG_CACHE") or "").strip() in {"1", "true", "True"}
            if log_cache:
                logger.info(
                    "FlexAttention2 decode cache alloc #%s: sig=%s",
                    int(self._flex2_decode_cache_batched_alloc_ct),
                    static_sig,
                )
            physical_pages = torch.empty((bsz, total_pages), device=device, dtype=torch.int32)
            physical_to_logical = torch.empty((bsz, total_pages + 1), device=device, dtype=torch.int32)
            kv_indices = torch.empty((bsz, 1, 1, total_pages + 1), device=device, dtype=torch.int32)
            kv_num_blocks = torch.empty((bsz, 1, 1), device=device, dtype=torch.int32)
            kv_start_rel = torch.empty((bsz,), device=device, dtype=torch.int32)
            q_kv_offset = torch.empty((bsz,), device=device, dtype=torch.int32)

            page_starts = torch.empty((bsz, total_pages), device=device, dtype=torch.int64)
            page_starts_safe = torch.empty((bsz, total_pages), device=device, dtype=torch.int64)
            valid = torch.empty((bsz, total_pages), device=device, dtype=torch.bool)

            # Scratch buffers to reduce per-step allocations (shape-stable decode).
            seq_len_kv_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)
            abs_qpos_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)
            kv_abs_start_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)
            kv_base_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)
            kv_base_i32 = torch.empty((bsz,), device=device, dtype=torch.int32)
            page_start_i32 = torch.empty((bsz,), device=device, dtype=torch.int32)
            page_end_i32 = torch.empty((bsz,), device=device, dtype=torch.int32)
            num_window_pages_i32 = torch.empty((bsz,), device=device, dtype=torch.int32)
            num_window_pages_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)
            req_pool_indices_i64 = torch.empty((bsz,), device=device, dtype=torch.int64)

            page_col = self._get_page_col(device=device, total_pages=total_pages)
            page_col_ps = (page_col * ps).to(torch.int64)

            # For decode (q_len=1), BlockMask transpose output has fixed shapes:
            #   q_num_blocks: [B, 1, max_cols], q_indices: [B, 1, max_cols, 1]
            # Maintain these tensors in-place each step so we can reuse a single
            # BlockMask object (no per-step BlockMask.from_kv_blocks allocations).
            max_cols = int(total_pages) + 1
            q_num_blocks = torch.empty((bsz, 1, max_cols), device=device, dtype=torch.int32)
            q_indices = torch.zeros((bsz, 1, max_cols, 1), device=device, dtype=torch.int32)
            scatter_ones = torch.ones((bsz, 1, max_cols), device=device, dtype=torch.int32)
            block_mask = BlockMask(
                seq_lengths=(1, int(kv_total_len)),
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=None,
                full_kv_indices=None,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_q_num_blocks=None,
                full_q_indices=None,
                BLOCK_SIZE=(int(q_block_size), int(self._page_size)),
                mask_mod=self._paged_mask_mod,
            )

            cache = {
                "static_sig": static_sig,
                "kv_total_len": kv_total_len,
                "total_pages": total_pages,
                "q_block_size": int(q_block_size),
                "physical_pages": physical_pages,
                "physical_to_logical": physical_to_logical,
                "kv_start_rel": kv_start_rel,
                "q_kv_offset": q_kv_offset,
                "kv_indices": kv_indices,
                "kv_num_blocks": kv_num_blocks,
                "page_starts": page_starts,
                "page_starts_safe": page_starts_safe,
                "valid": valid,
                "seq_len_kv_i64": seq_len_kv_i64,
                "abs_qpos_i64": abs_qpos_i64,
                "kv_abs_start_i64": kv_abs_start_i64,
                "kv_base_i64": kv_base_i64,
                "kv_base_i32": kv_base_i32,
                "page_start_i32": page_start_i32,
                "page_end_i32": page_end_i32,
                "num_window_pages_i32": num_window_pages_i32,
                "num_window_pages_i64": num_window_pages_i64,
                "req_pool_indices_i64": req_pool_indices_i64,
                "page_col": page_col,
                "page_col_ps": page_col_ps,
                "q_num_blocks": q_num_blocks,
                "q_indices": q_indices,
                "scatter_ones": scatter_ones,
                "block_mask": block_mask,
                "last_forward_batch_id": None,
            }
            # Insert into a small LRU-ish cache keyed by static_sig.
            self._flex2_decode_cache_batched_by_sig[static_sig] = cache
            self._flex2_decode_cache_batched_key_order.append(static_sig)
            max_keys = int(self._flex2_decode_cache_batched_max_keys)
            if max_keys > 0 and len(self._flex2_decode_cache_batched_key_order) > max_keys:
                evict = self._flex2_decode_cache_batched_key_order.pop(0)
                self._flex2_decode_cache_batched_by_sig.pop(evict, None)
            if log_cache:
                logger.info(
                    "FlexAttention2 decode cache shapes: physical_pages=%s kv_indices=%s q_num_blocks=%s",
                    tuple(cache["physical_pages"].shape),
                    tuple(cache["kv_indices"].shape),
                    tuple(cache["q_num_blocks"].shape),
                )

        # Avoid recomputation across layers within the same decode step: forward_batch is reused.
        fb_id = id(forward_batch)
        if cache.get("last_forward_batch_id") != fb_id:
            # Recompute into existing buffers (no reallocation).
            cache["last_forward_batch_id"] = fb_id
            physical_pages = cache["physical_pages"]
            physical_to_logical = cache["physical_to_logical"]
            kv_start_rel = cache["kv_start_rel"]
            q_kv_offset = cache["q_kv_offset"]
            kv_indices = cache["kv_indices"]
            kv_num_blocks = cache["kv_num_blocks"]
            page_starts = cache["page_starts"]
            page_starts_safe = cache["page_starts_safe"]
            valid = cache["valid"]
            q_num_blocks = cache["q_num_blocks"]

            seq_len_kv_i64 = cache["seq_len_kv_i64"]
            abs_qpos_i64 = cache["abs_qpos_i64"]
            kv_abs_start_i64 = cache["kv_abs_start_i64"]
            kv_base_i64 = cache["kv_base_i64"]
            kv_base_i32 = cache["kv_base_i32"]
            page_start_i32 = cache["page_start_i32"]
            page_end_i32 = cache["page_end_i32"]
            num_window_pages_i32 = cache["num_window_pages_i32"]
            num_window_pages_i64 = cache["num_window_pages_i64"]

            seq_len_kv_i64.copy_(seq_lens)
            abs_qpos_i64.copy_(seq_len_kv_i64)
            abs_qpos_i64.add_(-1)

            if wl >= 0:
                kv_abs_start_i64.copy_(abs_qpos_i64)
                kv_abs_start_i64.add_(-wl)
                kv_abs_start_i64.clamp_min_(0)
            else:
                kv_abs_start_i64.zero_()

            kv_base_i64.copy_(kv_abs_start_i64)
            kv_base_i64.div_(ps, rounding_mode="floor")
            kv_base_i64.mul_(ps)
            kv_base_i32.copy_(kv_base_i64)

            kv_start_rel.copy_(kv_abs_start_i64)
            kv_start_rel.sub_(kv_base_i32)
            q_kv_offset.copy_(abs_qpos_i64)
            q_kv_offset.sub_(kv_base_i32)

            page_start_i32.copy_(kv_base_i64)
            page_start_i32.div_(ps, rounding_mode="floor")
            page_end_i32.copy_(abs_qpos_i64)
            page_end_i32.div_(ps, rounding_mode="floor")
            num_window_pages_i32.copy_(page_end_i32)
            num_window_pages_i32.sub_(page_start_i32)
            num_window_pages_i32.add_(1)
            num_window_pages_i32.clamp_min_(0).clamp_max(total_pages)
            num_window_pages_i64.copy_(num_window_pages_i32)

            # page_starts = kv_base[:, None] + page_col_ps[None, :]
            page_starts.copy_(kv_base_i64.view(bsz, 1))
            page_starts.add_(cache["page_col_ps"].view(1, -1))
            torch.lt(cache["page_col"].view(1, -1), num_window_pages_i64.view(bsz, 1), out=valid)
            page_starts_safe.copy_(page_starts)
            page_starts_safe.masked_fill_(~valid, 0)

            cache["req_pool_indices_i64"].copy_(req_pool_indices)
            req_tokens = req_to_token.index_select(0, cache["req_pool_indices_i64"])
            page_tokens = req_tokens.gather(1, page_starts_safe)
            physical_pages.copy_(page_tokens.to(torch.int32))
            physical_pages.div_(ps, rounding_mode="floor")
            physical_pages.masked_fill_(~valid, total_pages)

            physical_to_logical.fill_(-1)
            logical_row = self._get_logical_pages_row(device=device, total_pages=total_pages)
            physical_to_logical.scatter_(1, physical_pages.to(torch.int64), logical_row.view(1, -1).expand(bsz, -1))

            kv_indices.fill_(total_pages)
            kv_indices[:, :, :, :total_pages] = physical_pages.view(bsz, 1, 1, total_pages)
            kv_num_blocks.copy_(num_window_pages_i32.view(bsz, 1, 1))

            # Decode transpose (q_len=1): q_indices is always zero; q_num_blocks is
            # just membership of each KV block column in kv_indices (0/1).
            # Keep shapes stable and avoid allocations.
            q_num_blocks.zero_()
            q_num_blocks.scatter_(2, kv_indices.squeeze(2), cache["scatter_ones"])
            q_num_blocks[:, :, int(total_pages)] = 0

        return cache

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if self._fa_delegate is not None:
            return self._fa_delegate.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=kwargs.get("q_rope", None),
                k_rope=kwargs.get("k_rope", None),
                sinks=kwargs.get("sinks", None),
            )
        if (os.environ.get("SGLANG_FLEX2_FORCE_FALLBACK") or "").strip() in {"1", "true", "True"}:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        # Mostly copied from TorchFlexAttnBackend.forward_decode, but caches
        # paging math across layers to reduce overhead.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        # If not using paged mode, fall back to base behavior unless explicitly allowed.
        if int(self._page_size) <= 1:
            if (os.environ.get("SGLANG_FLEX2_ALLOW_PAGE_SIZE1") or "").strip() not in _DISALLOW_FALLBACK_ENVS:
                return super().forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            if not self._logged_page_size1:
                self._logged_page_size1 = True
                logger.warning(
                    "FlexAttention2 forward_decode running with page_size=1 (requested via SGLANG_FLEX2_ALLOW_PAGE_SIZE1=1). "
                    "This can be significantly slower than page_size=64/128."
                )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        window_left = int(getattr(layer, "sliding_window_size", -1))

        ps = int(self._page_size)
        query = q_.movedim(0, q_.dim() - 2)
        output = o_
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        key_global = k_cache.movedim(0, query.dim() - 2)
        value_global = v_cache.movedim(0, query.dim() - 2)

        kv_total_len = int(k_cache.shape[0])

        # FP8 KV-cache compatibility:
        # FlexAttention2's fast path operates directly over the *global* KV cache pool.
        # If the KV cache is stored in float8 (fp8_e4m3/fp8_e5m2), some PyTorch builds
        # cannot run FlexAttention kernels directly on float8 K/V. Additionally, even
        # when supported, casting the entire global KV cache to bf16 would destroy the
        # memory advantage and can be prohibitively expensive.
        #
        # For correctness, when KV is float8 we fall back to the V1 gather path
        # (TorchFlexAttnBackend), which materializes only the per-request K/V window.
        # This is slower than the V2 path but avoids global-KV casts.
        try:
            if k_cache.dtype == torch.float8_e5m2 or v_cache.dtype == torch.float8_e5m2:
                raise RuntimeError(
                    "FlexAttention2 does not support fp8_e5m2 KV-cache in this build (quality/perf). "
                    "Use fp8_e4m3 or bf16 KV-cache."
                )
            is_fp8_kv = k_cache.dtype == torch.float8_e4m3fn or v_cache.dtype == torch.float8_e4m3fn
        except Exception:
            is_fp8_kv = False
        if is_fp8_kv:
            if not getattr(self, "_logged_fp8_kv_fallback", False):
                self._logged_fp8_kv_fallback = True
                logger.warning(
                    "FlexAttention2 KV-cache is float8; falling back to gather-based FlexAttention (V1) for correctness. "
                    "For best FP8-KV performance, prefer FA3/FA4 for the target model and keep draft KV in bf16."
                )
            # Explicitly call the V1 gather implementation to avoid re-entering V2.
            TorchFlexAttnBackend._run_flex_forward_decode(
                self,
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=True,
                sinks=kwargs.get("sinks", None),
                window_left=window_left,
            )
            return o
        # page_size==1 path: avoid O(total_pages) page-table buffers by using
        # dense token-block masking over the global KV pool.
        if int(self._page_size) <= 1:
            self._run_flex_forward_decode_tokenblock(
                query.movedim(query.dim() - 2, 0),  # back to [T, H, D]
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=True,
                sinks=sinks,
                window_left=window_left,
            )
            return o
        total_pages = kv_total_len // ps
        self._ensure_paged_buffers(total_pages=total_pages)

        q_block_size_env = (os.environ.get("SGLANG_FLEX2_Q_BLOCK") or "").strip()
        q_block_size = int(q_block_size_env) if q_block_size_env else 128

        sinks = kwargs.get("sinks", None)
        need_sinks = sinks is not None and getattr(sinks, "numel", lambda: 0)() > 0

        # Vectorized decode: a single flex_attention call over the batch.
        if (os.environ.get("SGLANG_FLEX2_DISABLE_VECTORIZED_DECODE") or "").strip() not in {"1", "true", "True"}:
            timing = (os.environ.get("SGLANG_FLEX2_TIMING") or "").strip() in {"1", "true", "True"}
            if timing and not hasattr(self, "_flex2_timing_decode"):
                # Aggregate timing to avoid spamming logs.
                self._flex2_timing_decode = {"cache_ms": 0.0, "flex_ms": 0.0, "ct": 0, "every": 10}

            if timing:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            cache_b = self._get_or_build_decode_cache_batched(
                forward_batch=forward_batch,
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                window_left=window_left,
                q_block_size=q_block_size,
                kv_total_len=kv_total_len,
            )
            if timing:
                ev1.record()
            if int(cache_b["total_pages"]) != int(total_pages):
                raise RuntimeError(
                    "FlexAttention2 decode cache total_pages mismatch: "
                    f"cache={int(cache_b['total_pages'])} vs expected={int(total_pages)}"
                )
            if (
                not self._logged_vectorized_decode
                and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
            ):
                self._logged_vectorized_decode = True
                logger.info(
                    "FlexAttention2 vectorized decode enabled. page_size=%s q_block=%s window_left=%s kv_total_len=%s",
                    int(self._page_size),
                    int(q_block_size),
                    int(window_left),
                    int(cache_b.get("kv_total_len", -1)),
                )
                print(
                    f"[flex2] vectorized_decode=1 page_size={int(self._page_size)} q_block={int(q_block_size)} "
                    f"window_left={int(window_left)} kv_total_len={int(cache_b.get('kv_total_len', -1))}",
                    flush=True,
                )
            bsz = int(seq_lens.shape[0])
            # Avoid per-layer copies: point the mask_mod buffers directly at the
            # cached per-step tensors.
            self._physical_to_logical_batched = cache_b["physical_to_logical"]
            self._kv_start_rel_batched = cache_b["kv_start_rel"]
            self._q_kv_offset_batched = cache_b["q_kv_offset"]
            # Reuse a single BlockMask instance across decode steps. Its backing
            # tensors are updated in-place in _get_or_build_decode_cache_batched.
            block_mask = cache_b["block_mask"]

            q_b = query.permute(1, 0, 2).unsqueeze(2)
            k_b = key_global.unsqueeze(0).expand(bsz, -1, -1, -1)
            v_b = value_global.unsqueeze(0).expand(bsz, -1, -1, -1)
            if timing:
                ev2 = torch.cuda.Event(enable_timing=True)
                ev3 = torch.cuda.Event(enable_timing=True)
                ev2.record()
            if need_sinks:
                out, lse = self._flex(
                    q_b,
                    k_b,
                    v_b,
                    block_mask=block_mask,
                    scale=layer.scaling,
                    enable_gqa=use_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    q_b,
                    k_b,
                    v_b,
                    block_mask=block_mask,
                    scale=layer.scaling,
                    enable_gqa=use_gqa,
                    return_lse=False,
                )
            if timing:
                ev3.record()
            output[:, :, :] = out.squeeze(2)
            if timing:
                torch.cuda.synchronize()
                self._flex2_timing_decode["cache_ms"] += float(ev0.elapsed_time(ev1))
                self._flex2_timing_decode["flex_ms"] += float(ev2.elapsed_time(ev3))
                self._flex2_timing_decode["ct"] += 1
                every = int(self._flex2_timing_decode.get("every", 10))
                if every > 0 and int(self._flex2_timing_decode["ct"]) % every == 0:
                    ct = float(self._flex2_timing_decode["ct"])
                    print(
                        "[flex2-timing] decode "
                        f"cache_ms_avg={self._flex2_timing_decode['cache_ms']/ct:.3f} "
                        f"flex_ms_avg={self._flex2_timing_decode['flex_ms']/ct:.3f} "
                        f"page_size={int(self._page_size)} window_left={int(window_left)}",
                        flush=True,
                    )
            return o

        # CUDA graph capture cannot tolerate the per-request fallback path below
        # (it has Python loops and historically used `.item()` on CUDA tensors).
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlexAttention2 decode fell back to per-request path during CUDA graph capture. "
                "This indicates the vectorized decode cache was not applicable; "
                "CUDA graphs require the vectorized path."
            )

        # Fallback: per-request loop.
        if (
            not self._logged_vectorized_decode
            and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
        ):
            self._logged_vectorized_decode = True
            logger.info(
                "FlexAttention2 vectorized decode disabled or not applicable; using per-request loop. "
                "page_size=%s window_left=%s",
                int(self._page_size),
                int(window_left),
            )
            print(
                f"[flex2] vectorized_decode=0 page_size={int(self._page_size)} window_left={int(window_left)}",
                flush=True,
            )
        cache_by_req = self._get_or_build_decode_cache(
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            window_left=window_left,
            kv_total_len=kv_total_len,
        )
        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            physical_pages, kv_start_rel, q_kv_offset = cache_by_req[seq_idx]

            assert self._physical_to_logical is not None
            assert self._kv_start_rel is not None
            assert self._q_kv_offset is not None
            self._physical_to_logical.fill_(-1)
            self._physical_to_logical[physical_pages] = torch.arange(
                int(physical_pages.numel()),
                device=self._physical_to_logical.device,
                dtype=torch.int32,
            )
            self._kv_start_rel.fill_(int(kv_start_rel))
            self._q_kv_offset.fill_(int(q_kv_offset))

            block_mask = self._build_block_mask_from_pages(
                q_len=1,
                kv_total_len=kv_total_len,
                total_pages=total_pages,
                physical_pages=physical_pages,
                q_block_size=q_block_size,
            )

            end_q = start_q + 1
            per_req_query = query[:, start_q:end_q, :]
            if need_sinks:
                out, lse = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=layer.scaling,
                    enable_gqa=use_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=layer.scaling,
                    enable_gqa=use_gqa,
                    return_lse=False,
                )

            per_req_out = out.squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if self._fa_delegate is not None:
            return self._fa_delegate.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=kwargs.get("q_rope", None),
                k_rope=kwargs.get("k_rope", None),
                sinks=kwargs.get("sinks", None),
            )
        return super().forward_extend(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def _run_flex_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        if (
            not self._logged_run_decode
            and (os.environ.get("SGLANG_FLEX2_LOG_VECTORIZED") or "").strip() in {"1", "true", "True"}
        ):
            self._logged_run_decode = True
            logger.info(
                "FlexAttention2 _run_flex_forward_decode entered. page_size=%s window_left=%s causal=%s",
                int(self._page_size),
                int(window_left),
                bool(causal),
            )
        # page_size==1 is slower (degenerates to token-blocking), but we allow it when
        # explicitly requested because DFLASH verify currently forces page_size=1.
        if int(self._page_size) <= 1:
            if (os.environ.get("SGLANG_FLEX2_ALLOW_PAGE_SIZE1") or "").strip() not in _DISALLOW_FALLBACK_ENVS:
                return super()._run_flex_forward_decode(
                    query,
                    output,
                    k_cache,
                    v_cache,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    scaling=scaling,
                    enable_gqa=enable_gqa,
                    causal=causal,
                    sinks=sinks,
                    window_left=window_left,
                )
            if not self._logged_page_size1:
                self._logged_page_size1 = True
                logger.warning(
                    "FlexAttention2 _run_flex_forward_decode running with page_size=1 (requested via SGLANG_FLEX2_ALLOW_PAGE_SIZE1=1). "
                    "This can be significantly slower than page_size=64/128."
                )
            return self._run_flex_forward_decode_tokenblock(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )

        ps = int(self._page_size)
        if int(k_cache.shape[0]) % ps != 0:
            if (os.environ.get("SGLANG_FLEX2_DISALLOW_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS:
                raise RuntimeError(
                    "TorchFlexAttnBackendV2: KV cache is not page-aligned "
                    f"(k_cache.shape[0]={int(k_cache.shape[0])}, page_size={ps}); "
                    "fallback to gather is disallowed by SGLANG_FLEX2_DISALLOW_FALLBACK=1."
                )
            logger.warning(
                "TorchFlexAttnBackendV2: KV cache is not page-aligned (k_cache.shape[0]=%s, page_size=%s). "
                "Falling back to flex_attention (gather).",
                int(k_cache.shape[0]),
                ps,
            )
            return super()._run_flex_forward_decode(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                sinks=sinks,
                window_left=window_left,
            )

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        key_global = k_cache.movedim(0, query.dim() - 2)
        value_global = v_cache.movedim(0, query.dim() - 2)

        kv_total_len = int(k_cache.shape[0])
        total_pages = kv_total_len // ps
        self._ensure_paged_buffers(total_pages=total_pages)

        q_block_size_env = (os.environ.get("SGLANG_FLEX2_Q_BLOCK") or "").strip()
        q_block_size = int(q_block_size_env) if q_block_size_env else 128

        # Vectorized decode: single flex_attention call over the batch.
        if (os.environ.get("SGLANG_FLEX2_DISABLE_VECTORIZED_DECODE") or "").strip() not in {"1", "true", "True"}:
            try:
                bsz = int(seq_lens.shape[0])
                device = req_to_token.device
                total_pages_i32 = int(total_pages)

                seq_len_kv = seq_lens.to(torch.int64)
                abs_qpos = seq_len_kv - 1
                wl = int(window_left)
                kv_abs_start = (abs_qpos - wl).clamp_min(0) if wl >= 0 else torch.zeros_like(abs_qpos)
                kv_abs_end = abs_qpos

                kv_base = torch.div(kv_abs_start, ps, rounding_mode="floor") * ps
                kv_start_rel = (kv_abs_start - kv_base).to(torch.int32)
                q_kv_offset = (abs_qpos - kv_base).to(torch.int32)

                page_start = torch.div(kv_base, ps, rounding_mode="floor")
                page_end = torch.div(kv_abs_end, ps, rounding_mode="floor")
                num_window_pages = (page_end - page_start + 1).to(torch.int32).clamp_min(0).clamp_max(total_pages_i32)

                page_col = self._get_page_col(device=device, total_pages=total_pages_i32)
                page_starts = kv_base.view(bsz, 1) + page_col.view(1, -1) * ps
                valid = page_col.view(1, -1) < num_window_pages.to(torch.int64).view(bsz, 1)
                page_starts_safe = torch.where(valid, page_starts, torch.zeros_like(page_starts))
                req_tokens = req_to_token.index_select(0, req_pool_indices.to(torch.int64))
                page_tokens = req_tokens.gather(1, page_starts_safe)
                physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(torch.int32)
                physical_pages = torch.where(
                    valid,
                    physical_pages,
                    torch.full_like(physical_pages, fill_value=total_pages_i32, dtype=torch.int32),
                )

                self._ensure_paged_buffers_batched(total_pages=total_pages_i32, batch_size=bsz)
                assert self._physical_to_logical_batched is not None
                assert self._kv_start_rel_batched is not None
                assert self._q_kv_offset_batched is not None
                self._physical_to_logical_batched.fill_(-1)
                logical_pages = (
                    self._get_logical_pages_row(device=device, total_pages=total_pages_i32)
                    .view(1, -1)
                    .expand(bsz, -1)
                )
                self._physical_to_logical_batched.scatter_(1, physical_pages.to(torch.int64), logical_pages)
                self._kv_start_rel_batched.copy_(kv_start_rel)
                self._q_kv_offset_batched.copy_(q_kv_offset)

                block_mask = self._build_block_mask_from_pages_batched(
                    q_len=1,
                    kv_total_len=kv_total_len,
                    total_pages=total_pages_i32,
                    physical_pages_padded=physical_pages,
                    kv_num_pages=num_window_pages.to(torch.int32),
                    q_block_size=q_block_size,
                )

                q_b = query.permute(1, 0, 2).unsqueeze(2)
                k_b = key_global.unsqueeze(0).expand(bsz, -1, -1, -1)
                v_b = value_global.unsqueeze(0).expand(bsz, -1, -1, -1)
                need_sinks = sinks is not None and sinks.numel() > 0
                if need_sinks:
                    out, lse = self._flex(
                        q_b,
                        k_b,
                        v_b,
                        block_mask=block_mask,
                        scale=scaling,
                        enable_gqa=enable_gqa,
                        return_lse=True,
                    )
                    out = apply_attention_sinks(out, lse, sinks)
                else:
                    out = self._flex(
                        q_b,
                        k_b,
                        v_b,
                        block_mask=block_mask,
                        scale=scaling,
                        enable_gqa=enable_gqa,
                        return_lse=False,
                    )
                output[:, :, :] = out.squeeze(2)
                return output
            except Exception:
                # Best-effort; fall back to per-request loop below.
                pass

        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = int(seq_lens[seq_idx])
            abs_qpos = seq_len_kv - 1

            wl = int(window_left)
            if wl >= 0:
                kv_abs_start = max(0, abs_qpos - wl)
            else:
                kv_abs_start = 0
            kv_abs_end = abs_qpos

            kv_base = (kv_abs_start // ps) * ps
            kv_start_rel = kv_abs_start - kv_base
            q_kv_offset = abs_qpos - kv_base

            page_start = kv_base // ps
            page_end = kv_abs_end // ps
            num_window_pages = int(page_end - page_start + 1)
            page_starts = torch.arange(
                num_window_pages, device=req_to_token.device, dtype=torch.int64
            ) * ps + int(kv_base)

            req_pool_idx = req_pool_indices[seq_idx]
            page_tokens = req_to_token[req_pool_idx, page_starts]
            physical_pages = torch.div(page_tokens, ps, rounding_mode="floor").to(torch.int64)

            assert self._physical_to_logical is not None
            assert self._kv_start_rel is not None
            assert self._q_kv_offset is not None
            self._physical_to_logical.fill_(-1)
            self._physical_to_logical[physical_pages] = torch.arange(
                num_window_pages, device=self._physical_to_logical.device, dtype=torch.int32
            )
            self._kv_start_rel.fill_(int(kv_start_rel))
            self._q_kv_offset.fill_(int(q_kv_offset))

            block_mask = self._build_block_mask_from_pages(
                q_len=1,
                kv_total_len=kv_total_len,
                total_pages=total_pages,
                physical_pages=physical_pages,
                q_block_size=q_block_size,
            )

            end_q = start_q + 1
            per_req_query = query[:, start_q:end_q, :]

            need_sinks = sinks is not None and sinks.numel() > 0
            if need_sinks:
                out, lse = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    per_req_query.unsqueeze(0),
                    key_global.unsqueeze(0),
                    value_global.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=False,
                )

            per_req_out = out.squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    def _run_flex_forward_decode_tokenblock(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa: bool = False,
        causal: bool = False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        """
        Token-block decode for page_size==1.

        This avoids building page tables of width kv_total_len (120k+) for extend/prefill and
        keeps CUDA-graph friendliness by reusing a cached BlockMask.
        """
        if not causal:
            raise NotImplementedError("FlexAttention2 tokenblock decode only supports causal=True.")

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        key_global = k_cache.movedim(0, query.dim() - 2)
        value_global = v_cache.movedim(0, query.dim() - 2)

        device = req_to_token.device
        bsz = int(seq_lens.shape[0])
        kv_total_len = int(k_cache.shape[0])

        # Token-block size for dense KV blocks.
        kv_block_size_env = (os.environ.get("SGLANG_FLEX2_TOKEN_KV_BLOCK") or "").strip()
        kv_block_size = int(kv_block_size_env) if kv_block_size_env else 128
        q_block_size_env = (os.environ.get("SGLANG_FLEX2_Q_BLOCK") or "").strip()
        q_block_size = int(q_block_size_env) if q_block_size_env else 128

        # Per-request (window-relative) physical->logical token mapping.
        self._ensure_paged_buffers_batched(total_pages=int(kv_total_len), batch_size=bsz)
        assert self._physical_to_logical_batched is not None
        assert self._kv_start_rel_batched is not None
        assert self._q_kv_offset_batched is not None

        # Choose a stable max_seq during CUDA-graph capture; otherwise use the runtime max.
        max_seq = 0
        if torch.cuda.is_current_stream_capturing():
            max_seq = int(getattr(self.model_runner.server_args, "context_length", 0) or 0)
        else:
            try:
                max_seq = int(seq_lens.max().item())
            except Exception:
                max_seq = int(getattr(self.model_runner.server_args, "context_length", 0) or 0)
        max_seq = max(1, int(min(int(max_seq), int(req_to_token.shape[1]))))

        req_pool_indices_i64 = req_pool_indices.to(torch.int64)
        req_tokens = req_to_token.index_select(0, req_pool_indices_i64)[:, :max_seq]
        pos = self._get_pos_row(device=device, n=max_seq).view(1, -1)

        seq_lens_i64 = seq_lens.to(torch.int64).view(bsz, 1)
        abs_qpos = (seq_lens.to(torch.int64) - 1).view(bsz, 1)
        wl = int(window_left)
        if wl >= 0:
            kv_abs_start = (abs_qpos - int(wl)).clamp_min(0)
        else:
            kv_abs_start = torch.zeros_like(abs_qpos)
        kv_base = kv_abs_start  # page_size==1
        q_kv_offset = (abs_qpos - kv_base).to(torch.int32).view(bsz)

        in_seq = pos < seq_lens_i64
        in_window = pos >= kv_base
        logical = (pos - kv_base).to(torch.int32)
        logical = torch.where(in_seq & in_window, logical, torch.full_like(logical, -1, dtype=torch.int32))

        self._physical_to_logical_batched.fill_(-1)
        self._physical_to_logical_batched.scatter_(1, req_tokens.to(torch.int64).clamp_max(kv_total_len), logical)
        self._kv_start_rel_batched.zero_()
        self._q_kv_offset_batched.copy_(q_kv_offset)

        cache = self._get_or_build_tokenblock_mask_batched(
            device=device,
            batch_size=bsz,
            q_len=1,
            kv_total_len=kv_total_len,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        )
        block_mask = cache["block_mask"]

        # Run flex attention.
        h = int(query.shape[0])
        q_b = query.view(h, bsz, 1, query.shape[-1]).permute(1, 0, 2, 3)
        k_b = key_global.unsqueeze(0).expand(bsz, -1, -1, -1)
        v_b = value_global.unsqueeze(0).expand(bsz, -1, -1, -1)
        need_sinks = sinks is not None and getattr(sinks, "numel", lambda: 0)() > 0
        if need_sinks:
            out, lse = self._flex(
                q_b,
                k_b,
                v_b,
                block_mask=block_mask,
                scale=scaling,
                enable_gqa=enable_gqa,
                return_lse=True,
            )
            out = apply_attention_sinks(out, lse, sinks)
        else:
            out = self._flex(
                q_b,
                k_b,
                v_b,
                block_mask=block_mask,
                scale=scaling,
                enable_gqa=enable_gqa,
                return_lse=False,
            )
        output[:, :, :] = out.squeeze(2)
        return output

    def _run_flex_forward_extend_tokenblock(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa: bool = False,
        causal: bool = False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        """
        Token-block extend/prefill for page_size==1.

        Fastpath supports the common uniform-lens extend case (benchmarking / chunked prefill).
        """
        if not causal:
            raise NotImplementedError("FlexAttention2 tokenblock extend only supports causal=True.")

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        key_global = k_cache.movedim(0, query.dim() - 2)
        value_global = v_cache.movedim(0, query.dim() - 2)

        device = req_to_token.device
        bsz = int(seq_lens.shape[0])
        kv_total_len = int(k_cache.shape[0])

        kv_block_size_env = (os.environ.get("SGLANG_FLEX2_TOKEN_KV_BLOCK") or "").strip()
        kv_block_size = int(kv_block_size_env) if kv_block_size_env else 128

        q_block_size_env = (os.environ.get("SGLANG_FLEX2_Q_BLOCK") or "").strip()
        q_block_size = int(q_block_size_env) if q_block_size_env else 128

        # Uniform-lens extend is the high-ROI case.
        if (
            not getattr(extend_seq_lens, "is_cuda", False)
            and not getattr(extend_prefix_lens, "is_cuda", False)
            and int(extend_seq_lens.numel()) > 0
            and bool(torch.all(extend_seq_lens == extend_seq_lens[0]).item())
            and bool(torch.all(extend_prefix_lens == extend_prefix_lens[0]).item())
            and int(extend_seq_lens[0].item()) > 0
        ):
            q_len = int(extend_seq_lens[0].item())
            prefix_len = int(extend_prefix_lens[0].item())
            if int(query.shape[1]) != bsz * q_len:
                raise RuntimeError(
                    f"Tokenblock extend expected query tokens={bsz*q_len}, got {int(query.shape[1])}."
                )

            self._ensure_paged_buffers_batched(total_pages=int(kv_total_len), batch_size=bsz)
            assert self._physical_to_logical_batched is not None
            assert self._kv_start_rel_batched is not None
            assert self._q_kv_offset_batched is not None

            # Stable max_seq during capture; otherwise runtime max.
            max_seq = 0
            if torch.cuda.is_current_stream_capturing():
                max_seq = int(getattr(self.model_runner.server_args, "context_length", 0) or 0)
            else:
                try:
                    max_seq = int(seq_lens.max().item())
                except Exception:
                    max_seq = int(getattr(self.model_runner.server_args, "context_length", 0) or 0)
            max_seq = max(1, int(min(int(max_seq), int(req_to_token.shape[1]))))

            req_pool_indices_i64 = req_pool_indices.to(torch.int64)
            req_tokens = req_to_token.index_select(0, req_pool_indices_i64)[:, :max_seq]
            pos = self._get_pos_row(device=device, n=max_seq).view(1, -1)

            seq_lens_i64 = seq_lens.to(torch.int64).view(bsz, 1)
            abs_q0 = torch.full((bsz, 1), int(prefix_len), device=device, dtype=torch.int64)
            abs_q_last = (seq_lens.to(torch.int64) - 1).view(bsz, 1)
            wl = int(window_left)
            if wl >= 0:
                kv_abs_start = (abs_q_last - int(wl)).clamp_min(0)
            else:
                kv_abs_start = torch.zeros_like(abs_q_last)
            kv_base = kv_abs_start
            q_kv_offset = (abs_q0 - kv_base).to(torch.int32).view(bsz)

            in_seq = pos < seq_lens_i64
            in_window = pos >= kv_base
            logical = (pos - kv_base).to(torch.int32)
            logical = torch.where(in_seq & in_window, logical, torch.full_like(logical, -1, dtype=torch.int32))

            self._physical_to_logical_batched.fill_(-1)
            self._physical_to_logical_batched.scatter_(1, req_tokens.to(torch.int64).clamp_max(kv_total_len), logical)
            self._kv_start_rel_batched.zero_()
            self._q_kv_offset_batched.copy_(q_kv_offset)

            cache = self._get_or_build_tokenblock_mask_batched(
                device=device,
                batch_size=bsz,
                q_len=int(q_len),
                kv_total_len=kv_total_len,
                q_block_size=int(q_block_size),
                kv_block_size=int(kv_block_size),
            )
            block_mask = cache["block_mask"]

            h = int(query.shape[0])
            q_b = query.view(h, bsz, int(q_len), query.shape[-1]).permute(1, 0, 2, 3)
            k_b = key_global.unsqueeze(0).expand(bsz, -1, -1, -1)
            v_b = value_global.unsqueeze(0).expand(bsz, -1, -1, -1)
            need_sinks = sinks is not None and getattr(sinks, "numel", lambda: 0)() > 0
            if need_sinks:
                out, lse = self._flex(
                    q_b,
                    k_b,
                    v_b,
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    q_b,
                    k_b,
                    v_b,
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=False,
                )
            out_tokens = out.permute(0, 2, 1, 3).reshape(bsz * int(q_len), out.shape[1], out.shape[-1])
            output[: out_tokens.shape[0], :, :] = out_tokens
            return output

        # Fallback: use the base (gather) path for non-uniform extend shapes.
        disallow = (os.environ.get("SGLANG_FLEX2_DISALLOW_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS
        disallow = disallow or (os.environ.get("SGLANG_FLEX2_DISALLOW_EXTEND_FALLBACK") or "").strip() in _DISALLOW_FALLBACK_ENVS
        if disallow:
            raise RuntimeError(
                "FlexAttention2 tokenblock extend hit non-uniform extend shapes and would fall back to gather, "
                "but fallback is disallowed."
            )
        self._flex2_fallback_extend_ct += 1
        if (os.environ.get("SGLANG_FLEX2_LOG_FALLBACK") or "").strip() in {"1", "true", "True"}:
            logger.warning(
                "FlexAttention2 tokenblock extend falling back to gather (ct=%s).",
                int(self._flex2_fallback_extend_ct),
            )
        return super()._run_flex_forward_extend(
            query.movedim(query.dim() - 2, 0),
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            scaling=scaling,
            enable_gqa=enable_gqa,
            causal=causal,
            sinks=sinks,
            window_left=window_left,
        )
