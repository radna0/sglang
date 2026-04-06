from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.jit_kernel.flash_attention_v4 import (
    fa4_hopper_stable_enabled,
    flash_attn_with_kvcache as flash_attn_with_kvcache_fa4,
)
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class FlexFlash4CuteBackend(FlashAttentionBackend):
    """
    Hopper-first custom backend that uses the in-tree CuTe/FA4 path with explicit
    block-sparse metadata for GPT-OSS decode.

    This is intentionally narrow for the first implementation:
    - optimized path: non-MLA decode with compact contiguous KV gathered from the
      request-to-token table, page_size=128, bf16/fp16 KV
    - fallback path: standard FlashAttentionBackend(FA4) for unsupported cases
    """

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, fa_impl_ver=4, **kwargs)
        major, _minor = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError(
                "flex_flash4 requires Hopper-or-newer (SM90+) for the CuTe FA4 path."
            )
        self._custom_block_size = max(1, int(self.page_size))
        self._decode_local_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._decode_full_page_cache: dict[tuple, dict[str, torch.Tensor]] = {}
        self._logged_custom_decode_once = False
        self._logged_skip_reasons: set[str] = set()
        self._logged_sparse_stats_once: set[tuple[str, int]] = set()
        self._logged_swa_index_clamp_once = False
        self._logged_swa_page_compact_once = False
        self._logged_full_graph_page_cap_once = False
        self._logged_dflash_spec_allow_once = False
        self._logged_dflash_force_extend_once = False
        self._sync_debug = str(
            os.environ.get("SGLANG_FLEX_FLASH4_SYNC_DEBUG", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._capture_sync_debug = False
        self._require_custom = str(
            os.environ.get("SGLANG_FLEX_FLASH4_REQUIRE_CUSTOM", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._log_sparse = str(
            os.environ.get("SGLANG_FLEX_FLASH4_LOG_SPARSE", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._disable_swa_page_compaction = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DISABLE_SWA_PAGE_COMPACTION", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._custom_decode_mode = (
            str(
                os.environ.get(
                    "SGLANG_FLEX_FLASH4_CUSTOM_DECODE_MODE", ""
                )
            )
            .strip()
            .lower()
        )
        self._delegate_full_decode = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_FULL_DECODE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        full_decode_delegate_fa_ver_env = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_FULL_DECODE_FA_VER", "4")
        ).strip()
        try:
            self._full_decode_delegate_fa_ver = int(full_decode_delegate_fa_ver_env)
        except ValueError:
            self._full_decode_delegate_fa_ver = 4
        if self._full_decode_delegate_fa_ver not in {3, 4}:
            self._full_decode_delegate_fa_ver = 4
        self._delegate_sliding_decode = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_SLIDING_DECODE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        sliding_decode_delegate_fa_ver_env = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_SLIDING_DECODE_FA_VER", "4")
        ).strip()
        try:
            self._sliding_decode_delegate_fa_ver = int(sliding_decode_delegate_fa_ver_env)
        except ValueError:
            self._sliding_decode_delegate_fa_ver = 4
        if self._sliding_decode_delegate_fa_ver not in {3, 4}:
            self._sliding_decode_delegate_fa_ver = 4
        self._delegate_full_extend = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_FULL_EXTEND", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._delegate_sliding_extend = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_SLIDING_EXTEND", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._delegate_quantized_kv_decode = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DELEGATE_QUANTIZED_KV", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._requested_kv_cache_dtype = str(
            getattr(model_runner.server_args, "kv_cache_dtype", "auto") or "auto"
        ).strip().lower()
        self._enable_custom_extend = str(
            os.environ.get("SGLANG_FLEX_FLASH4_ENABLE_CUSTOM_EXTEND", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._cuda_graph_enabled = not bool(
            getattr(model_runner.server_args, "disable_cuda_graph", False)
        )
        self._allow_full_decode_in_cuda_graph = str(
            os.environ.get("SGLANG_FLEX_FLASH4_ALLOW_FULL_GRAPH_DECODE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._graph_safe_full_decode = not bool(
            getattr(model_runner.server_args, "disable_cuda_graph", False)
        )
        full_graph_page_cap_env = str(
            os.environ.get("SGLANG_FLEX_FLASH4_FULL_GRAPH_PAGE_CAP", "0")
        ).strip()
        try:
            self._full_graph_page_cap = max(0, int(full_graph_page_cap_env or "0"))
        except ValueError:
            self._full_graph_page_cap = 0
        self._graph_cache_batch_capacity_hint = max(
            1,
            int(getattr(model_runner.server_args, "cuda_graph_max_bs", 0) or 0),
        )
        logger.warning(
            "FlexFlash4 init: custom_decode_mode=%s disable_swa_page_compaction=%s delegate_full_decode=%s delegate_full_decode_fa_ver=%s delegate_sliding_decode=%s delegate_sliding_decode_fa_ver=%s allow_full_graph_decode=%s kv_dtype=%s page_size=%s sync_debug=%s",
            self._custom_decode_mode,
            int(self._disable_swa_page_compaction),
            int(self._delegate_full_decode),
            int(self._full_decode_delegate_fa_ver),
            int(self._delegate_sliding_decode),
            int(self._sliding_decode_delegate_fa_ver),
            int(self._allow_full_decode_in_cuda_graph),
            self.kv_cache_dtype_str,
            int(self.page_size),
            int(self._sync_debug),
        )
        # CuTe block-sparse path does not support SplitKV here.
        self.num_splits = 1
        # Prefer FA4 delegation whenever the official Hopper/Blackwell interface is
        # actually available; otherwise fall back to FA3. This keeps paged H100
        # prefill/extend on the newer stable FA4 path instead of hard-clamping it
        # away purely by architecture generation.
        self._extend_delegate_fa_ver = self._resolve_delegate_fa_impl(4)
        self._extend_delegate = FlashAttentionBackend(
            model_runner, fa_impl_ver=self._extend_delegate_fa_ver
        )
        # Speculative DFlash quality is extremely sensitive to attention semantics. Until the
        # FlexFlash4 custom path is parity-proven for the DFlash draft/verify modes, earlier
        # bring-up routed DFlash speculative attention through a conservative FA3 delegate.
        # That changes the target distribution if the production backend is FlexFlash4, so
        # keep it only behind explicit env gates.
        self._dflash_delegate = FlashAttentionBackend(model_runner, fa_impl_ver=3)
        self._dflash_delegate_target_verify_to_fa3 = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DFLASH_DELEGATE_TARGET_VERIFY_FA3", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._dflash_delegate_decode_to_fa3 = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DFLASH_DELEGATE_DECODE_FA3", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._dflash_delegate_extend_to_fa3 = str(
            os.environ.get("SGLANG_FLEX_FLASH4_DFLASH_DELEGATE_EXTEND_FA3", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._full_decode_delegate_fa_ver = self._resolve_delegate_fa_impl(
            self._full_decode_delegate_fa_ver
        )
        self._sliding_decode_delegate_fa_ver = self._resolve_delegate_fa_impl(
            self._sliding_decode_delegate_fa_ver
        )

        self._full_decode_delegate = FlashAttentionBackend(
            model_runner, fa_impl_ver=self._full_decode_delegate_fa_ver
        )
        self._sliding_decode_delegate = FlashAttentionBackend(
            model_runner, fa_impl_ver=self._sliding_decode_delegate_fa_ver
        )
        self._logged_extend_delegate_once = False
        self._logged_full_decode_delegate_once = False
        self._logged_full_extend_delegate_once = False
        self._logged_sliding_decode_delegate_once = False
        self._logged_sliding_extend_delegate_once = False
        self._logged_quantized_kv_delegate_once = False
        self._native_paged_full_decode = str(
            os.environ.get("SGLANG_FLEX_FLASH4_NATIVE_PAGED_FULL_DECODE", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._allow_cute_paged_kv_sm90 = str(
            os.environ.get("SGLANG_FA4_CUTE_ALLOW_PAGED_KV_SM90", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._logged_native_paged_full_decode_once = False

    def _is_dflash_speculative(self, forward_batch: ForwardBatch) -> bool:
        spec_info = getattr(forward_batch, "spec_info", None)
        if getattr(
            forward_batch, "spec_algorithm", SpeculativeAlgorithm.NONE
        ) in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DFLASH_TREE):
            return True
        spec_type = getattr(spec_info, "spec_input_type", None)
        return spec_type in (SpecInputType.DFLASH_VERIFY, SpecInputType.DFLASH_DRAFT)

    def _hopper_fa4_delegate_enabled(self) -> bool:
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            return False
        if major >= 10:
            return True
        if major == 9 and fa4_hopper_stable_enabled():
            return True
        return False

    def _resolve_delegate_fa_impl(self, impl: int) -> int:
        if impl == 4 and not self._hopper_fa4_delegate_enabled():
            return 3
        return impl

    def _sync_dflash_delegate_metadata(self) -> None:
        self._dflash_delegate.forward_metadata = self.forward_metadata
        self._dflash_delegate.forward_metadata_spec_decode_expand = (
            self.forward_metadata_spec_decode_expand
        )

    def _uses_quantized_kv_cache(self) -> bool:
        return self._requested_kv_cache_dtype in {"fp8_e4m3", "fp8_e5m2"}

    def _log_skip_once(self, reason: str) -> None:
        if reason in self._logged_skip_reasons:
            return
        self._logged_skip_reasons.add(reason)
        logger.warning("FlexFlash4 custom decode disabled for path: %s", reason)
        if self._require_custom:
            raise RuntimeError(f"flex_flash4 custom decode required but unavailable: {reason}")

    def _maybe_sync_debug(self, tag: str) -> None:
        if not (self._sync_debug or self._capture_sync_debug):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        torch.cuda.synchronize()
        logger.warning("FlexFlash4 sync_debug checkpoint: %s", tag)

    def _cache_batch_capacity(self, requested_batch_size: int) -> int:
        batch_capacity = int(requested_batch_size)
        if self._cuda_graph_enabled:
            batch_capacity = max(batch_capacity, self._graph_cache_batch_capacity_hint)
        return max(1, batch_capacity)

    def _get_or_create_local_decode_cache(
        self,
        *,
        layer_id: int,
        batch_size: int,
        max_local_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        k_dtype: torch.dtype,
        v_dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        batch_capacity = self._cache_batch_capacity(batch_size)
        key = (
            str(device),
            int(batch_capacity),
            int(max_local_tokens),
            int(num_kv_heads),
            int(head_dim),
            int(v_head_dim),
            str(k_dtype),
            str(v_dtype),
        )
        cached = self._decode_local_cache.get(key)
        if cached is None:
            cached = {
                "batch_capacity": batch_capacity,
                "col_offsets": torch.arange(
                    int(max_local_tokens), dtype=torch.int64, device=device
                ).view(1, -1),
                "token_cols": torch.empty(
                    (batch_capacity, int(max_local_tokens)),
                    dtype=torch.int64,
                    device=device,
                ),
                "valid_cols": torch.empty(
                    (batch_capacity, int(max_local_tokens)),
                    dtype=torch.bool,
                    device=device,
                ),
                "safe_token_cols": torch.empty(
                    (batch_capacity, int(max_local_tokens)),
                    dtype=torch.int64,
                    device=device,
                ),
                "slot_indices": torch.empty(
                    (batch_capacity, int(max_local_tokens)),
                    dtype=torch.int64,
                    device=device,
                ),
                "local_seqlens": torch.empty(
                    (batch_capacity,), dtype=torch.int32, device=device
                ),
                "k_local": torch.empty(
                    (
                        batch_capacity,
                        int(max_local_tokens),
                        int(num_kv_heads),
                        int(head_dim),
                    ),
                    dtype=k_dtype,
                    device=device,
                ),
                "v_local": torch.empty(
                    (
                        batch_capacity,
                        int(max_local_tokens),
                        int(num_kv_heads),
                        int(v_head_dim),
                    ),
                    dtype=v_dtype,
                    device=device,
                ),
            }
            self._decode_local_cache[key] = cached
        return {
            "batch_capacity": cached["batch_capacity"],
            "col_offsets": cached["col_offsets"],
            "token_cols": cached["token_cols"][:batch_size],
            "valid_cols": cached["valid_cols"][:batch_size],
            "safe_token_cols": cached["safe_token_cols"][:batch_size],
            "slot_indices": cached["slot_indices"][:batch_size],
            "local_seqlens": cached["local_seqlens"][:batch_size],
            "k_local": cached["k_local"][:batch_size],
            "v_local": cached["v_local"][:batch_size],
        }

    def _get_or_create_full_decode_page_cache(
        self,
        *,
        layer_id: int,
        batch_size: int,
        page_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        k_dtype: torch.dtype,
        v_dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        batch_capacity = self._cache_batch_capacity(batch_size)
        key = (
            str(device),
            int(batch_capacity),
            int(page_capacity),
            int(num_kv_heads),
            int(head_dim),
            int(v_head_dim),
            str(k_dtype),
            str(v_dtype),
        )
        cached = self._decode_full_page_cache.get(key)
        if cached is None:
            cached = {
                "batch_capacity": batch_capacity,
                "page_offsets": torch.arange(
                    int(page_capacity), dtype=torch.int64, device=device
                ).view(1, -1),
                "token_offsets": torch.arange(
                    int(page_capacity) * int(self.page_size),
                    dtype=torch.int64,
                    device=device,
                ).view(1, -1),
                "page_indices": torch.empty(
                    (batch_capacity, int(page_capacity)),
                    dtype=torch.int64,
                    device=device,
                ),
                "valid_page_mask": torch.empty(
                    (batch_capacity, int(page_capacity)),
                    dtype=torch.bool,
                    device=device,
                ),
                "valid_token_mask": torch.empty(
                    (batch_capacity, int(page_capacity) * int(self.page_size)),
                    dtype=torch.bool,
                    device=device,
                ),
                "gather_pages": torch.empty(
                    (batch_capacity * int(page_capacity),),
                    dtype=torch.int64,
                    device=device,
                ),
                "k_pages": torch.empty(
                    (
                        batch_capacity,
                        int(page_capacity),
                        self.page_size,
                        int(num_kv_heads),
                        int(head_dim),
                    ),
                    dtype=k_dtype,
                    device=device,
                ),
                "v_pages": torch.empty(
                    (
                        batch_capacity,
                        int(page_capacity),
                        self.page_size,
                        int(num_kv_heads),
                        int(v_head_dim),
                    ),
                    dtype=v_dtype,
                    device=device,
                ),
            }
            cached["k_pages_flat"] = cached["k_pages"].view(
                batch_capacity * int(page_capacity),
                self.page_size,
                int(num_kv_heads),
                int(head_dim),
            )
            cached["v_pages_flat"] = cached["v_pages"].view(
                batch_capacity * int(page_capacity),
                self.page_size,
                int(num_kv_heads),
                int(v_head_dim),
            )
            self._decode_full_page_cache[key] = cached
        return {
            "batch_capacity": cached["batch_capacity"],
            "page_offsets": cached["page_offsets"],
            "token_offsets": cached["token_offsets"],
            "page_indices": cached["page_indices"][:batch_size],
            "valid_page_mask": cached["valid_page_mask"][:batch_size],
            "valid_token_mask": cached["valid_token_mask"][:batch_size],
            "gather_pages": cached["gather_pages"],
            "k_pages": cached["k_pages"][:batch_size],
            "v_pages": cached["v_pages"][:batch_size],
            "k_pages_flat": cached["k_pages_flat"],
            "v_pages_flat": cached["v_pages_flat"],
        }

    def _can_use_custom_decode(
        self,
        *,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> bool:
        # TARGET_VERIFY (DFLASH verifier) can have q_len > 1 per request. The custom local
        # decode kernel path is optimized for decode-style q_len==1 and is not a safe
        # drop-in for verifier blocks unless parity has been proven end-to-end.
        #
        # If this guard is wrong, it manifests exactly as "fishy" accept_len (near-max)
        # with token parity mismatches vs baseline under concurrency.
        if forward_batch.forward_mode.is_target_verify():
            allow_env = (os.environ.get("SGLANG_FLEX_FLASH4_ALLOW_CUSTOM_TARGET_VERIFY") or "").strip().lower()
            if allow_env not in {"1", "true", "yes", "on"}:
                self._log_skip_once("target_verify_force_delegate")
                return False

        is_sliding_layer = (
            layer.sliding_window_size is not None and int(layer.sliding_window_size) >= 0
        )
        # Current prompt-sensitive drift is isolated to the batched paged sliding decode
        # path: page_size=1 is exact, page_size>1 with bs=1 is exact, but page_size>1 with
        # bs>1 drifts early under GPT-OSS decode. Until the compact-KV sliding gather is
        # parity-proven for batched paged mode, keep that slice on the native FA path by
        # default and preserve the custom path for bs=1 / page_size=1 bring-up.
        if (
            is_sliding_layer
            and int(self.page_size) > 1
            and int(getattr(forward_batch, "batch_size", 0) or 0) > 1
        ):
            allow_env = (
                os.environ.get(
                    "SGLANG_FLEX_FLASH4_ALLOW_BATCHED_PAGED_SLIDING", ""
                )
            ).strip().lower()
            if allow_env not in {"1", "true", "yes", "on"}:
                self._log_skip_once("paged_sliding_batch")
                return False
        if (
            not is_sliding_layer
            and self._cuda_graph_enabled
            and not self._allow_full_decode_in_cuda_graph
        ):
            self._log_skip_once("full_layer_delegate_cuda_graph")
            return False
        if self._custom_decode_mode == "sliding_only" and not is_sliding_layer:
            self._log_skip_once("full_layer_delegate")
            return False
        if self._custom_decode_mode == "full_only" and is_sliding_layer:
            self._log_skip_once("sliding_layer_delegate")
            return False
        if self.use_mla:
            self._log_skip_once("mla")
            return False
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            self._log_skip_once("cross_attention_or_encoder_only")
            return False
        if forward_batch.spec_info is not None:
            # DFlash uses spec_info as a bookkeeping carrier for TARGET_VERIFY, but (in our
            # current GPT-OSS path) does not require a backend-level custom_mask. Treat
            # it as a "speculative" disqualifier only when masks / top-k expansion are
            # actually needed.
            spec_info = getattr(forward_batch, "spec_info", None)
            allow_target_verify_no_mask = (
                forward_batch.forward_mode.is_target_verify()
                and getattr(spec_info, "custom_mask", None) is None
            )
            if not allow_target_verify_no_mask:
                self._log_skip_once("speculative")
                return False
            if not self._logged_dflash_spec_allow_once and not torch.cuda.is_current_stream_capturing():
                self._logged_dflash_spec_allow_once = True
                logger.warning(
                    "FlexFlash4 custom path enabled for TARGET_VERIFY (spec_info present, custom_mask=None)."
                )
        if int(self.page_size) <= 0:
            self._log_skip_once(f"page_size_{self.page_size}")
            return False
        if self._uses_quantized_kv_cache():
            self._log_skip_once(f"kv_dtype_{self._requested_kv_cache_dtype}")
            return False
        if self.has_local_attention and getattr(self.forward_metadata, "local_attn_metadata", None) is not None:
            self._log_skip_once("local_attention_metadata")
            return False
        return True

    def _is_sliding_layer(self, layer: RadixAttention) -> bool:
        return (
            layer.sliding_window_size is not None and int(layer.sliding_window_size) >= 0
        )

    def _should_delegate_full_decode(self, layer: RadixAttention) -> bool:
        return self._delegate_full_decode and not self._is_sliding_layer(layer)

    def _should_delegate_full_extend(self, layer: RadixAttention) -> bool:
        return self._delegate_full_extend and not self._is_sliding_layer(layer)

    def _should_delegate_sliding_decode(self, layer: RadixAttention) -> bool:
        return self._delegate_sliding_decode and self._is_sliding_layer(layer)

    def _should_delegate_sliding_extend(self, layer: RadixAttention) -> bool:
        return (
            self._delegate_sliding_extend
            and self._is_sliding_layer(layer)
            and fa4_hopper_stable_enabled()
        )

    def _maybe_log_full_decode_delegate(self) -> None:
        if self._logged_full_decode_delegate_once:
            return
        self._logged_full_decode_delegate_once = True
        sliding_desc = (
            f"delegate to FA{int(self._sliding_decode_delegate_fa_ver)}"
            if self._delegate_sliding_decode
            else "stay on custom Flex/FA4"
        )
        logger.warning(
            "FlexFlash4 hybrid decode active: full GPT-OSS layers delegate to FA%s; sliding layers %s.",
            int(self._full_decode_delegate_fa_ver),
            sliding_desc,
        )

    def _maybe_log_full_extend_delegate(self) -> None:
        if self._logged_full_extend_delegate_once:
            return
        self._logged_full_extend_delegate_once = True
        logger.warning(
            "FlexFlash4 hybrid extend active: full GPT-OSS layers delegate to native paged FA4/FA3 extend while sliding layers use the custom policy."
        )

    def _maybe_log_sliding_decode_delegate(self) -> None:
        if self._logged_sliding_decode_delegate_once:
            return
        self._logged_sliding_decode_delegate_once = True
        logger.warning(
            "FlexFlash4 hybrid decode active: GPT-OSS sliding-window decode delegates to FA%s instead of custom Flex/FA4.",
            int(self._sliding_decode_delegate_fa_ver),
        )

    def _maybe_log_sliding_extend_delegate(self) -> None:
        if self._logged_sliding_extend_delegate_once:
            return
        self._logged_sliding_extend_delegate_once = True
        logger.warning(
            "FlexFlash4 fast path active: standard GPT-OSS sliding-window extend delegates to native paged Hopper FA4 when available."
        )

    def _maybe_log_quantized_kv_delegate(self) -> None:
        if self._logged_quantized_kv_delegate_once:
            return
        self._logged_quantized_kv_delegate_once = True
        logger.warning(
            "FlexFlash4 hybrid FP8/quantized KV path active: custom sliding decode is bypassed and native paged FA4 handles quantized KV semantics."
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._extend_delegate.init_cuda_graph_state(max_bs, max_num_tokens)
        self._full_decode_delegate.init_cuda_graph_state(max_bs, max_num_tokens)
        self._sliding_decode_delegate.init_cuda_graph_state(max_bs, max_num_tokens)

    def _build_compact_kv(
        self,
        *,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_device: torch.device,
        use_full_sequence: bool,
    ) -> dict[str, torch.Tensor | int | bool | str]:
        is_sliding_layer = (
            layer.sliding_window_size is not None and int(layer.sliding_window_size) >= 0
        )
        bs = int(forward_batch.seq_lens.shape[0])
        window_left = int(layer.sliding_window_size) if is_sliding_layer else -1
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        if not is_sliding_layer:
            metadata = self.forward_metadata
            page_table = metadata.page_table
            if page_table is None:
                raise RuntimeError("FlexFlash4 full-layer decode requires page_table metadata.")
            seq_lens = forward_batch.seq_lens.to(torch.int32)
            key_cache_pages = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache_pages = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            page_capacity = int(page_table.shape[1])
            if self._graph_safe_full_decode and self._full_graph_page_cap > 0:
                page_capacity = min(page_capacity, int(self._full_graph_page_cap))
                if not self._logged_full_graph_page_cap_once:
                    self._logged_full_graph_page_cap_once = True
                    logger.warning(
                        "FlexFlash4 full-layer graph page cap active: page_capacity=%s page_size=%s.",
                        int(page_capacity),
                        int(self.page_size),
                    )
            page_cache = self._get_or_create_full_decode_page_cache(
                layer_id=int(layer.layer_id),
                batch_size=bs,
                page_capacity=page_capacity,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.head_dim,
                v_head_dim=layer.v_head_dim,
                k_dtype=key_cache.dtype,
                v_dtype=value_cache.dtype,
                device=q_device,
            )
            page_offsets = page_cache["page_offsets"]
            token_offsets = page_cache["token_offsets"]
            page_indices = page_cache["page_indices"]
            valid_page_mask = page_cache["valid_page_mask"]
            valid_token_mask = page_cache["valid_token_mask"]
            gather_pages = page_cache["gather_pages"]
            k_pages = page_cache["k_pages"]
            v_pages = page_cache["v_pages"]
            k_pages_flat = page_cache["k_pages_flat"]
            v_pages_flat = page_cache["v_pages_flat"]
            max_used_pages = page_capacity
            if not self._graph_safe_full_decode:
                max_used_pages = max(
                    1,
                    min(
                        page_capacity,
                        int(
                            (int(seq_lens.max().item()) + self.page_size - 1)
                            // self.page_size
                        ),
                    ),
                )
            # SGLang only guarantees the used prefix of page_table is initialized.
            # Under CUDA graph we still need a fixed page_capacity shape, so pad the
            # unused tail with page 0 (reserved dummy page) instead of reading the
            # raw tail entries, which can be stale and trigger invalid KV gathers.
            pages_per_seq = (
                (seq_lens.to(torch.int64) + self.page_size - 1) // self.page_size
            )
            if self._graph_safe_full_decode and self._full_graph_page_cap > 0:
                required_pages = max(
                    1,
                    (int(forward_batch.seq_lens_cpu.max().item()) + self.page_size - 1)
                    // self.page_size,
                )
                if required_pages > int(page_capacity):
                    raise RuntimeError(
                        "FlexFlash4 full-layer graph page cap exceeded: "
                        f"required_pages={required_pages} "
                        f"configured_pages={int(page_capacity)}. "
                        "Raise SGLANG_FLEX_FLASH4_FULL_GRAPH_PAGE_CAP for this regime."
                    )
            valid_page_mask.copy_(page_offsets < pages_per_seq.view(-1, 1))
            page_indices.zero_()
            page_indices.copy_(page_table[:bs, :page_capacity].to(torch.int64))
            page_indices.masked_fill_(~valid_page_mask, 0)
            gather_page_count = page_capacity if self._graph_safe_full_decode else max_used_pages
            gather_pages = gather_pages[: bs * gather_page_count]
            gather_pages.copy_(page_indices[:, :gather_page_count].reshape(-1))
            # Defensive: page_table indices can be stale/uninitialized in some edge cases.
            # Out-of-range indices lead to CUDA illegal memory access inside index_select.
            max_page = max(0, int(key_cache_pages.shape[0]) - 1)
            gather_pages.clamp_(min=0, max=max_page)
            torch.index_select(
                key_cache_pages,
                0,
                gather_pages,
                out=k_pages_flat[: bs * gather_page_count],
            )
            torch.index_select(
                value_cache_pages,
                0,
                gather_pages,
                out=v_pages_flat[: bs * gather_page_count],
            )
            self._maybe_sync_debug(
                f"build_compact_kv_full layer={int(layer.layer_id)} bs={bs} gather_pages={gather_page_count}"
            )
            return {
                "is_sliding_layer": False,
                "window_left": -1,
                "max_local_tokens": page_capacity * self.page_size,
                "local_seqlens": seq_lens,
                "k_local": k_pages.view(
                    bs,
                    page_capacity * self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim,
                ),
                "v_local": v_pages.view(
                    bs,
                    page_capacity * self.page_size,
                    layer.tp_v_head_num,
                    layer.v_head_dim,
                ),
                "attn_mode": "full",
                "used_pages": max_used_pages,
            }

        max_local_tokens = int(window_left) + 1
        metadata = self.forward_metadata

        if (
            is_sliding_layer
            and self.use_sliding_window_kv_pool
            and getattr(metadata, "swa_page_table", None) is not None
            and not self._disable_swa_page_compaction
        ):
            page_table = metadata.swa_page_table
            page_capacity = max(
                1, (int(max_local_tokens) + int(self.page_size) - 1) // int(self.page_size)
            )
            page_cache = self._get_or_create_full_decode_page_cache(
                layer_id=int(layer.layer_id),
                batch_size=bs,
                page_capacity=page_capacity,
                num_kv_heads=layer.tp_k_head_num,
                head_dim=layer.head_dim,
                v_head_dim=layer.v_head_dim,
                k_dtype=key_cache.dtype,
                v_dtype=value_cache.dtype,
                device=q_device,
            )
            page_offsets = page_cache["page_offsets"]
            token_offsets = page_cache["token_offsets"]
            page_indices = page_cache["page_indices"]
            valid_page_mask = page_cache["valid_page_mask"]
            valid_token_mask = page_cache["valid_token_mask"]
            gather_pages = page_cache["gather_pages"]
            k_pages = page_cache["k_pages"]
            v_pages = page_cache["v_pages"]
            k_pages_flat = page_cache["k_pages_flat"]
            v_pages_flat = page_cache["v_pages_flat"]

            key_cache_pages = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache_pages = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            local_seqlens = torch.clamp(
                forward_batch.seq_lens.to(torch.int32),
                max=int(max_local_tokens),
            )
            pages_per_seq = (
                (local_seqlens.to(torch.int64) + int(self.page_size) - 1)
                // int(self.page_size)
            )
            valid_page_mask.copy_(page_offsets < pages_per_seq.view(-1, 1))
            page_indices.zero_()
            page_indices.copy_(page_table[:bs, :page_capacity].to(torch.int64))
            page_indices.masked_fill_(~valid_page_mask, 0)

            max_page = max(0, int(key_cache_pages.shape[0]) - 1)
            gather_pages = gather_pages[: bs * page_capacity]
            gather_pages.copy_(page_indices.reshape(-1))
            gather_pages.clamp_(min=0, max=max_page)
            torch.index_select(
                key_cache_pages,
                0,
                gather_pages,
                out=k_pages_flat[: bs * page_capacity],
            )
            torch.index_select(
                value_cache_pages,
                0,
                gather_pages,
                out=v_pages_flat[: bs * page_capacity],
            )

            k_local = k_pages.view(
                bs,
                page_capacity * self.page_size,
                layer.tp_k_head_num,
                layer.head_dim,
            )
            v_local = v_pages.view(
                bs,
                page_capacity * self.page_size,
                layer.tp_v_head_num,
                layer.v_head_dim,
            )
            valid_token_mask.copy_(
                token_offsets < local_seqlens.to(torch.int64).view(-1, 1)
            )
            k_local.masked_fill_(~valid_token_mask[:, :, None, None], 0)
            v_local.masked_fill_(~valid_token_mask[:, :, None, None], 0)
            self._maybe_sync_debug(
                f"build_compact_kv_sliding_pages layer={int(layer.layer_id)} bs={bs} page_capacity={page_capacity}"
            )
            if not self._logged_swa_page_compact_once:
                self._logged_swa_page_compact_once = True
                logger.warning(
                    "FlexFlash4 sliding decode switched to SWA page-table compaction (page_size=%s page_capacity=%s max_local_tokens=%s).",
                    int(self.page_size),
                    int(page_capacity),
                    int(max_local_tokens),
                )
            return {
                "is_sliding_layer": True,
                "window_left": window_left,
                "max_local_tokens": page_capacity * self.page_size,
                "local_seqlens": local_seqlens,
                "k_local": k_local,
                "v_local": v_local,
                "used_pages": int(page_capacity),
                "attn_mode": "sliding_pages",
            }

        cache = self._get_or_create_local_decode_cache(
            layer_id=int(layer.layer_id),
            batch_size=bs,
            max_local_tokens=max_local_tokens,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.head_dim,
            v_head_dim=layer.v_head_dim,
            k_dtype=key_cache.dtype,
            v_dtype=value_cache.dtype,
            device=q_device,
        )

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
        seq_lens = forward_batch.seq_lens.to(torch.int64)
        col_offsets = cache["col_offsets"]
        token_cols = cache["token_cols"]
        valid_cols = cache["valid_cols"]
        safe_token_cols = cache["safe_token_cols"]
        slot_indices = cache["slot_indices"]

        if use_full_sequence or not is_sliding_layer:
            token_cols.copy_(col_offsets.expand(bs, -1))
            valid_cols.copy_(token_cols < seq_lens.view(-1, 1))
        else:
            token_cols.copy_(seq_lens.view(-1, 1) - int(max_local_tokens) + col_offsets)
            valid_cols.copy_((token_cols >= 0) & (token_cols < seq_lens.view(-1, 1)))

        req_to_token_rows = req_to_token.index_select(0, req_pool_indices)
        max_token_col = max(0, int(req_to_token_rows.shape[1]) - 1)
        safe_token_cols.copy_(token_cols)
        safe_token_cols.clamp_(min=0, max=max_token_col)
        slot_indices.copy_(
            torch.gather(req_to_token_rows, 1, safe_token_cols)
        )
        slot_indices.masked_fill_(~valid_cols, 0)
        if is_sliding_layer and self.use_sliding_window_kv_pool:
            # The req_to_token table can contain stale/uninitialized tail values for some
            # capture/replay shapes. Clamp *before* SWA translation to avoid OOB indexing
            # into full_to_swa_index_mapping (which can surface as an H100 XID 31).
            mapping = getattr(
                forward_batch.token_to_kv_pool, "full_to_swa_index_mapping", None
            )
            if mapping is not None:
                mapping_max = int(mapping.shape[0]) - 2
                if mapping_max >= 0:
                    slot_indices.clamp_(min=-1, max=mapping_max)
            slot_indices.copy_(
                forward_batch.token_to_kv_pool.translate_loc_from_full_to_swa(
                    slot_indices
                ).to(torch.int64)
            )

        local_seqlens = cache["local_seqlens"]
        local_seqlens.copy_(forward_batch.seq_lens.to(torch.int32))
        local_seqlens.clamp_(max=int(max_local_tokens))

        k_local = cache["k_local"]
        v_local = cache["v_local"]
        gather_slots = slot_indices.view(-1)
        # Defensive: req_to_token tail entries can be stale for some shapes; avoid
        # out-of-range gather indices which cause CUDA illegal memory access.
        max_slot = max(0, int(key_cache.shape[0]) - 1)
        gather_slots.clamp_(min=0, max=max_slot)
        k_local.copy_(
            key_cache.index_select(0, gather_slots).view(
                bs, int(max_local_tokens), layer.tp_k_head_num, layer.head_dim
            )
        )
        v_local.copy_(
            value_cache.index_select(0, gather_slots).view(
                bs, int(max_local_tokens), layer.tp_v_head_num, layer.v_head_dim
            )
        )
        k_local.masked_fill_(~valid_cols[:, :, None, None], 0)
        v_local.masked_fill_(~valid_cols[:, :, None, None], 0)
        self._maybe_sync_debug(
            f"build_compact_kv_sliding layer={int(layer.layer_id)} bs={bs} local_tokens={int(max_local_tokens)}"
        )

        return {
            "is_sliding_layer": is_sliding_layer,
            "window_left": window_left,
            "max_local_tokens": max_local_tokens,
            "local_seqlens": local_seqlens,
            "k_local": k_local,
            "v_local": v_local,
            "used_pages": max(1, (max_local_tokens + self.page_size - 1) // self.page_size),
            "attn_mode": (
                "sliding_fullkv"
                if use_full_sequence and is_sliding_layer
                else ("sliding" if is_sliding_layer else "full")
            ),
        }

    def _sync_extend_delegate_metadata(self) -> None:
        self._extend_delegate.forward_metadata = self.forward_metadata
        self._extend_delegate.forward_metadata_spec_decode_expand = (
            self.forward_metadata_spec_decode_expand
        )

    def _sync_full_decode_delegate_metadata(self) -> None:
        self._full_decode_delegate.forward_metadata = self.forward_metadata
        self._full_decode_delegate.forward_metadata_spec_decode_expand = (
            self.forward_metadata_spec_decode_expand
        )

    def _delegate_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._sync_full_decode_delegate_metadata()
        self._maybe_sync_debug(
            f"delegate_full_decode_enter layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        out = self._full_decode_delegate.forward_decode(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=False,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        )
        self._maybe_sync_debug(
            f"delegate_full_decode_exit layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        return out

    def _delegate_sliding_decode_path(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._sliding_decode_delegate.forward_metadata = self.forward_metadata
        self._sliding_decode_delegate.forward_metadata_spec_decode_expand = (
            self.forward_metadata_spec_decode_expand
        )
        self._maybe_sync_debug(
            f"delegate_sliding_decode_enter layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        out = self._sliding_decode_delegate.forward_decode(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=False,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        )
        self._maybe_sync_debug(
            f"delegate_sliding_decode_exit layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        return out

    def _delegate_extend(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Delegate prefill/extend to the safer FA3/FA4 backend.

        Note: paged Hopper FA4 prefill is not yet stable for GPT-OSS on all stacks.
        For page_size>1 we default the delegate to FA3 (see __init__). Keeping this
        logic centralized prevents accidental routing back into paged FA4.
        """
        if not self._logged_extend_delegate_once:
            self._logged_extend_delegate_once = True
            logger.warning(
                "FlexFlash4 extend delegates to FA%s (page_size=%s) while custom Hopper extend is stabilized.",
                self._extend_delegate_fa_ver,
                int(self.page_size),
            )
        self._sync_extend_delegate_metadata()
        self._maybe_sync_debug(
            f"delegate_extend_enter layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        out = self._extend_delegate.forward_extend(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=False,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        )
        self._maybe_sync_debug(
            f"delegate_extend_exit layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        return out

    def _can_use_native_paged_full_decode(
        self,
        *,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
    ) -> bool:
        if not self._native_paged_full_decode:
            return False
        if self._is_sliding_layer(layer):
            return False
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            return False
        if forward_batch.spec_info is not None:
            return False
        if self._uses_quantized_kv_cache():
            return False
        if self.has_local_attention and getattr(self.forward_metadata, "local_attn_metadata", None) is not None:
            return False
        if not fa4_hopper_stable_enabled() and not self._allow_cute_paged_kv_sm90:
            return False
        metadata = getattr(self, "forward_metadata", None)
        if metadata is None or getattr(metadata, "page_table", None) is None:
            return False
        return True

    def _forward_decode_native_paged_full(
        self,
        *,
        q: torch.Tensor,
        q_rope: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        page_table = metadata.page_table
        refresh_scheduler_metadata = getattr(
            self, "_maybe_refresh_fa4_scheduler_metadata_replay", None
        )
        if callable(refresh_scheduler_metadata):
            refresh_scheduler_metadata(
                metadata,
                layer,
                q_reshaped.dtype,
                page_table,
                True,
                (-1, -1),
                is_swa_layer=False,
            )
        if not self._logged_native_paged_full_decode_once:
            self._logged_native_paged_full_decode_once = True
            logger.warning(
                "FlexFlash4 native paged full decode active inside backend: page_size=%s kv_dtype=%s cuda_graph=%s source=%s.",
                int(self.page_size),
                self.kv_cache_dtype_str,
                int(self._cuda_graph_enabled),
                "hopper_stable" if fa4_hopper_stable_enabled() else "vendored_cute",
            )
        result = flash_attn_with_kvcache_fa4(
            q=q_reshaped,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            causal=True,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            scheduler_metadata=getattr(metadata, "scheduler_metadata", None),
            sinks=sinks,
            num_splits=1,
        )
        self._maybe_sync_debug(
            f"forward_decode_native_paged_full layer={int(layer.layer_id)} bs={int(q.shape[0])}"
        )
        return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                self._maybe_sync_debug(
                    f"save_kv_decode layer={int(layer.layer_id)} bs={int(q.shape[0])}"
                )

        # DFlash TARGET_VERIFY is extremely sensitive to attention semantics and is used to
        # compute accept lengths. Even if FlexFlash4 custom decode is graph-stable, any
        # small numerical drift can collapse acceptance. Keep TARGET_VERIFY on FA3 unless
        # explicitly parity-proven.
        if forward_batch.forward_mode.is_target_verify():
            spec_info = getattr(forward_batch, "spec_info", None)
            if (
                self._dflash_delegate_target_verify_to_fa3
                and getattr(spec_info, "spec_input_type", None) == SpecInputType.DFLASH_VERIFY
            ):
                if (
                    not getattr(self, "_logged_dflash_verify_delegate_once", False)
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    setattr(self, "_logged_dflash_verify_delegate_once", True)
                    logger.warning(
                        "FlexFlash4 routing DFLASH TARGET_VERIFY attention through FA3 delegate for parity."
                    )
                self._sync_dflash_delegate_metadata()
                return self._dflash_delegate.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=False,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    sinks=sinks,
                )

        if self._dflash_delegate_decode_to_fa3 and self._is_dflash_speculative(forward_batch):
            if (
                not getattr(self, "_logged_dflash_decode_delegate_once", False)
                and not torch.cuda.is_current_stream_capturing()
            ):
                setattr(self, "_logged_dflash_decode_delegate_once", True)
                logger.warning("FlexFlash4 routing DFLASH attention through FA3 delegate for parity.")
            self._sync_dflash_delegate_metadata()
            return self._dflash_delegate.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if self._should_delegate_full_decode(layer):
            self._maybe_log_full_decode_delegate()
            return self._delegate_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        if self._should_delegate_sliding_decode(layer):
            self._maybe_log_sliding_decode_delegate()
            return self._delegate_sliding_decode_path(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        if self._delegate_quantized_kv_decode and self._uses_quantized_kv_cache():
            self._maybe_log_quantized_kv_delegate()
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        if self._can_use_native_paged_full_decode(
            layer=layer, forward_batch=forward_batch, q_rope=q_rope, k_rope=k_rope
        ):
            return self._forward_decode_native_paged_full(
                q=q,
                q_rope=q_rope,
                layer=layer,
                forward_batch=forward_batch,
                sinks=sinks,
            )

        if not self._can_use_custom_decode(
            layer=layer,
            forward_batch=forward_batch,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        ):
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        metadata = self.forward_metadata
        compact = self._build_compact_kv(
            layer=layer,
            forward_batch=forward_batch,
            q_device=q.device,
            use_full_sequence=False,
        )
        is_sliding_layer = bool(compact["is_sliding_layer"])
        window_left = int(compact["window_left"])
        max_local_tokens = int(compact["max_local_tokens"])
        local_seqlens = compact["local_seqlens"]
        k_local = compact["k_local"]
        v_local = compact["v_local"]
        used_pages = int(compact["used_pages"])
        attn_mode = str(compact["attn_mode"])
        if not self._logged_custom_decode_once:
            self._logged_custom_decode_once = True
            logger.warning(
                "FlexFlash4 custom local decode active: mode=%s layer=%s page_size=%s window_left=%s max_local_tokens=%s used_pages=%s kv_dtype=%s",
                attn_mode,
                int(layer.layer_id),
                self._custom_block_size,
                window_left,
                max_local_tokens,
                used_pages,
                self.kv_cache_dtype_str,
            )
        sparse_key = (attn_mode, int(local_seqlens.shape[0]))
        if (
            self._log_sparse
            and sparse_key not in self._logged_sparse_stats_once
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._logged_sparse_stats_once.add(sparse_key)
            local_len_max = int(local_seqlens.max().item())
            logger.warning(
                "FlexFlash4 local decode stats: mode=%s batch=%s local_len_max=%s used_pages=%s window_left=%s",
                attn_mode,
                int(local_seqlens.shape[0]),
                local_len_max,
                used_pages,
                window_left,
            )

        result = flash_attn_with_kvcache_fa4(
            q=q.contiguous().view(
                -1, layer.tp_q_head_num, layer.head_dim
            ),
            k_cache=k_local,
            v_cache=v_local,
            page_table=None,
            cache_seqlens=local_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=layer.scaling,
            causal=False,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            sinks=sinks,
            num_splits=1,
        )
        self._maybe_sync_debug(
            f"forward_decode layer={int(layer.layer_id)} mode={attn_mode} bs={int(local_seqlens.shape[0])}"
        )
        return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                self._maybe_sync_debug(
                    f"save_kv_extend layer={int(layer.layer_id)} bs={int(q.shape[0])}"
                )

        if self._dflash_delegate_extend_to_fa3 and self._is_dflash_speculative(forward_batch):
            if (
                not getattr(self, "_logged_dflash_extend_delegate_once", False)
                and not torch.cuda.is_current_stream_capturing()
            ):
                setattr(self, "_logged_dflash_extend_delegate_once", True)
                logger.warning("FlexFlash4 routing DFLASH extend attention through FA3 delegate for parity.")
            self._sync_dflash_delegate_metadata()
            return self._dflash_delegate.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if self._should_delegate_full_extend(layer):
            self._maybe_log_full_extend_delegate()
            return self._delegate_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        if self._should_delegate_sliding_extend(layer):
            self._maybe_log_sliding_extend_delegate()
            return self._delegate_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        if self._delegate_quantized_kv_decode and self._uses_quantized_kv_cache():
            self._maybe_log_quantized_kv_delegate()
            return self._delegate_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if not self._enable_custom_extend:
            return self._delegate_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        if not self._can_use_custom_decode(
            layer=layer,
            forward_batch=forward_batch,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        ):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        metadata = self.forward_metadata
        compact = self._build_compact_kv(
            layer=layer,
            forward_batch=forward_batch,
            q_device=q.device,
            use_full_sequence=True,
        )
        is_sliding_layer = bool(compact["is_sliding_layer"])
        window_left = int(compact["window_left"])
        local_seqlens = compact["local_seqlens"]
        k_local = compact["k_local"]
        v_local = compact["v_local"]

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False
        window_size = (window_left, 0) if is_sliding_layer else (-1, -1)

        result = flash_attn_with_kvcache_fa4(
            q=q.contiguous().view(
                -1, layer.tp_q_head_num, layer.head_dim
            ),
            k_cache=k_local,
            v_cache=v_local,
            page_table=None,
            cache_seqlens=local_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            sinks=sinks,
            num_splits=1,
        )
        return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)
