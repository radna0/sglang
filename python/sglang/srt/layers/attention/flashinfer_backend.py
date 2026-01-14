from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
import os
import shutil
import time
import sys
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

try:
    import torch.cuda.nvtx as nvtx
    _has_nvtx = True
except ImportError:
    _has_nvtx = False

_SGLANG_FLASHINFER_NVTX = os.environ.get("SGLANG_FLASHINFER_NVTX", "0") == "1"


class _NVTXRangeCompat:
    """Compatibility wrapper: torch.cuda.nvtx.range() returns a context manager, not an object with .end()."""

    def __init__(self, name: str):
        self._cm = nvtx.range(name)
        self._cm.__enter__()

    def end(self) -> None:
        self._cm.__exit__(None, None, None)


def _nvtx_range(name: str) -> "_NVTXRangeCompat | None":
    if not (_SGLANG_FLASHINFER_NVTX and _has_nvtx):
        return None
    try:
        return _NVTXRangeCompat(name)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# FlashInfer decode attribution timing (graphs-OFF only; for profiling).
# -----------------------------------------------------------------------------
_sglang_flashinfer_timing_ctr = 0
_fi_timing_step_ctr = 0
_fi_timing_step_do = False
_fi_timing_acc: Optional[dict] = None

# FlashInfer prefill attribution timing (graphs-OFF only; for profiling).
_sglang_flashinfer_prefill_timing_ctr = 0
_fi_prefill_timing_step_ctr = 0
_fi_prefill_timing_step_do = False
_fi_prefill_timing_acc: Optional[dict] = None


def _append_profile_log(line: str) -> None:
    """Append a single line to a per-run profile log file if configured.

    The benchmark harness sets `SGLANG_REMOTE_PROFILE_LOG=/logs/<file>.log` so
    worker-process timing can be collected alongside the benchmark output.
    """
    path = os.environ.get("SGLANG_REMOTE_PROFILE_LOG", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Never let profiling break inference.
        return

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_FLASHINFER_JIT_CACHE_INIT = False


def _flashinfer_cache_root() -> str:
    base = os.environ.get("FLASHINFER_WORKSPACE_BASE", "").strip()
    if not base:
        base = os.path.expanduser("~")
    return os.path.join(base, ".cache", "flashinfer")


def _flashinfer_build_fingerprint(flashinfer_mod) -> str:
    version = getattr(flashinfer_mod, "__version__", "unknown")

    ext_paths: list[str] = []
    for attr in ("_C", "_flashinfer"):
        try:
            m = getattr(flashinfer_mod, attr)
            p = getattr(m, "__file__", None)
            if isinstance(p, str) and p:
                ext_paths.append(p)
        except Exception:
            continue

    parts: list[str] = [f"version={version}"]
    for p in sorted(set(ext_paths)):
        try:
            st = os.stat(p)
            parts.append(f"ext={p}|size={st.st_size}|mtime={int(st.st_mtime)}")
        except Exception:
            parts.append(f"ext={p}|stat=err")
    return ";".join(parts)


def _ensure_flashinfer_jit_cache_compatible(flashinfer_mod) -> None:
    global _FLASHINFER_JIT_CACHE_INIT
    if _FLASHINFER_JIT_CACHE_INIT:
        return

    cache_root = _flashinfer_cache_root()
    sig_path = os.path.join(cache_root, ".sglang_flashinfer_fingerprint.txt")
    manual_clear = get_bool_env_var("SGLANG_FLASHINFER_CLEAR_JIT_CACHE", "false")
    fingerprint = _flashinfer_build_fingerprint(flashinfer_mod)

    try:
        prev = None
        if os.path.exists(sig_path):
            with open(sig_path, "r", encoding="utf-8") as rf:
                prev = rf.read().strip()

        has_unknown_existing_cache = False
        if prev is None and os.path.isdir(cache_root):
            try:
                for ent in os.scandir(cache_root):
                    if ent.name != os.path.basename(sig_path):
                        has_unknown_existing_cache = True
                        break
            except Exception:
                has_unknown_existing_cache = True

        if manual_clear or has_unknown_existing_cache or (prev is not None and prev != fingerprint):
            shutil.rmtree(cache_root, ignore_errors=True)
            logger.warning("[FlashInfer] Cleared JIT cache: %s", cache_root)

        os.makedirs(cache_root, exist_ok=True)
        with open(sig_path, "w", encoding="utf-8") as wf:
            wf.write(fingerprint + "\n")
    except Exception as e:
        logger.warning("[FlashInfer] JIT cache compatibility check failed: %s", e)

    _FLASHINFER_JIT_CACHE_INIT = True


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    def _import_flashinfer_rtld_global():
        # FlashInfer's JITed cached_ops may dlopen shared objects that expect to
        # resolve symbols from already-loaded FlashInfer extension modules. On
        # some setups this requires importing FlashInfer with RTLD_GLOBAL.
        if sys.platform == "win32":
            import flashinfer as m
            return m

        if not get_bool_env_var("SGLANG_FLASHINFER_RTLD_GLOBAL", "true"):
            import flashinfer as m
            return m

        try:
            import ctypes

            old_flags = sys.getdlopenflags()
            sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
            import flashinfer as m
            sys.setdlopenflags(old_flags)
            return m
        except Exception:
            import flashinfer as m
            return m

    _flashinfer_mod = _import_flashinfer_rtld_global()

    BatchDecodeWithPagedKVCacheWrapper = _flashinfer_mod.BatchDecodeWithPagedKVCacheWrapper
    BatchPrefillWithPagedKVCacheWrapper = _flashinfer_mod.BatchPrefillWithPagedKVCacheWrapper
    BatchPrefillWithRaggedKVCacheWrapper = _flashinfer_mod.BatchPrefillWithRaggedKVCacheWrapper
    fast_decode_plan = _flashinfer_mod.fast_decode_plan

    _ensure_flashinfer_jit_cache_compatible(_flashinfer_mod)
    from flashinfer.attention import BatchAttentionWithAttentionSinkWrapper
    from flashinfer.cascade import merge_state
    from flashinfer.jit.attention.modules import (
        get_batch_decode_attention_sink_uri,
        get_batch_prefill_attention_sink_uri,
    )
    from flashinfer.jit.attention.variants import attention_sink_decl


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[
        BatchDecodeWithPagedKVCacheWrapper | BatchPrefillWithPagedKVCacheWrapper
    ]
    q_workspace_buffers: Dict[int, torch.Tensor] = None  # Pre-allocated Q workspace per layer
    # Cache fp32-cast sinks tensors (keyed by (data_ptr, numel)) to avoid per-step
    # allocations and to keep CUDA-graph capture stable when using the decode wrapper.
    sinks_fp32_cache: Dict[tuple[int, int], torch.Tensor] = None
    # Cache XQA-formatted sinks tensors (keyed by (data_ptr, numel)).
    #
    # GPT-OSS stores sinks as *log* values per Q head. FlashInfer's XQA decode path
    # expects sinks as a *linear* additive term in the softmax denominator (per KV
    # head group), so we precompute `exp(sinks_log)` once and reuse it.
    sinks_xqa_cache: Dict[tuple[int, int], torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()

        # Keep a reference for runtime decisions (e.g. sliding-window size).
        self.model_runner = model_runner
        self.model_dtype = model_runner.dtype
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.head_dim = model_runner.model_config.head_dim

        # Default-enable the fused FP8 KV update path for FlashInfer FP8 KV cache.
        # This avoids a multi-kernel (scale/cast/scatter) sequence on the decode hotpath.
        if self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if os.environ.get("SGLANG_FP8_KV_SCATTER_QUANT") is None:
                os.environ["SGLANG_FP8_KV_SCATTER_QUANT"] = "1"
                logger.info(
                    "[FlashInfer] Enabling SGLANG_FP8_KV_SCATTER_QUANT=1 by default for FP8 KV cache."
                )

        hf_config = model_runner.model_config.hf_config
        self.requires_attention_sinks = (
            getattr(hf_config, "model_type", None) == "gpt_oss"
            or "GptOssForCausalLM" in (getattr(hf_config, "architectures", None) or [])
        )
        self._diag_disable_sinks = False
        if self.requires_attention_sinks and get_bool_env_var(
            "SGLANG_GPTOSS_DIAG_DISABLE_SINKS", "false"
        ):
            logger.warning(
                "[FlashInfer] DIAG: disabling GPT-OSS attention sinks via "
                "SGLANG_GPTOSS_DIAG_DISABLE_SINKS=1 (A/B validation only)."
            )
            self._diag_disable_sinks = True
            self.requires_attention_sinks = False

        # NOTE: On SM90, the AttentionSink FP8 path (Hopper GMMA) effectively requires
        # Q and KV to share a tensorcore dtype. BF16Q×FP8KV has no eligible GMMA op and
        # can fail JIT compilation when CUDA graphs are enabled.
        #
        # Workaround (opt-in, correctness must be validated):
        # Force Q to FP8 for BOTH prefill and decode in the sink-wrapper path when
        # KV cache is FP8, and route sinks to the FA3 backend (which supports FP8 Q
        # via the fp8-enabled Hopper templates).
        #
        # Controlled by:
        #   SGLANG_FLASHINFER_PREFILL_FP8Q_WITH_FP8KV=1  (default: 1)
        #
        # If disabled, Q stays in model dtype and FP8 sinks may still fail to compile
        # on SM90 due to BF16Q×FP8KV limitations.
        force_sinks_fp8_q = (
            self.requires_attention_sinks
            and get_bool_env_var("SGLANG_FLASHINFER_PREFILL_FP8Q_WITH_FP8KV", "true")
            and self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )

        if self.requires_attention_sinks:
            self.paged_q_data_type = self.kv_cache_dtype if force_sinks_fp8_q else self.model_dtype
        else:
            self.paged_q_data_type = (
                self.kv_cache_dtype
                if self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                else self.model_dtype
            )

        self.prefill_q_data_type = self.kv_cache_dtype if force_sinks_fp8_q else self.model_dtype
        self._sinks_force_fp8_q = bool(force_sinks_fp8_q)
        if self._sinks_force_fp8_q:
            logger.warning(
                "[FlashInfer] GPT-OSS sinks + FP8 KV: forcing Q dtype to %s for prefill+decode "
                "and preferring FA3 sinks backend "
                "(SGLANG_FLASHINFER_PREFILL_FP8Q_WITH_FP8KV=1)",
                str(self.kv_cache_dtype),
            )

        # Store multi-item scoring delimiter for efficient access
        self.multi_item_scoring_delimiter = (
            model_runner.server_args.multi_item_scoring_delimiter
        )

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_attention_tp_size(),
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_attention_tp_size()
            ),
        )
        # NOTE: FlashInfer's tensor-core decode path (which reuses prefill kernels)
        # currently does not support attention sinks correctly/robustly, and also
        # triggers unsupported BF16×FP8 compilation paths for GPT-OSS FP8 KV runs.
        # Force the non-TC decode path for GPT-OSS until a sink-aware TC decode
        # kernel is integrated.
        #
        # OPTIMIZATION: Allow TC with sinks via environment variable override
        # This restores FP8 tensor core performance for GPT-OSS models
        # Default to "true" to enable tensor cores by default
        allow_tc_with_sinks = get_bool_env_var(
            "SGLANG_FLASHINFER_DECODE_TC_WITH_SINKS", "true"
        )
        if self.requires_attention_sinks and not allow_tc_with_sinks:
            logger.warning(
                "[FlashInfer] Tensor cores DISABLED for attention sinks. "
                "Set SGLANG_FLASHINFER_DECODE_TC_WITH_SINKS=1 to enable."
            )
            self.decode_use_tensor_cores = False
        elif self.decode_use_tensor_cores:
            logger.info(
                f"[FlashInfer] Tensor cores ENABLED (decode_use_tensor_cores=True)"
            )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        # NOTE: hf_config + requires_attention_sinks are initialized above.
        self._logged_prefill_dispatch_key = False
        self._logged_decode_dispatch_key = False

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on B200
            if model_runner.server_args.enable_piecewise_cuda_graph:
                logger.warning(
                    "CUTLASS backend is disabled when piecewise cuda graph is enabled "
                    "due to TMA descriptor initialization issues on B200. "
                    "Using auto backend instead for stability."
                )
            else:
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []

        # IMPORTANT (GPT-OSS sinks + speculative / custom masks):
        # FlashInfer selects different kernel implementations depending on whether custom masks
        # are enabled. For EAGLE3, FlashInfer uses a flattened `custom_mask` for speculative
        # prefill/verify. If we instantiate the AttentionSink wrapper without `custom_mask_buf`,
        # it can route to a non-custom-mask specialization and yield severe quality regressions.
        #
        # Allocate small persistent buffers to ensure the custom-mask-capable path is selected.
        # The actual mask is still provided dynamically via `begin_forward(custom_mask=...)`.
        needs_sink_custom_mask = bool(
            self.requires_attention_sinks
            and (
                getattr(model_runner.server_args, "speculative_algorithm", None) is not None
                or model_runner.server_args.multi_item_scoring_delimiter is not None
            )
        )
        self._sink_custom_mask_buf: Optional[torch.Tensor] = None
        self._sink_mask_indptr_buf: Optional[torch.Tensor] = None
        if needs_sink_custom_mask:
            # Only the presence (not the size) of these buffers affects kernel selection
            # when CUDA graphs are disabled. Keep them minimal to reduce memory overhead.
            self._sink_custom_mask_buf = torch.empty(
                (1,), dtype=torch.uint8, device=model_runner.device
            )
            self._sink_mask_indptr_buf = torch.empty(
                (model_runner.server_args.max_running_requests + 1,),
                dtype=torch.int32,
                device=model_runner.device,
            )
        sink_backend = os.environ.get("SGLANG_FLASHINFER_SINK_BACKEND", "auto").lower()
        if sink_backend == "cutlass":
            # FlashInfer (0.6.x) does not accept "cutlass" as a sink backend.
            # Keep the env for forward-compat experiments, but remap to "auto" today.
            logger.warning(
                "[FlashInfer] SGLANG_FLASHINFER_SINK_BACKEND=cutlass is not supported by this "
                "FlashInfer build; falling back to auto."
            )
            sink_backend = "auto"
        if sink_backend not in ("auto", "fa2", "fa3"):
            logger.warning(
                "[FlashInfer] Unknown SGLANG_FLASHINFER_SINK_BACKEND=%s; using auto.",
                sink_backend,
            )
            sink_backend = "auto"
        if sink_backend == "auto" and self.requires_attention_sinks:
            if self._sinks_force_fp8_q:
                # Required to hit FlashInfer's FP8-enabled Hopper sink templates.
                sink_backend = "fa3"
            elif needs_sink_custom_mask:
                # Multi-item scoring / speculative verification workloads may require sink mask
                # modes that are not yet stable on SM90 FA3. Prefer the slower but stable FA2 path
                # in these cases.
                sink_backend = "fa2"
            else:
                # Empirically (GPT-OSS-20B long-KV, SM90), FA3 sink kernels can be slower than
                # FA2 when we are not using the tensor-core decode path (decode_use_tensor_cores=False).
                # Prefer FA2 in that case; otherwise default to FA3.
                sink_backend = "fa3" if self.decode_use_tensor_cores else "fa2"
        # Persist for later (e.g. CUDA-graph TARGET_VERIFY wrapper creation).
        self.sink_backend = sink_backend
        decode_sinks_wrapper = os.environ.get(
            "SGLANG_FLASHINFER_DECODE_SINKS_WRAPPER", "sink"
        ).lower()
        use_decode_wrapper_for_sinks = decode_sinks_wrapper in ("decode", "paged")
        # Persist selection for CUDA-graph capture and debugging.
        self.decode_sinks_wrapper = decode_sinks_wrapper
        # GPT-OSS requires AttentionSinks for correctness. The "decode-wrapper" sinks mode
        # is implemented by building a custom AttentionSink JIT module and running it via
        # `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)` (FlashInfer's TC decode
        # wrapper path reuses prefill kernels). This path MUST use tensor cores; the non-TC
        # decode module does not implement the same AttentionSink variant interface.
        if self.requires_attention_sinks and use_decode_wrapper_for_sinks and (not self.decode_use_tensor_cores):
            raise ValueError(
                "FlashInfer decode-wrapper sinks requires decode_use_tensor_cores=True. "
                "Set SGLANG_FLASHINFER_DECODE_TC_WITH_SINKS=1 (default) and ensure tensor cores are enabled."
            )
        self.use_decode_wrapper_for_sinks = use_decode_wrapper_for_sinks
        # Prefill backend for paged (supports sinks). For FP8 sinks runs where we
        # force Q to FP8, we must use FA3 to enable the FP8 Hopper templates.
        prefill_paged_backend = "fa3" if self._sinks_force_fp8_q else "fa2"
        self.prefill_paged_backend = prefill_paged_backend
        for wrapper_id in range(self.num_wrappers):
            if not skip_prefill:
                if self.requires_attention_sinks:
                    init_window_left = -1
                    if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
                        # For sinks models with FP8 KV, FlashInfer's "full attention" (use_swa=False)
                        # AttentionSink prefill specialization can fail to compile on SM90 for some
                        # BF16Q×FP8KV configurations. We don't need a separate full-attn wrapper for
                        # GPT-OSS: SWA with a large enough window is equivalent when seq_len <= window.
                        #
                        # Force SWA for all wrappers in this specific case to avoid compiling the
                        # failing use_swa=False variant.
                        if self.kv_cache_dtype in (
                            torch.float8_e4m3fn,
                            torch.float8_e5m2,
                        ):
                            init_window_left = model_runner.sliding_window_size
                        else:
                            init_window_left = (
                                model_runner.sliding_window_size if wrapper_id == 0 else -1
                            )
                    kwargs = dict(
                        backend=sink_backend,
                        q_data_type=self.prefill_q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=self.model_dtype,
                        head_dim_qk=model_runner.model_config.head_dim,
                        head_dim_vo=model_runner.model_config.head_dim,
                        window_left=init_window_left,
                        custom_mask_buf=self._sink_custom_mask_buf,
                        mask_indptr_buf=self._sink_mask_indptr_buf,
                    )
                    self.prefill_wrappers_paged.append(
                        BatchAttentionWithAttentionSinkWrapper(
                            self.workspace_buffer, "NHD", **kwargs
                        )
                    )
                    self.prefill_wrappers_verify.append(
                        BatchAttentionWithAttentionSinkWrapper(
                            self.workspace_buffer, "NHD", **kwargs
                        )
                    )
                    if os.environ.get("SGLANG_FLASHINFER_LOG_DISPATCH", "0") == "1":
                        try:
                            uri_hint = None
                            try:
                                uri_hint = get_batch_prefill_attention_sink_uri(
                                    backend=kwargs.get("backend", sink_backend),
                                    dtype_q=kwargs.get("q_data_type", self.prefill_q_data_type),
                                    dtype_kv=kwargs.get("kv_data_type", self.kv_cache_dtype),
                                    dtype_o=kwargs.get("o_data_type", self.model_dtype),
                                    dtype_idx=torch.int32,
                                    head_dim_qk=model_runner.model_config.head_dim,
                                    head_dim_vo=model_runner.model_config.head_dim,
                                    pos_encoding_mode=0,
                                    use_sliding_window=(int(kwargs.get("window_left", -1)) != -1),
                                )
                            except Exception:
                                uri_hint = None
                            msg = (
                                "[FI_DISPATCH] mode=prefill "
                                f"wrapper_id={wrapper_id} "
                                "sinks=1 "
                                f"backend={kwargs.get('backend')} "
                                f"q_dtype={kwargs.get('q_data_type')} kv_dtype={kwargs.get('kv_data_type')} "
                                f"o_dtype={kwargs.get('o_data_type')} head_dim={kwargs.get('head_dim_qk')} "
                                f"window_left={kwargs.get('window_left')} "
                                f"uri_hint={uri_hint}"
                            )
                            print(msg, flush=True)
                            _append_profile_log(msg)
                        except Exception:
                            pass
                else:
                    self.prefill_wrappers_paged.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=prefill_paged_backend,
                        )
                    )
                    self.prefill_wrappers_verify.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=prefill_paged_backend,
                        )
                    )
            if self.requires_attention_sinks:
                if use_decode_wrapper_for_sinks:
                    # Decode-wrapper path for GPT-OSS sinks (often faster), but it MUST be
                    # sink-aware. The stock tensor-core decode wrapper reuses the standard
                    # prefill module, whose fa2/fa3 paged_run wrapper ignores `sinks`.
                    # Build a custom AttentionSink JIT module and pass (sink, sm_scale)
                    # via positional args at runtime.
                    if sink_backend == "auto":
                        raise ValueError(
                            "FlashInfer sinks decode-wrapper requires explicit sink_backend (fa2/fa3)."
                        )

                    init_window_left = -1
                    if (
                        self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW
                        and model_runner.sliding_window_size is not None
                    ):
                        # Keep the module specialization aligned with wrapper dispatch:
                        # - wrapper_id=0 => SWA (window_left = sliding_window_size)
                        # - wrapper_id=1 => full attention (window_left = -1)
                        init_window_left = (
                            model_runner.sliding_window_size if wrapper_id == 0 else -1
                        )
                    use_sliding_window = init_window_left != -1
                    o_dtype = self.model_dtype
                    if o_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        o_dtype = torch.bfloat16
                    fp8_enabled = sink_backend == "fa3" and self.paged_q_data_type in (
                        torch.float8_e4m3fn,
                        torch.float8_e5m2,
                    )
                    uri_prefill = get_batch_prefill_attention_sink_uri(
                        backend=sink_backend,
                        dtype_q=self.paged_q_data_type,
                        dtype_kv=self.kv_cache_dtype,
                        dtype_o=o_dtype,
                        dtype_idx=torch.int32,
                        head_dim_qk=model_runner.model_config.head_dim,
                        head_dim_vo=model_runner.model_config.head_dim,
                        pos_encoding_mode=0,
                        use_sliding_window=use_sliding_window,
                    )
                    uri_decode = get_batch_decode_attention_sink_uri(
                        backend=sink_backend,
                        dtype_q=self.paged_q_data_type,
                        dtype_kv=self.kv_cache_dtype,
                        dtype_o=o_dtype,
                        dtype_idx=torch.int32,
                        head_dim_qk=model_runner.model_config.head_dim,
                        head_dim_vo=model_runner.model_config.head_dim,
                        pos_encoding_mode=0,
                        use_sliding_window=use_sliding_window,
                        use_logits_soft_cap=False,
                    )
                    jit_args = [
                        # NOTE: The first JIT arg is the module URI. For `use_tensor_cores=True`,
                        # FlashInfer's BatchDecode wrapper uses the *prefill* module generator; for
                        # `use_tensor_cores=False`, it uses the *decode* module generator. These
                        # must not collide in the JIT cache.
                        uri_prefill,
                        self.paged_q_data_type,  # dtype_q
                        self.kv_cache_dtype,  # dtype_kv
                        o_dtype,  # dtype_o
                        torch.int32,  # idtype
                        model_runner.model_config.head_dim,  # head_dim_qk
                        model_runner.model_config.head_dim,  # head_dim_vo
                        ["sink"],  # additional_tensor_names
                        ["__nv_bfloat16"],  # additional_tensor_dtypes
                        ["sm_scale"],  # additional_scalar_names
                        ["double"],  # additional_scalar_dtypes
                        "AttentionSink",
                        attention_sink_decl[sink_backend],
                        0,  # pos_encoding_mode
                        use_sliding_window,
                        False,  # use_logits_soft_cap
                        False,  # use_fp16_qk_reduction
                        fp8_enabled,
                    ]
                    # IMPORTANT: When using the "decode-wrapper" sinks path for GPT-OSS,
                    # constructing `BatchDecodeWithPagedKVCacheWrapper` with
                    # `use_tensor_cores=True` selects FlashInfer's prefill-JIT module
                    # path (see FlashInfer wrapper implementation). This is intentional:
                    # it provides the fastest decode-throughput path on SM90, and the
                    # custom module is responsible for correctly applying sinks.
                    use_tc = bool(self.decode_use_tensor_cores)
                    if not use_tc:
                        raise ValueError(
                            "decode-wrapper sinks requires tensor cores, but decode_use_tensor_cores=False"
                        )
                    jit_args_for_wrapper = jit_args
                    self.decode_wrappers.append(
                        BatchDecodeWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            use_tensor_cores=use_tc,
                            backend=sink_backend,
                            jit_args=jit_args_for_wrapper,
                        )
                    )
                    try:
                        # Fail-fast marker: sinks + decode-wrapper must be sink-aware.
                        self.decode_wrappers[-1]._fi_sink_uri = uri_prefill  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_backend = sink_backend  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_use_swa = bool(use_sliding_window)  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_q_dtype = self.paged_q_data_type  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_kv_dtype = self.kv_cache_dtype  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_o_dtype = o_dtype  # type: ignore[attr-defined]
                        self.decode_wrappers[-1]._fi_sink_fp8_enabled = bool(fp8_enabled)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    if os.environ.get("SGLANG_FLASHINFER_LOG_DISPATCH", "0") == "1":
                        try:
                            msg = (
                                "[FI_DISPATCH] mode=decode "
                                f"wrapper_id={wrapper_id} "
                                "sinks=1 "
                                "wrapper=BatchDecodeWithPagedKVCacheWrapper "
                                f"tc={int(bool(use_tc))} "
                                f"backend={sink_backend} "
                                f"q_dtype={self.paged_q_data_type} kv_dtype={self.kv_cache_dtype} o_dtype={o_dtype} "
                                f"use_swa={int(bool(use_sliding_window))} fp8_enabled={int(bool(fp8_enabled))} "
                                f"jit_args_n={len(jit_args_for_wrapper)} "
                                f"uri={uri_prefill}"
                            )
                            print(msg, flush=True)
                            _append_profile_log(msg)
                        except Exception:
                            pass
                else:
                    # Default: use the attention-sink wrapper (prefill-style kernels).
                    init_window_left = -1
                    if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
                        # Same rationale as prefill: avoid compiling the use_swa=False sink wrapper
                        # variant for FP8 KV on SM90 (can fail GMMA operator selection).
                        if self.kv_cache_dtype in (
                            torch.float8_e4m3fn,
                            torch.float8_e5m2,
                        ):
                            init_window_left = model_runner.sliding_window_size
                        else:
                            init_window_left = (
                                model_runner.sliding_window_size if wrapper_id == 0 else -1
                            )
                    self.decode_wrappers.append(
                        BatchAttentionWithAttentionSinkWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=sink_backend,
                            q_data_type=self.paged_q_data_type,
                            kv_data_type=self.kv_cache_dtype,
                            o_data_type=self.model_dtype,
                            head_dim_qk=model_runner.model_config.head_dim,
                            head_dim_vo=model_runner.model_config.head_dim,
                            window_left=init_window_left,
                        )
                    )
                    if os.environ.get("SGLANG_FLASHINFER_LOG_DISPATCH", "0") == "1":
                        try:
                            uri_hint = None
                            try:
                                uri_hint = get_batch_prefill_attention_sink_uri(
                                    backend=sink_backend,
                                    dtype_q=self.paged_q_data_type,
                                    dtype_kv=self.kv_cache_dtype,
                                    dtype_o=self.model_dtype,
                                    dtype_idx=torch.int32,
                                    head_dim_qk=model_runner.model_config.head_dim,
                                    head_dim_vo=model_runner.model_config.head_dim,
                                    pos_encoding_mode=0,
                                    use_sliding_window=(int(init_window_left) != -1),
                                )
                            except Exception:
                                uri_hint = None
                            msg = (
                                "[FI_DISPATCH] mode=decode "
                                f"wrapper_id={wrapper_id} "
                                "sinks=1 "
                                "wrapper=BatchAttentionWithAttentionSinkWrapper "
                                f"backend={sink_backend} q_dtype={self.paged_q_data_type} kv_dtype={self.kv_cache_dtype} "
                                f"o_dtype={self.model_dtype} head_dim={model_runner.model_config.head_dim} "
                                f"window_left={init_window_left} "
                                f"uri_hint={uri_hint}"
                            )
                            print(msg, flush=True)
                            _append_profile_log(msg)
                        except Exception:
                            pass
            else:
                self.decode_wrappers.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_tensor_cores=self.decode_use_tensor_cores,
                    )
                )
                if os.environ.get("SGLANG_FLASHINFER_LOG_DISPATCH", "0") == "1":
                    try:
                        msg = (
                            "[FI_DISPATCH] mode=decode "
                            f"wrapper_id={wrapper_id} "
                            f"sinks={int(bool(self.requires_attention_sinks))} "
                            "wrapper=BatchDecodeWithPagedKVCacheWrapper "
                            f"tc={int(bool(self.decode_use_tensor_cores))}"
                        )
                        print(msg, flush=True)
                        _append_profile_log(msg)
                    except Exception:
                        pass

        if os.environ.get("SGLANG_DEBUG_FLASHINFER_BACKEND", "0") == "1":
            prefill_backend0 = (
                getattr(self.prefill_wrappers_paged[0], "_backend", None)
                if self.prefill_wrappers_paged
                else None
            )
            verify_backend0 = (
                getattr(self.prefill_wrappers_verify[0], "_backend", None)
                if self.prefill_wrappers_verify
                else None
            )
            decode_tc0 = (
                getattr(self.decode_wrappers[0], "_use_tensor_cores", None)
                if self.decode_wrappers
                else None
            )
            decode_wrapper0 = (
                self.decode_wrappers[0].__class__.__name__ if self.decode_wrappers else None
            )
            print(
                "[SGLANG][FlashInfer] init "
                f"model_dtype={self.model_dtype} kv_cache_dtype={self.kv_cache_dtype} "
                f"paged_q_dtype={self.paged_q_data_type} "
                f"prefill_paged_backend={self.prefill_paged_backend} "
                f"prefill_wrapper_backend0={prefill_backend0} "
                f"verify_wrapper_backend0={verify_backend0} "
                f"decode_use_tensor_cores={self.decode_use_tensor_cores} "
                f"decode_sinks_wrapper={decode_sinks_wrapper} "
                f"decode_wrapper0={decode_wrapper0} "
                f"decode_wrapper_tc0={decode_tc0}",
                flush=True,
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        delimiter = self.multi_item_scoring_delimiter
        if delimiter is None or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        delimiter_mask = forward_batch.input_ids == delimiter
        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            mask = delimiter_mask[seq_start:seq_end]
            pos = forward_batch.positions[seq_start:seq_end]
            delimiter_indices = torch.nonzero(mask, as_tuple=True)[0]

            if len(delimiter_indices) > 0:
                first_delim = delimiter_indices[0]
                # Prefix length: store as scalar
                prefix_len = first_delim + (
                    prefix_cache_lens[i] if prefix_cache_lens is not None else 0
                )
                prefix_len_ptr.append(
                    prefix_len.item() if torch.is_tensor(prefix_len) else prefix_len
                )

                # Compute relative positions within items after delimiters
                diff = pos[first_delim:] - torch.cummax(mask[first_delim:], 0)[1]
                token_pos = (diff - pos[first_delim]).to(torch.uint16)
                token_pos_in_items_ptr.append(token_pos)

                # Update forward_batch positions in-place
                pos[first_delim:] = diff - 1
                forward_batch.positions[seq_start:seq_end] = pos

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            device = forward_batch.input_ids.device
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        # Build final params
        device = forward_batch.input_ids.device
        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        elif forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged, False, False
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify, False, False
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if (
                self.requires_attention_sinks
                or self.is_multimodal
                or self.multi_item_scoring_delimiter is not None
            ):
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = not self.enable_deterministic
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.multi_item_scoring_delimiter is not None:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrappers = []
            for i in range(self.num_wrappers):
                if self.requires_attention_sinks and not getattr(
                    self, "use_decode_wrapper_for_sinks", False
                ):
                    decode_wrappers.append(
                        BatchAttentionWithAttentionSinkWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend="auto",
                            use_cuda_graph=True,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: num_tokens + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: num_tokens + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:num_tokens],
                            q_data_type=self.paged_q_data_type,
                            kv_data_type=self.kv_cache_dtype,
                            head_dim_qk=self.head_dim,
                            head_dim_vo=self.head_dim,
                            window_left=-1,
                        )
                    )
                else:
                    if self.requires_attention_sinks and getattr(
                        self, "use_decode_wrapper_for_sinks", False
                    ):
                        sink_backend = getattr(self, "sink_backend", "auto")
                        if sink_backend == "auto":
                            raise ValueError(
                                "FlashInfer sinks decode-wrapper requires explicit sink_backend (fa2/fa3)."
                            )

                        init_window_left = -1
                        if (
                            self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW
                            and self.model_runner.sliding_window_size is not None
                        ):
                            init_window_left = (
                                self.model_runner.sliding_window_size if i == 0 else -1
                            )
                        use_sliding_window = init_window_left != -1

                        o_dtype = self.model_dtype
                        if o_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            o_dtype = torch.bfloat16
                        fp8_enabled = sink_backend == "fa3" and self.paged_q_data_type in (
                            torch.float8_e4m3fn,
                            torch.float8_e5m2,
                        )
                        jit_args = [
                            get_batch_prefill_attention_sink_uri(
                                backend=sink_backend,
                                dtype_q=self.paged_q_data_type,
                                dtype_kv=self.kv_cache_dtype,
                                dtype_o=o_dtype,
                                dtype_idx=torch.int32,
                                head_dim_qk=self.head_dim,
                                head_dim_vo=self.head_dim,
                                pos_encoding_mode=0,
                                use_sliding_window=use_sliding_window,
                            ),
                            self.paged_q_data_type,  # dtype_q
                            self.kv_cache_dtype,  # dtype_kv
                            o_dtype,  # dtype_o
                            torch.int32,  # idtype
                            self.head_dim,  # head_dim_qk
                            self.head_dim,  # head_dim_vo
                            ["sink"],  # additional_tensor_names
                            ["__nv_bfloat16"],  # additional_tensor_dtypes
                            ["sm_scale"],  # additional_scalar_names
                            ["double"],  # additional_scalar_dtypes
                            "AttentionSink",
                            attention_sink_decl[sink_backend],
                            0,  # pos_encoding_mode
                            use_sliding_window,
                            False,  # use_logits_soft_cap
                            False,  # use_fp16_qk_reduction
                            fp8_enabled,
                        ]
                        decode_wrappers.append(
                            BatchDecodeWithPagedKVCacheWrapper(
                                self.workspace_buffer,
                                "NHD",
                                use_cuda_graph=True,
                                use_tensor_cores=True,
                                backend=sink_backend,
                                jit_args=jit_args,
                                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                            )
                        )
                    else:
                        decode_wrappers.append(
                            BatchDecodeWithPagedKVCacheWrapper(
                                self.workspace_buffer,
                                "NHD",
                                use_cuda_graph=True,
                                use_tensor_cores=self.decode_use_tensor_cores,
                                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                            )
                        )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                decode_wrappers=decode_wrappers,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
            for i in range(self.num_wrappers):
                if isinstance(decode_wrappers[i], BatchDecodeWithPagedKVCacheWrapper):
                    decode_wrappers[i].begin_forward = partial(
                        fast_decode_plan, decode_wrappers[i]
                    )
        elif forward_mode.is_target_verify():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                if self.requires_attention_sinks:
                    # IMPORTANT (GPT-OSS + EAGLE TARGET_VERIFY):
                    # TARGET_VERIFY must run with the same attention-sinks semantics as normal
                    # GPT-OSS attention. Capturing the CUDA graph with the non-sink prefill wrapper
                    # leads to severe quality collapse under EAGLE (bs>1).
                    prefill_wrappers.append(
                        BatchAttentionWithAttentionSinkWrapper(
                            self.workspace_buffer,
                            "NHD",
                            backend=getattr(self, "sink_backend", "auto"),
                            use_cuda_graph=True,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                            custom_mask_buf=self.cuda_graph_custom_mask,
                            mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                            q_data_type=self.prefill_q_data_type,
                            kv_data_type=self.kv_cache_dtype,
                            o_data_type=self.model_dtype,
                            head_dim_qk=self.head_dim,
                            head_dim_vo=self.head_dim,
                            # Compile the SWA-enabled specialization for GPT-OSS.
                            window_left=self.model_runner.sliding_window_size,
                        )
                    )
                else:
                    prefill_wrappers.append(
                        BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace_buffer,
                            "NHD",
                            use_cuda_graph=True,
                            qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                            paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                            paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                            paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                            custom_mask_buf=self.cuda_graph_custom_mask,
                            mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                        )
                    )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_draft_extend():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens,
                spec_info=spec_info,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_dllm_extend():
            prefill_wrappers = []
            for i in range(self.num_wrappers):
                prefill_wrappers.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2",
                        use_cuda_graph=True,
                        qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                        paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                        paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                        paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    )
                )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens.cpu(),  # may add a little overhead in capture stage
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=prefill_wrappers,
                use_ragged=(not self.requires_attention_sinks),
                encoder_lens=encoder_lens,
                spec_info=None,
            )
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, (not self.requires_attention_sinks), False
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=True,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        global _sglang_flashinfer_prefill_timing_ctr
        global _fi_prefill_timing_step_ctr, _fi_prefill_timing_step_do, _fi_prefill_timing_acc

        do_timing = False
        timing_enabled = os.environ.get("SGLANG_FLASHINFER_TIMING_PREFILL", "0") == "1"
        try:
            timing_layer = int(os.environ.get("SGLANG_FLASHINFER_TIMING_LAYER", "0"))
        except Exception:
            timing_layer = 0
        try:
            sample_every = int(os.environ.get("SGLANG_FLASHINFER_TIMING_SAMPLE_EVERY", "64"))
        except Exception:
            sample_every = 64

        timing_all_layers = timing_layer == -1
        if timing_enabled and (timing_all_layers or layer.layer_id == timing_layer):
            if not getattr(self.forward_metadata, "_fi_prefill_timing_banner", False):
                setattr(self.forward_metadata, "_fi_prefill_timing_banner", True)
                banner = (
                    "[FI_TIMING] prefill enabled "
                    f"pid={os.getpid()} layer={timing_layer} sample_every={sample_every}"
                )
                print(banner, flush=True)
                _append_profile_log(banner)

            if timing_all_layers and layer.layer_id == 0:
                # Flush any previous step if we didn't observe the final layer (best-effort).
                if _fi_prefill_timing_acc is not None and _fi_prefill_timing_acc.get("do", False):
                    gpu_total_ms = float(_fi_prefill_timing_acc.get("gpu_total_ms", 0.0) or 0.0)
                    qcopy_ms = float(_fi_prefill_timing_acc.get("qcopy_ms", 0.0) or 0.0)
                    setkv_ms = float(_fi_prefill_timing_acc.get("setkv_ms", 0.0) or 0.0)
                    attn_ms = float(_fi_prefill_timing_acc.get("attn_ms", 0.0) or 0.0)
                    other_ms = float(_fi_prefill_timing_acc.get("other_ms", 0.0) or 0.0)

                    def _pct(x: float, denom: float) -> float:
                        return (100.0 * x / denom) if denom > 0 else 0.0

                    agg_line = (
                        "[FI_TIMING_AGG] "
                        "mode=prefill "
                        f"step={_fi_prefill_timing_acc.get('step', -1)} "
                        f"layers={_fi_prefill_timing_acc.get('layers', 0)} "
                        f"q_tokens={_fi_prefill_timing_acc.get('q_tokens', -1)} "
                        f"qcopy_ms_sum={qcopy_ms:.3f} "
                        f"setkv_ms_sum={setkv_ms:.3f} "
                        f"attn_ms_sum={attn_ms:.3f} "
                        f"other_ms_sum={other_ms:.3f} "
                        f"gpu_total_ms_sum={gpu_total_ms:.3f} "
                        f"share_qcopy_pct={_pct(qcopy_ms, gpu_total_ms):.1f} "
                        f"share_setkv_pct={_pct(setkv_ms, gpu_total_ms):.1f} "
                        f"share_attn_pct={_pct(attn_ms, gpu_total_ms):.1f} "
                        f"share_other_pct={_pct(other_ms, gpu_total_ms):.1f} "
                        f"cpu_total_ms_sum={_fi_prefill_timing_acc.get('cpu_total_ms', 0.0):.3f}"
                    )
                    print(agg_line, flush=True)
                    _append_profile_log(agg_line)

                _fi_prefill_timing_step_ctr += 1
                _fi_prefill_timing_step_do = (_fi_prefill_timing_step_ctr % max(1, sample_every)) == 0
                _fi_prefill_timing_acc = {
                    "do": bool(_fi_prefill_timing_step_do),
                    "step": _fi_prefill_timing_step_ctr,
                    "layers": 0,
                    "q_tokens": -1,
                    "qcopy_ms": 0.0,
                    "setkv_ms": 0.0,
                    "attn_ms": 0.0,
                    "other_ms": 0.0,
                    "gpu_total_ms": 0.0,
                    "cpu_total_ms": 0.0,
                }

            if timing_all_layers:
                do_timing = bool(_fi_prefill_timing_step_do)
            else:
                _sglang_flashinfer_prefill_timing_ctr += 1
                do_timing = (_sglang_flashinfer_prefill_timing_ctr % max(1, sample_every)) == 0

        # Best-effort last-layer detection so we can emit an aggregate line even if
        # there is only one prefill/extend call.
        last_layer_id = -1
        try:
            last_layer_id = int(getattr(self.model_runner.model_config, "num_hidden_layers", -1)) - 1
        except Exception:
            last_layer_id = -1
        if last_layer_id < 0:
            try:
                last_layer_id = int(getattr(self.model_runner.model_config, "num_layers", -1)) - 1
            except Exception:
                last_layer_id = -1
        is_last_layer = bool(last_layer_id >= 0 and layer.layer_id == last_layer_id)

        if do_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            cpu_t0 = time.perf_counter()
            timing_stream = torch.cuda.current_stream()
            ev_total_start = torch.cuda.Event(enable_timing=True)
            ev_total_end = torch.cuda.Event(enable_timing=True)
            ev_total_start.record(timing_stream)
            ev_qcopy_start = torch.cuda.Event(enable_timing=True)
            ev_qcopy_end = torch.cuda.Event(enable_timing=True)
            ev_setkv_start = torch.cuda.Event(enable_timing=True)
            ev_setkv_end = torch.cuda.Event(enable_timing=True)
            ev_attn_start = torch.cuda.Event(enable_timing=True)
            ev_attn_end = torch.cuda.Event(enable_timing=True)
        else:
            cpu_t0 = None
            timing_stream = None
            ev_total_start = ev_total_end = None
            ev_qcopy_start = ev_qcopy_end = None
            ev_setkv_start = ev_setkv_end = None
            ev_attn_start = ev_attn_end = None

        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        sinks = kwargs.get("sinks", None)
        if self._diag_disable_sinks:
            sinks = None
        multi_item_enabled = bool(
            self.forward_metadata.multi_item_params
            and self.forward_metadata.multi_item_params.is_enabled()
        )
        # IMPORTANT for GPT-OSS:
        # GPT-OSS uses sliding-window attention as part of its trained semantics. Disabling SWA
        # for multi-item scoring changes the model and has shown severe quality regressions in
        # speculative/EAGLE bs>1 workloads.
        #
        # For models without GPT-OSS sinks, we keep the old behavior (disable SWA for multi-item
        # scoring) unless the caller explicitly requests otherwise via the model config.
        if (
            multi_item_enabled
            and not self.requires_attention_sinks
            and not self._diag_disable_sinks
        ):
            window_left = -1
        else:
            window_left = layer.sliding_window_size

        # Log the effective prefill dispatch key once (otherwise it can spam logs and
        # distort throughput in long-decode benchmarks).
        if (
            (not self._logged_prefill_dispatch_key)
            and os.environ.get("SGLANG_FLASHINFER_LOG_DISPATCH", "0") == "1"
        ):
            self._logged_prefill_dispatch_key = True
            wrapper_backend = getattr(prefill_wrapper_paged, "_backend", None)
            if wrapper_backend is None:
                wrapper_backend = (
                    getattr(self, "sink_backend", None)
                    if self.requires_attention_sinks
                    else self.prefill_paged_backend
                )
            uri = getattr(prefill_wrapper_paged, "_fi_sink_uri", None)
            jit_name = getattr(prefill_wrapper_paged, "_jit_module_name", None)
            _append_profile_log(
                "[FI_DISPATCH] prefill "
                f"wrapper={prefill_wrapper_paged.__class__.__name__} "
                f"q_dtype={self.prefill_q_data_type} kv_dtype={self.kv_cache_dtype} o_dtype={self.model_dtype} "
                f"sinks={int(sinks is not None)} window_left={int(window_left)} "
                f"multi_item={int(multi_item_enabled)} backend={wrapper_backend} uri={uri} jit_module_name={jit_name}"
            )

        nvtx_range_q = _nvtx_range(f"flashinfer.prefill.q_transform_layer{layer.layer_id}")

        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        if timing_enabled and timing_all_layers and layer.layer_id == 0:
            if _fi_prefill_timing_acc is not None and _fi_prefill_timing_acc.get("do", False):
                try:
                    _fi_prefill_timing_acc["q_tokens"] = int(q_reshaped.shape[0])
                except Exception:
                    pass

        q_workspace_buffers = getattr(self.forward_metadata, "q_workspace_buffers", None)
        if q_workspace_buffers is None:
            q_workspace_buffers = {}
            setattr(self.forward_metadata, "q_workspace_buffers", q_workspace_buffers)

        q_workspace = q_workspace_buffers.get(layer.layer_id, None)
        if (
            q_workspace is None
            or q_workspace.shape != q_reshaped.shape
            or q_workspace.dtype != self.prefill_q_data_type
        ):
            q_workspace = torch.empty_like(
                q_reshaped, dtype=self.prefill_q_data_type, device=q_reshaped.device
            )
            q_workspace_buffers[layer.layer_id] = q_workspace

        nvtx_range_q_copy = _nvtx_range(f"flashinfer.prefill.q_copy_layer{layer.layer_id}")
        if ev_qcopy_start is not None:
            ev_qcopy_start.record(timing_stream)
        if q_reshaped.is_contiguous():
            q_workspace.copy_(q_reshaped)
        else:
            q_for_kernel = q_reshaped.contiguous()
            q_workspace.copy_(q_for_kernel)
        if ev_qcopy_end is not None:
            ev_qcopy_end.record(timing_stream)
        if nvtx_range_q_copy is not None:
            nvtx_range_q_copy.end()

        if nvtx_range_q is not None:
            nvtx_range_q.end()

        if ev_setkv_start is not None:
            ev_setkv_start.record(timing_stream)
        if k is not None:
            assert v is not None
            if save_kv_cache:
                nvtx_range_kv = _nvtx_range(f"flashinfer.prefill.set_kv_layer{layer.layer_id}")
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
                if nvtx_range_kv is not None:
                    nvtx_range_kv.end()
        if ev_setkv_end is not None:
            ev_setkv_end.record(timing_stream)

        nvtx_range_attn = _nvtx_range(f"flashinfer.prefill.attention_kernel_layer{layer.layer_id}")
        if ev_attn_start is not None:
            ev_attn_start.record(timing_stream)
        o = prefill_wrapper_paged.forward(
            q_workspace,
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=not layer.is_cross_attention,
            sm_scale=layer.scaling,
            # Disable sliding window attention for multi-item scoring:
            # - Sliding window could cut across item boundaries, breaking semantic coherence
            # - Multi-item sequences need full attention to properly handle delimiter tokens
            # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
            #   provide more precise attention control than simple sliding windows
            # - Item-aware masking takes precedence over window-based masking
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            sinks=sinks,
        )
        if ev_attn_end is not None:
            ev_attn_end.record(timing_stream)
        if nvtx_range_attn is not None:
            nvtx_range_attn.end()

        if ev_total_end is not None:
            ev_total_end.record(timing_stream)
            torch.cuda.synchronize()
            cpu_t1 = time.perf_counter()
            qcopy_ms = (
                ev_qcopy_start.elapsed_time(ev_qcopy_end) if ev_qcopy_start is not None else 0.0
            )
            setkv_ms = (
                ev_setkv_start.elapsed_time(ev_setkv_end) if ev_setkv_start is not None else 0.0
            )
            attn_ms = (
                ev_attn_start.elapsed_time(ev_attn_end) if ev_attn_start is not None else 0.0
            )
            total_ms = (
                ev_total_start.elapsed_time(ev_total_end) if ev_total_start is not None else 0.0
            )
            cpu_ms = (cpu_t1 - cpu_t0) * 1000.0 if cpu_t0 is not None else 0.0
            other_ms = max(
                0.0, float(total_ms) - float(qcopy_ms) - float(setkv_ms) - float(attn_ms)
            )

            sample_id = (
                _fi_prefill_timing_step_ctr
                if timing_all_layers
                else _sglang_flashinfer_prefill_timing_ctr
            )
            try:
                q_tokens = int(q_reshaped.shape[0])
            except Exception:
                q_tokens = -1

            timing_line = (
                "[FI_TIMING] "
                f"mode=prefill layer={layer.layer_id} sample={sample_id} "
                f"wrapper={prefill_wrapper_paged.__class__.__name__} "
                f"kv_cache_dtype={self.kv_cache_dtype} q_dtype={self.prefill_q_data_type} "
                f"q_tokens={q_tokens} "
                f"qcopy_ms={qcopy_ms:.3f} setkv_ms={setkv_ms:.3f} attn_ms={attn_ms:.3f} "
                f"other_ms={other_ms:.3f} gpu_total_ms={total_ms:.3f} cpu_total_ms={cpu_ms:.3f}"
            )
            print(timing_line, flush=True)
            _append_profile_log(timing_line)

            if timing_all_layers and _fi_prefill_timing_acc is not None and _fi_prefill_timing_acc.get("do", False):
                _fi_prefill_timing_acc["layers"] += 1
                _fi_prefill_timing_acc["qcopy_ms"] += float(qcopy_ms)
                _fi_prefill_timing_acc["setkv_ms"] += float(setkv_ms)
                _fi_prefill_timing_acc["attn_ms"] += float(attn_ms)
                _fi_prefill_timing_acc["other_ms"] += float(other_ms)
                _fi_prefill_timing_acc["gpu_total_ms"] += float(total_ms)
                _fi_prefill_timing_acc["cpu_total_ms"] += float(cpu_ms)
                if _fi_prefill_timing_acc.get("q_tokens", -1) < 0 and q_tokens >= 0:
                    _fi_prefill_timing_acc["q_tokens"] = int(q_tokens)

                if is_last_layer:
                    gpu_total_ms = float(_fi_prefill_timing_acc.get("gpu_total_ms", 0.0) or 0.0)
                    qcopy_ms_sum = float(_fi_prefill_timing_acc.get("qcopy_ms", 0.0) or 0.0)
                    setkv_ms_sum = float(_fi_prefill_timing_acc.get("setkv_ms", 0.0) or 0.0)
                    attn_ms_sum = float(_fi_prefill_timing_acc.get("attn_ms", 0.0) or 0.0)
                    other_ms_sum = float(_fi_prefill_timing_acc.get("other_ms", 0.0) or 0.0)

                    def _pct(x: float, denom: float) -> float:
                        return (100.0 * x / denom) if denom > 0 else 0.0

                    agg_line = (
                        "[FI_TIMING_AGG] "
                        "mode=prefill "
                        f"step={_fi_prefill_timing_acc.get('step', -1)} "
                        f"layers={_fi_prefill_timing_acc.get('layers', 0)} "
                        f"q_tokens={_fi_prefill_timing_acc.get('q_tokens', -1)} "
                        f"qcopy_ms_sum={qcopy_ms_sum:.3f} "
                        f"setkv_ms_sum={setkv_ms_sum:.3f} "
                        f"attn_ms_sum={attn_ms_sum:.3f} "
                        f"other_ms_sum={other_ms_sum:.3f} "
                        f"gpu_total_ms_sum={gpu_total_ms:.3f} "
                        f"share_qcopy_pct={_pct(qcopy_ms_sum, gpu_total_ms):.1f} "
                        f"share_setkv_pct={_pct(setkv_ms_sum, gpu_total_ms):.1f} "
                        f"share_attn_pct={_pct(attn_ms_sum, gpu_total_ms):.1f} "
                        f"share_other_pct={_pct(other_ms_sum, gpu_total_ms):.1f} "
                        f"cpu_total_ms_sum={_fi_prefill_timing_acc.get('cpu_total_ms', 0.0):.3f}"
                    )
                    print(agg_line, flush=True)
                    _append_profile_log(agg_line)
                    _fi_prefill_timing_acc = None

        # Default path: paged prefill (supports attention sinks).
        # If sinks are present, always force the paged path even if the metadata
        # suggests ragged, because ragged prefill does not support sinks.
        if sinks is not None or not self.forward_metadata.use_ragged:
            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

        # Ragged prefill path (used by specific modes like DLLM); sinks are unsupported here.
        if sinks is not None:
            raise NotImplementedError(
                "FlashInfer ragged prefill does not support attention sinks; "
                "use the paged prefill path instead."
            )

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False
        if save_kv_cache and layer.attn_type == AttentionType.ENCODER_ONLY:
            save_kv_cache = False

        if self.forward_metadata.extend_no_prefix:
            # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
            # The FlashInfer head_dim limitation itself is tracked here:
            # https://github.com/flashinfer-ai/flashinfer/issues/1048
            o = self.prefill_wrapper_ragged.forward(
                q_workspace,
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
        else:
            if not self.is_dllm_model:
                # TODO: design a better interface
                # For other models, use causal attention for the ragged part as previously
                causal = True

            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q_workspace,
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
            o2, s2 = prefill_wrapper_paged.forward_return_lse(
                q_workspace,
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
            o, _ = merge_state(o1, s1, o2, s2)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        global _sglang_flashinfer_timing_ctr
        global _fi_timing_step_ctr, _fi_timing_step_do, _fi_timing_acc

        do_timing = False
        timing_enabled = os.environ.get("SGLANG_FLASHINFER_TIMING", "0") == "1"
        failfast = os.environ.get("SGLANG_FLASHINFER_FAILFAST", "0") == "1"
        try:
            timing_layer = int(os.environ.get("SGLANG_FLASHINFER_TIMING_LAYER", "0"))
        except Exception:
            timing_layer = 0
        try:
            sample_every = int(os.environ.get("SGLANG_FLASHINFER_TIMING_SAMPLE_EVERY", "64"))
        except Exception:
            sample_every = 64

        timing_all_layers = timing_layer == -1
        if timing_enabled and (timing_all_layers or layer.layer_id == timing_layer):
            if not getattr(self.forward_metadata, "_fi_timing_banner", False):
                setattr(self.forward_metadata, "_fi_timing_banner", True)
                banner = (
                    "[FI_TIMING] enabled "
                    f"pid={os.getpid()} layer={timing_layer} sample_every={sample_every}"
                )
                print(banner, flush=True)
                _append_profile_log(banner)

            if timing_all_layers:
                # Aggregate across all layers for one token step. Use layer0 as the
                # token-step boundary (it is the first decode layer).
                if layer.layer_id == 0:
                    if _fi_timing_acc is not None and _fi_timing_acc.get("do", False):
                        gpu_total_ms = float(_fi_timing_acc.get("gpu_total_ms", 0.0) or 0.0)
                        setkv_ms = float(_fi_timing_acc.get("setkv_ms", 0.0) or 0.0)
                        qcopy_ms = float(_fi_timing_acc.get("qcopy_ms", 0.0) or 0.0)
                        attn_ms = float(_fi_timing_acc.get("attn_ms", 0.0) or 0.0)
                        other_ms = float(_fi_timing_acc.get("other_ms", 0.0) or 0.0)

                        def _pct(x: float, denom: float) -> float:
                            return (100.0 * x / denom) if denom > 0 else 0.0

                        agg_line = (
                            "[FI_TIMING_AGG] "
                            f"token_step={_fi_timing_acc.get('step', -1)} "
                            f"layers={_fi_timing_acc.get('layers', 0)} "
                            f"pre_setkv_ms_sum={_fi_timing_acc.get('pre_setkv_ms', 0.0):.3f} "
                            f"setkv_ms_sum={setkv_ms:.3f} "
                            f"setkv_quant_ms_sum={_fi_timing_acc.get('setkv_quant_ms', 0.0):.3f} "
                            f"setkv_scatter_ms_sum={_fi_timing_acc.get('setkv_scatter_ms', 0.0):.3f} "
                            f"setkv_other_ms_sum={_fi_timing_acc.get('setkv_other_ms', 0.0):.3f} "
                            f"setkv_pre_scatter_ms_sum={_fi_timing_acc.get('setkv_pre_scatter_ms', 0.0):.3f} "
                            f"setkv_post_scatter_ms_sum={_fi_timing_acc.get('setkv_post_scatter_ms', 0.0):.3f} "
                            f"setkv_contig_k_ms_sum={_fi_timing_acc.get('setkv_contig_k_ms', 0.0):.3f} "
                            f"setkv_contig_v_ms_sum={_fi_timing_acc.get('setkv_contig_v_ms', 0.0):.3f} "
                            f"setkv_scale_fill_k_ms_sum={_fi_timing_acc.get('setkv_scale_fill_k_ms', 0.0):.3f} "
                            f"setkv_scale_fill_v_ms_sum={_fi_timing_acc.get('setkv_scale_fill_v_ms', 0.0):.3f} "
                            f"setkv_misc_ms_sum={_fi_timing_acc.get('setkv_misc_ms', 0.0):.3f} "
                            f"setkv_cpu_total_ms_sum={_fi_timing_acc.get('setkv_cpu_total_ms', 0.0):.3f} "
                            f"setkv_cpu_prepare_ms_sum={_fi_timing_acc.get('setkv_cpu_prepare_ms', 0.0):.3f} "
                            f"setkv_cpu_launch_ms_sum={_fi_timing_acc.get('setkv_cpu_launch_ms', 0.0):.3f} "
                            f"setkv_cpu_post_ms_sum={_fi_timing_acc.get('setkv_cpu_post_ms', 0.0):.3f} "
                            f"setkv_pack_ms_sum={_fi_timing_acc.get('setkv_pack_ms', 0.0):.3f} "
                            f"setkv_k_scatter_ms_sum={_fi_timing_acc.get('setkv_k_scatter_ms', 0.0):.3f} "
                            f"setkv_v_scatter_ms_sum={_fi_timing_acc.get('setkv_v_scatter_ms', 0.0):.3f} "
                            f"between_setkv_qcopy_ms_sum={_fi_timing_acc.get('between_setkv_qcopy_ms', 0.0):.3f} "
                            f"qcopy_ms_sum={qcopy_ms:.3f} "
                            f"between_qcopy_attn_ms_sum={_fi_timing_acc.get('between_qcopy_attn_ms', 0.0):.3f} "
                            f"attn_ms_sum={attn_ms:.3f} "
                            f"post_attn_ms_sum={_fi_timing_acc.get('post_attn_ms', 0.0):.3f} "
                            f"other_ms_sum={other_ms:.3f} "
                            f"gpu_total_ms_sum={gpu_total_ms:.3f} "
                            f"share_setkv_pct={_pct(setkv_ms, gpu_total_ms):.1f} "
                            f"share_qcopy_pct={_pct(qcopy_ms, gpu_total_ms):.1f} "
                            f"share_attn_pct={_pct(attn_ms, gpu_total_ms):.1f} "
                            f"share_other_pct={_pct(other_ms, gpu_total_ms):.1f} "
                            f"cpu_total_ms_sum={_fi_timing_acc.get('cpu_total_ms', 0.0):.3f}"
                        )
                        print(agg_line, flush=True)
                        _append_profile_log(agg_line)

                    _fi_timing_step_ctr += 1
                    _fi_timing_step_do = (_fi_timing_step_ctr % max(1, sample_every)) == 0
                    _fi_timing_acc = {
                        "do": _fi_timing_step_do,
                        "step": _fi_timing_step_ctr,
                        "layers": 0,
                        "pre_setkv_ms": 0.0,
                        "between_setkv_qcopy_ms": 0.0,
                        "between_qcopy_attn_ms": 0.0,
                        "post_attn_ms": 0.0,
                        "setkv_ms": 0.0,
                        "setkv_quant_ms": 0.0,
                        "setkv_scatter_ms": 0.0,
                        "setkv_other_ms": 0.0,
                        "setkv_pre_scatter_ms": 0.0,
                        "setkv_post_scatter_ms": 0.0,
                        "setkv_contig_k_ms": 0.0,
                        "setkv_contig_v_ms": 0.0,
                        "setkv_scale_fill_k_ms": 0.0,
                        "setkv_scale_fill_v_ms": 0.0,
                        "setkv_misc_ms": 0.0,
                        "setkv_cpu_total_ms": 0.0,
                        "setkv_cpu_prepare_ms": 0.0,
                        "setkv_cpu_launch_ms": 0.0,
                        "setkv_cpu_post_ms": 0.0,
                        "setkv_pack_ms": 0.0,
                        "setkv_k_scatter_ms": 0.0,
                        "setkv_v_scatter_ms": 0.0,
                        "qcopy_ms": 0.0,
                        "attn_ms": 0.0,
                        "other_ms": 0.0,
                        "gpu_total_ms": 0.0,
                        "cpu_total_ms": 0.0,
                    }
                do_timing = bool(_fi_timing_step_do)
            else:
                _sglang_flashinfer_timing_ctr += 1
                do_timing = (_sglang_flashinfer_timing_ctr % max(1, sample_every)) == 0

        if do_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            cpu_t0 = time.perf_counter()
            # Record all timing events on a single, explicit stream to avoid
            # cross-stream timestamp artifacts (which can inflate the "other_ms"
            # bucket and make setkv/attn attribution unreliable).
            timing_stream = torch.cuda.current_stream()
            ev_total_start = torch.cuda.Event(enable_timing=True)
            ev_total_end = torch.cuda.Event(enable_timing=True)
            ev_total_start.record(timing_stream)
            ev_setkv_start = torch.cuda.Event(enable_timing=True)
            ev_setkv_end = torch.cuda.Event(enable_timing=True)
            ev_qcopy_start = torch.cuda.Event(enable_timing=True)
            ev_qcopy_end = torch.cuda.Event(enable_timing=True)
            ev_attn_start = torch.cuda.Event(enable_timing=True)
            ev_attn_end = torch.cuda.Event(enable_timing=True)
        else:
            cpu_t0 = None
            timing_stream = None
            ev_total_start = ev_total_end = None
            ev_setkv_start = ev_setkv_end = None
            ev_qcopy_start = ev_qcopy_end = None
            ev_attn_start = ev_attn_end = None

        setkv_profile = {"__timing_stream": timing_stream} if do_timing else None

        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        nvtx_range_kv = _nvtx_range(f"flashinfer.decode.set_kv_layer{layer.layer_id}")
        if ev_setkv_start is not None:
            ev_setkv_start.record(timing_stream)
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                    profile_events=setkv_profile,
                )
        if ev_setkv_end is not None:
            ev_setkv_end.record(timing_stream)
        if nvtx_range_kv is not None:
            nvtx_range_kv.end()

        nvtx_range_q = _nvtx_range(f"flashinfer.decode.q_transform_layer{layer.layer_id}")

        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Fast path: if Q is already contiguous and dtype matches the kernel, avoid
        # an extra device-to-device copy into q_workspace.
        needs_q_workspace = not (q_reshaped.is_contiguous() and q_reshaped.dtype == self.paged_q_data_type)
        if needs_q_workspace:
            if self.forward_metadata.q_workspace_buffers is None:
                self.forward_metadata.q_workspace_buffers = {}

            q_workspace = self.forward_metadata.q_workspace_buffers.get(layer.layer_id, None)
            if (
                q_workspace is None
                or q_workspace.shape != q_reshaped.shape
                or q_workspace.dtype != self.paged_q_data_type
            ):
                q_workspace = torch.empty_like(
                    q_reshaped, dtype=self.paged_q_data_type, device=q_reshaped.device
                )
                self.forward_metadata.q_workspace_buffers[layer.layer_id] = q_workspace

            nvtx_range_q_copy = _nvtx_range(f"flashinfer.decode.q_copy_layer{layer.layer_id}")
            if ev_qcopy_start is not None:
                ev_qcopy_start.record(timing_stream)
            if q_reshaped.is_contiguous():
                q_workspace.copy_(q_reshaped)
            else:
                q_workspace.copy_(q_reshaped.contiguous())
            if ev_qcopy_end is not None:
                ev_qcopy_end.record(timing_stream)
            if nvtx_range_q_copy is not None:
                nvtx_range_q_copy.end()

            q_for_kernel = q_workspace
        else:
            if ev_qcopy_start is not None:
                ev_qcopy_start.record(timing_stream)
            if ev_qcopy_end is not None:
                ev_qcopy_end.record(timing_stream)
            q_for_kernel = q_reshaped

        if nvtx_range_q is not None:
            nvtx_range_q.end()

        nvtx_range_attn = _nvtx_range(f"flashinfer.decode.attention_kernel_layer{layer.layer_id}")
        if (layer.layer_id == 0) and (not self._logged_decode_dispatch_key):
            self._logged_decode_dispatch_key = True
            sinks = None if self._diag_disable_sinks else kwargs.get("sinks", None)
            window_left = -1
            if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
                try:
                    window_left = int(layer.sliding_window_size)
                except Exception:
                    window_left = int(self.model_runner.sliding_window_size or -1)
            wrapper_backend = getattr(decode_wrapper, "_backend", None)
            uri = getattr(decode_wrapper, "_fi_sink_uri", None)
            jit_name = getattr(decode_wrapper, "_jit_module_name", None)
            _append_profile_log(
                "[FI_DISPATCH] decode "
                f"wrapper={decode_wrapper.__class__.__name__} "
                f"q_dtype={self.paged_q_data_type} kv_dtype={self.kv_cache_dtype} o_dtype={self.model_dtype} "
                f"sinks={int(sinks is not None)} window_left={int(window_left)} "
                f"decode_use_tensor_cores={int(bool(self.decode_use_tensor_cores))} "
                f"decode_sinks_wrapper={getattr(self, 'decode_sinks_wrapper', 'unknown')} "
                f"backend={wrapper_backend} uri={uri} jit_module_name={jit_name}"
            )
        if ev_attn_start is not None:
            ev_attn_start.record(timing_stream)

        if isinstance(decode_wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # NOTE: `BatchDecodeWithPagedKVCacheWrapper.forward` is deprecated and
            # can overwrite plan-time invariants (e.g. `window_left`). Use `run()`
            # and only mutate the truly per-call scalars.
            try:
                decode_wrapper._sm_scale = layer.scaling
                decode_wrapper._logits_soft_cap = layer.logit_cap
            except Exception:
                pass
            sinks = None if self._diag_disable_sinks else kwargs.get("sinks", None)
            kv_buf = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if sinks is not None:
                # Decode-wrapper sinks MUST use the AttentionSink JIT module; otherwise sinks are
                # silently ignored in the underlying fa2/fa3 paged_run wrapper and long-context
                # quality collapses catastrophically.
                if getattr(decode_wrapper, "_jit_module", None) is None:
                    raise RuntimeError(
                        "FlashInfer decode-wrapper is missing the AttentionSink JIT module; "
                        "this path ignores sinks and will destroy long-context quality. "
                        "Use SGLANG_FLASHINFER_DECODE_SINKS_WRAPPER=sink or rebuild decode wrappers with JIT."
                    )

                # Prefer robust sink-awareness detection from the FlashInfer wrapper itself
                # (stored at construction-time) rather than relying only on SGLang-side
                # monkey-patched markers, which may be absent on some wrapper instances
                # (e.g. CUDA graph capture/replay paths).
                sink_aware = False
                try:
                    if hasattr(decode_wrapper, "_fi_sink_uri"):
                        sink_aware = True
                    elif bool(getattr(decode_wrapper, "_jit_is_attention_sink", False)):
                        sink_aware = True
                    else:
                        name = str(getattr(decode_wrapper, "_jit_module_name", "") or "")
                        if "attention_sink" in name.lower():
                            sink_aware = True
                except Exception:
                    sink_aware = False

                if not sink_aware:
                    raise RuntimeError(
                        "FlashInfer sinks decode-wrapper selected, but wrapper does not appear sink-aware. "
                        "This likely means sinks would be ignored (silent quality collapse). "
                        f"jit_module_name={getattr(decode_wrapper, '_jit_module_name', None)!r} "
                        f"has_fi_sink_uri={int(hasattr(decode_wrapper, '_fi_sink_uri'))}"
                    )

                # AttentionSink semantics: sinks is bf16 log-sink per Q head.
                num_q_heads = int(layer.tp_q_head_num)
                if sinks.numel() != num_q_heads:
                    raise ValueError(
                        "FlashInfer decode-wrapper sinks mismatch: expected sinks per Q head. "
                        f"got numel={int(sinks.numel())} expected={num_q_heads} "
                        f"(layer_id={layer.layer_id} num_q_heads={num_q_heads})"
                    )
                if sinks.dtype != torch.bfloat16:
                    raise TypeError(
                        "FlashInfer decode-wrapper (AttentionSink) requires bf16 sinks. "
                        f"got dtype={sinks.dtype} (layer_id={layer.layer_id})"
                    )
                if failfast:
                    finite = torch.isfinite(sinks).all()
                    try:
                        is_capturing = bool(torch.cuda.is_current_stream_capturing())
                    except Exception:
                        is_capturing = False
                    if is_capturing:
                        if hasattr(torch, "_assert_async"):
                            torch._assert_async(finite, "[FI_FAILFAST] sinks contains NaN/Inf")
                    else:
                        try:
                            if not bool(finite.item()):
                                raise RuntimeError("sinks contains NaN/Inf")
                        except Exception as e:
                            msg = (
                                "[FI_FAILFAST] sinks_invalid "
                                f"layer={layer.layer_id} "
                                f"sinks_shape={tuple(getattr(sinks, 'shape', ())) if hasattr(sinks, 'shape') else None} "
                                f"sinks_dtype={getattr(sinks, 'dtype', None)} "
                                f"err={e}"
                            )
                            _append_profile_log(msg)
                            raise

                sm_scale_val = (
                    float(layer.scaling)
                    if layer.scaling is not None
                    else float(q_for_kernel.size(-1)) ** -0.5
                )
                if layer.k_scale_float is not None:
                    sm_scale_val *= float(layer.k_scale_float)

                window_left_for_run = -1
                try:
                    window_left_for_run = int(layer.sliding_window_size)
                except Exception:
                    try:
                        window_left_for_run = int(self.model_runner.sliding_window_size or -1)
                    except Exception:
                        window_left_for_run = -1

                # Pass (sinks, sm_scale) via positional args to the custom JIT module.
                o = decode_wrapper.run(
                    q_for_kernel,
                    kv_buf,
                    sinks,
                    sm_scale_val,
                    window_left=window_left_for_run,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=None,
                    v_scale=layer.v_scale_float,
                )
            else:
                o = decode_wrapper.run(
                    q_for_kernel,
                    kv_buf,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                    sinks=None,
                )
        else:
            # When using the AttentionSink wrapper path, pass `window_left` explicitly so
            # the runtime behavior matches the wrapper specialization compiled in __init__.
            window_left = -1
            if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
                window_left = self.model_runner.sliding_window_size
            o = decode_wrapper.forward(
                q_for_kernel,
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                sinks=None if self._diag_disable_sinks else kwargs.get("sinks", None),
                window_left=window_left,
            )
        if nvtx_range_attn is not None:
            nvtx_range_attn.end()

        if failfast:
            finite = torch.isfinite(o).all()
            try:
                is_capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception:
                is_capturing = False
            if is_capturing:
                if hasattr(torch, "_assert_async"):
                    torch._assert_async(
                        finite, "[FI_FAILFAST] attention output contains NaN/Inf"
                    )
            else:
                try:
                    if not bool(finite.item()):
                        raise RuntimeError("attention output contains NaN/Inf")
                except Exception as e:
                    msg = (
                        "[FI_FAILFAST] attn_output_invalid "
                        f"layer={layer.layer_id} "
                        f"o_shape={tuple(o.shape)} o_dtype={o.dtype} "
                        f"q_dtype={q_for_kernel.dtype} kv_dtype={self.kv_cache_dtype} "
                        f"wrapper={decode_wrapper.__class__.__name__} "
                        f"err={e}"
                    )
                    _append_profile_log(msg)
                    raise

        if ev_attn_end is not None:
            ev_attn_end.record(timing_stream)
        if ev_total_end is not None:
            ev_total_end.record(timing_stream)
            torch.cuda.synchronize()
            cpu_t1 = time.perf_counter()
            setkv_ms = (
                ev_setkv_start.elapsed_time(ev_setkv_end) if ev_setkv_start is not None else 0.0
            )
            qcopy_ms = (
                ev_qcopy_start.elapsed_time(ev_qcopy_end) if ev_qcopy_start is not None else 0.0
            )
            attn_ms = (
                ev_attn_start.elapsed_time(ev_attn_end) if ev_attn_start is not None else 0.0
            )
            total_ms = (
                ev_total_start.elapsed_time(ev_total_end) if ev_total_start is not None else 0.0
            )
            cpu_ms = (cpu_t1 - cpu_t0) * 1000.0 if cpu_t0 is not None else 0.0
            other_ms = max(0.0, float(total_ms) - float(setkv_ms) - float(qcopy_ms) - float(attn_ms))
            sample_id = _fi_timing_step_ctr if timing_all_layers else _sglang_flashinfer_timing_ctr

            pre_setkv_ms = 0.0
            between_setkv_qcopy_ms = 0.0
            between_qcopy_attn_ms = 0.0
            post_attn_ms = 0.0
            if (
                ev_total_start is not None
                and ev_setkv_start is not None
                and ev_setkv_end is not None
                and ev_qcopy_start is not None
                and ev_qcopy_end is not None
                and ev_attn_start is not None
                and ev_attn_end is not None
            ):
                # These gaps (GPU time) explain `other_ms` for decode: they cover all
                # work that happens outside the explicit setkv/qcopy/attn buckets.
                pre_setkv_ms = float(ev_total_start.elapsed_time(ev_setkv_start))
                between_setkv_qcopy_ms = float(
                    ev_setkv_end.elapsed_time(ev_qcopy_start)
                )
                between_qcopy_attn_ms = float(ev_qcopy_end.elapsed_time(ev_attn_start))
                post_attn_ms = float(ev_attn_end.elapsed_time(ev_total_end))

            setkv_quant_ms = 0.0
            setkv_scatter_ms = 0.0
            setkv_pre_scatter_ms = 0.0
            setkv_post_scatter_ms = 0.0
            setkv_contig_k_ms = 0.0
            setkv_contig_v_ms = 0.0
            setkv_scale_fill_k_ms = 0.0
            setkv_scale_fill_v_ms = 0.0
            setkv_pack_ms = 0.0
            setkv_k_scatter_ms = 0.0
            setkv_v_scatter_ms = 0.0
            setkv_cpu_total_ms = 0.0
            setkv_cpu_prepare_ms = 0.0
            setkv_cpu_launch_ms = 0.0
            setkv_cpu_post_ms = 0.0
            if isinstance(setkv_profile, dict):
                qev = setkv_profile.get("setkv_quant_ev", None)
                sev = setkv_profile.get("setkv_scatter_ev", None)
                ckev = setkv_profile.get("setkv_contig_k_ev", None)
                cvev = setkv_profile.get("setkv_contig_v_ev", None)
                sfkev = setkv_profile.get("setkv_scale_fill_k_ev", None)
                sfvev = setkv_profile.get("setkv_scale_fill_v_ev", None)
                pev = setkv_profile.get("setkv_pack_ev", None)
                kev = setkv_profile.get("setkv_k_scatter_ev", None)
                vev = setkv_profile.get("setkv_v_scatter_ev", None)
                has_quant_ev = (
                    isinstance(qev, tuple)
                    and len(qev) == 2
                    and qev[0] is not None
                    and qev[1] is not None
                )
                has_scatter_ev = (
                    isinstance(sev, tuple)
                    and len(sev) == 2
                    and sev[0] is not None
                    and sev[1] is not None
                )
                has_pack_ev = (
                    isinstance(pev, tuple)
                    and len(pev) == 2
                    and pev[0] is not None
                    and pev[1] is not None
                )
                has_contig_k_ev = (
                    isinstance(ckev, tuple)
                    and len(ckev) == 2
                    and ckev[0] is not None
                    and ckev[1] is not None
                )
                has_contig_v_ev = (
                    isinstance(cvev, tuple)
                    and len(cvev) == 2
                    and cvev[0] is not None
                    and cvev[1] is not None
                )
                has_scale_fill_k_ev = (
                    isinstance(sfkev, tuple)
                    and len(sfkev) == 2
                    and sfkev[0] is not None
                    and sfkev[1] is not None
                )
                has_scale_fill_v_ev = (
                    isinstance(sfvev, tuple)
                    and len(sfvev) == 2
                    and sfvev[0] is not None
                    and sfvev[1] is not None
                )
                has_k_scatter_ev = (
                    isinstance(kev, tuple)
                    and len(kev) == 2
                    and kev[0] is not None
                    and kev[1] is not None
                )
                has_v_scatter_ev = (
                    isinstance(vev, tuple)
                    and len(vev) == 2
                    and vev[0] is not None
                    and vev[1] is not None
                )
                try:
                    setkv_cpu_total_ms = float(
                        setkv_profile.get("setkv_cpu_total_ms", 0.0) or 0.0
                    )
                except Exception:
                    setkv_cpu_total_ms = 0.0
                try:
                    setkv_cpu_prepare_ms = float(
                        setkv_profile.get("setkv_cpu_prepare_ms", 0.0) or 0.0
                    )
                except Exception:
                    setkv_cpu_prepare_ms = 0.0
                try:
                    setkv_cpu_launch_ms = float(
                        setkv_profile.get("setkv_cpu_launch_ms", 0.0) or 0.0
                    )
                except Exception:
                    setkv_cpu_launch_ms = 0.0
                try:
                    setkv_cpu_post_ms = float(
                        setkv_profile.get("setkv_cpu_post_ms", 0.0) or 0.0
                    )
                except Exception:
                    setkv_cpu_post_ms = 0.0
                if has_quant_ev:
                    setkv_quant_ms = float(qev[0].elapsed_time(qev[1]))
                if has_scatter_ev:
                    setkv_scatter_ms = float(sev[0].elapsed_time(sev[1]))
                    # Additional breakdown (helps debug fused FP8 setkv): isolate time before/after
                    # the scatter region inside the setkv envelope.
                    try:
                        if ev_setkv_start is not None:
                            setkv_pre_scatter_ms = float(
                                ev_setkv_start.elapsed_time(sev[0])
                            )
                        if ev_setkv_end is not None:
                            setkv_post_scatter_ms = float(
                                sev[1].elapsed_time(ev_setkv_end)
                            )
                    except Exception:
                        setkv_pre_scatter_ms = 0.0
                        setkv_post_scatter_ms = 0.0
                elif has_quant_ev:
                    # Fallback: if we failed to isolate scatter on-stream, approximate it
                    # as the remainder of setkv time after quantization.
                    setkv_scatter_ms = max(0.0, float(setkv_ms) - float(setkv_quant_ms))
                else:
                    setkv_scatter_ms = float(setkv_ms)

                if has_pack_ev:
                    setkv_pack_ms = float(pev[0].elapsed_time(pev[1]))
                if has_contig_k_ev:
                    setkv_contig_k_ms = float(ckev[0].elapsed_time(ckev[1]))
                if has_contig_v_ev:
                    setkv_contig_v_ms = float(cvev[0].elapsed_time(cvev[1]))
                if has_scale_fill_k_ev:
                    setkv_scale_fill_k_ms = float(sfkev[0].elapsed_time(sfkev[1]))
                if has_scale_fill_v_ev:
                    setkv_scale_fill_v_ms = float(sfvev[0].elapsed_time(sfvev[1]))
                if has_k_scatter_ev:
                    setkv_k_scatter_ms = float(kev[0].elapsed_time(kev[1]))
                if has_v_scatter_ev:
                    setkv_v_scatter_ms = float(vev[0].elapsed_time(vev[1]))
            setkv_other_ms = max(
                0.0, float(setkv_ms) - float(setkv_quant_ms) - float(setkv_scatter_ms)
            )
            setkv_misc_ms = max(
                0.0,
                float(setkv_ms)
                - float(setkv_quant_ms)
                - float(setkv_scatter_ms)
                - float(setkv_contig_k_ms)
                - float(setkv_contig_v_ms)
                - float(setkv_scale_fill_k_ms)
                - float(setkv_scale_fill_v_ms),
            )

            timing_line = (
                "[FI_TIMING] "
                f"mode=decode layer={layer.layer_id} sample={sample_id} "
                f"wrapper={decode_wrapper.__class__.__name__} "
                f"kv_cache_dtype={self.kv_cache_dtype} q_dtype={self.paged_q_data_type} "
                f"pre_setkv_ms={pre_setkv_ms:.3f} between_setkv_qcopy_ms={between_setkv_qcopy_ms:.3f} "
                f"between_qcopy_attn_ms={between_qcopy_attn_ms:.3f} post_attn_ms={post_attn_ms:.3f} "
                f"setkv_ms={setkv_ms:.3f} setkv_quant_ms={setkv_quant_ms:.3f} setkv_scatter_ms={setkv_scatter_ms:.3f} "
                f"setkv_other_ms={setkv_other_ms:.3f} "
                f"setkv_pre_scatter_ms={setkv_pre_scatter_ms:.3f} setkv_post_scatter_ms={setkv_post_scatter_ms:.3f} "
                f"setkv_contig_k_ms={setkv_contig_k_ms:.3f} setkv_contig_v_ms={setkv_contig_v_ms:.3f} "
                f"setkv_scale_fill_k_ms={setkv_scale_fill_k_ms:.3f} setkv_scale_fill_v_ms={setkv_scale_fill_v_ms:.3f} "
                f"setkv_misc_ms={setkv_misc_ms:.3f} "
                f"setkv_cpu_total_ms={setkv_cpu_total_ms:.3f} "
                f"setkv_cpu_prepare_ms={setkv_cpu_prepare_ms:.3f} "
                f"setkv_cpu_launch_ms={setkv_cpu_launch_ms:.3f} "
                f"setkv_cpu_post_ms={setkv_cpu_post_ms:.3f} "
                f"setkv_pack_ms={setkv_pack_ms:.3f} setkv_k_scatter_ms={setkv_k_scatter_ms:.3f} setkv_v_scatter_ms={setkv_v_scatter_ms:.3f} "
                f"qcopy_ms={qcopy_ms:.3f} attn_ms={attn_ms:.3f} "
                f"other_ms={other_ms:.3f} gpu_total_ms={total_ms:.3f} cpu_total_ms={cpu_ms:.3f}"
            )
            print(timing_line, flush=True)
            _append_profile_log(timing_line)

            if timing_all_layers and _fi_timing_acc is not None and _fi_timing_acc.get("do", False):
                _fi_timing_acc["layers"] += 1
                _fi_timing_acc["pre_setkv_ms"] += float(pre_setkv_ms)
                _fi_timing_acc["between_setkv_qcopy_ms"] += float(between_setkv_qcopy_ms)
                _fi_timing_acc["between_qcopy_attn_ms"] += float(between_qcopy_attn_ms)
                _fi_timing_acc["post_attn_ms"] += float(post_attn_ms)
                _fi_timing_acc["setkv_ms"] += float(setkv_ms)
                _fi_timing_acc["setkv_quant_ms"] += float(setkv_quant_ms)
                _fi_timing_acc["setkv_scatter_ms"] += float(setkv_scatter_ms)
                _fi_timing_acc["setkv_other_ms"] += float(setkv_other_ms)
                _fi_timing_acc["setkv_pre_scatter_ms"] += float(setkv_pre_scatter_ms)
                _fi_timing_acc["setkv_post_scatter_ms"] += float(setkv_post_scatter_ms)
                _fi_timing_acc["setkv_contig_k_ms"] += float(setkv_contig_k_ms)
                _fi_timing_acc["setkv_contig_v_ms"] += float(setkv_contig_v_ms)
                _fi_timing_acc["setkv_scale_fill_k_ms"] += float(setkv_scale_fill_k_ms)
                _fi_timing_acc["setkv_scale_fill_v_ms"] += float(setkv_scale_fill_v_ms)
                _fi_timing_acc["setkv_misc_ms"] += float(setkv_misc_ms)
                _fi_timing_acc["setkv_cpu_total_ms"] += float(setkv_cpu_total_ms)
                _fi_timing_acc["setkv_cpu_prepare_ms"] += float(setkv_cpu_prepare_ms)
                _fi_timing_acc["setkv_cpu_launch_ms"] += float(setkv_cpu_launch_ms)
                _fi_timing_acc["setkv_cpu_post_ms"] += float(setkv_cpu_post_ms)
                _fi_timing_acc["setkv_pack_ms"] += float(setkv_pack_ms)
                _fi_timing_acc["setkv_k_scatter_ms"] += float(setkv_k_scatter_ms)
                _fi_timing_acc["setkv_v_scatter_ms"] += float(setkv_v_scatter_ms)
                _fi_timing_acc["qcopy_ms"] += float(qcopy_ms)
                _fi_timing_acc["attn_ms"] += float(attn_ms)
                _fi_timing_acc["other_ms"] += float(other_ms)
                _fi_timing_acc["gpu_total_ms"] += float(total_ms)
                _fi_timing_acc["cpu_total_ms"] += float(cpu_ms)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = attn_backend.paged_q_data_type
        self.o_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        self._use_prefill_wrapper_for_decode = bool(
            attn_backend.requires_attention_sinks
            and not getattr(attn_backend, "use_decode_wrapper_for_sinks", False)
        )
        self._qo_indptr = None
        self._qo_indptr_range = None
        if self._use_prefill_wrapper_for_decode:
            max_bs = model_runner.req_to_token_pool.size
            self._qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(attn_backend.num_wrappers)
            ]
            self._qo_indptr_range = torch.arange(
                max_bs + 1, dtype=torch.int32, device=model_runner.device
            )

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            qo_indptr=(self._qo_indptr[0] if self._use_prefill_wrapper_for_decode else None),
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                qo_indptr=(
                    self._qo_indptr[wrapper_id]
                    if self._use_prefill_wrapper_for_decode
                    else None
                ),
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=seq_lens_cpu,
                qo_indptr=(
                    self._qo_indptr[wrapper_id]
                    if self._use_prefill_wrapper_for_decode
                    else None
                ),
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper | BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool and spec_info is not None:
            # Diagnostic toggle: if set, skip SWA translation for speculative runs.
            # (Kept off by default; correctness investigation only.)
            if os.getenv("SGLANG_FLASHINFER_SPEC_SKIP_SWA_TRANSLATE", "0") == "1":
                use_sliding_window_kv_pool = False

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper):
            window_left = -1
            if (
                getattr(self.attn_backend, "dispatch_reason", None)
                == WrapperDispatch.SLIDING_WINDOW
                and kv_start_idx is not None
                and self.sliding_window_size is not None
            ):
                window_left = int(self.sliding_window_size)

            # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
            # by checking if it's a partial function with fast_decode_plan as the func
            wrapper_uses_fast_decode_plan = (
                hasattr(wrapper.begin_forward, "func")
                and wrapper.begin_forward.func == fast_decode_plan
            )

            if wrapper_uses_fast_decode_plan:
                # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
                wrapper.begin_forward(
                    kv_indptr,
                    kv_indices,
                    self.kv_last_page_len[:bs],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    1,
                    data_type=self.data_type,
                    q_data_type=self.q_data_type,
                    o_data_type=self.o_data_type,
                    window_left=window_left,
                    non_blocking=True,
                    fixed_split_size=fixed_split_size,
                    disable_split_kv=(
                        disable_split_kv if disable_split_kv is not None else False
                    ),
                    global_override_indptr_cpu=global_override_indptr_cpu,
                )
            else:
                # When using original begin_forward, don't pass global_override_indptr_cpu
                wrapper.begin_forward(
                    kv_indptr,
                    kv_indices,
                    self.kv_last_page_len[:bs],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    1,
                    data_type=self.data_type,
                    q_data_type=self.q_data_type,
                    window_left=window_left,
                    o_data_type=self.o_data_type,
                    non_blocking=True,
                    fixed_split_size=fixed_split_size,
                    disable_split_kv=(
                        disable_split_kv if disable_split_kv is not None else False
                    ),
                )
        else:
            if qo_indptr is None:
                raise ValueError(
                    "Prefill-style decode wrapper requires qo_indptr buffer."
                )

            # Decode always has q_len=1 per request, so qo_indptr = [0, 1, 2, ..., bs].
            assert self._qo_indptr_range is not None
            qo_indptr[: bs + 1].copy_(self._qo_indptr_range[: bs + 1], non_blocking=True)
            qo_indptr = qo_indptr[: bs + 1]

            wrapper.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                o_data_type=self.o_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type_ragged = model_runner.dtype
        self.q_data_type_paged = attn_backend.prefill_q_data_type
        self.o_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
    ):
        if use_ragged:
            # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
            # and forward_batch.extend_seq_lens_cpu
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
    ):
        if prefix_lens is None:
            prefix_lens = torch.zeros_like(seq_lens)

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                multi_item_params=multi_item_params,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                multi_item_params=multi_item_params,
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            assert isinstance(spec_info, SpecInput)
            paged_kernel_lens_for_spec = paged_kernel_lens
            paged_kernel_lens_sum_for_spec = paged_kernel_lens_sum
            # IMPORTANT (EAGLE verify + SWA):
            # Some scheduler paths pass `seq_lens` that already includes the verify tokens
            # (i.e. `seq_lens - prefix_lens == draft_token_num`). EagleVerifyInput's
            # `generate_attn_arg_prefill()` adds `draft_token_num` internally, so passing a
            # lens that already includes them will double-count and mis-size the custom mask.
            if spec_info.spec_input_type == SpecInputType.EAGLE_VERIFY:
                draft_token_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
                if draft_token_num > 0:
                    extend_lens = (seq_lens - prefix_lens)[:bs]
                    if torch.all(extend_lens == draft_token_num):
                        paged_kernel_lens_for_spec = paged_kernel_lens - draft_token_num
                        paged_kernel_lens_sum_for_spec = paged_kernel_lens_sum - (
                            draft_token_num * bs
                        )
                        if not getattr(self, "_logged_eagle_verify_len_fix", False):
                            self._logged_eagle_verify_len_fix = True
                            _append_profile_log(
                                "[FI_EAGLE_VERIFY] Adjust paged_kernel_lens to exclude "
                                f"draft_token_num={draft_token_num} (avoid double-count)."
                            )
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens_for_spec,
                    paged_kernel_lens_sum_for_spec,
                    self.req_to_token,
                )
            )
            # IMPORTANT (SWA + speculative/EAGLE):
            # When sliding-window attention is active, `kv_start_idx` must be applied so
            # that the KV indices correspond to the *tail window* (plus any newly-extended
            # tokens) rather than the prefix window [0..window). Some SpecInput
            # implementations generate KV indices assuming full-context; for SWA wrappers
            # we must rebuild KV indices using `kv_start_idx`.
            if kv_start_idx is not None:
                kv_total = int(kv_indptr[-1].item())
                kv_lens = (kv_indptr[1 : bs + 1] - kv_indptr[:bs]).to(
                    dtype=paged_kernel_lens.dtype, device=req_pool_indices.device
                )
                kv_indices_swa = torch.empty(
                    kv_total + 256, dtype=torch.int32, device=req_pool_indices.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    kv_start_idx,
                    kv_indices_swa,
                    self.req_to_token.shape[1],
                )
                kv_indices = kv_indices_swa

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type_ragged,
            )

        if use_sliding_window_kv_pool and spec_info is not None:
            # Diagnostic toggle: if set, skip SWA translation for speculative runs.
            # (Use for correctness triage only; keep off by default.)
            if os.getenv("SGLANG_FLASHINFER_SPEC_SKIP_SWA_TRANSLATE", "0") == "1":
                use_sliding_window_kv_pool = False

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters.
        #
        # IMPORTANT for GPT-OSS sinks + EAGLE:
        # FlashInfer's MaskMode::kMultiItemScoring path has shown severe quality regressions
        # for speculative decoding when GPT-OSS attention sinks are enabled (bs>1). Until we
        # establish parity, force the custom-mask path for speculative runs and sinks models.
        multi_item_enabled = bool(multi_item_params is not None and multi_item_params.is_enabled())
        allow_multi_item_with_sinks = get_bool_env_var(
            "SGLANG_FLASHINFER_ALLOW_MULTI_ITEM_WITH_SINKS", "false"
        )
        force_custom_mask = bool(
            spec_info is not None
            and self.attn_backend.requires_attention_sinks
            and not allow_multi_item_with_sinks
        )
        if multi_item_enabled and not force_custom_mask:
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type_paged,
            kv_data_type=self.data_type,
            o_data_type=self.o_data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
        )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # IMPORTANT (CUDA graphs):
        # Avoid any device->host copies here (e.g. `.cpu()` on a CUDA tensor). In CUDA-graph
        # replay this can trigger illegal memory access / MMU faults (XID 31) on H100.
        #
        # For draft decode, kv_indptr is a pure prefix-sum over per-sequence KV lengths.
        # We can reconstruct the exact CPU indptr deterministically from seq_lens_cpu and
        # the speculative step index, without touching CUDA memory.
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        indptr_cpu_whole = None
        if seq_lens_cpu is not None:
            # seq_lens_cpu: (num_seqs,) -> expand to (bs,) by repeating per topk beam.
            expanded_seq_lens = seq_lens_cpu[:num_seqs].repeat_interleave(self.topk)
            indptr_cpu_whole = torch.empty(
                (self.speculative_num_steps, bs + 1),
                dtype=self.kv_indptr.dtype,
                device="cpu",
            )
            indptr_cpu_whole.zero_()
            for step_i in range(self.speculative_num_steps - 1):
                # At speculative step_i, each beam has (base_len + (step_i + 1)) KV items.
                indptr_cpu_whole[step_i, 0] = 0
                indptr_cpu_whole[step_i, 1 : bs + 1] = torch.cumsum(
                    expanded_seq_lens + (step_i + 1), dim=0
                )
        else:
            # Fallback (non-graph / debugging): materialize from CUDA.
            indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # `max_bs` is the number of sequences (requests) captured by the EAGLE draft runner.
        # In EAGLE, the draft model processes `topk` beams per sequence, so the draft-decode
        # "token batch size" is `max_num_tokens = max_bs * topk`.
        #
        # The KV-indices buffer is indexed by (spec_step, token_batch, max_context_len),
        # so it must be sized by `max_num_tokens * max_context_len`, not `max_bs * max_context_len`.
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
