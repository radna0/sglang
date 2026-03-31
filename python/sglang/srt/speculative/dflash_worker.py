import logging
import math
import os
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker

from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput

from sglang.srt.speculative.dflash_controller import (
    DFlashAdaptivePqConfig,
    DFlashAdaptivePqController,
    DFlashDifficultySignals,
    DFlashReqDifficultyState,
    compute_adaptive_max_steps_for_req,
    req_is_hard_enough_for_fanout,
    survival_should_force_target_only,
)
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_shared_pool_verify_append,
    can_dflash_use_fused_qkv_proj,
    gather_dflash_committed_hidden,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_append_path,
    resolve_dflash_verify_mask_policy,
    update_dflash_req_verify_bookkeeping,
)
from sglang.srt.speculative.pq_filter import filter_topk_probs_like_sglang_sampler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


@dataclass
class _DFlashDraftProposal:
    draft_tokens: torch.Tensor
    draft_token_num: int
    verify_mode: str
    draft_topk: int = 0
    draft_topk_ids: torch.Tensor | None = None
    draft_topk_probs: torch.Tensor | None = None
    max_steps_per_req: torch.Tensor | None = None
    draft_conf_debug: dict | None = None


@dataclass
class _DFlashSsdCachedProposal:
    expected_seq_len: int
    expected_verified_id: int
    draft_tokens: torch.Tensor
    draft_token_num: int
    verify_mode: str
    draft_topk: int = 0
    draft_topk_ids: torch.Tensor | None = None
    draft_topk_probs: torch.Tensor | None = None
    max_steps: int | None = None
    draft_conf_debug: dict | None = None


@dataclass
class _DFlashDraftBlockBuffers:
    block_ids: torch.Tensor
    positions: torch.Tensor
    block_tokens: torch.Tensor
    block_end: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens_plus_cpu: torch.Tensor


@dataclass
class _DFlashSingleReqSamplingBatchView:
    reqs: list
    device: torch.device


@dataclass
class _DFlashSingleReqBatchView:
    reqs: list
    req_pool_indices: torch.Tensor
    sampling_info: SamplingBatchInfo

    def batch_size(self) -> int:
        return len(self.reqs)


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._logged_first_verify = False

        self._ssd_enabled = (os.environ.get("SGLANG_DFLASH_SSD_ENABLE") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        self._ssd_prepare_next = (os.environ.get("SGLANG_DFLASH_SSD_PREPARE_NEXT") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        self._ssd_prepare_failures = 0
        self._ssd_batch_hits = 0
        self._ssd_batch_misses = 0
        self._ssd_req_hits = 0
        self._ssd_req_misses = 0
        self._verify_step = 0
        try:
            self._ssd_max_entries = max(
                1, int((os.environ.get("SGLANG_DFLASH_SSD_MAX_ENTRIES") or "4").strip())
            )
        except Exception:
            self._ssd_max_entries = 4
        try:
            self._ssd_fanout = max(
                1, int((os.environ.get("SGLANG_DFLASH_SSD_FANOUT") or "1").strip())
            )
        except Exception:
            self._ssd_fanout = 1
        try:
            self._ssd_branch_search_topk = max(
                int(self._ssd_fanout),
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_SSD_BRANCH_SEARCH_TOPK")
                        or str(max(8, min(64, int(self._ssd_fanout) * 4)))
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_branch_search_topk = max(
                int(self._ssd_fanout), max(8, min(64, int(self._ssd_fanout) * 4))
            )
        self._ssd_branch_mode = (
            (os.environ.get("SGLANG_DFLASH_SSD_BRANCH_MODE") or "topk")
            .strip()
            .lower()
        )
        if self._ssd_branch_mode not in {"topk", "sample"}:
            self._ssd_branch_mode = "topk"
        try:
            self._ssd_sampler_x = float(
                (os.environ.get("SGLANG_DFLASH_SSD_SAMPLER_X") or "1.0").strip()
            )
        except Exception:
            self._ssd_sampler_x = 1.0
        if not math.isfinite(self._ssd_sampler_x) or self._ssd_sampler_x <= 0:
            self._ssd_sampler_x = 1.0
        try:
            self._ssd_fanout_target_alt_mass = float(
                (os.environ.get("SGLANG_DFLASH_SSD_FANOUT_TARGET_ALT_MASS") or "0.0").strip()
            )
        except Exception:
            self._ssd_fanout_target_alt_mass = 0.0
        if (
            not math.isfinite(self._ssd_fanout_target_alt_mass)
            or self._ssd_fanout_target_alt_mass <= 0.0
        ):
            self._ssd_fanout_target_alt_mass = 0.0
        self._ssd_fanout_target_alt_mass = min(1.0, self._ssd_fanout_target_alt_mass)
        try:
            self._ssd_fanout_min_alt_prob = float(
                (os.environ.get("SGLANG_DFLASH_SSD_FANOUT_MIN_ALT_PROB") or "0.0").strip()
            )
        except Exception:
            self._ssd_fanout_min_alt_prob = 0.0
        if (
            not math.isfinite(self._ssd_fanout_min_alt_prob)
            or self._ssd_fanout_min_alt_prob < 0.0
        ):
            self._ssd_fanout_min_alt_prob = 0.0
        try:
            _fanout_max_alt_entropy = (
                os.environ.get("SGLANG_DFLASH_SSD_FANOUT_MAX_ALT_ENTROPY") or ""
            ).strip()
            self._ssd_fanout_max_alt_entropy = (
                float(_fanout_max_alt_entropy)
                if _fanout_max_alt_entropy
                else math.inf
            )
        except Exception:
            self._ssd_fanout_max_alt_entropy = math.inf
        if (
            not math.isfinite(self._ssd_fanout_max_alt_entropy)
            or self._ssd_fanout_max_alt_entropy <= 0.0
        ):
            self._ssd_fanout_max_alt_entropy = math.inf
        try:
            self._ssd_fanout_skip_if_actual_prob_ge = float(
                (
                    os.environ.get("SGLANG_DFLASH_SSD_FANOUT_SKIP_IF_ACTUAL_PROB_GE")
                    or "1.0"
                ).strip()
            )
        except Exception:
            self._ssd_fanout_skip_if_actual_prob_ge = 1.0
        if not math.isfinite(self._ssd_fanout_skip_if_actual_prob_ge):
            self._ssd_fanout_skip_if_actual_prob_ge = 1.0
        self._ssd_fanout_skip_if_actual_prob_ge = min(
            1.0, max(0.0, self._ssd_fanout_skip_if_actual_prob_ge)
        )
        self._ssd_max_entries = max(
            int(self._ssd_max_entries), int(self._ssd_fanout)
        )
        self._ssd_async_overlap = (os.environ.get("SGLANG_DFLASH_SSD_ASYNC_OVERLAP") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        try:
            self._ssd_hit_wait_ms = max(
                0,
                int((os.environ.get("SGLANG_DFLASH_SSD_HIT_WAIT_MS") or "10").strip()),
            )
        except Exception:
            self._ssd_hit_wait_ms = 10
        self._draft_worker_lock = threading.Lock()
        self._ssd_overlap_executor: ThreadPoolExecutor | None = None
        self._ssd_overlap_future: Future | None = None
        self._ssd_overlap_lock = threading.Lock()
        self._ssd_overlap_launch_ct = 0
        self._ssd_overlap_wait_ct = 0
        try:
            self._ssd_fanout_miss_streak_trigger = max(
                0,
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_SSD_FANOUT_MISS_STREAK_TRIGGER")
                        or "0"
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_fanout_miss_streak_trigger = 0
        try:
            self._ssd_fanout_difficulty_accept_ema_le = float(
                (
                    os.environ.get("SGLANG_DFLASH_SSD_FANOUT_DIFFICULTY_ACCEPT_EMA_LE")
                    or "-1.0"
                ).strip()
            )
        except Exception:
            self._ssd_fanout_difficulty_accept_ema_le = -1.0
        try:
            self._ssd_fanout_difficulty_accept_last_le = float(
                (
                    os.environ.get("SGLANG_DFLASH_SSD_FANOUT_DIFFICULTY_ACCEPT_LAST_LE")
                    or "-1.0"
                ).strip()
            )
        except Exception:
            self._ssd_fanout_difficulty_accept_last_le = -1.0
        try:
            self._ssd_fanout_difficulty_min_verify_ct = max(
                0,
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_SSD_FANOUT_DIFFICULTY_MIN_VERIFY_CT")
                        or "0"
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_fanout_difficulty_min_verify_ct = 0
        try:
            self._ssd_fanout_long_horizon_tokens = max(
                0,
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_SSD_FANOUT_LONG_HORIZON_TOKENS")
                        or "0"
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_fanout_long_horizon_tokens = 0
        try:
            self._ssd_fanout_miss_streak_trigger_long = max(
                0,
                int(
                    (
                        os.environ.get(
                            "SGLANG_DFLASH_SSD_FANOUT_MISS_STREAK_TRIGGER_LONG"
                        )
                        or str(int(self._ssd_fanout_miss_streak_trigger))
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_fanout_miss_streak_trigger_long = int(
                self._ssd_fanout_miss_streak_trigger
            )
        try:
            self._dflash_enable_after_decoded_tokens = max(
                0,
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_ENABLE_AFTER_DECODED_TOKENS")
                        or "0"
                    ).strip()
                ),
            )
        except Exception:
            self._dflash_enable_after_decoded_tokens = 0
        try:
            self._ssd_enable_after_decoded_tokens = max(
                0,
                int(
                    (
                        os.environ.get("SGLANG_DFLASH_SSD_ENABLE_AFTER_DECODED_TOKENS")
                        or str(int(self._dflash_enable_after_decoded_tokens))
                    ).strip()
                ),
            )
        except Exception:
            self._ssd_enable_after_decoded_tokens = int(
                self._dflash_enable_after_decoded_tokens
            )
        self._ssd_shadow_req_to_token_pool: ReqToTokenPool | None = None
        self._ssd_shadow_req_pool_indices: torch.Tensor | None = None
        self._ssd_shadow_block_cache_loc: torch.Tensor | None = None
        self._adaptive_pq = DFlashAdaptivePqController(
            DFlashAdaptivePqConfig.from_env(
                default_temp_mul=float(
                    getattr(server_args, "speculative_dflash_pq_draft_temp_mul", 1.0)
                    or 1.0
                )
            )
        )

        # Draft runner (separate KV cache + attention backend).
        #
        # Historically we shared req_to_token_pool + token_to_kv_pool_allocator with the target
        # worker (EAGLE3-style) while keeping a separate draft KV pool. That requires the draft
        # KV pool to have the *same* capacity as the shared allocator (otherwise cache_loc indices
        # can exceed the draft pool and crash in KV-store with CUDA illegal memory access).
        #
        # For single-GPU bring-up and debugging we often want a *smaller* draft KV pool to fit
        # alongside the target worker. Allow opting out of pool-sharing so the draft worker can
        # use its own allocator + req_to_token_pool safely.
        share_pools_env = (
            os.environ.get("SGLANG_DFLASH_DRAFT_SHARE_POOLS") or "1"
        ).strip().lower()
        share_pools = share_pools_env not in {"0", "false", "no", "off"}
        # Whether the draft worker shares req_to_token_pool + KV allocator semantics with the target.
        # When False, draft KV locations must be allocated and tracked independently.
        self._dflash_draft_share_pools = bool(share_pools)
        shared_req_to_token_pool = None
        shared_token_to_kv_pool_allocator = None
        if share_pools:
            shared_req_to_token_pool, shared_token_to_kv_pool_allocator = (
                target_worker.get_memory_pool()
            )
        elif self.tp_rank == 0:
            logger.warning(
                "DFLASH draft runner pool sharing disabled (SGLANG_DFLASH_DRAFT_SHARE_POOLS=%r). "
                "Draft worker will allocate its own req_to_token_pool + KV allocator.",
                share_pools_env,
            )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True

        # Allow the draft worker to use a different page_size than the target worker.
        # This is particularly useful when the target runs paged KV (page_size>1) but the
        # DFlash draft was trained/validated mostly in the non-paged regime (page_size=1).
        target_page_size = int(getattr(server_args, "page_size", 1) or 1)
        draft_page_size_source = "inherit_target"
        cli_draft_page_size = getattr(server_args, "speculative_draft_page_size", None)
        draft_page_size_env = (
            os.environ.get("SGLANG_DFLASH_DRAFT_PAGE_SIZE") or ""
        ).strip()
        if cli_draft_page_size is not None:
            draft_page_size = int(cli_draft_page_size)
            draft_page_size_source = "cli"
            draft_server_args.page_size = draft_page_size
            if int(draft_page_size) != target_page_size:
                # Pool sharing requires allocator semantics (including page_size) to match.
                share_pools = False
                self._dflash_draft_share_pools = False
                shared_req_to_token_pool = None
                shared_token_to_kv_pool_allocator = None
                if self.tp_rank == 0:
                    logger.warning(
                        "DFLASH draft page_size overridden via CLI to %s (target page_size=%s); disabling pool sharing.",
                        int(draft_page_size),
                        int(target_page_size),
                    )
        elif draft_page_size_env:
            try:
                draft_page_size = int(draft_page_size_env)
                if draft_page_size < 1:
                    raise ValueError("page_size must be >= 1")
                draft_page_size_source = "env"
                draft_server_args.page_size = draft_page_size
                if int(draft_page_size) != target_page_size:
                    # Pool sharing requires allocator semantics (including page_size) to match.
                    share_pools = False
                    self._dflash_draft_share_pools = False
                    shared_req_to_token_pool = None
                    shared_token_to_kv_pool_allocator = None
                    if self.tp_rank == 0:
                        logger.warning(
                            "DFLASH draft page_size overridden to %s (target page_size=%s); disabling pool sharing.",
                            int(draft_page_size),
                            int(target_page_size),
                        )
            except Exception:
                if self.tp_rank == 0:
                    logger.warning(
                        "Ignoring invalid SGLANG_DFLASH_DRAFT_PAGE_SIZE=%r",
                        draft_page_size_env,
                    )
        elif target_page_size > 1:
            # By default, the draft worker inherits target `page_size`. For GPT-OSS DFlash,
            # we have observed lower acceptance when the *draft* runs paged-KV compared to
            # non-paged draft KV (page_size=1) while keeping the target paged-KV.
            #
            # Keep the default behavior for backward compatibility, but allow an opt-in auto
            # override to force draft page_size=1 when the target is paged.
            auto_env = (os.environ.get("SGLANG_DFLASH_DRAFT_PAGE_SIZE_AUTO") or "").strip().lower()
            auto_enable = auto_env not in ("", "0", "false", "off", "no")
            if auto_enable:
                draft_page_size_source = "env_auto"
                draft_server_args.page_size = 1
                # Pool sharing requires allocator semantics (including page_size) to match.
                share_pools = False
                self._dflash_draft_share_pools = False
                shared_req_to_token_pool = None
                shared_token_to_kv_pool_allocator = None
                if self.tp_rank == 0:
                    logger.warning(
                        "DFLASH auto draft page_size override enabled: draft page_size=1 (target page_size=%s); disabling pool sharing.",
                        int(target_page_size),
                    )
            elif self.tp_rank == 0:
                logger.warning(
                    "DFLASH target runs paged-KV (page_size=%s) with draft page_size unset. "
                    "If acceptance is low, try setting SGLANG_DFLASH_DRAFT_PAGE_SIZE=1 (draft non-paged) "
                    "or SGLANG_DFLASH_DRAFT_PAGE_SIZE_AUTO=1.",
                    int(target_page_size),
                )
        if self.tp_rank == 0:
            logger.info(
                "DFLASH draft page config: target_page_size=%s draft_page_size=%s share_pools=%s source=%s",
                int(target_page_size),
                int(getattr(draft_server_args, "page_size", target_page_size)),
                bool(self._dflash_draft_share_pools),
                draft_page_size_source,
            )
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' yet; "
                "falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"

        elif draft_backend in ("flex_flash4",):
            # FlexFlash4 is a Hopper-first decode backend; draft proposal quality depends on
            # matching the target distribution as closely as possible. In practice, using
            # FlexFlash4 for the *draft worker* has shown severe acceptance collapse for GPT-OSS
            # DFlash (even under deterministic sampling). Keep a safe default and require an
            # explicit opt-in to use FlexFlash4 on the draft side.
            allow_flex_draft = (os.environ.get("SGLANG_DFLASH_ALLOW_FLEX_DRAFT") or "").strip().lower() not in (
                "",
                "0",
                "false",
                "off",
                "no",
            )
            if not allow_flex_draft:
                logger.warning(
                    "DFLASH draft attention_backend=%r is disabled by default (acceptance collapse observed). "
                    "Falling back to 'fa3'. Set SGLANG_DFLASH_ALLOW_FLEX_DRAFT=1 to force-enable.",
                    draft_backend,
                )
                draft_backend = "fa3"
        elif draft_backend not in (
            "flashinfer",
            "fa3",
            "fa4",
            "flex_attention",
            "flex_attention2",
            "flex_flash",
            "flex_flash2",
            "flex_flash2_delegate_fa3",
            "flex_flash4",
        ):
            logger.warning(
                "DFLASH draft worker only supports attention_backend in {'flashinfer', 'fa3', 'fa4', 'flex_attention', 'flex_attention2', 'flex_flash', 'flex_flash2', 'flex_flash2_delegate_fa3', 'flex_flash4'} for now, "
                "but got %r. Falling back to 'flashinfer'.",
                supported_draft_backends,
                draft_backend,
            )
            draft_backend = "flashinfer"

        self._draft_attention_backend = draft_backend

        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )

        # The default speculative plumbing sets `draft_runner_cache_size` to the target worker's
        # KV-cache capacity. On single-GPU bring-up (especially with high --mem-fraction-static),
        # spawning a second worker with a target-sized KV cache can OOM. The draft model proposes
        # fixed-size blocks; allow an explicit cap (in tokens) to keep the draft KV cache small.
        draft_cache_cap_env = (
            os.environ.get("SGLANG_DFLASH_DRAFT_RUNNER_CACHE_SIZE") or ""
        ).strip()
        if draft_cache_cap_env:
            try:
                draft_server_args.draft_runner_cache_size = max(1, int(draft_cache_cap_env))
                if self.tp_rank == 0:
                    logger.warning(
                        "DFLASH draft runner cache size overridden: draft_runner_cache_size=%s",
                        int(draft_server_args.draft_runner_cache_size),
                    )
            except Exception:
                if self.tp_rank == 0:
                    logger.warning(
                        "Ignoring invalid SGLANG_DFLASH_DRAFT_RUNNER_CACHE_SIZE=%r",
                        draft_cache_cap_env,
                    )
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=shared_token_to_kv_pool_allocator,
        )
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        if hasattr(self.draft_model, "set_embed"):
            self.draft_model.set_embed(
                self.target_worker.model_runner.model.get_input_embeddings()
            )
            # Keep linear DFLASH on the original explicit-input-embed path.
            # This matches the stable production behavior and avoids relying on
            # the newer self-embedding draft path while we validate regressions.
            self.draft_model.requires_input_embeds = True
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_seq_lens_plus_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._ssd_overlap_block_ids_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_block_positions_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_block_tokens_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_block_end_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_seq_lens_cpu_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_seq_lens_plus_cpu_buf: Optional[torch.Tensor] = None
        self._ssd_overlap_stream = None
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0


        self._draft_block_cache_loc_scratch: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_cache_loc_scratch_cap: int = 0

        # The fused KV materialization helper uses Triton kernels. This is independent
        # of the attention backend and is a major throughput lever for DFlash (it
        # avoids per-layer Python loops when appending target hidden states into the
        # draft KV cache). Disable via SGLANG_DFLASH_DISABLE_FUSED_KV_MATERIALIZE=1.
        self._use_fused_kv_materialize = is_cuda()
        if (os.environ.get("SGLANG_DFLASH_DISABLE_FUSED_KV_MATERIALIZE") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        ):
            self._use_fused_kv_materialize = False
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()
        elif is_cuda() and self.tp_rank == 0:
            logger.info(
                "DFLASH: fused_kv_materialize disabled (draft attention_backend=%s).",
                self._draft_attention_backend,
            )
        if self._ssd_enabled and self.tp_rank == 0:
            logger.info(
                "DFLASH SSD enabled. prepare_next=%s max_entries=%s fanout=%s branch_search_topk=%s branch_mode=%s sampler_x=%s async_overlap=%s.",
                bool(self._ssd_prepare_next),
                int(self._ssd_max_entries),
                int(self._ssd_fanout),
                int(self._ssd_branch_search_topk),
                str(self._ssd_branch_mode),
                float(self._ssd_sampler_x),
                bool(self._ssd_async_overlap),
            )
            if (
                float(self._ssd_fanout_min_alt_prob) > 0.0
                or float(self._ssd_fanout_skip_if_actual_prob_ge) < 1.0
                or math.isfinite(float(self._ssd_fanout_max_alt_entropy))
            ):
                logger.info(
                    "DFLASH SSD fanout gate: min_alt_prob=%s max_alt_entropy=%s skip_if_actual_prob_ge=%s",
                    float(self._ssd_fanout_min_alt_prob),
                    (
                        float(self._ssd_fanout_max_alt_entropy)
                        if math.isfinite(float(self._ssd_fanout_max_alt_entropy))
                        else "inf"
                    ),
                    float(self._ssd_fanout_skip_if_actual_prob_ge),
                )
            if float(self._ssd_fanout_target_alt_mass) > 0.0:
                logger.info(
                    "DFLASH SSD adaptive fanout target_alt_mass=%s",
                    float(self._ssd_fanout_target_alt_mass),
                )
            if int(self._ssd_fanout_miss_streak_trigger) > 0:
                logger.info(
                    "DFLASH SSD miss-streak-triggered fanout=%s",
                    int(self._ssd_fanout_miss_streak_trigger),
                )
            if (
                int(self._ssd_fanout_long_horizon_tokens) > 0
                and int(self._ssd_fanout_miss_streak_trigger_long)
                != int(self._ssd_fanout_miss_streak_trigger)
            ):
                logger.info(
                    "DFLASH SSD long-horizon trigger: tokens>=%s -> miss_streak_trigger=%s",
                    int(self._ssd_fanout_long_horizon_tokens),
                    int(self._ssd_fanout_miss_streak_trigger_long),
                )
            if (
                float(self._ssd_fanout_difficulty_accept_ema_le) >= 0.0
                or float(self._ssd_fanout_difficulty_accept_last_le) >= 0.0
                or int(self._ssd_fanout_difficulty_min_verify_ct) > 0
            ):
                logger.info(
                    "DFLASH SSD difficulty gate: accept_ema_le=%s accept_last_le=%s min_verify_ct=%s",
                    float(self._ssd_fanout_difficulty_accept_ema_le),
                    float(self._ssd_fanout_difficulty_accept_last_le),
                    int(self._ssd_fanout_difficulty_min_verify_ct),
                )
            if int(self._dflash_enable_after_decoded_tokens) > 0:
                logger.info(
                    "DFLASH late-enable gate: full_dflash_after_decoded_tokens=%s ssd_after_decoded_tokens=%s",
                    int(self._dflash_enable_after_decoded_tokens),
                    int(self._ssd_enable_after_decoded_tokens),
                )
        if self._ssd_enabled and self._ssd_async_overlap:
            shadow_bs = max(1, int(self._ssd_fanout))
            self._ssd_overlap_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dflash_ssd_overlap"
            )
            if is_cuda():
                self._ssd_overlap_stream = torch.cuda.Stream()
            self._ssd_shadow_req_to_token_pool = ReqToTokenPool(
                size=int(shadow_bs),
                max_context_len=int(target_worker.model_runner.model_config.context_len),
                device=str(self.device),
                enable_memory_saver=False,
            )
            self._ssd_shadow_req_pool_indices = torch.arange(
                int(shadow_bs), dtype=torch.int32, device=self.device
            )
            shadow_need_tokens = int(shadow_bs) * int(self.block_size)
            if int(getattr(shared_token_to_kv_pool_allocator, "page_size", 1)) == 1:
                scratch_loc = shared_token_to_kv_pool_allocator.alloc(int(shadow_need_tokens))
            else:
                fake_prefix = torch.zeros((shadow_bs,), dtype=torch.int32, device=self.device)
                fake_prefix_cpu = torch.zeros((shadow_bs,), dtype=torch.int32, device="cpu")
                fake_seq = torch.full(
                    (shadow_bs,), int(self.block_size), dtype=torch.int32, device=self.device
                )
                fake_seq_cpu = torch.full(
                    (shadow_bs,), int(self.block_size), dtype=torch.int32, device="cpu"
                )
                fake_last_loc = torch.full(
                    (shadow_bs,), -1, dtype=torch.int64, device=self.device
                )
                scratch_loc = shared_token_to_kv_pool_allocator.alloc_extend(
                    fake_prefix,
                    fake_prefix_cpu,
                    fake_seq,
                    fake_seq_cpu,
                    fake_last_loc,
                    int(shadow_need_tokens),
                )
            if scratch_loc is None:
                raise RuntimeError(
                    f"DFLASH SSD async overlap could not reserve {int(shadow_need_tokens)} scratch KV slots."
                )
            self._ssd_shadow_block_cache_loc = scratch_loc.to(torch.int64)
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH SSD async shadow scratch reserved. block_size=%d slots=%d",
                    int(self.block_size),
                    int(self._ssd_shadow_block_cache_loc.numel()),
                )

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = None
                v_scale = None
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                target_fc_weight=self.draft_model.fc.weight,
                target_hidden_norm_weight=self.draft_model.hidden_norm.weight,
                target_hidden_norm_eps=self.draft_model.hidden_norm.variance_epsilon,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )
        self._draft_seq_lens_plus_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def _ensure_ssd_overlap_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._ssd_overlap_block_ids_buf is None
            else int(self._ssd_overlap_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._ssd_overlap_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._ssd_overlap_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._ssd_overlap_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._ssd_overlap_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._ssd_overlap_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )
        self._ssd_overlap_seq_lens_plus_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def _get_draft_block_buffers(
        self, *, bs: int, use_ssd_overlap_path: bool
    ) -> _DFlashDraftBlockBuffers:
        if use_ssd_overlap_path:
            self._ensure_ssd_overlap_block_buffers(bs)
            assert self._ssd_overlap_block_ids_buf is not None
            assert self._ssd_overlap_block_positions_buf is not None
            assert self._ssd_overlap_block_tokens_buf is not None
            assert self._ssd_overlap_block_end_buf is not None
            assert self._ssd_overlap_seq_lens_cpu_buf is not None
            assert self._ssd_overlap_seq_lens_plus_cpu_buf is not None
            return _DFlashDraftBlockBuffers(
                block_ids=self._ssd_overlap_block_ids_buf[:bs],
                positions=self._ssd_overlap_block_positions_buf[:bs],
                block_tokens=self._ssd_overlap_block_tokens_buf[:bs],
                block_end=self._ssd_overlap_block_end_buf[:bs],
                seq_lens_cpu=self._ssd_overlap_seq_lens_cpu_buf[:bs],
                seq_lens_plus_cpu=self._ssd_overlap_seq_lens_plus_cpu_buf[:bs],
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None
        assert self._draft_seq_lens_plus_cpu_buf is not None
        return _DFlashDraftBlockBuffers(
            block_ids=self._draft_block_ids_buf[:bs],
            positions=self._draft_block_positions_buf[:bs],
            block_tokens=self._draft_block_tokens_buf[:bs],
            block_end=self._draft_block_end_buf[:bs],
            seq_lens_cpu=self._draft_seq_lens_cpu_buf[:bs],
            seq_lens_plus_cpu=self._draft_seq_lens_plus_cpu_buf[:bs],
        )

    def _ensure_draft_block_cache_loc_scratch(self, bs: int) -> None:
        """Reserve reusable KV slots for the draft block forward.

        The draft block KV is transient (only needed within the draft forward). Reusing a fixed
        scratch region avoids per-verify allocator state backup/restore and keeps paged alloc
        overhead out of the hot path.
        """

        bs = int(bs)
        if self._draft_block_cache_loc_scratch is not None and int(
            self._draft_block_cache_loc_scratch_cap
        ) >= bs:
            return

        block_size = int(self.block_size)
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        page_size = int(getattr(allocator, "page_size", 1))

        # Let the scheduler's idle-time memory checker know about our persistent scratch reservation.
        # (Used by SchedulerRuntimeCheckerMixin._get_dflash_reserved_tokens()).
        reserved_env_key = "SGLANG_DFLASH_SSD_RESERVED_TOKENS"

        # If we are growing the scratch region, release the previous reservation first to avoid
        # leaking pages/tokens across long-running servers.
        if self._draft_block_cache_loc_scratch is not None:
            try:
                allocator.free(self._draft_block_cache_loc_scratch.reshape(-1))
            finally:
                self._draft_block_cache_loc_scratch = None
                self._draft_block_cache_loc_scratch_cap = 0

        if page_size == 1:
            need_tokens = bs * block_size
            loc = allocator.alloc(int(need_tokens))
            if loc is None or int(loc.numel()) < int(need_tokens):
                raise RuntimeError(
                    "DFLASH draft scratch allocation failed: "
                    f"need_tokens={int(need_tokens)} page_size={page_size}"
                )
            scratch = loc[: int(need_tokens)].to(torch.int64).view(bs, block_size)
            try:
                prev = int((os.environ.get(reserved_env_key) or "0").strip() or 0)
            except Exception:
                prev = 0
            os.environ[reserved_env_key] = str(max(prev, int(need_tokens)))
        else:
            # IMPORTANT: For paged KV, each request's token indices must be page-aligned.
            # If we pack multiple requests into one "fake" sequence, per-request token positions
            # will no longer align with KV page offsets (e.g., position%page_size != loc%page_size),
            # which can corrupt attention reads/writes and collapse DFlash acceptance.
            #
            # We need a page-aligned scratch window *for any* prefix_len % page_size:
            # the block we write spans positions [prefix_len, prefix_len+block_size), so for paged
            # KV we must ensure `loc % page_size == position % page_size` for each token.
            #
            # Reserve TWO pages per request so we can take a length-`block_size` slice starting at
            # offset `prefix_len % page_size` even when it crosses a page boundary.
            scratch_width = 2 * page_size
            need_tokens_alloc = bs * scratch_width
            fake_prefix = torch.zeros((1,), dtype=torch.int32, device=self.device)
            fake_prefix_cpu = torch.zeros((1,), dtype=torch.int32, device="cpu")
            fake_seq = torch.tensor([int(need_tokens_alloc)], dtype=torch.int32, device=self.device)
            fake_seq_cpu = torch.tensor([int(need_tokens_alloc)], dtype=torch.int32, device="cpu")
            fake_last_loc = torch.full((1,), -1, dtype=torch.int64, device=self.device)
            loc = allocator.alloc_extend(
                fake_prefix,
                fake_prefix_cpu,
                fake_seq,
                fake_seq_cpu,
                fake_last_loc,
                int(need_tokens_alloc),
            )
            if loc is None or int(loc.numel()) < int(need_tokens_alloc):
                raise RuntimeError(
                    "DFLASH draft scratch allocation failed: "
                    f"need_tokens_alloc={int(need_tokens_alloc)} page_size={page_size}"
                )

            scratch = loc[: int(need_tokens_alloc)].to(torch.int64).view(bs, scratch_width)

            try:
                prev = int((os.environ.get(reserved_env_key) or "0").strip() or 0)
            except Exception:
                prev = 0
            os.environ[reserved_env_key] = str(max(prev, int(need_tokens_alloc)))

            if os.environ.get("SGLANG_DFLASH_DEBUG_SCRATCH", "").strip():
                # Ensure each request starts at a page boundary, and pages are not shared.
                assert int(scratch.shape[0]) == bs and int(scratch.shape[1]) == scratch_width
                assert torch.all((scratch[:, 0] % page_size) == 0), (
                    "DFLASH scratch must be page-aligned per request when page_size>1. "
                    f"page_size={page_size}"
                )
                page_ids = (scratch.reshape(-1) // page_size).to(torch.int64)
                # Every request owns 2 unique pages.
                assert int(torch.unique(page_ids).numel()) == int(bs * 2), (
                    "DFLASH scratch pages are unexpectedly shared across requests. "
                    f"page_size={page_size} bs={bs}"
                )

        self._draft_block_cache_loc_scratch = scratch
        self._draft_block_cache_loc_scratch_cap = bs

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # allocator and req_to_token_pool are shared with target worker
        pass


    def on_req_finished(self, req):
        # allocator and req_to_token_pool are shared with the target worker;
        # there is no separate draft allocation to release here.
        if hasattr(req, "dflash_draft_seq_len"):
            req.dflash_draft_seq_len = 0
        if hasattr(req, "dflash_ssd_cache"):
            req.dflash_ssd_cache = {}
        if hasattr(req, "dflash_ssd_cached_proposal"):
            req.dflash_ssd_cached_proposal = None

    def _maybe_finalize_ssd_overlap(self, *, block: bool) -> None:
        fut = self._ssd_overlap_future
        if fut is None:
            return
        if not fut.done():
            if not block:
                return
            self._ssd_overlap_wait_ct += 1
        try:
            fut.result(timeout=None if block else 0)
        except Exception as e:
            self._ssd_prepare_failures += 1
            if self.tp_rank == 0 and not getattr(self, "_logged_ssd_async_failure", False):
                logger.warning(
                    "DFLASH SSD async overlap task failed; falling back to sequential path: %s",
                    e,
                )
                setattr(self, "_logged_ssd_async_failure", True)
        finally:
            if fut.done():
                self._ssd_overlap_future = None

    def _get_req_ssd_cache(
        self, req, *, create: bool = False
    ) -> dict[tuple[int, int], _DFlashSsdCachedProposal] | None:
        cache = getattr(req, "dflash_ssd_cache", None)
        if cache is None:
            if not create:
                return None
            cache = {}
            setattr(req, "dflash_ssd_cache", cache)
            return cache
        if not isinstance(cache, dict):
            if not create:
                return None
            cache = {}
            setattr(req, "dflash_ssd_cache", cache)
        return cache

    def _get_req_ssd_cache_pending_count(self, req) -> int:
        cache = self._get_req_ssd_cache(req, create=False)
        if cache:
            return int(len(cache))
        return 1 if getattr(req, "dflash_ssd_cached_proposal", None) is not None else 0

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _maybe_consume_ssd_cached_proposal(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ) -> _DFlashDraftProposal | None:
        if not self._ssd_enabled:
            return None
        if any(not self._req_ssd_enabled_for_phase(req) for req in batch.reqs):
            return None


        entries: list[_DFlashSsdCachedProposal | None] = []
        seq_lens_cpu = batch.seq_lens_cpu.tolist()
        verified_ids_cpu = draft_input.verified_id.detach().to("cpu").to(torch.int64).tolist()

        for req, seq_len, verified_id in zip(
            batch.reqs, seq_lens_cpu, verified_ids_cpu, strict=True
        ):
            cache_key = (int(seq_len), int(verified_id))
            entry = None
            cache = self._get_req_ssd_cache(req, create=False)
            if cache is not None:
                entry = cache.pop(cache_key, None)
            if entry is None:
                legacy_entry = getattr(req, "dflash_ssd_cached_proposal", None)
                if (
                    isinstance(legacy_entry, _DFlashSsdCachedProposal)
                    and int(legacy_entry.expected_seq_len) == int(seq_len)
                    and int(legacy_entry.expected_verified_id) == int(verified_id)
                ):
                    entry = legacy_entry
            if entry is None or not isinstance(entry, _DFlashSsdCachedProposal):
                req.spec_ssd_miss_streak = int(getattr(req, "spec_ssd_miss_streak", 0)) + 1
                entries.append(None)
                continue
            entries.append(entry)

        if not entries or not any(e is not None for e in entries):
            self._ssd_batch_misses += 1
            self._ssd_req_misses += int(batch.batch_size())
            return None

        ref_entry = next(e for e in entries if e is not None)
        draft_topk = int(ref_entry.draft_topk)
        verify_mode = str(ref_entry.verify_mode)
        if any(
            e is not None
            and (int(e.draft_topk) != draft_topk or str(e.verify_mode) != verify_mode)
            for e in entries
        ):
            for req in batch.reqs:
                setattr(req, "dflash_ssd_cached_proposal", None)
            self._ssd_batch_misses += 1
            self._ssd_req_misses += int(batch.batch_size())
            return None

        bs = int(batch.batch_size())
        single_proposals: dict[int, _DFlashDraftProposal] = {}
        proposal_token_num = max(
            1, int(getattr(ref_entry, "draft_token_num", int(self.block_size)))
        )
        for i, req in enumerate(batch.reqs):
            if entries[i] is not None:
                continue
            self._ssd_req_misses += 1
            single_batch = self._build_single_req_batch_view(
                req=req,
                req_pool_index=batch.req_pool_indices[i : i + 1],
            )
            single_input = DFlashDraftInput(
                verified_id=draft_input.verified_id[i : i + 1],
                target_hidden=torch.empty((0,), dtype=torch.float32, device=self.device),
                ctx_lens=torch.zeros((1,), dtype=torch.int32, device=self.device),
                draft_seq_lens=torch.zeros((1,), dtype=torch.int32, device=self.device),
            )
            single_proposal = self._build_draft_proposal(
                single_batch,
                single_input,
                prefix_lens=batch.seq_lens[i : i + 1],
                seq_lens_cpu=batch.seq_lens_cpu[i : i + 1],
            )
            if (
                str(single_proposal.verify_mode) != verify_mode
                or int(single_proposal.draft_topk) != draft_topk
            ):
                self._ssd_batch_misses += 1
                self._ssd_req_misses += int(batch.batch_size())
                return None
            single_proposals[i] = single_proposal
            proposal_token_num = max(
                proposal_token_num, int(single_proposal.draft_token_num)
            )

        draft_tokens = torch.full(
            (bs, int(proposal_token_num)),
            fill_value=int(self._mask_token_id),
            dtype=torch.long,
            device=self.device,
        )

        draft_topk_ids = None
        draft_topk_probs = None
        if draft_topk > 0:
            if ref_entry.draft_topk_ids is None or ref_entry.draft_topk_probs is None:
                # Unexpected: cached proposal claims topk but lacks tensors.
                self._ssd_batch_misses += 1
                self._ssd_req_misses += int(batch.batch_size())
                return None
            ids_dtype = ref_entry.draft_topk_ids.dtype
            probs_dtype = ref_entry.draft_topk_probs.dtype
            draft_topk_ids = torch.empty(
                (bs, max(0, int(proposal_token_num) - 1), int(draft_topk)),
                dtype=ids_dtype,
                device=self.device,
            )
            draft_topk_probs = torch.empty(
                (bs, max(0, int(proposal_token_num) - 1), int(draft_topk)),
                dtype=probs_dtype,
                device=self.device,
            )

        max_steps_per_req = None
        if any((e is not None and e.max_steps is not None) for e in entries) or bool(
            single_proposals
        ) or int(proposal_token_num) != int(self.block_size):
            max_steps_per_req = torch.full(
                (bs,),
                max(0, int(proposal_token_num) - 1),
                dtype=torch.int32,
                device=self.device,
            )

        # Partial-SSD support: allow hits on a subset of requests in the batch.
        # For misses, fall back to per-request draft proposal and then merge.
        for i, req in enumerate(batch.reqs):
            entry = entries[i]
            if entry is not None:
                row_token_num = int(
                    getattr(entry, "draft_token_num", entry.draft_tokens.shape[0])
                )
                draft_tokens[i, :row_token_num].copy_(entry.draft_tokens[:row_token_num])
                if draft_topk_ids is not None and entry.draft_topk_ids is not None:
                    row_steps = max(0, row_token_num - 1)
                    draft_topk_ids[i, :row_steps].copy_(entry.draft_topk_ids[:row_steps])
                if draft_topk_probs is not None and entry.draft_topk_probs is not None:
                    row_steps = max(0, row_token_num - 1)
                    draft_topk_probs[i, :row_steps].copy_(
                        entry.draft_topk_probs[:row_steps]
                    )
                if max_steps_per_req is not None:
                    row_max_steps = (
                        int(entry.max_steps)
                        if entry.max_steps is not None
                        else max(0, row_token_num - 1)
                    )
                    max_steps_per_req[i] = int(row_max_steps)
                req.spec_ssd_hit_ct = int(getattr(req, "spec_ssd_hit_ct", 0)) + 1
                req.spec_ssd_miss_streak = 0
                setattr(req, "dflash_ssd_cached_proposal", None)
                continue

            single_proposal = single_proposals[i]
            row_token_num = int(single_proposal.draft_token_num)
            draft_tokens[i, :row_token_num].copy_(
                single_proposal.draft_tokens[0, :row_token_num]
            )
            if draft_topk_ids is not None:
                if single_proposal.draft_topk_ids is None:
                    self._ssd_batch_misses += 1
                    self._ssd_req_misses += int(batch.batch_size())
                    return None
                draft_topk_ids[i, : max(0, row_token_num - 1)].copy_(
                    single_proposal.draft_topk_ids[0, : max(0, row_token_num - 1)]
                )
            if draft_topk_probs is not None:
                if single_proposal.draft_topk_probs is None:
                    self._ssd_batch_misses += 1
                    self._ssd_req_misses += int(batch.batch_size())
                    return None
                draft_topk_probs[i, : max(0, row_token_num - 1)].copy_(
                    single_proposal.draft_topk_probs[0, : max(0, row_token_num - 1)]
                )
            if max_steps_per_req is not None:
                if single_proposal.max_steps_per_req is not None:
                    max_steps_per_req[i] = int(single_proposal.max_steps_per_req[0].item())
                else:
                    max_steps_per_req[i] = max(0, row_token_num - 1)

        hit_ct = int(sum(1 for e in entries if e is not None))
        if hit_ct >= int(batch.batch_size()):
            self._ssd_batch_hits += 1
            self._ssd_req_hits += int(batch.batch_size())
        else:
            self._ssd_batch_misses += 1
            self._ssd_req_hits += int(hit_ct)
        if self.tp_rank == 0 and not getattr(self, "_logged_ssd_hit", False):
            logger.info(
                "DFLASH SSD cache hit. bs=%d hit_ct=%d verify_mode=%s draft_topk=%d",
                int(batch.batch_size()),
                int(hit_ct),
                verify_mode,
                draft_topk,
            )
            setattr(self, "_logged_ssd_hit", True)

        return _DFlashDraftProposal(
            draft_tokens=draft_tokens,
            draft_token_num=int(proposal_token_num),
            verify_mode=verify_mode,
            draft_topk=draft_topk,
            draft_topk_ids=draft_topk_ids,
            draft_topk_probs=draft_topk_probs,
            max_steps_per_req=max_steps_per_req,
            draft_conf_debug=ref_entry.draft_conf_debug,
        )

    def _store_ssd_cached_proposal(
        self,
        batch: ScheduleBatch,
        proposal: _DFlashDraftProposal,
        *,
        expected_seq_lens_cpu: torch.Tensor,
        expected_verified_id: torch.Tensor,
    ) -> None:
        if not self._ssd_enabled:
            return

        seq_lens_cpu = expected_seq_lens_cpu.to("cpu", non_blocking=False).to(torch.int64)
        verified_cpu = expected_verified_id.to("cpu", non_blocking=False).to(torch.int64)
        draft_tokens = proposal.draft_tokens.detach().clone()
        draft_topk_ids = (
            proposal.draft_topk_ids.detach().clone()
            if proposal.draft_topk_ids is not None
            else None
        )
        draft_topk_probs = (
            proposal.draft_topk_probs.detach().clone()
            if proposal.draft_topk_probs is not None
            else None
        )
        max_steps_cpu = (
            proposal.max_steps_per_req.detach().to("cpu", non_blocking=False).to(torch.int64)
            if proposal.max_steps_per_req is not None
            else None
        )

        for i, req in enumerate(batch.reqs):
            if not self._req_ssd_enabled_for_phase(req):
                continue
            req.spec_ssd_prepare_ct = int(getattr(req, "spec_ssd_prepare_ct", 0)) + 1
            cache_key = (int(seq_lens_cpu[i].item()), int(verified_cpu[i].item()))
            entry = _DFlashSsdCachedProposal(
                expected_seq_len=cache_key[0],
                expected_verified_id=cache_key[1],
                draft_tokens=draft_tokens[i].clone(),
                draft_token_num=int(proposal.draft_token_num),
                verify_mode=str(proposal.verify_mode),
                draft_topk=int(proposal.draft_topk),
                draft_topk_ids=(
                    draft_topk_ids[i].clone() if draft_topk_ids is not None else None
                ),
                draft_topk_probs=(
                    draft_topk_probs[i].clone() if draft_topk_probs is not None else None
                ),
                max_steps=(
                    int(max_steps_cpu[i].item()) if max_steps_cpu is not None else None
                ),
                draft_conf_debug=proposal.draft_conf_debug,
            )
            cache = self._get_req_ssd_cache(req, create=True)
            assert cache is not None
            cache[cache_key] = entry
            while len(cache) > int(self._ssd_max_entries):
                oldest_key = next(iter(cache))
                cache.pop(oldest_key, None)
            # Keep the legacy single-entry mirror for debug/compatibility until the
            # rest of the stack no longer references it.
            setattr(req, "dflash_ssd_cached_proposal", entry)
            if self.tp_rank == 0 and not getattr(self, "_logged_ssd_store", False):
                logger.info(
                    "DFLASH SSD cache store. key=(seq_len=%d, verified_id=%d) pending=%d",
                    int(cache_key[0]),
                    int(cache_key[1]),
                    int(len(cache)),
                )
                setattr(self, "_logged_ssd_store", True)

    def _build_single_req_sampling_info(self, req) -> SamplingBatchInfo:
        proxy = _DFlashSingleReqSamplingBatchView(reqs=[req], device=self.device)
        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        return SamplingBatchInfo.from_schedule_batch(proxy, vocab_size=vocab_size)

    def _build_single_req_batch_view(
        self,
        *,
        req,
        req_pool_index: torch.Tensor,
    ) -> _DFlashSingleReqBatchView:
        sampling_info = self._build_single_req_sampling_info(req)
        return _DFlashSingleReqBatchView(
            reqs=[req],
            req_pool_indices=req_pool_index.view(1),
            sampling_info=sampling_info,
        )

    def _build_repeated_req_batch_view(
        self,
        *,
        req,
        req_pool_indices: torch.Tensor,
    ) -> _DFlashSingleReqBatchView:
        bs = int(req_pool_indices.numel())
        proxy = _DFlashSingleReqSamplingBatchView(reqs=[req] * bs, device=self.device)
        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        sampling_info = SamplingBatchInfo.from_schedule_batch(proxy, vocab_size=vocab_size)
        return _DFlashSingleReqBatchView(
            reqs=[req] * bs,
            req_pool_indices=req_pool_indices.view(bs),
            sampling_info=sampling_info,
        )

    def _get_req_fanout_miss_streak_trigger(self, req) -> int:
        trigger = int(self._ssd_fanout_miss_streak_trigger)
        if trigger <= 0:
            return 0
        long_tokens = int(self._ssd_fanout_long_horizon_tokens)
        long_trigger = int(self._ssd_fanout_miss_streak_trigger_long)
        if long_tokens <= 0 or long_trigger <= 0:
            return trigger
        decoded_tokens = 0
        try:
            decoded_tokens = int(len(getattr(req, "output_ids", []) or []))
        except Exception:
            decoded_tokens = 0
        if decoded_tokens >= long_tokens:
            return long_trigger
        return trigger

    def _get_req_decoded_tokens(self, req) -> int:
        try:
            return int(len(getattr(req, "output_ids", []) or []))
        except Exception:
            return 0

    def _req_full_dflash_enabled(self, req) -> bool:
        threshold = int(self._dflash_enable_after_decoded_tokens)
        if threshold <= 0:
            return True
        return self._get_req_decoded_tokens(req) >= threshold

    def _req_ssd_enabled_for_phase(self, req) -> bool:
        if not self._ssd_enabled:
            return False
        threshold = int(self._ssd_enable_after_decoded_tokens)
        if threshold <= 0:
            return True
        return self._get_req_decoded_tokens(req) >= threshold

    def _apply_late_enable_dflash_cap(
        self,
        *,
        batch: ScheduleBatch,
        max_steps_per_req: torch.Tensor | None,
        step_count: int,
    ) -> torch.Tensor | None:
        threshold = int(self._dflash_enable_after_decoded_tokens)
        if threshold <= 0 or step_count <= 0:
            return max_steps_per_req

        late_enable_mask = torch.tensor(
            [not self._req_full_dflash_enabled(req) for req in batch.reqs],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(late_enable_mask.any()):
            return max_steps_per_req

        if max_steps_per_req is None:
            max_steps_per_req = torch.full(
                (int(batch.batch_size()),),
                int(step_count),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            max_steps_per_req = max_steps_per_req.clone()

        one_step_cap = torch.ones_like(max_steps_per_req, dtype=torch.int32)
        max_steps_per_req = torch.where(
            late_enable_mask,
            torch.minimum(max_steps_per_req, one_step_cap),
            max_steps_per_req,
        )
        return max_steps_per_req

    def _apply_adaptive_logical_cap(
        self,
        *,
        batch: ScheduleBatch,
        max_steps_per_req: torch.Tensor | None,
        step_count: int,
    ) -> torch.Tensor | None:
        force_max_steps = (os.environ.get("SGLANG_DFLASH_FORCE_MAX_STEPS") or "").strip()
        if force_max_steps:
            try:
                force_max_steps_i = int(force_max_steps)
            except Exception:
                force_max_steps_i = -1
            if force_max_steps_i > 0 and step_count > 0 and int(batch.batch_size()) > 0:
                force_max_steps_i = max(1, min(int(step_count), int(force_max_steps_i)))
                if max_steps_per_req is None:
                    max_steps_per_req = torch.full(
                        (int(batch.batch_size()),),
                        int(force_max_steps_i),
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    max_steps_per_req = torch.full_like(
                        max_steps_per_req, int(force_max_steps_i), dtype=torch.int32
                    )
                if self.tp_rank == 0 and not getattr(self, "_logged_force_max_steps", False):
                    logger.info(
                        "DFLASH forcing logical max_steps_per_req=%s with physical_block=%s",
                        int(force_max_steps_i),
                        int(step_count + 1),
                    )
                    setattr(self, "_logged_force_max_steps", True)
                return max_steps_per_req

        enabled = (
            os.environ.get("SGLANG_DFLASH_ADAPTIVE_CAP_ENABLE") or ""
        ).strip().lower() not in ("", "0", "false", "off", "no")
        if not enabled or step_count <= 0 or int(batch.batch_size()) <= 0:
            return max_steps_per_req

        def _env_float(name: str, default: float) -> float:
            try:
                return float((os.environ.get(name) or str(default)).strip())
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            try:
                return int((os.environ.get(name) or str(default)).strip())
            except Exception:
                return int(default)

        verify_ct_ge = _env_int("SGLANG_DFLASH_ADAPTIVE_CAP_VERIFY_CT_GE", 8)
        accept_ema_hard_le = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_EMA_HARD_LE", 2.0
        )
        accept_ema_medium_le = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_EMA_MEDIUM_LE", 5.0
        )
        hard_cap_steps = _env_int("SGLANG_DFLASH_ADAPTIVE_CAP_HARD_STEPS", 4)
        medium_cap_steps = _env_int("SGLANG_DFLASH_ADAPTIVE_CAP_MEDIUM_STEPS", 6)
        q_entropy_hard_le = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_Q_ENTROPY_HARD_LE", -1.0
        )
        q_entropy_hard_ge = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_Q_ENTROPY_HARD_GE", -1.0
        )
        q_max_hard_ge = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_Q_MAX_HARD_GE", -1.0
        )
        q_max_hard_le = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_Q_MAX_HARD_LE", -1.0
        )
        tv_hard_ge = _env_float(
            "SGLANG_DFLASH_ADAPTIVE_CAP_TV_HARD_GE", -1.0
        )

        if max_steps_per_req is None:
            max_steps_per_req = torch.full(
                (int(batch.batch_size()),),
                int(step_count),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            max_steps_per_req = max_steps_per_req.clone()

        caps_cpu = []
        for req in batch.reqs:
            caps_cpu.append(
                int(
                    compute_adaptive_max_steps_for_req(
                        getattr(req, "dflash_difficulty_state", None),
                        step_count=int(step_count),
                        verify_ct_ge=int(verify_ct_ge),
                        accept_ema_hard_le=float(accept_ema_hard_le),
                        accept_ema_medium_le=float(accept_ema_medium_le),
                        hard_cap_steps=int(hard_cap_steps),
                        medium_cap_steps=int(medium_cap_steps),
                        q_entropy_hard_le=float(q_entropy_hard_le),
                        q_entropy_hard_ge=float(q_entropy_hard_ge),
                        q_max_hard_ge=float(q_max_hard_ge),
                        q_max_hard_le=float(q_max_hard_le),
                        tv_hard_ge=float(tv_hard_ge),
                    )
                )
            )
        caps = torch.tensor(caps_cpu, dtype=torch.int32, device=self.device)
        max_steps_per_req = torch.minimum(max_steps_per_req, caps)
        return max_steps_per_req

    @staticmethod
    def _update_req_running_mean(req, attr_name: str, value: float | None) -> None:
        if value is None:
            return
        try:
            value_f = float(value)
        except Exception:
            return
        if not math.isfinite(value_f):
            return
        sum_attr = f"{attr_name}_sum"
        ct_attr = f"{attr_name}_ct"
        new_sum = float(getattr(req, sum_attr, 0.0)) + value_f
        new_ct = int(getattr(req, ct_attr, 0)) + 1
        setattr(req, sum_attr, new_sum)
        setattr(req, ct_attr, new_ct)
        setattr(req, attr_name, new_sum / float(new_ct))

    def _update_req_dflash_debug_stats(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DFlashVerifyInput,
        accept_length_per_req_cpu: list[int],
        dflash_debug: dict | None,
        append_path: str | None = None,
    ) -> None:
        verify_mode = str(getattr(verify_input, "verify_mode", "target_only") or "target_only")
        draft_conf_debug = getattr(verify_input, "draft_conf_debug", None)
        pq_ctrl_debug = None
        if dflash_debug is not None:
            try:
                sig = DFlashDifficultySignals.from_debug(
                    verify_mode=verify_mode,
                    dflash_debug=dflash_debug,
                    draft_conf_debug=draft_conf_debug,
                )
            except Exception:
                sig = None
        else:
            sig = None
        if verify_mode == "pq" and sig is not None:
            try:
                _, pq_ctrl_debug = self._adaptive_pq.on_verify_end(sig)
            except Exception:
                pq_ctrl_debug = None

        max_steps_cpu = None
        if getattr(verify_input, "max_steps_per_req", None) is not None:
            with torch.no_grad():
                max_steps_cpu = (
                    verify_input.max_steps_per_req.detach()
                    .to("cpu", non_blocking=False)
                    .to(torch.int64)
                    .tolist()
                )

        effective_draft_token_num = int(
            getattr(verify_input, "draft_token_num", self.block_size)
        )
        update_dflash_req_verify_bookkeeping(
            reqs=list(batch.reqs),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            verify_mode=verify_mode,
            append_path=append_path,
            dflash_debug=dflash_debug,
            draft_conf_debug=draft_conf_debug,
            max_steps_per_req_cpu=max_steps_cpu,
            default_max_steps=int(self.block_size - 1),
            default_effective_draft_token_num=effective_draft_token_num,
            default_effective_step_count=max(0, effective_draft_token_num - 1),
        )

        if pq_ctrl_debug is not None:
            for req in batch.reqs:
                self._update_req_running_mean(
                    req,
                    "spec_dflash_adaptive_temp_mul",
                    pq_ctrl_debug.get("temp_mul"),
                )
                req.spec_dflash_pq_disabled_rounds_left = int(
                    pq_ctrl_debug.get("pq_disabled_rounds_left", 0) or 0
                )

    def _prepare_ssd_shadow_req_to_token(
        self,
        *,
        real_req_pool_index: torch.Tensor,
        prefix_len: torch.Tensor,
        rows: int = 1,
    ) -> None:
        if (
            self._ssd_shadow_req_to_token_pool is None
            or self._ssd_shadow_req_pool_indices is None
            or self._ssd_shadow_block_cache_loc is None
        ):
            raise RuntimeError("DFLASH SSD async shadow buffers are not initialized.")

        prefix_len_i = int(prefix_len.reshape(-1)[0].item())
        if prefix_len_i < 0:
            raise RuntimeError(f"Invalid SSD shadow prefix_len={prefix_len_i}.")

        rows_i = max(
            1,
            min(
                int(rows),
                int(self._ssd_shadow_req_pool_indices.numel()),
            ),
        )
        shadow_rows = self._ssd_shadow_req_to_token_pool.req_to_token[
            self._ssd_shadow_req_pool_indices[:rows_i]
        ]
        real_row = self.draft_model_runner.req_to_token_pool.req_to_token[
            real_req_pool_index.reshape(-1)[0]
        ]

        with self._draft_worker_lock:
            if prefix_len_i > 0:
                shadow_rows[:, :prefix_len_i].copy_(
                    real_row[:prefix_len_i].unsqueeze(0).expand(rows_i, -1)
                )
            block_end = prefix_len_i + int(self.block_size)
            shadow_rows[:, prefix_len_i:block_end].zero_()

    def _collect_ssd_branch_candidates(
        self,
        *,
        batch: ScheduleBatch,
        logits_output,
        commit_lens: torch.Tensor,
        new_verified_id: torch.Tensor,
    ) -> list[list[int]]:
        fanout = int(self._ssd_fanout)
        if fanout <= 1:
            return [[int(tok)] for tok in new_verified_id.detach().to("cpu").tolist()]

        logits = logits_output.next_token_logits
        if logits is None or logits.numel() == 0:
            return [[int(tok)] for tok in new_verified_id.detach().to("cpu").tolist()]

        bs = int(batch.batch_size())
        step_width = int(self.block_size)
        search_topk = max(int(fanout), int(self._ssd_branch_search_topk))
        search_topk = min(search_topk, int(logits.shape[-1]))
        results: list[list[int]] = []
        sampling_info = batch.sampling_info

        commit_lens_cpu = commit_lens.detach().to("cpu", non_blocking=False).tolist()
        verified_cpu = (
            new_verified_id.detach().to("cpu", non_blocking=False).to(torch.int64).tolist()
        )

        for req_idx in range(bs):
            actual_id = int(verified_cpu[req_idx])
            commit_len = int(commit_lens_cpu[req_idx])
            branch_ids: list[int] = [actual_id]
            req = batch.reqs[req_idx]
            req_trigger = int(self._get_req_fanout_miss_streak_trigger(req))
            if (
                req_trigger > 0
                and int(getattr(req, "spec_ssd_miss_streak", 0))
                < req_trigger
            ):
                results.append(branch_ids)
                continue
            if not req_is_hard_enough_for_fanout(
                getattr(req, "dflash_difficulty_state", None),
                accept_ema_le=float(self._ssd_fanout_difficulty_accept_ema_le),
                accept_last_le=float(self._ssd_fanout_difficulty_accept_last_le),
                min_verify_ct=int(self._ssd_fanout_difficulty_min_verify_ct),
            ):
                req.spec_ssd_difficulty_gate_skip_ct = int(
                    getattr(req, "spec_ssd_difficulty_gate_skip_ct", 0)
                ) + 1
                if self.tp_rank == 0 and not getattr(
                    self, "_logged_ssd_difficulty_gate_skip", False
                ):
                    st = getattr(req, "dflash_difficulty_state", None)
                    logger.info(
                        "DFLASH SSD difficulty gate skip: accept_last=%s accept_ema=%s verify_ct=%s",
                        (
                            float(getattr(st, "accept_len_last", 0.0))
                            if st is not None
                            else None
                        ),
                        (
                            float(getattr(st, "accept_len_ema", 0.0))
                            if st is not None
                            else None
                        ),
                        (
                            int(getattr(st, "verify_ct_last", 0))
                            if st is not None
                            else None
                        ),
                    )
                    setattr(self, "_logged_ssd_difficulty_gate_skip", True)
                results.append(branch_ids)
                continue
            if commit_len <= 0 or search_topk <= 1:
                results.append(branch_ids)
                continue

            row = req_idx * step_width + (commit_len - 1)
            row_logits = logits[row : row + 1]
            topk_vals, topk_ids = torch.topk(row_logits, k=search_topk, dim=-1)

            req_is_greedy = (
                sampling_info is None
                or int(sampling_info.top_ks[req_idx].item()) <= 1
            )
            if req_is_greedy:
                results.append(branch_ids)
                continue
            else:
                req_temp = sampling_info.temperatures[req_idx : req_idx + 1].to(
                    row_logits.device
                )
                req_top_p = sampling_info.top_ps[req_idx : req_idx + 1].to(
                    row_logits.device
                )
                req_top_k = sampling_info.top_ks[req_idx : req_idx + 1].to(
                    row_logits.device
                )
                req_min_p = sampling_info.min_ps[req_idx : req_idx + 1].to(
                    row_logits.device
                )

                scaled = topk_vals.to(torch.float32) / req_temp.to(torch.float32).view(
                    -1, 1
                )
                probs = torch.softmax(scaled, dim=-1)
                probs = filter_topk_probs_like_sglang_sampler(
                    probs,
                    temperatures=torch.ones_like(req_temp, device=row_logits.device),
                    top_ks=req_top_k,
                    top_ps=req_top_p,
                    min_ps=req_min_p,
                    need_min_p_sampling=bool(sampling_info.need_min_p_sampling),
                    no_min_p_filter_apply_order="joint",
                )
                probs_row = probs[0].to(torch.float32)
                actual_mask = topk_ids[0] == int(actual_id)
                actual_prob = (
                    float(probs_row[actual_mask].max().item())
                    if bool(actual_mask.any().item())
                    else 0.0
                )
                alt_probs = probs_row.clone()
                if bool(actual_mask.any().item()):
                    alt_probs[actual_mask] = 0.0
                alt_mass = float(alt_probs.sum().item())
                if alt_mass <= 0.0:
                    results.append(branch_ids)
                    continue
                alt_top1 = float(alt_probs.max().item())
                alt_norm = alt_probs / alt_probs.sum().clamp_min(1e-20)
                alt_entropy = float(
                    -(
                        alt_norm.clamp_min(1e-20)
                        * torch.log(alt_norm.clamp_min(1e-20))
                    ).sum().item()
                )
                gate_skip = False
                if (
                    float(self._ssd_fanout_skip_if_actual_prob_ge) < 1.0
                    and actual_prob >= float(self._ssd_fanout_skip_if_actual_prob_ge)
                ):
                    gate_skip = True
                if alt_top1 < float(self._ssd_fanout_min_alt_prob):
                    gate_skip = True
                if (
                    math.isfinite(float(self._ssd_fanout_max_alt_entropy))
                    and alt_entropy > float(self._ssd_fanout_max_alt_entropy)
                ):
                    gate_skip = True
                if gate_skip:
                    req.spec_ssd_fanout_gate_skip_ct = int(
                        getattr(req, "spec_ssd_fanout_gate_skip_ct", 0)
                    ) + 1
                    if self.tp_rank == 0 and not getattr(
                        self, "_logged_ssd_fanout_gate_skip", False
                    ):
                        logger.info(
                            "DFLASH SSD fanout gate skip: actual_prob=%s alt_top1=%s alt_entropy=%s",
                            actual_prob,
                            alt_top1,
                            alt_entropy,
                        )
                        setattr(self, "_logged_ssd_fanout_gate_skip", True)
                    results.append(branch_ids)
                    continue
                effective_fanout = int(fanout)
                if float(self._ssd_fanout_target_alt_mass) > 0.0:
                    alt_sorted = torch.sort(alt_norm, descending=True).values
                    target_mass = float(self._ssd_fanout_target_alt_mass)
                    cum_mass = 0.0
                    needed_alts = 0
                    max_alts = max(1, int(fanout) - 1)
                    for p in alt_sorted.tolist():
                        p = float(p)
                        if p <= 0.0:
                            break
                        cum_mass += p
                        needed_alts += 1
                        if needed_alts >= max_alts or cum_mass >= target_mass:
                            break
                    effective_fanout = max(
                        1,
                        min(int(fanout), 1 + int(max(0, needed_alts))),
                    )
                if self.tp_rank == 0 and not getattr(
                    self, "_logged_ssd_effective_fanout", False
                ):
                    logger.info(
                        "DFLASH SSD adaptive fanout decision: configured=%s effective=%s actual_prob=%s alt_top1=%s alt_entropy=%s",
                        int(fanout),
                        int(effective_fanout),
                        actual_prob,
                        alt_top1,
                        alt_entropy,
                    )
                    setattr(self, "_logged_ssd_effective_fanout", True)
                if int(effective_fanout) > 1:
                    req.spec_ssd_fanout_escalation_ct = int(
                        getattr(req, "spec_ssd_fanout_escalation_ct", 0)
                    ) + 1
                    req.spec_ssd_fanout_alt_budget = int(
                        getattr(req, "spec_ssd_fanout_alt_budget", 0)
                    ) + max(0, int(effective_fanout) - 1)
                if float(self._ssd_sampler_x) != 1.0:
                    topf = min(int(search_topk), max(1, int(fanout)))
                    topf_idx = torch.topk(alt_probs, k=topf, dim=-1).indices
                    probs_row = alt_probs.clone()
                    probs_row[topf_idx] *= float(self._ssd_sampler_x)
                    probs_row = probs_row / probs_row.sum().clamp_min(1e-20)
                else:
                    probs_row = alt_probs

                if (
                    self._ssd_branch_mode == "sample"
                    and int(effective_fanout) > 1
                    and float(probs_row.sum().item()) > 0.0
                ):
                    sample_count = min(
                        int(effective_fanout) - 1,
                        int((probs_row > 0).sum().item()),
                    )
                    sampled_cols = torch.multinomial(
                        probs_row / probs_row.sum().clamp_min(1e-20),
                        num_samples=sample_count,
                        replacement=False,
                    )
                    ranked_ids = (
                        topk_ids[0]
                        .gather(0, sampled_cols.to(torch.int64))
                        .detach()
                        .to("cpu", non_blocking=False)
                        .tolist()
                    )
                else:
                    ranked = torch.argsort(probs_row, descending=True)
                    ranked_ids = (
                        topk_ids[0]
                        .gather(0, ranked.to(torch.int64))
                        .detach()
                        .to("cpu", non_blocking=False)
                        .tolist()
                    )

            for token_id in ranked_ids:
                token_id = int(token_id)
                if token_id in branch_ids:
                    continue
                branch_ids.append(token_id)
                if len(branch_ids) >= effective_fanout:
                    break

            results.append(branch_ids)

        return results

    def _prepare_ssd_fanout_branches(
        self,
        *,
        batch: ScheduleBatch,
        logits_output,
        commit_lens: torch.Tensor,
        new_verified_id: torch.Tensor,
        next_prefix_lens: torch.Tensor,
        next_seq_lens_cpu: torch.Tensor,
    ) -> None:
        if not self._ssd_enabled or not self._ssd_prepare_next:
            return

        branch_ids_per_req = self._collect_ssd_branch_candidates(
            batch=batch,
            logits_output=logits_output,
            commit_lens=commit_lens,
            new_verified_id=new_verified_id,
        )
        if not branch_ids_per_req:
            return

        for req_idx, req in enumerate(batch.reqs):
            if not self._req_ssd_enabled_for_phase(req):
                continue
            branch_ids = branch_ids_per_req[req_idx]

            req_pool_index = batch.req_pool_indices[req_idx : req_idx + 1]
            single_batch = self._build_single_req_batch_view(
                req=req,
                req_pool_index=req_pool_index,
            )
            prefix_len = next_prefix_lens[req_idx : req_idx + 1]
            seq_len_cpu = next_seq_lens_cpu[req_idx : req_idx + 1]

            seen_branch_tokens: set[int] = set()
            for branch_token in branch_ids:
                branch_token = int(branch_token)
                if branch_token in seen_branch_tokens:
                    continue
                seen_branch_tokens.add(branch_token)
                branch_verified = torch.tensor(
                    [branch_token], dtype=torch.int64, device=self.device
                )
                branch_input = DFlashDraftInput(
                    verified_id=branch_verified,
                    target_hidden=torch.empty(
                        (0,), dtype=torch.float32, device=self.device
                    ),
                    ctx_lens=torch.zeros((1,), dtype=torch.int32, device=self.device),
                    draft_seq_lens=torch.zeros(
                        (1,), dtype=torch.int32, device=self.device
                    ),
                )
                branch_proposal = self._build_draft_proposal(
                    single_batch,
                    branch_input,
                    prefix_lens=prefix_len,
                    seq_lens_cpu=seq_len_cpu,
                )
                self._store_ssd_cached_proposal(
                    single_batch,
                    branch_proposal,
                    expected_seq_lens_cpu=seq_len_cpu,
                    expected_verified_id=branch_verified,
                )

        if self.tp_rank == 0 and not getattr(self, "_logged_ssd_fanout", False):
            logger.info(
                "DFLASH SSD fanout enabled. fanout=%d first_branch_sizes=%s",
                int(self._ssd_fanout),
                [int(len(x)) for x in branch_ids_per_req[: min(8, len(branch_ids_per_req))]],
            )
            setattr(self, "_logged_ssd_fanout", True)

    def _schedule_ssd_fanout_overlap(
        self,
        *,
        batch: ScheduleBatch,
        logits_output,
        commit_lens: torch.Tensor,
        new_verified_id: torch.Tensor,
        next_prefix_lens: torch.Tensor,
        next_seq_lens_cpu: torch.Tensor,
    ) -> None:
        branch_ids_per_req = self._collect_ssd_branch_candidates(
            batch=batch,
            logits_output=logits_output,
            commit_lens=commit_lens,
            new_verified_id=new_verified_id,
        )
        if not branch_ids_per_req:
            return

        if self._ssd_overlap_executor is None or not self._ssd_async_overlap:
            self._prepare_ssd_fanout_branches(
                batch=batch,
                logits_output=logits_output,
                commit_lens=commit_lens,
                new_verified_id=new_verified_id,
                next_prefix_lens=next_prefix_lens,
                next_seq_lens_cpu=next_seq_lens_cpu,
            )
            return

        payloads = []
        for req_idx, req in enumerate(batch.reqs):
            if not self._req_ssd_enabled_for_phase(req):
                continue
            branch_ids = branch_ids_per_req[req_idx]
            deduped_branch_ids: list[int] = []
            seen_branch_tokens: set[int] = set()
            for branch_token in branch_ids:
                branch_token = int(branch_token)
                if branch_token in seen_branch_tokens:
                    continue
                seen_branch_tokens.add(branch_token)
                deduped_branch_ids.append(branch_token)
            if not deduped_branch_ids:
                continue
            payloads.append(
                {
                    "req": req,
                    "req_pool_index": batch.req_pool_indices[
                        req_idx : req_idx + 1
                    ].detach().clone(),
                    "prefix_len": next_prefix_lens[req_idx : req_idx + 1]
                    .detach()
                    .clone(),
                    "seq_len_cpu": next_seq_lens_cpu[req_idx : req_idx + 1]
                    .detach()
                    .clone(),
                    "branch_ids": deduped_branch_ids,
                }
            )

        if not payloads:
            return

        self._maybe_finalize_ssd_overlap(block=False)

        def _task() -> None:
            for item in payloads:
                req = item["req"]
                try:
                    if hasattr(req, "finished") and callable(req.finished) and bool(req.finished()):
                        continue
                except Exception:
                    pass

                if (
                    self._ssd_shadow_req_pool_indices is None
                    or self._ssd_shadow_req_to_token_pool is None
                    or self._ssd_shadow_block_cache_loc is None
                ):
                    raise RuntimeError("DFLASH SSD async overlap shadow state is missing.")

                self._prepare_ssd_shadow_req_to_token(
                    real_req_pool_index=item["req_pool_index"],
                    prefix_len=item["prefix_len"],
                    rows=int(len(item.get("branch_ids") or [])),
                )

                branch_ids = list(item.get("branch_ids") or [])
                branch_ct = int(len(branch_ids))
                if branch_ct <= 0:
                    continue
                if int(self._ssd_shadow_req_pool_indices.numel()) < int(branch_ct):
                    raise RuntimeError(
                        "DFLASH SSD async overlap shadow pool is too small for fanout. "
                        f"shadow_bs={int(self._ssd_shadow_req_pool_indices.numel())} branch_ct={int(branch_ct)}"
                    )
                need_slots = int(branch_ct) * int(self.block_size)
                if int(self._ssd_shadow_block_cache_loc.numel()) < int(need_slots):
                    raise RuntimeError(
                        "DFLASH SSD async overlap shadow scratch is too small for fanout. "
                        f"slots={int(self._ssd_shadow_block_cache_loc.numel())} need_slots={int(need_slots)}"
                    )

                branch_pool_indices = self._ssd_shadow_req_pool_indices[:branch_ct]
                branch_batch = self._build_repeated_req_batch_view(
                    req=req,
                    req_pool_indices=branch_pool_indices,
                )
                branch_verified = torch.tensor(
                    [int(x) for x in branch_ids], dtype=torch.int64, device=self.device
                )
                branch_input = DFlashDraftInput(
                    verified_id=branch_verified,
                    target_hidden=torch.empty((0,), dtype=torch.float32, device=self.device),
                    ctx_lens=torch.zeros((branch_ct,), dtype=torch.int32, device=self.device),
                    draft_seq_lens=torch.zeros((branch_ct,), dtype=torch.int32, device=self.device),
                )
                prefix_lens = item["prefix_len"].reshape(1).repeat(branch_ct)
                seq_lens_cpu = item["seq_len_cpu"].reshape(1).repeat(branch_ct)
                out_cache_loc = self._ssd_shadow_block_cache_loc[:need_slots]
                branch_proposal = self._build_draft_proposal(
                    branch_batch,
                    branch_input,
                    prefix_lens=prefix_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    req_to_token_pool_override=self._ssd_shadow_req_to_token_pool,
                    out_cache_loc_override=out_cache_loc,
                    use_ssd_overlap_path=True,
                )
                self._store_ssd_cached_proposal(
                    branch_batch,
                    branch_proposal,
                    expected_seq_lens_cpu=seq_lens_cpu,
                    expected_verified_id=branch_verified,
                )

            if self.tp_rank == 0 and not getattr(self, "_logged_ssd_async_launch", False):
                logger.info(
                    "DFLASH SSD async overlap scheduled. payload_reqs=%d",
                    int(len(payloads)),
                )
                setattr(self, "_logged_ssd_async_launch", True)

        with self._ssd_overlap_lock:
            self._maybe_finalize_ssd_overlap(block=False)
            self._ssd_overlap_future = self._ssd_overlap_executor.submit(_task)
            self._ssd_overlap_launch_ct += 1

    def _build_draft_proposal(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
        *,
        prefix_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_to_token_pool_override: ReqToTokenPool | None = None,
        out_cache_loc_override: torch.Tensor | None = None,
        use_ssd_overlap_path: bool = False,
    ) -> _DFlashDraftProposal:
        bs = batch.batch_size()
        physical_step_count = int(self.block_size - 1)
        max_steps_per_req = self._apply_adaptive_logical_cap(
            batch=batch,
            max_steps_per_req=None,
            step_count=physical_step_count,
        )
        max_steps_per_req = self._apply_late_enable_dflash_cap(
            batch=batch,
            max_steps_per_req=max_steps_per_req,
            step_count=physical_step_count,
        )
        effective_step_count = int(physical_step_count)
        if max_steps_per_req is not None and int(max_steps_per_req.numel()) > 0:
            effective_step_count = int(
                torch.clamp(
                    max_steps_per_req.max().to(torch.int64),
                    min=1,
                    max=int(physical_step_count),
                ).item()
            )
        draft_token_num = int(effective_step_count + 1)


        # --- 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and `shard_indices` attributes."
            )

        req_to_token_pool = (
            req_to_token_pool_override
            if req_to_token_pool_override is not None
            else self.draft_model_runner.req_to_token_pool
        )
        buffers = self._get_draft_block_buffers(
            bs=bs, use_ssd_overlap_path=bool(use_ssd_overlap_path)
        )

        block_ids = buffers.block_ids[:, :draft_token_num]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))
        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        pos_offsets = self._block_pos_offsets[:draft_token_num]
        positions_2d = buffers.positions[:, :draft_token_num]
        torch.add(prefix_lens.unsqueeze(1), pos_offsets, out=positions_2d)

        block_start = prefix_lens
        block_end = buffers.block_end
        torch.add(block_start, int(draft_token_num), out=block_end)

        seq_lens_cpu_buf = buffers.seq_lens_cpu
        if seq_lens_cpu.dtype == torch.int32:
            seq_lens_cpu_buf.copy_(seq_lens_cpu)
        else:
            seq_lens_cpu_buf.copy_(seq_lens_cpu.to(torch.int32))

        allocator = self.draft_model_runner.token_to_kv_pool_allocator

        draft_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(draft_token_num),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        seq_lens_sum = int(prefix_lens.to(torch.int64).sum().item())
        device_module = torch.get_device_module(self.device)
        stream_ctx = nullcontext()
        lock_ctx = nullcontext()
        if not use_ssd_overlap_path:
            lock_ctx = self._draft_worker_lock
        if use_ssd_overlap_path and self._ssd_overlap_stream is not None:
            self._ssd_overlap_stream.wait_stream(device_module.current_stream())
            stream_ctx = device_module.stream(self._ssd_overlap_stream)

        with lock_ctx, stream_ctx:
            if out_cache_loc_override is None:
                self._ensure_draft_block_cache_loc_scratch(bs)
                if self._draft_block_cache_loc_scratch is None:
                    raise RuntimeError("DFLASH draft scratch KV buffer is missing.")

                scratch = self._draft_block_cache_loc_scratch[:bs]
                if int(getattr(allocator, "page_size", 1)) == 1:
                    block_cache_loc = scratch[:, :draft_token_num].reshape(-1)
                else:
                    page_size = int(getattr(allocator, "page_size", 1))
                    # Align scratch locs so `loc % page_size == position % page_size` for each
                    # token position in the draft block.
                    start = (prefix_lens.to(torch.int64) % page_size).to(torch.int64)
                    offs = start.unsqueeze(1) + pos_offsets.unsqueeze(0)
                    block_cache_loc_2d = torch.gather(scratch, 1, offs)
                    if os.environ.get("SGLANG_DFLASH_DEBUG_SCRATCH", "").strip():
                        pos_mod = (
                            prefix_lens.to(torch.int64).unsqueeze(1)
                            + pos_offsets.unsqueeze(0)
                        ) % page_size
                        loc_mod = block_cache_loc_2d % page_size
                        assert torch.all(loc_mod == pos_mod), (
                            "DFLASH scratch loc mod mismatch for paged KV. "
                            f"page_size={page_size}"
                        )
                    block_cache_loc = block_cache_loc_2d.reshape(-1)
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    block_start,
                    block_end,
                    block_cache_loc,
                    bs,
                )

                forward_batch = ForwardBatch(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    batch_size=bs,
                    input_ids=block_ids.flatten(),
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=prefix_lens,
                    out_cache_loc=block_cache_loc,
                    seq_lens_sum=seq_lens_sum,
                    seq_lens_cpu=seq_lens_cpu_buf,
                    positions=positions_2d.reshape(-1),
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                    attn_backend=self.draft_model_runner.attn_backend,
                    input_embeds=input_embeds,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                    spec_info=draft_spec_info,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    num_token_non_padded=torch.tensor(
                        [int(block_ids.numel())],
                        dtype=torch.int32,
                        device=block_ids.device,
                    ),
                )

                with torch.inference_mode():
                    draft_hidden = self.draft_model_runner.forward(forward_batch).logits_output
            else:
                block_cache_loc = out_cache_loc_override.view(-1)
                if int(block_cache_loc.numel()) < int(bs * draft_token_num):
                    raise RuntimeError(
                        "DFLASH SSD shadow scratch size mismatch: "
                        f"expected at least {int(bs * draft_token_num)} slots, got {int(block_cache_loc.numel())}."
                    )
                block_cache_loc = block_cache_loc[: int(bs * draft_token_num)]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    block_start,
                    block_end,
                    block_cache_loc,
                    bs,
                )

                forward_batch = ForwardBatch(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    batch_size=bs,
                    input_ids=block_ids.flatten(),
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=prefix_lens,
                    out_cache_loc=block_cache_loc,
                    seq_lens_sum=seq_lens_sum,
                    seq_lens_cpu=seq_lens_cpu_buf,
                    positions=positions_2d.reshape(-1),
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                    attn_backend=self.draft_model_runner.attn_backend,
                    input_embeds=input_embeds,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                    spec_info=draft_spec_info,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    num_token_non_padded=torch.tensor(
                        [int(block_ids.numel())],
                        dtype=torch.int32,
                        device=block_ids.device,
                    ),
                )


                with torch.inference_mode():
                    draft_hidden = self.draft_model_runner.forward(forward_batch).logits_output

            draft_hidden = draft_hidden.view(bs, draft_token_num, -1).clone()
            first_verified = block_ids[:, 0].clone()

        draft_tokens = torch.empty(
            (bs, draft_token_num), dtype=torch.long, device=draft_hidden.device
        )
        draft_tokens[:, 0].copy_(first_verified)

        sampling_info = batch.sampling_info
        verify_mode = str(
            getattr(self.server_args, "speculative_dflash_verify_mode", "target_only")
            or "target_only"
        )
        draft_conf_debug = None
        if sampling_info is not None:
            temperatures = sampling_info.temperatures.to(draft_hidden.device)
            top_ps = sampling_info.top_ps.to(draft_hidden.device)
            top_ks = sampling_info.top_ks.to(draft_hidden.device)
            min_ps = sampling_info.min_ps.to(draft_hidden.device)
            need_min_p_sampling = bool(sampling_info.need_min_p_sampling)
            is_all_greedy = bool(torch.all(top_ks.to(torch.int64) == 1).item())
        else:
            temps_list = []
            top_p_list = []
            top_k_list = []
            min_p_list = []
            for req in batch.reqs:
                sp = req.sampling_params
                temps_list.append(float(getattr(sp, "temperature", 1.0)))
                top_p_list.append(float(getattr(sp, "top_p", 1.0)))
                top_k_list.append(int(getattr(sp, "top_k", 1)))
                min_p_list.append(float(getattr(sp, "min_p", 0.0)))
            temperatures = torch.tensor(
                temps_list, dtype=torch.float32, device=draft_hidden.device
            )
            top_ps = torch.tensor(
                top_p_list, dtype=torch.float32, device=draft_hidden.device
            )
            top_ks = torch.tensor(
                top_k_list, dtype=torch.int32, device=draft_hidden.device
            )
            min_ps = torch.tensor(
                min_p_list, dtype=torch.float32, device=draft_hidden.device
            )
            need_min_p_sampling = bool(any(mp > 0.0 for mp in min_p_list))
            is_all_greedy = bool(all(int(k) == 1 for k in top_k_list))

        if verify_mode == "pq" and self._adaptive_pq.should_force_target_only():
            verify_mode = "target_only"

        use_pq = verify_mode == "pq" and not is_all_greedy
        draft_topk = 0
        draft_topk_ids = None
        draft_topk_probs = None

        if use_pq:
            draft_temp_mul = float(self._adaptive_pq.cfg.temp_mul or 1.0)
            try:
                draft_topk_cap = int(
                    getattr(self.server_args, "speculative_dflash_pq_draft_topk_cap", 0) or 0
                )
            except Exception:
                draft_topk_cap = 0

            draft_temp_mul_env = (os.environ.get("DFLASH_PQ_DRAFT_TEMP_MUL") or "").strip()
            draft_topk_cap_env = (os.environ.get("DFLASH_PQ_DRAFT_TOPK_CAP") or "").strip()
            if draft_temp_mul_env:
                try:
                    draft_temp_mul = float(draft_temp_mul_env)
                except Exception:
                    pass
            if draft_topk_cap_env:
                try:
                    draft_topk_cap = int(draft_topk_cap_env)
                except Exception:
                    pass
            if not math.isfinite(draft_temp_mul) or draft_temp_mul <= 0:
                draft_temp_mul = 1.0
            if draft_topk_cap < 0:
                draft_topk_cap = 0

            topk = int(torch.max(top_ks).item())
            if draft_topk_cap > 0:
                topk = min(topk, int(draft_topk_cap))
            if topk <= 0 or topk >= (1 << 20):
                use_pq = False
                verify_mode = "target_only"
            else:
                step_count = int(draft_token_num - 1)
                hs = draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1])
                topk_p_flat, topk_id_flat = self._topk_from_vocab_parallel_head(
                    hidden_states=hs,
                    lm_head=lm_head,
                    topk=topk,
                )

                temps = torch.repeat_interleave(
                    temperatures, step_count, dim=0
                ).to(topk_p_flat.device)
                if draft_temp_mul != 1.0:
                    temps = temps * float(draft_temp_mul)
                top_ps_rep = torch.repeat_interleave(
                    top_ps, step_count, dim=0
                ).to(topk_p_flat.device)
                top_ks_rep = torch.repeat_interleave(
                    top_ks, step_count, dim=0
                ).to(topk_p_flat.device)
                if draft_topk_cap > 0:
                    cap = torch.tensor(int(draft_topk_cap), device=top_ks_rep.device, dtype=top_ks_rep.dtype)
                    top_ks_rep = torch.minimum(top_ks_rep, cap)
                min_ps_rep = torch.repeat_interleave(
                    min_ps, step_count, dim=0
                ).to(topk_p_flat.device)

                filtered_p = self._filter_topk_probs_for_sampling(
                    topk_p_flat,
                    temperatures=temps,
                    top_ks=top_ks_rep,
                    top_ps=top_ps_rep,
                    min_ps=min_ps_rep,
                    need_min_p_sampling=bool(need_min_p_sampling),
                )

                # Robustify the proposal distribution for:
                # - draft token sampling (`torch.multinomial`)
                # - pq verification (residual sampling uses q_probs; NaNs here can crash verify)
                #
                # If filtering yields an all-zero row or propagates NaN/Inf, fall back to a safe
                # one-hot at top1.
                filtered_p = torch.where(
                    torch.isfinite(filtered_p), filtered_p, torch.zeros_like(filtered_p)
                ).clamp_min(0.0)
                denom = filtered_p.sum(dim=1, keepdim=True)
                bad = denom <= 0
                if bad.any():
                    base = torch.where(
                        torch.isfinite(topk_p_flat), topk_p_flat, torch.zeros_like(topk_p_flat)
                    ).clamp_min(0.0)
                    arg = torch.argmax(base, dim=1, keepdim=True)
                    fallback = torch.zeros_like(filtered_p)
                    fallback.scatter_(1, arg, 1.0)
                    filtered_p = torch.where(
                        bad, fallback, filtered_p / denom.clamp_min(1e-20)
                    )
                else:
                    filtered_p = filtered_p / denom

                sample_mode = (
                    os.environ.get("SGLANG_DFLASH_DRAFT_SAMPLE_MODE") or ""
                ).strip().lower()
                use_multinomial = sample_mode in ("", "multinomial")
                if use_multinomial:
                    sampled_col = torch.multinomial(filtered_p, num_samples=1)
                else:
                    sampled_col = torch.argmax(filtered_p, dim=1, keepdim=True)
                sampled_ids = topk_id_flat.gather(1, sampled_col.to(torch.int64)).view(-1)

                draft_tokens[:, 1:].copy_(
                    sampled_ids.view(bs, step_count).to(torch.long)
                )
                draft_topk = int(topk)
                draft_topk_ids = topk_id_flat.view(bs, step_count, int(topk))
                if use_multinomial:
                    # True pq: proposals are sampled from q.
                    draft_topk_probs = filtered_p.view(bs, step_count, int(topk))
                else:
                    # If we propose deterministically (argmax), then the *proposal distribution* q is a delta
                    # at argmax. For pq correctness, the verifier must use that same q (not the full filtered
                    # distribution), otherwise p/q acceptance becomes invalid.
                    if self.tp_rank == 0 and not getattr(self, "_warned_pq_argmax", False):
                        logger.warning(
                            "DFLASH pq: SGLANG_DFLASH_DRAFT_SAMPLE_MODE=%r uses argmax proposals; treating q as a delta distribution for verifier correctness.",
                            sample_mode,
                        )
                        setattr(self, "_warned_pq_argmax", True)
                    delta = torch.zeros_like(filtered_p)
                    delta.scatter_(1, sampled_col.to(torch.int64), 1.0)
                    draft_topk_probs = delta.view(bs, step_count, int(topk))

                dawn_enable = (os.environ.get("SGLANG_DFLASH_DAWN_ENABLE") or "").strip().lower() not in (
                    "",
                    "0",
                    "false",
                    "off",
                    "no",
                )
                if dawn_enable:
                    try:
                        qmax_lt = float(os.environ.get("SGLANG_DFLASH_DAWN_QMAX_LT") or "0.25")
                    except Exception:
                        qmax_lt = 0.25
                    try:
                        cap_steps = int(os.environ.get("SGLANG_DFLASH_DAWN_CAP_STEPS") or "4")
                    except Exception:
                        cap_steps = 4
                    try:
                        look = int(os.environ.get("SGLANG_DFLASH_DAWN_LOOKAHEAD") or "4")
                    except Exception:
                        look = 4
                    cap_steps = max(1, min(int(step_count), int(cap_steps)))
                    look = max(1, min(int(step_count), int(look)))
                    with torch.no_grad():
                        q = draft_topk_probs.to(torch.float32)
                        qmax = q.max(dim=-1).values
                        min_first = qmax[:, :look].min(dim=1).values
                        hard = min_first < float(qmax_lt)
                        dawn_caps = torch.where(
                            hard,
                            torch.full((bs,), int(cap_steps), device=q.device, dtype=torch.int32),
                            torch.full((bs,), int(step_count), device=q.device, dtype=torch.int32),
                        )
                        if max_steps_per_req is None:
                            max_steps_per_req = dawn_caps
                        else:
                            max_steps_per_req = torch.minimum(
                                max_steps_per_req, dawn_caps
                            )
                if (os.environ.get("SGLANG_DFLASH_DRAFT_CONF_DEBUG") or "").strip().lower() not in (
                    "",
                    "0",
                    "false",
                    "off",
                    "no",
                ):
                    with torch.no_grad():
                        q = draft_topk_probs.to(torch.float32)
                        q_safe = q.clamp_min(1e-20)
                        q_max = q.max(dim=-1).values
                        q_ent = -(q_safe * torch.log(q_safe)).sum(dim=-1)
                        first = min(int(step_count), 4)
                        draft_conf_debug = {
                            "q_max_mean_first": float(q_max[:, :first].mean().item()),
                            "q_max_min_first": float(q_max[:, :first].min().item()),
                            "q_ent_mean_first": float(q_ent[:, :first].mean().item()),
                        }

        if not use_pq:
            draft_next, draft_conf_debug = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
                lm_head=lm_head,
                return_conf_debug=True,
            )
            draft_tokens[:, 1:].copy_(draft_next.view(bs, draft_token_num - 1))

        return _DFlashDraftProposal(
            draft_tokens=draft_tokens.clone(),
            draft_token_num=int(draft_token_num),
            verify_mode=verify_mode,
            draft_topk=draft_topk,
            draft_topk_ids=(draft_topk_ids.clone() if draft_topk_ids is not None else None),
            draft_topk_probs=(draft_topk_probs.clone() if draft_topk_probs is not None else None),
            max_steps_per_req=(
                max_steps_per_req.clone() if max_steps_per_req is not None else None
            ),
            draft_conf_debug=draft_conf_debug,
        )

    def _apply_draft_proposal_to_batch(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        proposal: _DFlashDraftProposal,
    ) -> None:
        bs = batch.batch_size()
        draft_token_num = int(proposal.draft_token_num)
        positions_2d = self._draft_block_positions_buf[:bs, :draft_token_num]
        torch.add(
            batch.seq_lens.unsqueeze(1),
            self._block_pos_offsets[:draft_token_num],
            out=positions_2d,
        )
        positions = positions_2d.reshape(-1)

        verify_input = DFlashVerifyInput(
            draft_token=proposal.draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=draft_token_num,

            verify_mode=proposal.verify_mode,
            draft_topk=proposal.draft_topk,
            draft_topk_ids=proposal.draft_topk_ids,
            draft_topk_probs=proposal.draft_topk_probs,
            max_steps_per_req=proposal.max_steps_per_req,
        )
        if proposal.draft_conf_debug is not None:
            setattr(verify_input, "draft_conf_debug", proposal.draft_conf_debug)

        backend_name = type(self.model_runner.attn_backend).__name__
        skip_custom_mask = backend_name in {
            "FlashInferAttnBackend",
            "FlashInferMLAAttnBackend",
            "FlashAttentionBackend",
            "FlexFlash4CuteBackend",
            "TorchFlexAttnBackend",
            "TorchFlexAttnBackendV2",
            "TRTLLMHAAttnBackend",
            "TRTLLMMLABackend",
        }
        build_custom_mask = not skip_custom_mask
        force_custom_mask_paged = (os.environ.get("SGLANG_DFLASH_FORCE_CUSTOM_MASK_PAGED") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if force_custom_mask_paged and int(self.page_size) > 1:
            # Diagnostic bring-up: some paged-KV TARGET_VERIFY kernels require an explicit
            # allow-mask to enforce correct intra-block causality. This mask can be large
            # (prefix_len * block_size), so keep it behind an env flag.
            build_custom_mask = True
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _prepare_for_speculative_decoding(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch], draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        # Async SSD overlap now uses a shadow req_to_token row plus reserved
        # scratch KV slots, so it no longer needs to fence the target verify
        # critical section on the shared allocator.
        self._maybe_finalize_ssd_overlap(block=False)

        if batch.has_grammar:
            raise ValueError(
                "DFLASH does not support grammar-constrained decoding yet."
            )

        if hasattr(batch, "maybe_evict_swa"):
            batch.maybe_evict_swa()

        bs = batch.batch_size()

        # --- 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        proposal = self._maybe_consume_ssd_cached_proposal(batch, draft_input)
        if (
            proposal is None
            and self._ssd_enabled
            and self._ssd_async_overlap
            and int(self._ssd_hit_wait_ms) > 0
        ):
            fut = self._ssd_overlap_future
            if fut is not None:
                try:
                    self._ssd_overlap_wait_ct += 1
                    fut.result(timeout=float(self._ssd_hit_wait_ms) / 1000.0)
                except FutureTimeoutError:
                    pass
                except Exception as e:
                    self._ssd_prepare_failures += 1
                    if self.tp_rank == 0 and not getattr(self, "_logged_ssd_async_failure", False):
                        logger.warning(
                            "DFLASH SSD async overlap task failed during hit-wait; falling back to sequential path: %s",
                            e,
                        )
                        setattr(self, "_logged_ssd_async_failure", True)
                finally:
                    if fut.done():
                        self._ssd_overlap_future = None
                proposal = self._maybe_consume_ssd_cached_proposal(batch, draft_input)
        # Survival-weighted scheduling (FailFast/SSD-style): if recent accept lengths are low,
        # avoid spending extra pq bookkeeping and fall back to target_only until we recover.
        verify_mode = str(
            getattr(self.server_args, "speculative_dflash_verify_mode", "target_only")
            or "target_only"
        )
        surv_flag = (os.environ.get("SGLANG_DFLASH_SURVIVAL_FAILFAST") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if surv_flag and verify_mode == "pq":
            try:
                thr = float(os.environ.get("SGLANG_DFLASH_SURVIVAL_ACCEPT_EMA_LE") or "1.0")
            except Exception:
                thr = 1.0
            states = []
            for r in batch.reqs:
                st = getattr(r, "dflash_difficulty_state", None)
                if st is not None:
                    states.append(st)
            if survival_should_force_target_only(states, accept_ema_le=float(thr)):
                    if self.tp_rank == 0 and not getattr(self, "_logged_survival_failfast", False):
                        logger.info(
                            "DFLASH survival failfast: forcing target_only (accept_len_ema <= %.4f)",
                            float(thr),
                        )
                        setattr(self, "_logged_survival_failfast", True)
                    verify_mode = "target_only"
        if proposal is None:
            proposal = self._build_draft_proposal(
                batch,
                draft_input,
                prefix_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
            )
            if verify_mode == "target_only" and proposal.verify_mode == "pq":
                proposal.verify_mode = "target_only"
        else:
            if verify_mode == "target_only" and proposal.verify_mode == "pq":
                proposal.verify_mode = "target_only"

        if (
            proposal.verify_mode == "pq"
            and self.tp_rank == 0
            and not getattr(self, "_logged_pq_draft_gate", False)
        ):
            logger.info(
                "DFLASH pq draft gate active. cache_hit=%s verify_mode=%s",
                bool(getattr(self, "_logged_ssd_hit", False)),
                proposal.verify_mode,
            )
            setattr(self, "_logged_pq_draft_gate", True)

        if (
            proposal.draft_conf_debug is not None
            and self.tp_rank == 0
            and not getattr(self, "_logged_draft_conf_debug", False)
        ):
            logger.info("DFLASH draft confidence (debug): %s", proposal.draft_conf_debug)
            setattr(self, "_logged_draft_conf_debug", True)

        self._apply_draft_proposal_to_batch(batch, proposal)

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
        return_conf_debug: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=hidden_states.device),
                None,
            )

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )
        conf_budget = 0
        if return_conf_debug:
            try:
                conf_budget = int(
                    (os.environ.get("SGLANG_DFLASH_DRAFT_CONF_TOKENS") or "4").strip()
                )
            except Exception:
                conf_budget = 4
            if conf_budget <= 0:
                conf_budget = 4
            conf_budget = min(conf_budget, num_tokens)

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        q_max_sum = 0.0
        q_ent_sum = 0.0
        q_max_first_vals: list[torch.Tensor] = []
        q_ent_first_vals: list[torch.Tensor] = []
        conf_token_ct = 0

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    max_idx = torch.argmax(base_logits, dim=-1)
                    out_token_ids[start:end] = max_idx.to(torch.long) + org_vocab_start
                    if return_conf_debug and conf_token_ct < conf_budget:
                        take = min(conf_budget - conf_token_ct, int(base_logits.shape[0]))
                        logits_f = base_logits[:take].to(torch.float32)
                        max_idx_f = max_idx[:take].to(torch.int64)
                        max_vals = logits_f.gather(1, max_idx_f.view(-1, 1)).view(-1)
                        log_z = torch.logsumexp(logits_f, dim=-1)
                        log_probs = torch.log_softmax(logits_f, dim=-1)
                        probs = torch.exp(log_probs)
                        q_max = torch.exp(max_vals - log_z)
                        q_ent = -(probs * log_probs).sum(dim=-1)
                        q_max_sum += float(q_max.sum().item())
                        q_ent_sum += float(q_ent.sum().item())
                        conf_token_ct += int(q_max.numel())
                        q_max_first_vals.append(q_max.detach().cpu())
                        q_ent_first_vals.append(q_ent.detach().cpu())
                else:
                    out_token_ids[start:end] = 0
            conf_debug = None
            if return_conf_debug and conf_token_ct > 0:
                q_max_first = torch.cat(q_max_first_vals, dim=0)
                q_ent_first = torch.cat(q_ent_first_vals, dim=0)
                first = min(int(q_max_first.numel()), 4)
                conf_debug = {
                    "q_max_mean": q_max_sum / float(conf_token_ct),
                    "q_entropy_mean": q_ent_sum / float(conf_token_ct),
                    "q_max_mean_first": float(q_max_first[:first].mean().item()),
                    "q_max_min_first": float(q_max_first[:first].min().item()),
                    "q_ent_mean_first": float(q_ent_first[:first].mean().item()),
                }
            return out_token_ids, conf_debug

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids, None

    def _project_and_write_target_hidden_to_draft_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

    def _append_verified_hidden_from_cache_plan(
        self,
        *,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        draft_input: DFlashDraftInput,
        verify_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_plan,
        commit_lens: torch.Tensor,
        draft_token_num: int,
    ) -> bool:
        if not bool(getattr(self, "_dflash_draft_share_pools", True)):
            return False
        apply_dflash_shared_pool_verify_append(
            draft_input=draft_input,
            verify_positions=verify_positions,
            hidden_states=hidden_states,
            cache_plan=cache_plan,
            commit_lens=commit_lens,
            write_selected_hidden=self._project_and_write_verified_hidden_selected_to_draft_kv,
        )
        return True

    def _project_and_write_verified_hidden_selected_to_draft_kv(
        self,
        *,
        hidden_states: torch.Tensor,
        accepted_indices: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Shared-pool fast path seam for accepted verify tokens.

        This is the current replacement point for a future DFlash-owned mixed-precision
        fused verify append path: selected target hidden -> draft projection -> BF16 KV write.
        """
        with torch.inference_mode():
            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_verified_hidden_selected_fused(
                        hidden_states=hidden_states,
                        accepted_indices=accepted_indices,
                        ctx_positions=ctx_positions,
                        ctx_cache_loc=ctx_cache_loc,
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH fused selected verify append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                if self._use_fused_kv_materialize:
                    return

            ctx_hidden = self.draft_model.project_target_hidden_selected(
                hidden_states, accepted_indices
            )
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    "DFLASH selected verify hidden/cache_loc mismatch: "
                    f"{ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )
            self._append_target_hidden_sequential(
                ctx_hidden, ctx_positions, ctx_cache_loc
            )

    def _append_verified_hidden_selected_fused(
        self,
        *,
        hidden_states: torch.Tensor,
        accepted_indices: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        if self._fused_kv_helper is None:
            raise RuntimeError("DFLASH fused selected verify append requires fused helper.")

        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
            )

        self._fused_kv_helper.materialize_from_target_hidden_selected(
            target_hidden=hidden_states,
            accepted_indices=accepted_indices,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _append_target_hidden_to_draft_kv(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        validate_kv = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.new_seq_lens = draft_input.draft_seq_lens.clone()
            if validate_kv and self.tp_rank == 0 and not getattr(self, "_logged_validate_kv_skip", False):
                logger.warning(
                    "DFLASH validate_draft_kv: KV append skipped (total_ctx<=0). page_size=%s bs=%s",
                    int(getattr(self.server_args, "page_size", 1) or 1),
                    int(bs),
                )
                setattr(self, "_logged_validate_kv_skip", True)
            return

        if validate_kv and self.tp_rank == 0 and not getattr(self, "_logged_validate_kv_cfg", False):
            logger.info(
                "DFLASH validate_draft_kv enabled (one-shot): page_size=%s bs=%s total_ctx=%s",
                int(getattr(self.server_args, "page_size", 1) or 1),
                int(bs),
                int(total_ctx),
            )
            setattr(self, "_logged_validate_kv_cfg", True)

        with self._draft_worker_lock:
            req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

            req_pool_indices = batch.req_pool_indices
            if req_pool_indices.dtype != torch.int64:
                req_pool_indices = req_pool_indices.to(torch.int64)

            ctx_lens = draft_input.ctx_lens
            draft_seq_lens = draft_input.draft_seq_lens
            if ctx_lens.dtype != torch.int32:
                ctx_lens = ctx_lens.to(torch.int32)
            if draft_seq_lens.dtype != torch.int32:
                draft_seq_lens = draft_seq_lens.to(torch.int32)
            if ctx_lens.device != device:
                ctx_lens = ctx_lens.to(device, non_blocking=True)
            if draft_seq_lens.device != device:
                draft_seq_lens = draft_seq_lens.to(device, non_blocking=True)

            share_pools = bool(getattr(self, "_dflash_draft_share_pools", True))

            if bs == 1:
                # Fast path for single request.
                max_ctx = int(total_ctx)
                if max_ctx <= self._block_pos_offsets.numel():
                    r = self._block_pos_offsets[:max_ctx]
                else:
                    r = torch.arange(max_ctx, device=device, dtype=torch.int64)
                pos2d = draft_seq_lens.to(torch.int64)[:, None] + r[None, :]  # [1, ctx]
                ctx_positions = pos2d.reshape(-1)  # [ctx]
                if share_pools:
                    cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
                    ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
                else:
                    allocator = self.draft_model_runner.token_to_kv_pool_allocator
                    page_size = int(getattr(allocator, "page_size", 1) or 1)
                    if page_size > 1:
                        prefix_lens = draft_seq_lens.to(torch.int32)
                        seq_lens = (draft_seq_lens + ctx_lens).to(torch.int32)
                        prefix_lens_cpu = prefix_lens.to("cpu")
                        seq_lens_cpu = seq_lens.to("cpu")
                        prefix_i64 = prefix_lens.to(torch.int64)
                        last_loc = torch.full(
                            (1,), -1, dtype=torch.int64, device=device
                        )
                        if int(prefix_i64.item()) > 0:
                            last_pos = int(prefix_i64.item()) - 1
                            last_loc[0] = req_to_token[
                                req_pool_indices.reshape(-1)[0],
                                last_pos,
                            ].to(torch.int64)
                        alloc = allocator.alloc_extend(
                            prefix_lens,
                            prefix_lens_cpu,
                            seq_lens,
                            seq_lens_cpu,
                            last_loc,
                            int(total_ctx),
                        )
                    else:
                        alloc = allocator.alloc(int(total_ctx))

                    if alloc is None or int(alloc.numel()) < int(total_ctx):
                        raise RuntimeError(
                            "DFLASH draft KV alloc OOM (non-shared pools): "
                            f"need_tokens={int(total_ctx)} available={int(allocator.available_size())}"
                        )
                    ctx_cache_loc = alloc[: int(total_ctx)].to(torch.int64)
                    req_to_token[
                        req_pool_indices.reshape(-1)[0],
                        ctx_positions,
                    ] = ctx_cache_loc.to(torch.int32)
            else:
                # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
                if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                    max_ctx = int(ctx_lens.max().item())
                else:
                    max_ctx = int(self.block_size)
                if max_ctx <= 0:
                    raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

                if max_ctx <= self._block_pos_offsets.numel():
                    r = self._block_pos_offsets[:max_ctx]
                else:
                    r = torch.arange(max_ctx, device=device, dtype=torch.int64)
                r = r[None, :]  # [1, max_ctx]
                pos2d = draft_seq_lens.to(torch.int64)[:, None] + r  # [bs, max_ctx]
                mask = r < ctx_lens[:, None]

                ctx_positions = pos2d[mask]  # [sum(ctx_lens)]
                if share_pools:
                    # Batched gather of cache locations and positions (shared allocator semantics).
                    cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [bs, max_ctx]
                    ctx_cache_loc = cache2d[mask].to(torch.int64)  # [sum(ctx_lens)]
                else:
                    allocator = self.draft_model_runner.token_to_kv_pool_allocator
                    page_size = int(getattr(allocator, "page_size", 1) or 1)
                    if page_size > 1:
                        prefix_lens = draft_seq_lens.to(torch.int32)
                        seq_lens = (draft_seq_lens + ctx_lens).to(torch.int32)
                        prefix_lens_cpu = prefix_lens.to("cpu")
                        seq_lens_cpu = seq_lens.to("cpu")

                        prefix_i64 = prefix_lens.to(torch.int64)
                        last_loc = torch.full(
                            (bs,), -1, dtype=torch.int64, device=device
                        )
                        has_prefix = prefix_i64 > 0
                        if bool(has_prefix.any()):
                            last_pos = (prefix_i64 - 1).clamp_min(0)
                            last_loc[has_prefix] = req_to_token[
                                req_pool_indices[has_prefix],
                                last_pos[has_prefix],
                            ].to(torch.int64)

                        alloc = allocator.alloc_extend(
                            prefix_lens,
                            prefix_lens_cpu,
                            seq_lens,
                            seq_lens_cpu,
                            last_loc,
                            int(total_ctx),
                        )
                    else:
                        alloc = allocator.alloc(int(total_ctx))

                    if alloc is None or int(alloc.numel()) < int(total_ctx):
                        raise RuntimeError(
                            "DFLASH draft KV alloc OOM (non-shared pools): "
                            f"need_tokens={int(total_ctx)} available={int(allocator.available_size())}"
                        )
                    ctx_cache_loc = alloc[: int(total_ctx)].to(torch.int64)
                    row_ids = req_pool_indices.repeat_interleave(
                        ctx_lens.to(torch.int64), dim=0
                    )
                    req_to_token[row_ids, ctx_positions] = ctx_cache_loc.to(torch.int32)

            self._project_and_write_target_hidden_to_draft_kv(
                target_hidden=draft_input.target_hidden,
                ctx_positions=ctx_positions,
                ctx_cache_loc=ctx_cache_loc,
            )

        draft_input.draft_seq_lens = draft_seq_lens + ctx_lens
        draft_input.new_seq_lens = draft_input.draft_seq_lens.clone()
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        validate_kv = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        validate_once = validate_kv and not getattr(self, "_validated_draft_kv_once", False)
        validate_tokens_env = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV_TOKENS") or "").strip()
        try:
            validate_tokens = int(validate_tokens_env) if validate_tokens_env else 8
        except Exception:
            validate_tokens = 8
        validate_tokens = max(1, min(int(validate_tokens), int(ctx_hidden.shape[0] if ctx_hidden is not None else 0)))

        if validate_once and self.tp_rank == 0:
            logger.info(
                "DFLASH draft_kv_append(sequential) validate_once: ctx_tokens=%s validate_tokens=%s",
                int(ctx_hidden.shape[0]) if ctx_hidden is not None else -1,
                int(validate_tokens),
            )

        for layer in self.draft_model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
            )
            if validate_once:
                try:
                    # Synchronize to ensure KV writes are visible before we read back.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    loc = ctx_cache_loc[:validate_tokens].to(torch.int64)
                    k_buf = self.draft_model_runner.token_to_kv_pool.get_key_buffer(attn.attn.layer_id)
                    v_buf = self.draft_model_runner.token_to_kv_pool.get_value_buffer(attn.attn.layer_id)
                    k_read = k_buf.index_select(0, loc).to(torch.float32)
                    v_read = v_buf.index_select(0, loc).to(torch.float32)
                    k_exp = k[:validate_tokens].to(torch.float32)
                    v_exp = v[:validate_tokens].to(torch.float32)
                    k_diff = (k_read - k_exp).abs()
                    v_diff = (v_read - v_exp).abs()
                    k_max = float(k_diff.max().item()) if k_diff.numel() else 0.0
                    v_max = float(v_diff.max().item()) if v_diff.numel() else 0.0
                    k_mean = float(k_diff.mean().item()) if k_diff.numel() else 0.0
                    v_mean = float(v_diff.mean().item()) if v_diff.numel() else 0.0
                    if self.tp_rank == 0:
                        logger.warning(
                            "DFLASH draft KV validate (layer=%s tokens=%s): k_max=%.6g k_mean=%.6g v_max=%.6g v_mean=%.6g page_size=%s",
                            int(attn.attn.layer_id),
                            int(validate_tokens),
                            k_max,
                            k_mean,
                            v_max,
                            v_mean,
                            int(getattr(self.server_args, "page_size", 1) or 1),
                        )
                        self._draft_kv_validate_stats = {
                            "path": "sequential",
                            "layer_id": int(attn.attn.layer_id),
                            "tokens": int(validate_tokens),
                            "page_size": int(getattr(self.server_args, "page_size", 1) or 1),
                            "k_max": float(k_max),
                            "k_mean": float(k_mean),
                            "v_max": float(v_max),
                            "v_mean": float(v_mean),
                        }
                except Exception as e:
                    if self.tp_rank == 0:
                        logger.warning("DFLASH draft KV validate failed: %s", e)
                        self._draft_kv_validate_stats = {
                            "path": "sequential",
                            "error": str(e),
                            "page_size": int(getattr(self.server_args, "page_size", 1) or 1),
                        }
                setattr(self, "_validated_draft_kv_once", True)
                validate_once = False

    def _build_future_draft_input(
        self,
        draft_input: DFlashDraftInput,
        *,
        verify_done: torch.cuda.Event | None = None,
    ) -> DFlashDraftInput:
        """Build the stable post-append DFLASH draft state for future replay.

        This is the DFLASH-native future payload candidate for spec-v2. It must
        represent the post-append state only: no transient target_hidden and no
        nonzero ctx_lens.
        """
        draft_seq_lens = draft_input.draft_seq_lens
        new_seq_lens = (
            draft_input.new_seq_lens
            if draft_input.new_seq_lens is not None
            else draft_seq_lens
        )
        return DFlashDraftInput(
            verified_id=draft_input.verified_id,
            target_hidden=draft_input.target_hidden,
            ctx_lens=draft_input.ctx_lens,
            draft_seq_lens=draft_seq_lens,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        validate_kv = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        validate_once = validate_kv and not getattr(self, "_validated_draft_kv_once", False)
        validate_tokens_env = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV_TOKENS") or "").strip()
        try:
            validate_tokens = int(validate_tokens_env) if validate_tokens_env else 8
        except Exception:
            validate_tokens = 8
        validate_tokens = max(1, min(int(validate_tokens), int(ctx_hidden.shape[0] if ctx_hidden is not None else 0)))

        if validate_once and self.tp_rank == 0:
            logger.info(
                "DFLASH draft_kv_append(fused) validate_once: ctx_tokens=%s validate_tokens=%s",
                int(ctx_hidden.shape[0]) if ctx_hidden is not None else -1,
                int(validate_tokens),
            )

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
            )

            nonlocal validate_once
            if validate_once and layer_idx == 0:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    loc = ctx_cache_loc[:validate_tokens].to(torch.int64)
                    k_buf = token_to_kv_pool.get_key_buffer(attn.layer_id)
                    v_buf = token_to_kv_pool.get_value_buffer(attn.layer_id)
                    k_read = k_buf.index_select(0, loc).to(torch.float32)
                    v_read = v_buf.index_select(0, loc).to(torch.float32)
                    k_exp = cache_k[:validate_tokens].to(torch.float32)
                    v_exp = cache_v[:validate_tokens].to(torch.float32)
                    k_diff = (k_read - k_exp).abs()
                    v_diff = (v_read - v_exp).abs()
                    k_max = float(k_diff.max().item()) if k_diff.numel() else 0.0
                    v_max = float(v_diff.max().item()) if v_diff.numel() else 0.0
                    k_mean = float(k_diff.mean().item()) if k_diff.numel() else 0.0
                    v_mean = float(v_diff.mean().item()) if v_diff.numel() else 0.0
                    if self.tp_rank == 0:
                        logger.warning(
                            "DFLASH draft KV validate(fused) (layer=%s tokens=%s): k_max=%.6g k_mean=%.6g v_max=%.6g v_mean=%.6g page_size=%s",
                            int(attn.layer_id),
                            int(validate_tokens),
                            k_max,
                            k_mean,
                            v_max,
                            v_mean,
                            int(getattr(self.server_args, "page_size", 1) or 1),
                        )
                        self._draft_kv_validate_stats = {
                            "path": "fused",
                            "layer_id": int(attn.layer_id),
                            "tokens": int(validate_tokens),
                            "page_size": int(getattr(self.server_args, "page_size", 1) or 1),
                            "k_max": float(k_max),
                            "k_mean": float(k_mean),
                            "v_max": float(v_max),
                            "v_mean": float(v_mean),
                        }
                except Exception as e:
                    if self.tp_rank == 0:
                        logger.warning("DFLASH draft KV validate(fused) failed: %s", e)
                        self._draft_kv_validate_stats = {
                            "path": "fused",
                            "error": str(e),
                            "page_size": int(getattr(self.server_args, "page_size", 1) or 1),
                        }
                setattr(self, "_validated_draft_kv_once", True)
                validate_once = False

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def _coerce_model_worker_batch(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        *,
        overlap_v2: bool,
    ) -> ModelWorkerBatch:
        if overlap_v2:
            if not isinstance(batch, ModelWorkerBatch):
                raise TypeError(
                    "DFLASH overlap-v2 expects ModelWorkerBatch at worker entry."
                )
            return batch

        return (
            batch.get_model_worker_batch()
            if isinstance(batch, ScheduleBatch)
            else batch
        )

    def _forward_batch_generation_impl(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        *,
        overlap_v2: bool,
        **kwargs,
    ) -> GenerationBatchResult:
        self._maybe_finalize_ssd_overlap(block=False)
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = self._coerce_model_worker_batch(
                batch, overlap_v2=overlap_v2
            )
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=_to_int32_device_tensor(model_worker_batch.extend_seq_lens),
                draft_seq_lens=_to_int32_device_tensor(
                    model_worker_batch.extend_prefix_lens
                ),
            )
            self._maybe_finalize_ssd_overlap(block=True)
            self._append_target_hidden_to_draft_kv(batch, draft_input)

            # Surface one-shot KV materialization validation stats via response meta_info.
            validate_kv = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV") or "").strip().lower() not in (
                "",
                "0",
                "false",
                "off",
                "no",
            )
            if validate_kv and not getattr(self, "_attached_draft_kv_validate_once", False):
                stats = getattr(self, "_draft_kv_validate_stats", None)
                if stats is not None and logits_output is not None:
                    try:
                        if getattr(logits_output, "customized_info", None) is None:
                            logits_output.customized_info = {}
                        logits_output.customized_info.setdefault(
                            "dflash_draft_kv_validate",
                            [stats for _ in batch.reqs],
                        )
                        setattr(self, "_attached_draft_kv_validate_once", True)
                    except Exception:
                        pass
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=self._build_future_draft_input(draft_input),
                num_accepted_tokens=0,
                spec_ssd_hit_ct=[
                    int(getattr(req, "spec_ssd_hit_ct", 0)) for req in batch.reqs
                ],
                spec_ssd_prepare_ct=[
                    int(getattr(req, "spec_ssd_prepare_ct", 0)) for req in batch.reqs
                ],
                spec_ssd_prepare_failure_ct=[
                    int(getattr(req, "spec_ssd_prepare_failure_ct", 0))
                    for req in batch.reqs
                ],
                spec_ssd_cache_pending=[
                    int(self._get_req_ssd_cache_pending_count(req))
                    for req in batch.reqs
                ],
                spec_ssd_overlap_launch_ct=[
                    int(self._ssd_overlap_launch_ct) for _ in batch.reqs
                ],
                spec_ssd_overlap_wait_ct=[
                    int(self._ssd_overlap_wait_ct) for _ in batch.reqs
                ],
                spec_ssd_difficulty_gate_skip_ct=[
                    int(getattr(req, "spec_ssd_difficulty_gate_skip_ct", 0))
                    for req in batch.reqs
                ],
                spec_ssd_fanout_gate_skip_ct=[
                    int(getattr(req, "spec_ssd_fanout_gate_skip_ct", 0))
                    for req in batch.reqs
                ],
                spec_ssd_fanout_escalation_ct=[
                    int(getattr(req, "spec_ssd_fanout_escalation_ct", 0))
                    for req in batch.reqs
                ],
                spec_ssd_fanout_alt_budget=[
                    int(getattr(req, "spec_ssd_fanout_alt_budget", 0))
                    for req in batch.reqs
                ],
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = self._coerce_model_worker_batch(
            batch, overlap_v2=overlap_v2
        )
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        timing_flag = (os.environ.get("SGLANG_DFLASH_TIMING") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        t0 = time.perf_counter() if timing_flag else 0.0

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        direct_verify_append_enabled = bool(
            getattr(self, "_dflash_draft_share_pools", True)
        ) and (os.environ.get("SGLANG_DFLASH_ENABLE_DIRECT_VERIFY_APPEND") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        verify_result = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            # Keep plain linear DFLASH on the stable staged verify-append path by
            # default. The newer direct cache-plan append is still available behind
            # an explicit env gate for targeted experiments.
            need_target_hidden=not direct_verify_append_enabled,
        )
        new_verified_id = verify_result.new_verified_id
        commit_lens = verify_result.commit_lens
        next_target_hidden = verify_result.next_target_hidden
        accept_length_per_req_cpu = verify_result.accept_length_per_req_cpu
        dflash_debug = verify_result.dflash_debug
        cache_plan = verify_result.cache_plan
        verify_done = None
        if torch.device(self.device).type != "cpu":
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
        append_path = "staged"
        if timing_flag and self.tp_rank == 0 and not getattr(self, "_logged_verify_wall", False):
            dt = time.perf_counter() - t0
            logger.info(
                "DFLASH verify wall timing (one-shot): %.6fs bs=%d accept_len_sum=%d verify_mode=%s",
                float(dt),
                int(batch.batch_size()),
                int(sum(accept_length_per_req_cpu)),
                str(getattr(verify_input, "verify_mode", None)),
            )
            setattr(self, "_logged_verify_wall", True)

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        t1 = time.perf_counter() if timing_flag else 0.0
        appended_from_verify = False
        if direct_verify_append_enabled:
            try:
                appended_from_verify = self._append_verified_hidden_from_cache_plan(
                    batch=batch,
                    draft_input=draft_input,
                    verify_positions=verify_input.positions,
                    hidden_states=logits_output.hidden_states,
                    cache_plan=cache_plan,
                    commit_lens=commit_lens,
                    draft_token_num=int(verify_input.draft_token_num),
                )
            except Exception as e:
                logger.warning(
                    "DFLASH direct verify append failed; falling back to staged path: %s",
                    e,
                )

        if not appended_from_verify:
            if next_target_hidden is None:
                next_target_hidden = gather_dflash_committed_hidden(
                    hidden_states=logits_output.hidden_states,
                    keep_mask=cache_plan.keep_mask,
                    draft_token_num=int(verify_input.draft_token_num),
                    accepted_indices=cache_plan.accepted_indices,
                )
            draft_input.target_hidden = next_target_hidden
            draft_input.ctx_lens = commit_lens
            self._append_target_hidden_to_draft_kv(batch, draft_input)
        append_path = resolve_dflash_verify_append_path(
            appended_from_verify=appended_from_verify,
            fused_helper_active=bool(
                self._use_fused_kv_materialize and self._fused_kv_helper is not None
            ),
        )

        self._update_req_dflash_debug_stats(
            batch=batch,
            verify_input=verify_input,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            dflash_debug=dflash_debug,
            append_path=append_path,
        )

        logits_output.hidden_states = None
        if timing_flag and self.tp_rank == 0 and not getattr(self, "_logged_append_wall", False):
            dt_append = time.perf_counter() - t1
            logger.info(
                "DFLASH draft_kv_append wall timing (one-shot): %.6fs bs=%d ctx_tokens=%d",
                float(dt_append),
                int(batch.batch_size()),
                int(commit_lens.sum().item()) if int(commit_lens.numel()) > 0 else 0,
            )
            setattr(self, "_logged_append_wall", True)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        # Surface one-shot KV validation stats via response meta_info (decode stage too).
        validate_kv = (os.environ.get("SGLANG_DFLASH_VALIDATE_DRAFT_KV") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if validate_kv and not getattr(self, "_attached_draft_kv_validate_once", False):
            stats = getattr(self, "_draft_kv_validate_stats", None)
            if stats is not None and logits_output is not None:
                try:
                    if getattr(logits_output, "customized_info", None) is None:
                        logits_output.customized_info = {}
                    logits_output.customized_info.setdefault(
                        "dflash_draft_kv_validate",
                        [stats for _ in batch.reqs],
                    )
                    setattr(self, "_attached_draft_kv_validate_once", True)
                except Exception:
                    pass

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        self._verify_step += 1
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        log_every = int((os.environ.get("SGLANG_DFLASH_LOG_EVERY") or "0").strip() or 0)
        if log_every > 0 and self.tp_rank == 0 and (self._verify_step % log_every) == 0:
            try:
                mean_acc = float(num_accepted_tokens) / float(max(1, len(accept_length_per_req_cpu)))
            except Exception:
                mean_acc = -1.0
            logger.info(
                "DFLASH verify stats: step=%s bs=%s accept_mean=%.3f accept_sum=%s commit_lens=%s",
                self._verify_step,
                batch.batch_size(),
                mean_acc,
                num_accepted_tokens,
                commit_lens.detach().to("cpu", non_blocking=True).tolist()
                if hasattr(commit_lens, "detach")
                else commit_lens,
            )

        trace_first_steps = int(
            (os.environ.get("SGLANG_DFLASH_TRACE_FIRST_STEPS") or "0").strip() or 0
        )
        if (
            trace_first_steps > 0
            and self.tp_rank == 0
            and int(self._verify_step) <= int(trace_first_steps)
        ):
            try:
                max_steps_trace = None
                if getattr(verify_input, "max_steps_per_req", None) is not None:
                    max_steps_trace = (
                        verify_input.max_steps_per_req.detach()
                        .to("cpu", non_blocking=False)
                        .to(torch.int64)
                        .tolist()
                    )
                req0 = batch.reqs[0] if batch.reqs else None
                req0_state = getattr(req0, "dflash_difficulty_state", None) if req0 is not None else None
                logger.info(
                    "DFLASH trace: step=%s block=%s verify_mode=%s accept=%s commit=%s max_steps=%s verified=%s seq_lens=%s req0_accept_ema=%s req0_q_ent=%s req0_q_max=%s",
                    int(self._verify_step),
                    int(self.block_size),
                    str(getattr(verify_input, "verify_mode", None)),
                    [int(x) for x in accept_length_per_req_cpu],
                    commit_lens.detach().to("cpu", non_blocking=False).tolist()
                    if hasattr(commit_lens, "detach")
                    else commit_lens,
                    max_steps_trace,
                    new_verified_id.detach()
                    .to("cpu", non_blocking=False)
                    .to(torch.int64)
                    .tolist(),
                    batch.seq_lens.detach().to("cpu", non_blocking=False).tolist()
                    if hasattr(batch.seq_lens, "detach")
                    else None,
                    (
                        float(getattr(req0_state, "accept_len_ema", 0.0))
                        if req0_state is not None
                        else None
                    ),
                    (
                        getattr(req0_state, "q_entropy_mean_last", None)
                        if req0_state is not None
                        else None
                    ),
                    (
                        getattr(req0_state, "q_max_mean_last", None)
                        if req0_state is not None
                        else None
                    ),
                )
            except Exception as e:
                logger.warning("DFLASH trace logging failed: %s", e)

        if self._ssd_enabled and self._ssd_prepare_next:
            try:
                # DFlash verify mutates batch.seq_lens / batch.seq_lens_cpu in place.
                # Use the already-updated current sequence lengths as the prefix for
                # the next speculative round; adding commit_lens again would skip a
                # block and poison SSD cache keys/proposals.
                next_prefix_lens = batch.seq_lens.to(torch.int32)
                next_seq_lens_cpu = batch.seq_lens_cpu.to(torch.int32)
                self._schedule_ssd_fanout_overlap(
                    batch=batch,
                    logits_output=logits_output,
                    commit_lens=commit_lens,
                    new_verified_id=new_verified_id,
                    next_prefix_lens=next_prefix_lens,
                    next_seq_lens_cpu=next_seq_lens_cpu,
                )
            except Exception as e:
                self._ssd_prepare_failures += 1
                for req in batch.reqs:
                    req.spec_ssd_prepare_failure_ct = int(
                        getattr(req, "spec_ssd_prepare_failure_ct", 0)
                    ) + 1
                if self.tp_rank == 0 and not getattr(self, "_logged_ssd_prepare_failure", False):
                    logger.warning("DFLASH SSD eager-next prepare failed; falling back to sequential path: %s", e)
                    setattr(self, "_logged_ssd_prepare_failure", True)

        if self._ssd_enabled and self.tp_rank == 0 and not getattr(self, "_logged_ssd_stats", False):
            logger.info(
                "DFLASH SSD stats (first): batch_hits=%d batch_misses=%d req_hits=%d req_misses=%d prepare_failures=%d",
                int(self._ssd_batch_hits),
                int(self._ssd_batch_misses),
                int(self._ssd_req_hits),
                int(self._ssd_req_misses),
                int(self._ssd_prepare_failures),
            )
            setattr(self, "_logged_ssd_stats", True)

        return GenerationBatchResult(
            logits_output=logits_output,
            # On the overlap-preprocessed decode path, verify already committed
            # request/output state and the scheduler consumes only the copied logits
            # + customized info. The compact verified-token payload is not read again.
            next_token_ids=None if overlap_v2 else new_verified_id,
            next_draft_input=self._build_future_draft_input(
                draft_input, verify_done=verify_done
            ),
            next_out_cache_loc=batch.out_cache_loc if overlap_v2 else None,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            accept_lens=None,
            dflash_overlap_preprocessed=overlap_v2,
            requires_output_processing_barrier=overlap_v2,
            spec_ssd_hit_ct=[
                int(getattr(req, "spec_ssd_hit_ct", 0)) for req in batch.reqs
            ],
            spec_ssd_prepare_ct=[
                int(getattr(req, "spec_ssd_prepare_ct", 0)) for req in batch.reqs
            ],
            spec_ssd_prepare_failure_ct=[
                int(getattr(req, "spec_ssd_prepare_failure_ct", 0)) for req in batch.reqs
            ],
            spec_ssd_cache_pending=[
                int(self._get_req_ssd_cache_pending_count(req))
                for req in batch.reqs
            ],
            spec_ssd_overlap_launch_ct=[
                int(self._ssd_overlap_launch_ct) for _ in batch.reqs
            ],
            spec_ssd_overlap_wait_ct=[
                int(self._ssd_overlap_wait_ct) for _ in batch.reqs
            ],
            spec_ssd_difficulty_gate_skip_ct=[
                int(getattr(req, "spec_ssd_difficulty_gate_skip_ct", 0))
                for req in batch.reqs
            ],
            spec_ssd_fanout_gate_skip_ct=[
                int(getattr(req, "spec_ssd_fanout_gate_skip_ct", 0))
                for req in batch.reqs
            ],
            spec_ssd_fanout_escalation_ct=[
                int(getattr(req, "spec_ssd_fanout_escalation_ct", 0))
                for req in batch.reqs
            ],
            spec_ssd_fanout_alt_budget=[
                int(getattr(req, "spec_ssd_fanout_alt_budget", 0))
                for req in batch.reqs
            ],
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        return self._forward_batch_generation_impl(
            batch,
            overlap_v2=isinstance(batch, ModelWorkerBatch),
            **kwargs,
        )
