import contextlib
import logging
import os
import math
import copy
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_tree_draft_cuda_graph_runner import (
    DFlashTreeDraftCudaGraphRunner,
)
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_shared_pool_verify_append,
    apply_dflash_commit_mapping_updates,
    apply_dflash_indexed_cache_plan,
    apply_dflash_target_only_req_kv_accounting,
    build_dflash_tree_candidates_from_per_step_topk,
    build_dflash_indexed_cache_plan,
    can_dflash_use_fused_qkv_proj,
    commit_dflash_proposed_tokens_to_req,
    commit_dflash_target_only_batch,
    gather_dflash_committed_hidden,
    materialize_dflash_target_only_commit_metadata,
    pack_dflash_indexed_commits,
    resolve_dflash_verify_append_path,
    resolve_dflash_mask_token,
    resolve_dflash_mask_token_id,
    resolve_dflash_indexed_accept_indices,
    sample_dflash_tree_branch_candidates_from_support,
    snapshot_dflash_request_sampling_params,
    update_dflash_req_verify_bookkeeping,
    verify_dflash_tree_greedy_fallback,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
    verify_tree_greedy_func,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.sampling_utils import min_p_renorm_prob
from sglang.srt.speculative.spec_utils import (
    TREE_SPEC_KERNEL_AVAILABLE,
    assign_req_to_token_pool_func,
    select_top_k_tokens,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


def _get_plan_stream(
    device: str,
):
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    return None, contextlib.nullcontext()


class DFlashTreeWorker:
    """DFlash speculative decoding worker with fused-tree verification (spec-v1, tp>=1/pp=1).

    Drafting runs the DFlash non-causal block model once per iteration, then builds a bounded
    top-k tree over the next `speculative_num_steps` positions and verifies it using SGLang's
    tree speculative kernel (EAGLE-style) to reduce verify overhead at higher concurrency.
    """

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
        **_unused_kwargs,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        self._warned_forced_greedy = False
        self._warned_draft_sampling_fallback = False
        self._logged_first_verify = False
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # DFLASH_TREE config (validated in ServerArgs).
        self.block_size = int(server_args.speculative_dflash_block_size or 16)
        self.spec_steps = int(
            server_args.speculative_num_steps or (self.block_size - 1)
        )
        self.topk = int(server_args.speculative_eagle_topk or 4)
        self.num_verify_tokens = int(
            server_args.speculative_num_draft_tokens or self.block_size
        )
        self._draft_branch_mode = (
            (os.environ.get("SGLANG_DFLASH_TREE_DRAFT_BRANCH_MODE") or "topk")
            .strip()
            .lower()
        )
        if self._draft_branch_mode not in ("topk", "sample"):
            raise ValueError(
                "Invalid SGLANG_DFLASH_TREE_DRAFT_BRANCH_MODE. "
                f"Expected one of {{'topk','sample'}}, got {self._draft_branch_mode!r}."
            )
        self._disable_draft_tree_fastpath = (
            (os.environ.get("SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._force_explicit_input_embeds = (
            (os.environ.get("SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._use_legacy_block_alloc = (
            (os.environ.get("SGLANG_DFLASH_TREE_USE_LEGACY_BLOCK_ALLOC") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._use_legacy_verify_commit_path = (
            (os.environ.get("SGLANG_DFLASH_TREE_USE_LEGACY_VERIFY_COMMIT") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._use_legacy_tree_builder = (
            (os.environ.get("SGLANG_DFLASH_TREE_USE_LEGACY_TREE_BUILDER") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._use_safe_greedy_verify = (
            (os.environ.get("SGLANG_DFLASH_TREE_USE_SAFE_GREEDY_VERIFY") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._safe_greedy_from_hidden = (
            (os.environ.get("SGLANG_DFLASH_TREE_SAFE_GREEDY_FROM_HIDDEN") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self._assert_builder_equiv = (
            (os.environ.get("SGLANG_DFLASH_TREE_ASSERT_BUILDER_EQUIV") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )

        # Draft runner (separate KV cache + attention backend).
        # Share req_to_token_pool + token_to_kv_pool_allocator with the target worker (EAGLE3-style),
        # while keeping a separate draft KV cache pool (the draft model has different KV values).
        shared_req_to_token_pool, shared_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.req_to_token_pool = shared_req_to_token_pool
        self.token_to_kv_pool_allocator = shared_token_to_kv_pool_allocator
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        if draft_server_args.speculative_draft_kv_cache_dtype is not None:
            draft_server_args.kv_cache_dtype = (
                draft_server_args.speculative_draft_kv_cache_dtype
            )
        if draft_server_args.speculative_draft_mem_fraction_static is not None:
            draft_server_args.mem_fraction_static = (
                draft_server_args.speculative_draft_mem_fraction_static
            )
        if draft_server_args.speculative_draft_page_size is not None:
            draft_server_args.page_size = draft_server_args.speculative_draft_page_size
        draft_backend = draft_server_args.speculative_draft_attention_backend
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH_TREE draft worker does not support 'trtllm_mha' yet; "
                "falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"
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
                "DFLASH_TREE draft worker only supports attention_backend in {'flashinfer', 'fa3', 'fa4', 'flex_attention', 'flex_attention2', 'flex_flash', 'flex_flash2', 'flex_flash2_delegate_fa3', 'flex_flash4'} for now, "
                "but got %r. Falling back to 'flashinfer'.",
                draft_backend,
            )
            draft_backend = "flashinfer"

        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        # IMPORTANT: for DFLASH_TREE we overload `speculative_num_draft_tokens` to
        # mean "verify-node budget" (num_verify_tokens). The draft worker,
        # however, always runs a fixed-size `block_size` forward (same as DFLASH),
        # and SGLang's cuda-graph capture for DFlash relies on
        # `speculative_num_draft_tokens == block_size` to size buffers + build
        # DFlashVerifyInput correctly. So configure the *draft worker* with DFLASH
        # semantics to keep graph capture safe and shape-stable.
        draft_server_args.speculative_algorithm = "DFLASH"
        draft_server_args.speculative_num_draft_tokens = int(self.block_size)
        draft_server_args.speculative_num_steps = 1
        draft_server_args.speculative_eagle_topk = 1

        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
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

        self._mask_token = resolve_dflash_mask_token(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        self._mask_token_id_override = resolve_dflash_mask_token_id(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH_TREE draft runner. attention_backend=%s, model=%s, block_size=%s, spec_steps=%s, topk=%s, num_verify_tokens=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.spec_steps,
                self.topk,
                self.num_verify_tokens,
            )
            logger.info(
                "DFLASH_TREE draft branch mode: %s",
                self._draft_branch_mode,
            )
            if self._force_explicit_input_embeds:
                logger.info(
                    "DFLASH_TREE forcing legacy explicit input_embeds path via SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1"
                )
            if self._use_legacy_block_alloc:
                logger.info(
                    "DFLASH_TREE forcing legacy per-round draft block alloc/restore via SGLANG_DFLASH_TREE_USE_LEGACY_BLOCK_ALLOC=1"
                )
            if self._use_legacy_verify_commit_path:
                logger.info(
                    "DFLASH_TREE forcing legacy non-overlap verify commit path via SGLANG_DFLASH_TREE_USE_LEGACY_VERIFY_COMMIT=1"
                )
            if self._use_legacy_tree_builder:
                logger.info(
                    "DFLASH_TREE forcing legacy tree builder path via SGLANG_DFLASH_TREE_USE_LEGACY_TREE_BUILDER=1"
                )
            if self._disable_draft_tree_fastpath:
                logger.info(
                    "DFLASH_TREE dedicated draft fast path disabled via SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH=1"
                )
            if self._use_safe_greedy_verify:
                logger.info(
                    "DFLASH_TREE forcing safe greedy verify fallback via SGLANG_DFLASH_TREE_USE_SAFE_GREEDY_VERIFY=1"
                )
            if self._safe_greedy_from_hidden:
                logger.info(
                    "DFLASH_TREE safe greedy verify will derive target argmax from "
                    "hidden_states + lm_head via SGLANG_DFLASH_TREE_SAFE_GREEDY_FROM_HIDDEN=1"
                )
            if self._assert_builder_equiv:
                logger.info(
                    "DFLASH_TREE asserting optimized tree-builder equivalence against legacy builder via SGLANG_DFLASH_TREE_ASSERT_BUILDER_EQUIV=1"
                )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_cache_loc_scratch: Optional[torch.Tensor] = None
        self._draft_block_cache_loc_scratch_cap: int = 0
        self._tree_candidate_scores_buf: Optional[torch.Tensor] = None
        self._tree_candidate_tokens_buf: Optional[torch.Tensor] = None
        self._tree_parent_list_buf: Optional[torch.Tensor] = None
        self._tree_top_scores_index_buf: Optional[torch.Tensor] = None
        self._draft_tree_cuda_graph_runner = None
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        if (
            is_cuda()
            and not bool(self.server_args.disable_cuda_graph)
            and int(getattr(self.draft_model_runner, "tp_size", 1)) == 1
            and int(getattr(self.draft_model_runner, "dp_size", 1)) == 1
            and self._draft_branch_mode == "topk"
            and not self._disable_draft_tree_fastpath
        ):
            try:
                self._draft_tree_cuda_graph_runner = DFlashTreeDraftCudaGraphRunner(
                    self
                )
                if self.tp_rank == 0:
                    logger.info(
                        "Initialized DFLASH_TREE dedicated draft cuda graph runner. block_size=%s spec_steps=%s topk=%s",
                        self.block_size,
                        self.spec_steps,
                        self.topk,
                    )
            except Exception as e:
                logger.warning(
                    "DFLASH_TREE dedicated draft cuda graph runner init failed; falling back to generic draft replay: %s",
                    e,
                )
                self._draft_tree_cuda_graph_runner = None

        self._use_fused_kv_materialize = is_cuda()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
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

            if fused_disable_reason is not None:
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            helper_cls = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            self._fused_kv_helper = helper_cls(
                layers=layers,
                rotary_emb=first_attn.rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                target_fc_weight=self.draft_model.fc.weight,
                target_hidden_norm_weight=self.draft_model.hidden_norm.weight,
                target_hidden_norm_eps=self.draft_model.hidden_norm.variance_epsilon,
            )
        except Exception as e:
            logger.warning(
                "DFLASH_TREE fused KV initialization failed, falling back to sequential path: %s",
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
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def _ensure_draft_block_cache_loc_scratch(self, bs: int) -> None:
        """Reserve reusable KV slots for transient tree draft-block forwards."""
        bs = int(bs)
        if (
            self._draft_block_cache_loc_scratch is not None
            and int(self._draft_block_cache_loc_scratch_cap) >= bs
        ):
            return

        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        page_size = int(getattr(allocator, "page_size", 1))
        block_size = int(self.block_size)
        reserved_env_key = "SGLANG_DFLASH_SSD_RESERVED_TOKENS"

        if self._draft_block_cache_loc_scratch is not None:
            try:
                allocator.free(self._draft_block_cache_loc_scratch.reshape(-1))
            finally:
                self._draft_block_cache_loc_scratch = None
                self._draft_block_cache_loc_scratch_cap = 0

        if page_size == 1:
            need_tokens = int(bs * block_size)
            loc = allocator.alloc(need_tokens)
            if loc is None or int(loc.numel()) < need_tokens:
                raise RuntimeError(
                    "DFLASH_TREE draft scratch allocation failed: "
                    f"need_tokens={need_tokens} page_size={page_size}"
                )
            scratch = loc[:need_tokens].to(torch.int64).view(bs, block_size)
            try:
                prev = int((os.environ.get(reserved_env_key) or "0").strip() or 0)
            except Exception:
                prev = 0
            os.environ[reserved_env_key] = str(max(prev, need_tokens))
        else:
            scratch_width = 2 * page_size
            need_tokens_alloc = int(bs * scratch_width)
            fake_prefix = torch.zeros((1,), dtype=torch.int32, device=self.device)
            fake_prefix_cpu = torch.zeros((1,), dtype=torch.int32, device="cpu")
            fake_seq = torch.tensor(
                [need_tokens_alloc], dtype=torch.int32, device=self.device
            )
            fake_seq_cpu = torch.tensor(
                [need_tokens_alloc], dtype=torch.int32, device="cpu"
            )
            fake_last_loc = torch.full(
                (1,), -1, dtype=torch.int64, device=self.device
            )
            loc = allocator.alloc_extend(
                fake_prefix,
                fake_prefix_cpu,
                fake_seq,
                fake_seq_cpu,
                fake_last_loc,
                need_tokens_alloc,
            )
            if loc is None or int(loc.numel()) < need_tokens_alloc:
                raise RuntimeError(
                    "DFLASH_TREE draft scratch allocation failed: "
                    f"need_tokens_alloc={need_tokens_alloc} page_size={page_size}"
                )
            scratch = loc[:need_tokens_alloc].to(torch.int64).view(bs, scratch_width)
            try:
                prev = int((os.environ.get(reserved_env_key) or "0").strip() or 0)
            except Exception:
                prev = 0
            os.environ[reserved_env_key] = str(max(prev, need_tokens_alloc))

            if os.environ.get("SGLANG_DFLASH_DEBUG_SCRATCH", "").strip():
                assert torch.all((scratch[:, 0] % page_size) == 0), (
                    "DFLASH_TREE scratch must be page-aligned per request when page_size>1. "
                    f"page_size={page_size}"
                )

        self._draft_block_cache_loc_scratch = scratch
        self._draft_block_cache_loc_scratch_cap = bs

    def _ensure_tree_candidate_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._tree_candidate_scores_buf is None
            else int(self._tree_candidate_scores_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        candidate_count = int(self.topk) + max(0, int(self.spec_steps) - 1) * (
            int(self.topk) ** 2
        )
        parent_count = (
            0
            if int(self.spec_steps) == 1
            else (int(self.topk) + 1) + max(0, int(self.spec_steps) - 2) * int(self.topk)
        )
        verify_token_count = max(0, int(self.num_verify_tokens) - 1)

        self._tree_candidate_scores_buf = torch.empty(
            (new_cap, candidate_count), dtype=torch.float32, device=self.device
        )
        self._tree_candidate_tokens_buf = torch.empty(
            (new_cap, candidate_count), dtype=torch.int64, device=self.device
        )
        self._tree_parent_list_buf = torch.empty(
            (new_cap, parent_count), dtype=torch.long, device=self.device
        )
        self._tree_top_scores_index_buf = torch.empty(
            (new_cap, verify_token_count), dtype=torch.long, device=self.device
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # allocator and req_to_token_pool are shared with target worker
        pass

    def on_req_finished(self, req):
        if hasattr(req, "dflash_draft_seq_len"):
            req.dflash_draft_seq_len = 0

    def _coerce_model_worker_batch(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        *,
        overlap_v2: bool,
    ) -> ModelWorkerBatch:
        if overlap_v2:
            if not isinstance(batch, ModelWorkerBatch):
                raise TypeError(
                    "DFLASH_TREE overlap-v2 expects ModelWorkerBatch at worker entry."
                )
            return batch

        return (
            batch.get_model_worker_batch()
            if isinstance(batch, ScheduleBatch)
            else batch
        )

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH_TREE mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH_TREE mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH_TREE config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH_TREE requires tokenizer initialization when dflash_config.mask_token_id is not set."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)
        if resolved_id is None:
            resolved_id = tokenizer.get_vocab().get(mask_token, None)
        if resolved_id is None:
            tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH_TREE requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )
        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH_TREE mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}."
            )
        return int(resolved_id)

    def _topk_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        topk: int,
        chunk_size: int = 128,
        return_logits: bool = False,
        allow_local_only_fast_path: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k token selection over the target LM head in a TP-safe way.

        Returns:
            topk_p: (num_tokens, topk) float32, normalized over top-k logits unless
                `return_logits=True`, in which case this is the exact top-k logits.
            topk_index: (num_tokens, topk) int64, global token ids.
        """
        if hidden_states.numel() == 0:
            empty_p = torch.empty(
                (0, int(topk)), dtype=torch.float32, device=hidden_states.device
            )
            empty_i = torch.empty(
                (0, int(topk)), dtype=torch.int64, device=hidden_states.device
            )
            return empty_p, empty_i

        if int(topk) <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH_TREE top-k sampling requires a vocab-parallel head with `weight` and `shard_indices`."
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
        out_p = torch.empty(
            (num_tokens, int(topk)), dtype=torch.float32, device=hidden_states.device
        )
        out_ids = torch.empty(
            (num_tokens, int(topk)), dtype=torch.int64, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        min_val = torch.finfo(weight_dtype).min
        local_only_fast_path = (
            allow_local_only_fast_path
            and
            tp_size == 1
            and num_added == 0
            and org_vocab_start == 0
            and num_org > 0
        )

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            if local_only_fast_path:
                logits = torch.matmul(hs, weight[:num_org].T)
                k_local = min(int(topk), int(num_org))
                local_vals, local_ids = torch.topk(logits, k=k_local, dim=-1)
                local_vals = local_vals.to(torch.float32)
                local_ids = local_ids.to(torch.int64)
                if k_local < int(topk):
                    pad = int(topk) - k_local
                    pad_vals = torch.full(
                        (chunk_len, pad),
                        float(min_val),
                        dtype=torch.float32,
                        device=hs.device,
                    )
                    pad_ids = torch.zeros(
                        (chunk_len, pad), dtype=torch.int64, device=hs.device
                    )
                    local_vals = torch.cat([local_vals, pad_vals], dim=1)
                    local_ids = torch.cat([local_ids, pad_ids], dim=1)
                out_p[start:end].copy_(
                    local_vals if return_logits else torch.softmax(local_vals, dim=1)
                )
                out_ids[start:end].copy_(local_ids)
                continue

            local_vals_list: List[torch.Tensor] = []
            local_ids_list: List[torch.Tensor] = []

            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                k_base = min(int(topk), int(num_org))
                base_vals, base_idx = torch.topk(base_logits, k=k_base, dim=-1)
                local_vals_list.append(base_vals)
                local_ids_list.append(base_idx.to(torch.int64) + org_vocab_start)

            if num_added > 0:
                added_weight = weight[num_org_padded : num_org_padded + num_added]
                added_logits = torch.matmul(hs, added_weight.T)
                k_added = min(int(topk), int(num_added))
                added_vals, added_idx = torch.topk(added_logits, k=k_added, dim=-1)
                local_vals_list.append(added_vals)
                local_ids_list.append(added_idx.to(torch.int64) + added_vocab_start)

            if not local_vals_list:
                local_vals = torch.full(
                    (chunk_len, int(topk)),
                    min_val,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_ids = torch.zeros(
                    (chunk_len, int(topk)), dtype=torch.int64, device=hs.device
                )
            else:
                local_vals = torch.cat(local_vals_list, dim=1)
                local_ids = torch.cat(local_ids_list, dim=1)

                # Keep exactly topk candidates per rank.
                if local_vals.shape[1] > int(topk):
                    local_vals, sel = torch.topk(local_vals, k=int(topk), dim=-1)
                    local_ids = local_ids.gather(1, sel.to(torch.int64))
                elif local_vals.shape[1] < int(topk):
                    pad = int(topk) - int(local_vals.shape[1])
                    pad_vals = torch.full(
                        (chunk_len, pad),
                        min_val,
                        dtype=local_vals.dtype,
                        device=hs.device,
                    )
                    pad_ids = torch.zeros(
                        (chunk_len, pad), dtype=torch.int64, device=hs.device
                    )
                    local_vals = torch.cat([local_vals, pad_vals], dim=1)
                    local_ids = torch.cat([local_ids, pad_ids], dim=1)

            # Gather local candidates across TP ranks and select global topk.
            if tp_size > 1:
                gathered_vals = tp_group.all_gather(local_vals.to(torch.float32), dim=1)
                gathered_ids = tp_group.all_gather(local_ids, dim=1)
                global_vals, global_sel = torch.topk(
                    gathered_vals, k=int(topk), dim=1
                )
                global_ids = gathered_ids.gather(1, global_sel.to(torch.int64))
            else:
                global_vals = local_vals.to(torch.float32)
                global_ids = local_ids

            out_p[start:end].copy_(
                global_vals if return_logits else torch.softmax(global_vals, dim=1)
            )
            out_ids[start:end].copy_(global_ids)

        return out_p, out_ids

    def _draft_tree_build_candidates_legacy(
        self,
        *,
        forward_batch: ForwardBatch,
        lm_head,
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            draft_hidden = self.draft_model_runner.forward(forward_batch).logits_output

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        step_count = int(self.spec_steps)
        step_hidden = draft_hidden[:, 1 : 1 + step_count, :].reshape(
            -1, draft_hidden.shape[-1]
        )
        topk_p_flat, topk_index_flat = self._topk_from_vocab_parallel_head(
            hidden_states=step_hidden,
            lm_head=lm_head,
            topk=self.topk,
            allow_local_only_fast_path=False,
        )
        topk_p = topk_p_flat.view(bs, step_count, self.topk)
        topk_index = topk_index_flat.view(bs, step_count, self.topk)

        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        hidden_dummy = torch.empty((0, 0), device=step_hidden.device, dtype=torch.float32)
        for i in range(step_count):
            step_p = topk_p[:, i, :]
            step_ids = topk_index[:, i, :]
            if i == 0:
                _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                    i, step_p, step_ids, hidden_dummy, scores, self.topk
                )
            else:
                step_p_rep = step_p.repeat_interleave(self.topk, dim=0)
                step_ids_rep = step_ids.repeat_interleave(self.topk, dim=0)
                _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                    i, step_p_rep, step_ids_rep, hidden_dummy, scores, self.topk
                )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

        candidate_count = int(self.topk) + max(0, step_count - 1) * (
            int(self.topk) ** 2
        )
        max_verify_tokens = 1 + candidate_count
        if int(self.num_verify_tokens) > max_verify_tokens:
            raise ValueError(
                "DFLASH_TREE speculative_num_draft_tokens is too large for the configured (topk, spec_steps) tree. "
                f"speculative_num_draft_tokens={self.num_verify_tokens}, max_allowed={max_verify_tokens}, "
                f"topk={self.topk}, speculative_num_steps={step_count}."
            )

        return organize_draft_results(
            score_list, token_list, parents_list, self.num_verify_tokens
        )

    def _build_tree_candidates_reference_from_per_step_topk(
        self,
        *,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        hidden_dummy = torch.empty((0, 0), device=topk_p.device, dtype=torch.float32)

        for i in range(int(topk_p.shape[1])):
            step_p = topk_p[:, i, :]
            step_ids = topk_index[:, i, :]
            if i == 0:
                _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                    i, step_p, step_ids, hidden_dummy, scores, self.topk
                )
            else:
                step_p_rep = step_p.repeat_interleave(self.topk, dim=0)
                step_ids_rep = step_ids.repeat_interleave(self.topk, dim=0)
                _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                    i, step_p_rep, step_ids_rep, hidden_dummy, scores, self.topk
                )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

        return organize_draft_results(
            score_list, token_list, parents_list, self.num_verify_tokens
        )

    def _assert_tree_builder_equiv(
        self,
        *,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        parent_list: torch.Tensor,
        top_scores_index: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        ref_parent_list, ref_top_scores_index, ref_draft_tokens = (
            self._build_tree_candidates_reference_from_per_step_topk(
                topk_p=topk_p,
                topk_index=topk_index,
            )
        )
        if not torch.equal(parent_list, ref_parent_list):
            raise AssertionError(
                "DFLASH_TREE optimized parent_list mismatch vs legacy reference. "
                f"optimized.shape={tuple(parent_list.shape)} ref.shape={tuple(ref_parent_list.shape)}"
            )
        if not torch.equal(top_scores_index, ref_top_scores_index):
            raise AssertionError(
                "DFLASH_TREE optimized top_scores_index mismatch vs legacy reference. "
                f"optimized.shape={tuple(top_scores_index.shape)} ref.shape={tuple(ref_top_scores_index.shape)}"
            )
        if not torch.equal(draft_tokens, ref_draft_tokens):
            raise AssertionError(
                "DFLASH_TREE optimized draft_tokens mismatch vs legacy reference. "
                f"optimized.shape={tuple(draft_tokens.shape)} ref.shape={tuple(ref_draft_tokens.shape)}"
            )

    def _draft_tree_build_candidates(
        self,
        *,
        forward_batch: ForwardBatch,
        lm_head,
        bs: int,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._use_legacy_tree_builder:
            return self._draft_tree_build_candidates_legacy(
                forward_batch=forward_batch,
                lm_head=lm_head,
                bs=bs,
            )
        if (
            self._draft_tree_cuda_graph_runner is not None
            and forward_batch.input_embeds is None
            and self._draft_tree_cuda_graph_runner.can_run(forward_batch)
        ):
            return self._draft_tree_cuda_graph_runner.replay(forward_batch)

        with torch.inference_mode():
            draft_hidden = self.draft_model_runner.forward(forward_batch).logits_output

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        step_hidden = draft_hidden[:, 1 : 1 + int(self.spec_steps), :].reshape(
            -1, draft_hidden.shape[-1]
        )
        if self._draft_branch_mode == "sample":
            topk_p, topk_index = self._sample_tree_candidates_from_vocab_parallel_head(
                hidden_states=step_hidden,
                lm_head=lm_head,
                batch=batch,
                bs=bs,
                step_count=int(self.spec_steps),
                topk=self.topk,
            )
        else:
            topk_p_flat, topk_index_flat = self._topk_from_vocab_parallel_head(
                hidden_states=step_hidden,
                lm_head=lm_head,
                topk=self.topk,
            )
            topk_p = topk_p_flat.view(bs, int(self.spec_steps), self.topk)
            topk_index = topk_index_flat.view(bs, int(self.spec_steps), self.topk)

        self._ensure_tree_candidate_buffers(bs)
        assert self._tree_candidate_scores_buf is not None
        assert self._tree_candidate_tokens_buf is not None
        assert self._tree_parent_list_buf is not None
        assert self._tree_top_scores_index_buf is not None
        parent_list, top_scores_index, draft_tokens = (
            build_dflash_tree_candidates_from_per_step_topk(
            topk_p=topk_p,
            topk_index=topk_index,
            num_verify_tokens=int(self.num_verify_tokens),
            candidate_scores_buf=self._tree_candidate_scores_buf[:bs],
            candidate_tokens_buf=self._tree_candidate_tokens_buf[:bs],
            parent_list_buf=self._tree_parent_list_buf[:bs],
            top_scores_index_buf=self._tree_top_scores_index_buf[:bs],
            )
        )
        if self._assert_builder_equiv:
            self._assert_tree_builder_equiv(
                topk_p=topk_p,
                topk_index=topk_index,
                parent_list=parent_list,
                top_scores_index=top_scores_index,
                draft_tokens=draft_tokens,
            )
        return parent_list, top_scores_index, draft_tokens

    def _sample_tree_candidates_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        bs: int,
        step_count: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size != 1:
            if not self._warned_draft_sampling_fallback and self.tp_rank == 0:
                logger.warning(
                    "DFLASH_TREE draft sampled-branch mode currently supports tp_size==1 only; falling back to deterministic top-k draft branches."
                )
                self._warned_draft_sampling_fallback = True
            topk_p_flat, topk_index_flat = self._topk_from_vocab_parallel_head(
                hidden_states=hidden_states,
                lm_head=lm_head,
                topk=topk,
            )
            return (
                topk_p_flat.view(bs, step_count, topk),
                topk_index_flat.view(bs, step_count, topk),
            )

        (
            temperatures,
            top_ps,
            top_ks,
            min_ps,
            need_min_p_sampling,
        ) = snapshot_dflash_request_sampling_params(
            batch.reqs, device=hidden_states.device
        )

        expanded_temperature = torch.repeat_interleave(
            temperatures.view(-1, 1), step_count, dim=0
        )
        expanded_top_ps = torch.repeat_interleave(top_ps, step_count, dim=0)
        support_k = max(int(topk), int(top_ks.max().item()))
        topk_logits, topk_index = self._topk_from_vocab_parallel_head(
            hidden_states=hidden_states,
            lm_head=lm_head,
            topk=support_k,
            return_logits=True,
        )
        expanded_top_ks = torch.repeat_interleave(top_ks, step_count, dim=0).clamp(
            min=1, max=support_k
        )
        expanded_min_ps = torch.repeat_interleave(min_ps, step_count, dim=0)

        probs = F.softmax(
            topk_logits.to(torch.float32) / expanded_temperature.clamp_min(1e-6),
            dim=-1,
        )
        if support_k > 1:
            probs = top_k_renorm_prob(probs, expanded_top_ks)
        if not torch.all(expanded_top_ps == 1.0):
            probs = top_p_renorm_prob(probs, expanded_top_ps)
        if need_min_p_sampling:
            probs = min_p_renorm_prob(probs, expanded_min_ps)

        sampled_probs, sampled_ids = sample_dflash_tree_branch_candidates_from_support(
            probs=probs,
            token_ids=topk_index,
            topk=topk,
            fallback_probs=torch.softmax(topk_logits.to(torch.float32), dim=-1),
        )

        return (
            sampled_probs.view(bs, step_count, topk),
            sampled_ids.view(bs, step_count, topk),
        )

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache."""
        bs = batch.batch_size()
        device = self.model_runner.device

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            draft_input.new_seq_lens = draft_input.draft_seq_lens.clone()
            return

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

        if bs == 1:
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = draft_seq_lens.to(torch.int64)[:, None] + r[None, :]  # [1, ctx]
            cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = draft_seq_lens.to(torch.int64)[:, None] + r  # [bs, max_ctx]
            mask = r < ctx_lens[:, None]
            cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [bs, max_ctx]
            ctx_cache_loc = cache2d[mask].to(torch.int64)  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

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

    def _build_future_draft_input(
        self,
        draft_input: DFlashDraftInput,
        *,
        verify_done: torch.cuda.Event | None = None,
    ) -> DFlashDraftInput:
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

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

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
                    "DFLASH_TREE ctx_hidden/cache_loc mismatch: "
                    f"{ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH_TREE fused KV append failed; falling back to sequential path: %s",
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

    def _append_verified_hidden_selected_fused(
        self,
        *,
        hidden_states: torch.Tensor,
        accepted_indices: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        if self._fused_kv_helper is None:
            raise RuntimeError(
                "DFLASH_TREE fused selected verify append requires fused helper."
            )

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

    def _project_and_write_verified_hidden_selected_to_draft_kv(
        self,
        *,
        hidden_states: torch.Tensor,
        accepted_indices: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
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
                        "DFLASH_TREE fused selected verify append failed; falling back to sequential path: %s",
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
                    "DFLASH_TREE selected verify hidden/cache_loc mismatch: "
                    f"{ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )
            self._append_target_hidden_sequential(
                ctx_hidden, ctx_positions, ctx_cache_loc
            )

    def _append_verified_hidden_from_cache_plan(
        self,
        *,
        draft_input: DFlashDraftInput,
        verify_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_plan,
        commit_lens: torch.Tensor,
    ) -> bool:
        apply_dflash_shared_pool_verify_append(
            draft_input=draft_input,
            verify_positions=verify_positions,
            hidden_states=hidden_states,
            cache_plan=cache_plan,
            commit_lens=commit_lens,
            write_selected_hidden=self._project_and_write_verified_hidden_selected_to_draft_kv,
        )
        return True

    def _prepare_for_speculative_decoding(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        draft_input: DFlashDraftInput,
        *,
        overlap_v2: bool,
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return None, None

        if batch.has_grammar:
            raise ValueError("DFLASH_TREE does not support grammar-constrained decoding yet.")

        bs = batch.batch_size()
        device = self.model_runner.device

        # 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = None
        if self._force_explicit_input_embeds:
            embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH_TREE requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # 2) Draft a non-causal block with the draft model (window = block_size).
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))
        input_embeds = None
        if embed_module is not None:
            input_embeds = embed_module(block_ids).view(bs * self.block_size, -1)

        prefix_lens = batch.seq_lens  # int32, device
        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d)
        positions = positions_2d.reshape(-1)

        block_start = prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, int(self.block_size), out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if batch.seq_lens_cpu.dtype == torch.int32:
            seq_lens_cpu.copy_(batch.seq_lens_cpu)
        else:
            seq_lens_cpu.copy_(batch.seq_lens_cpu.to(torch.int32))

        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        page_size = int(getattr(allocator, "page_size", 1))
        if self._use_legacy_block_alloc:
            token_to_kv_pool_state_backup = allocator.backup_state()
            try:
                block_cache_loc = allocator.alloc(bs * self.block_size)
                if block_cache_loc is None:
                    raise RuntimeError(
                        f"DFLASH_TREE draft OOM when allocating {bs * self.block_size} block tokens."
                    )

                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    self.draft_model_runner.req_to_token_pool.req_to_token,
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
                    seq_lens_sum=int(batch.seq_lens_sum),
                    seq_lens_cpu=seq_lens_cpu,
                    positions=positions,
                    req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                    token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                    attn_backend=self.draft_model_runner.attn_backend,
                    input_embeds=input_embeds,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                    spec_info=self._draft_block_spec_info,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                )

                parent_list, top_scores_index, draft_tokens = (
                    self._draft_tree_build_candidates(
                        forward_batch=forward_batch,
                        lm_head=lm_head,
                        bs=bs,
                        batch=batch,
                    )
                )
            finally:
                allocator.restore_state(token_to_kv_pool_state_backup)
        else:
            self._ensure_draft_block_cache_loc_scratch(bs)
            if self._draft_block_cache_loc_scratch is None:
                raise RuntimeError("DFLASH_TREE draft scratch KV buffer is missing.")
            scratch = self._draft_block_cache_loc_scratch[:bs]
            if page_size == 1:
                block_cache_loc = scratch[:, : self.block_size].reshape(-1)
            else:
                offs = (
                    (prefix_lens.to(torch.int64) % page_size).unsqueeze(1)
                    + self._block_pos_offsets.unsqueeze(0)
                )
                block_cache_loc = torch.gather(scratch, 1, offs).reshape(-1)
                if os.environ.get("SGLANG_DFLASH_DEBUG_SCRATCH", "").strip():
                    pos_mod = (
                        prefix_lens.to(torch.int64).unsqueeze(1)
                        + self._block_pos_offsets.unsqueeze(0)
                    ) % page_size
                    loc_mod = block_cache_loc.view(bs, self.block_size) % page_size
                    assert torch.all(loc_mod == pos_mod), (
                        "DFLASH_TREE scratch loc mod mismatch for paged KV. "
                        f"page_size={page_size}"
                    )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
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
                seq_lens_sum=int(batch.seq_lens_sum),
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            parent_list, top_scores_index, draft_tokens = self._draft_tree_build_candidates(
                forward_batch=forward_batch,
                lm_head=lm_head,
                bs=bs,
                batch=batch,
            )

        step_count = int(self.spec_steps)
        if step_count <= 0 or step_count >= int(self.block_size):
            raise RuntimeError(
                f"DFLASH_TREE invalid spec_steps={step_count} for block_size={self.block_size}."
            )

        # 3) Build a bounded beam-style tree from per-step top-k candidates.
        tree_mask_buf = None
        position_buf = None
        if overlap_v2:
            tree_mask_buf, position_buf = (
                self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            verify_tokens_flat,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id.to(torch.int64),
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            int(batch.seq_lens_sum),
            self.topk,
            step_count,
            self.num_verify_tokens,
            TreeMaskMode.FULL_MASK,
            tree_mask_buf,
            position_buf,
        )

        (
            sampling_temperatures,
            sampling_top_ps,
            sampling_top_ks,
            sampling_min_ps,
            _,
        ) = snapshot_dflash_request_sampling_params(batch.reqs, device="cpu")
        verify_input = EagleVerifyInput(
            draft_token=verify_tokens_flat,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=step_count,
            topk=self.topk,
            draft_token_num=self.num_verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=int(batch.seq_lens_sum),
            seq_lens_cpu=batch.seq_lens_cpu,
            sampling_temperatures=sampling_temperatures,
            sampling_top_ps=sampling_top_ps,
            sampling_top_ks=sampling_top_ks,
            sampling_min_ps=sampling_min_ps,
        )
        if self._use_safe_greedy_verify:
            verify_input._dflash_tree_candidates_cpu = (
                verify_tokens_flat.reshape(bs, self.num_verify_tokens)
                .detach()
                .to("cpu", dtype=torch.int64, non_blocking=False)
            )
            verify_input._dflash_tree_retrive_index_cpu = retrive_index.detach().to(
                "cpu", dtype=torch.int64, non_blocking=False
            )
            verify_input._dflash_tree_retrive_next_token_cpu = (
                retrive_next_token.detach().to(
                    "cpu", dtype=torch.int64, non_blocking=False
                )
            )
            verify_input._dflash_tree_retrive_next_sibling_cpu = (
                retrive_next_sibling.detach().to(
                    "cpu", dtype=torch.int64, non_blocking=False
                )
            )
        batch.spec_info = verify_input
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        if not overlap_v2:
            verify_input.prepare_for_verify(batch, self.page_size)
        return None, None

    def _verify_tree(
        self,
        *,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        verify_input: EagleVerifyInput,
        logits_output,
        overlap_v2: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
        sampling_info = batch.sampling_info
        debug_sync_verify = (os.environ.get("SGLANG_DFLASH_TREE_DEBUG_VERIFY_SYNC") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )

        def _debug_sync(label: str) -> None:
            if not debug_sync_verify or not torch.cuda.is_available():
                return
            try:
                torch.cuda.synchronize(device)
            except Exception as e:
                raise RuntimeError(
                    "DFLASH_TREE sampled verify failed after "
                    f"{label}: bs={int(bs)} draft_token_num={int(verify_input.draft_token_num)} "
                    f"logits_shape={tuple(logits_output.next_token_logits.shape)}"
                ) from e

        _debug_sync("verify_entry")
        if sampling_info is not None:
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=verify_input.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required
            ) or sampling_info.logit_bias is not None:
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=batch.device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        linear_penalty, verify_input.draft_token_num, dim=0
                    )
                )
        # Do not depend on carried `sampling_info` / verify-input sampling tensors here.
        # On the tree path, especially at higher concurrency, those tensors have been the
        # fragile boundary. Rebuild request sampling parameters directly from `batch.reqs`
        # at verify time instead.
        top_ks_cpu = [
            int(getattr(req.sampling_params, "top_k", 1)) for req in batch.reqs
        ]
        is_all_greedy = all(top_k == 1 for top_k in top_ks_cpu)
        temperatures = top_ps = top_ks = min_ps = None
        need_min_p_sampling = False
        if not is_all_greedy:
            temperatures, top_ps, top_ks, min_ps, need_min_p_sampling = (
                snapshot_dflash_request_sampling_params(batch.reqs, device=device)
            )
            temperatures = temperatures.view(-1, 1)
            _debug_sync("sampling_param_snapshot")

        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            if not self._warned_forced_greedy and self.tp_rank == 0:
                logger.warning(
                    "DFLASH_TREE non-greedy verification requested but the tree sampling kernel is unavailable; "
                    "falling back to greedy verification."
                )
                self._warned_forced_greedy = True
            is_all_greedy = True

        predict = torch.empty(
            (int(bs) * int(verify_input.draft_token_num) + 1,),
            dtype=torch.int32,
            device=batch.device,
        )
        accept_index = torch.full(
            (bs, int(verify_input.spec_steps) + 1),
            -1,
            dtype=torch.int32,
            device=batch.device,
        )
        accept_token_num = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            if self._use_safe_greedy_verify:
                if (
                    self._safe_greedy_from_hidden
                    and (
                        getattr(logits_output, "_dflash_final_hidden_states", None)
                        is not None
                        or (
                            logits_output.hidden_states is not None
                            and logits_output.hidden_states.numel() > 0
                        )
                    )
                ):
                    target_model = self.target_worker.model_runner.model
                    lm_head = getattr(target_model, "lm_head", None)
                    if lm_head is None:
                        raise RuntimeError(
                            "DFLASH_TREE safe greedy-from-hidden verify requires the "
                            "target model to expose `lm_head`."
                        )
                    final_hidden_states = getattr(
                        logits_output, "_dflash_final_hidden_states", None
                    )
                    hidden_for_logits = (
                        final_hidden_states
                        if final_hidden_states is not None
                        else logits_output.hidden_states
                    )
                    # The graph-backed target-verify path can hand us a replay-owned view.
                    # Make the safe debug lane fully own a stable copy before reading LM-head
                    # logits so we can distinguish lifetime issues from tree semantics bugs.
                    hidden_for_logits = hidden_for_logits.detach().clone()
                    _, target_predict = self._topk_from_vocab_parallel_head(
                        hidden_states=hidden_for_logits,
                        lm_head=lm_head,
                        topk=1,
                        return_logits=True,
                    )
                    target_predict_cpu = (
                        target_predict.reshape(bs, verify_input.draft_token_num)
                        .detach()
                        .to("cpu", dtype=torch.int64, non_blocking=False)
                    )
                else:
                    target_predict_cpu = (
                        torch.argmax(logits_output.next_token_logits, dim=-1)
                        .reshape(bs, verify_input.draft_token_num)
                        .detach()
                        .to("cpu", dtype=torch.int64, non_blocking=False)
                    )
                predict, accept_index, accept_token_num = (
                    verify_dflash_tree_greedy_fallback(
                        candidates=getattr(
                            verify_input, "_dflash_tree_candidates_cpu", candidates
                        ),
                        retrive_index=getattr(
                            verify_input,
                            "_dflash_tree_retrive_index_cpu",
                            verify_input.retrive_index,
                        ),
                        retrive_next_token=getattr(
                            verify_input,
                            "_dflash_tree_retrive_next_token_cpu",
                            verify_input.retrive_next_token,
                        ),
                        retrive_next_sibling=getattr(
                            verify_input,
                            "_dflash_tree_retrive_next_sibling_cpu",
                            verify_input.retrive_next_sibling,
                        ),
                        target_predict=target_predict_cpu,
                        num_speculative_tokens=int(accept_index.shape[1]),
                        device=batch.device,
                    )
                )
            else:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).reshape(bs, verify_input.draft_token_num)
                verify_tree_greedy_func(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=accept_token_num,
                    candidates=candidates,
                    retrive_index=verify_input.retrive_index,
                    retrive_next_token=verify_input.retrive_next_token,
                    retrive_next_sibling=verify_input.retrive_next_sibling,
                    target_predict=target_predict,
                    topk=int(verify_input.topk),
                )
        else:
            expanded_temperature = torch.repeat_interleave(
                temperatures, verify_input.draft_token_num, dim=0
            )
            _debug_sync("repeat_interleave(temperatures)")
            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )
            _debug_sync("softmax")
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    top_ks, verify_input.draft_token_num, dim=0
                ),
            )
            _debug_sync("top_k_renorm_prob")
            if not torch.all(top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        top_ps, verify_input.draft_token_num, dim=0
                    ),
                )
                _debug_sync("top_p_renorm_prob")
            if need_min_p_sampling:
                target_probs = min_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        min_ps, verify_input.draft_token_num, dim=0
                    ),
                )
                _debug_sync("min_p_renorm_prob")
            target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
            _debug_sync("reshape(target_probs)")

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )
            coins = torch.rand_like(candidates, dtype=torch.float32, device=batch.device)
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            server_args = get_global_server_args()
            tree_speculative_sampling_target_only(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=verify_input.retrive_index,
                retrive_next_token=verify_input.retrive_next_token,
                retrive_next_sibling=verify_input.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=server_args.speculative_accept_threshold_single,
                threshold_acc=server_args.speculative_accept_threshold_acc,
                deterministic=True,
            )
            _debug_sync("tree_speculative_sampling_target_only")

        if self._use_legacy_verify_commit_path and not overlap_v2:
            accept_index_cpu = accept_index.tolist()
            predict_cpu = predict.tolist()

            accept_length_per_req_cpu: List[int] = []
            commit_results = []

            for i, req in enumerate(batch.reqs):
                proposed: List[int] = []
                for idx in accept_index_cpu[i]:
                    if idx == -1:
                        break
                    proposed.append(int(predict_cpu[idx]))

                outcome = commit_dflash_proposed_tokens_to_req(
                    req=req,
                    proposed=proposed,
                    empty_error_prefix="DFLASH_TREE verify",
                )

                if outcome.commit_len < len(proposed):
                    accept_index[i, outcome.commit_len:] = -1

                commit_results.append(outcome)
                accept_length_per_req_cpu.append(outcome.accepted_draft_tokens)

            commit_metadata = materialize_dflash_target_only_commit_metadata(
                commit_results=commit_results,
                device=device,
            )
            commit_lens = commit_metadata.commit_lens
            new_verified_id = commit_metadata.new_verified_id

            if self.page_size != 1:
                raise NotImplementedError(
                    "DFLASH_TREE legacy verify commit path currently requires page_size == 1."
                )

            accept_index_flat = accept_index[accept_index != -1].to(torch.int64)
            cache_plan = build_dflash_indexed_cache_plan(
                out_cache_loc=batch.out_cache_loc,
                accepted_indices=accept_index_flat,
            )
            apply_dflash_indexed_cache_plan(
                batch=batch,
                cache_plan=cache_plan,
            )

            commit_lens_cpu = [result.commit_len for result in commit_results]
            apply_dflash_target_only_req_kv_accounting(
                reqs=batch.reqs,
                commit_lens_cpu=commit_lens_cpu,
            )
            apply_dflash_commit_mapping_updates(
                batch=batch,
                commit_lens=commit_lens,
                commit_lens_cpu=commit_lens_cpu,
            )

            return (
                new_verified_id,
                commit_lens,
                accept_index_flat,
                accept_length_per_req_cpu,
                cache_plan,
            )

        packed_indexed = pack_dflash_indexed_commits(
            predict=predict.to(torch.int64),
            accept_index=accept_index,
        )
        accept_length_per_req_cpu: List[int] = []
        commit_results = commit_dflash_target_only_batch(
            reqs=batch.reqs,
            proposed_flat_cpu=packed_indexed.proposed_flat.cpu(),
            commit_offsets_cpu=packed_indexed.commit_offsets.cpu(),
            empty_error_prefix="DFLASH_TREE verify",
        )
        accept_length_per_req_cpu.extend(
            result.accepted_draft_tokens for result in commit_results
        )

        commit_lens_cpu = [result.commit_len for result in commit_results]
        commit_metadata = materialize_dflash_target_only_commit_metadata(
            commit_results=commit_results,
            device=device,
            default_commit_lens=packed_indexed.commit_lens,
            default_new_verified_id=packed_indexed.default_new_verified_id,
        )
        commit_lens = commit_metadata.commit_lens
        new_verified_id = commit_metadata.new_verified_id

        accept_index_flat = resolve_dflash_indexed_accept_indices(
            accept_index=accept_index,
            commit_lens=commit_lens,
        )
        cache_plan = build_dflash_indexed_cache_plan(
            out_cache_loc=batch.out_cache_loc,
            accepted_indices=accept_index_flat,
            page_size=int(self.page_size),
            borrowed_out_cache_loc=bool(overlap_v2),
        )
        apply_dflash_indexed_cache_plan(
            batch=batch,
            cache_plan=cache_plan,
            page_size=int(self.page_size),
        )

        apply_dflash_target_only_req_kv_accounting(
            reqs=batch.reqs,
            commit_lens_cpu=commit_lens_cpu,
            preserve_allocated_len=bool(overlap_v2),
        )
        apply_dflash_commit_mapping_updates(
            batch=batch,
            commit_lens=commit_lens,
            commit_lens_cpu=commit_lens_cpu,
        )

        return (
            new_verified_id,
            commit_lens,
            accept_index_flat,
            accept_length_per_req_cpu,
            cache_plan,
        )

    def _forward_batch_generation_impl(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        *,
        overlap_v2: bool,
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH_TREE speculative decoding does not support return_logprob yet."
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
                    "DFLASH_TREE requires target aux hidden capture for prefill, but got None."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH_TREE expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

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
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            if isinstance(batch, ScheduleBatch):
                for req, draft_len in zip(batch.reqs, batch.seq_lens_cpu, strict=True):
                    req.dflash_draft_seq_len = int(draft_len)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=self._build_future_draft_input(draft_input),
                num_accepted_tokens=0,
                dflash_overlap_preprocessed=overlap_v2,
                requires_output_processing_barrier=overlap_v2,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH_TREE decode requires DFlashDraftInput state on the running batch."
            )

        _verify_forward_batch, _prepared_can_run_cuda_graph = (
            self._prepare_for_speculative_decoding(
                batch,
                draft_input,
                overlap_v2=overlap_v2,
            )
        )

        model_worker_batch = self._coerce_model_worker_batch(
            batch, overlap_v2=overlap_v2
        )
        if not overlap_v2 and isinstance(batch, ScheduleBatch):
            model_worker_batch = batch.get_model_worker_batch(
                seq_lens_cpu_cache=getattr(batch.spec_info, "seq_lens_cpu", None)
            )
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, EagleVerifyInput)

        if overlap_v2:
            model_worker_batch.seq_lens.record_stream(
                torch.get_device_module(self.device).current_stream()
            )
            with self.plan_stream_ctx:
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_v2_verify(
                        self.req_to_token_pool,
                        model_worker_batch,
                        self.target_worker,
                    )
                )
            if self.plan_stream is not None:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )
                self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                    verify_input,
                    (
                        self.target_worker.model_runner.graph_runner.bs
                        if can_run_cuda_graph
                        else None
                    ),
                )
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
                **kwargs,
            )
        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True, **kwargs
            )
            can_run_cuda_graph = batch_result.can_run_cuda_graph
        logits_output = batch_result.logits_output

        (
            new_verified_id,
            commit_lens,
            accept_index_flat,
            accept_length_per_req_cpu,
            cache_plan,
        ) = self._verify_tree(
            batch=batch,
            verify_input=verify_input,
            logits_output=logits_output,
            overlap_v2=overlap_v2,
        )
        verify_done = None
        if torch.device(self.device).type != "cpu":
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()

        draft_input.verified_id = new_verified_id
        append_path = "staged"
        appended_from_verify = False
        if not (self._use_legacy_verify_commit_path and not overlap_v2):
            try:
                appended_from_verify = self._append_verified_hidden_from_cache_plan(
                    draft_input=draft_input,
                    verify_positions=verify_input.positions,
                    hidden_states=logits_output.hidden_states,
                    cache_plan=cache_plan,
                    commit_lens=commit_lens,
                )
            except Exception as e:
                logger.warning(
                    "DFLASH_TREE direct verify append failed; falling back to staged path: %s",
                    e,
                )

        if not appended_from_verify:
            if (
                self._use_legacy_verify_commit_path
                and not overlap_v2
                and draft_input.target_hidden is not None
                and draft_input.target_hidden.numel() > 0
            ):
                self._append_target_hidden_to_draft_kv(batch, draft_input)
            else:
                next_target_hidden = gather_dflash_committed_hidden(
                    hidden_states=logits_output.hidden_states,
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
        update_dflash_req_verify_bookkeeping(
            reqs=list(batch.reqs),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            verify_mode="tree_target_only",
            append_path=append_path,
            default_max_steps=int(getattr(verify_input, "spec_steps", self.spec_steps)),
            default_effective_draft_token_num=int(
                getattr(verify_input, "draft_token_num", self.num_verify_tokens)
            ),
            default_effective_step_count=int(
                getattr(verify_input, "spec_steps", self.spec_steps)
            ),
        )
        logits_output.hidden_states = None
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH_TREE verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            # Tree verify follows the same overlap-preprocessed contract as linear
            # DFlash: request/output state is already committed in verify.
            next_token_ids=None if overlap_v2 else new_verified_id,
            next_draft_input=self._build_future_draft_input(
                draft_input, verify_done=verify_done
            ),
            next_out_cache_loc=batch.out_cache_loc if overlap_v2 else None,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            dflash_overlap_preprocessed=overlap_v2,
            requires_output_processing_barrier=overlap_v2,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        return self._forward_batch_generation_impl(
            batch, overlap_v2=False, **kwargs
        )
