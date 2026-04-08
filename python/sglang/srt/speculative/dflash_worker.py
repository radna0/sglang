import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
import cProfile
import io
import pstats
from typing import List, Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.sampler import (
    _sanitize_sampling_probs_for_multinomial_,
    multinomial_with_seed,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_shared_pool_verify_append,
    build_dflash_filtered_sampling_distribution_from_probs,
    can_dflash_use_fused_qkv_proj,
    dflash_sampling_info_uses_sampled_target,
    gather_dflash_committed_hidden,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
    resolve_dflash_verify_append_path,
    sample_dflash_filtered_distribution,
    update_dflash_req_verify_bookkeeping,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


def _parse_env_int(name: str, default: int) -> int:
    try:
        return max(1, int((os.environ.get(name) or str(default)).strip()))
    except Exception:
        return int(default)


def _should_cprofile_verify_step(enabled: bool, step_index: int) -> bool:
    if not enabled:
        return False
    every = _parse_env_int("SGLANG_DFLASH_CPROFILE_VERIFY_EVERY", 50)
    return ((int(step_index) + 1) % int(every)) == 0


def _env_enabled(name: str) -> bool:
    return (str(os.environ.get(name) or "").strip().lower()) in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )


def _log_cprofile_stats(
    *,
    profile: cProfile.Profile,
    prefix: str,
) -> None:
    sort_key = (os.environ.get("SGLANG_DFLASH_CPROFILE_VERIFY_SORT") or "cumtime").strip() or "cumtime"
    top_n = _parse_env_int("SGLANG_DFLASH_CPROFILE_VERIFY_TOP", 20)
    stream = io.StringIO()
    try:
        stats = pstats.Stats(profile, stream=stream).sort_stats(sort_key)
    except Exception:
        stats = pstats.Stats(profile, stream=stream).sort_stats("cumtime")
    stats.print_stats(int(top_n))
    for raw_line in stream.getvalue().splitlines():
        line = raw_line.rstrip()
        if line:
            logger.info("%s %s", prefix, line)

_FusedKVMaterializeHelper = None
_DFLASH_RESERVED_ENV_KEY = "SGLANG_DFLASH_SSD_RESERVED_TOKENS"


@dataclass
class DFlashDraftSamplingResult:
    token_ids: torch.Tensor
    selected_probs: Optional[torch.Tensor] = None
    proposal_indices: Optional[torch.Tensor] = None
    proposal_probs: Optional[torch.Tensor] = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )
        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """
    DFlash speculative decoding worker for SGLang.

    Design highlights:
    - Target layer capture uses +1 mapping: config target_layer_ids = [1,9,17,25,33]
      means we capture after layers 0,8,16,24,32. The worker sets capture_layer_ids
      = [l-1 for l in target_layer_ids].
    - Mixed precision: target KV can be FP8_E4M3 (2x FA3 speed), draft KV stays BF16.
    - Three sampling modes: greedy‑greedy, greedy‑sampled, sampled‑sampled.
    - Supports both shared KV pool (radix‑cache friendly) and compact sliding‑window.
    - Block size flexible (4,8,12,16) – defaults to training block size (16).
    - Pins FA3 attention and Triton MoE backends for optimal performance.
    - Uses TARGET_VERIFY mode with CUDA‑graph reuse (no separate graph capture).
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
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.draft_window_size: Optional[int] = (
            int(server_args.speculative_dflash_draft_window_size)
            if server_args.speculative_dflash_draft_window_size is not None
            else None
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        # ---------- Target layer capture mapping ----------
        # ModelRunner already resolves the authoritative DFLASH capture contract from the
        # draft checkpoint config. Reuse it here instead of re-deriving capture layers.
        raw_target_layer_ids = list(
            getattr(self.target_worker.model_runner, "dflash_target_layer_ids", []) or []
        )
        self.capture_layer_ids = list(
            getattr(self.target_worker.model_runner, "dflash_capture_layer_ids", []) or []
        )
        if self.tp_rank == 0 and raw_target_layer_ids:
            logger.info(
                "DFLASH target layer capture mapping: checkpoint target_layer_ids=%s -> runtime capture_layer_ids=%s",
                raw_target_layer_ids,
                self.capture_layer_ids,
            )
        self.model_runner.capture_layer_ids = self.capture_layer_ids

        # ---------- Draft worker initialization ----------
        self._warned_sampling_fallback = False
        self._logged_first_verify = False
        self._profile_enabled = (os.environ.get("SGLANG_DFLASH_PROFILE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            self._profile_every = max(
                1, int((os.environ.get("SGLANG_DFLASH_PROFILE_EVERY") or "50").strip())
            )
        except Exception:
            self._profile_every = 50
        self._profile_step = 0

        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        shared_req_to_token_pool = (
            None if self.use_compact_draft_cache else target_req_to_token_pool
        )
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
        supported_draft_backends = ("flashinfer", "fa3", "fa4")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in supported_draft_backends:
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to 'flashinfer'.",
                supported_draft_backends,
                draft_backend,
            )
            draft_backend = "flashinfer"
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
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
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model

        # Parse the draft checkpoint config from the actual draft model config.
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )

        # Block size (training default 16, can be overridden)
        if server_args.speculative_num_draft_tokens is None:
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
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s",
                self._mask_token,
                self._mask_token_id,
            )

        # Buffers for draft generation
        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None
        self._draft_block_positions_buf: Optional[torch.Tensor] = None
        self._draft_block_tokens_buf: Optional[torch.Tensor] = None
        self._draft_block_end_buf: Optional[torch.Tensor] = None
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None
        self._draft_input_embeds_buf: Optional[torch.Tensor] = None
        self._draft_block_cache_loc_buf: Optional[torch.Tensor] = None
        self._draft_block_cache_loc_cap: int = 0
        input_embeddings = self.target_worker.model_runner.model.get_input_embeddings()
        embed_weight = getattr(input_embeddings, "weight", None)
        if embed_weight is None:
            raise RuntimeError(
                "DFLASH requires the target model input embedding module to expose a weight tensor."
            )
        self._draft_embed_dim = int(embed_weight.shape[-1])
        self._draft_embed_dtype = embed_weight.dtype
        self._draft_mask_token_embed: Optional[torch.Tensor] = None
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        # Greedy sampling buffers
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

        # Fused KV materialization (CUDA only)
        self._use_fused_kv_materialize = is_cuda()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

        # Linear sampling mode (env override)
        draft_mode = (
            os.environ.get("SGLANG_DFLASH_LINEAR_DRAFT_MODE") or "auto"
        ).strip().lower()
        if draft_mode not in ("auto", "greedy", "sample"):
            raise ValueError(
                "SGLANG_DFLASH_LINEAR_DRAFT_MODE must be one of {'auto','greedy','sample'}, "
                f"got {draft_mode!r}."
            )
        self._linear_draft_mode = draft_mode
        self._last_logged_linear_mode: Optional[str] = None

    # ------------------------------------------------------------------
    #  Helper methods (KV materialization, buffer management, sampling)
    # ------------------------------------------------------------------

    def _resolve_linear_sampling_mode(self, batch: ScheduleBatch) -> str:
        if not dflash_sampling_info_uses_sampled_target(
            getattr(batch, "sampling_info", None),
            reqs=getattr(batch, "reqs", None),
        ):
            return "draft=greedy,target=greedy"

        if self._linear_draft_mode == "greedy":
            return "draft=greedy,target=sampled"

        tp_size = int(get_tp_group().world_size)
        if self._linear_draft_mode == "sample":
            if tp_size != 1:
                raise RuntimeError(
                    "DFLASH linear sampled-draft mode currently requires tp_size == 1."
                )
            return "draft=sampled,target=sampled"

        # auto: use sampled draft only when TP=1, else greedy draft
        if tp_size == 1:
            return "draft=sampled,target=sampled"
        return "draft=greedy,target=sampled"

    def _init_fused_kv_helper(self) -> None:
        """Initialize fused KV materialization helper (CUDA only)."""
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

                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        f"non-unit k_scale not supported: layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        f"non-unit v_scale not supported: layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        f"non-neox RoPE not supported: layer={layer_idx}"
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
        if cap >= bs:
            return
        new_cap = max(bs, cap * 2 if cap > 0 else bs)
        device = self.device
        block_size = self.block_size
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
        self._draft_input_embeds_buf = torch.empty(
            (new_cap, block_size, self._draft_embed_dim),
            dtype=self._draft_embed_dtype,
            device=device,
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=bool(torch.cuda.is_available()),
        )

    def _get_mask_token_embedding(self, embed_module) -> torch.Tensor:
        if self._draft_mask_token_embed is None:
            with torch.inference_mode():
                self._draft_mask_token_embed = (
                    embed_module(
                        torch.tensor(
                            [self._mask_token_id],
                            dtype=torch.long,
                            device=self.device,
                        )
                    )
                    .view(-1)
                    .detach()
                    .to(self._draft_embed_dtype)
                )
        return self._draft_mask_token_embed

    def _ensure_draft_block_cache_slots(self, needed_tokens: int) -> torch.Tensor:
        needed_tokens = int(needed_tokens)
        if needed_tokens <= 0:
            raise ValueError(f"DFLASH scratch capacity must be positive, got {needed_tokens}.")
        if (
            self._draft_block_cache_loc_buf is not None
            and self._draft_block_cache_loc_cap >= needed_tokens
        ):
            return self._draft_block_cache_loc_buf[:needed_tokens]

        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        current_cap = self._draft_block_cache_loc_cap
        new_cap = max(needed_tokens, current_cap * 2 if current_cap > 0 else needed_tokens)
        extra_needed = new_cap - current_cap
        extra_loc = allocator.alloc(extra_needed)
        if extra_loc is None:
            raise RuntimeError(
                f"DFLASH draft scratch OOM when allocating {extra_needed} persistent block tokens."
            )
        extra_loc = extra_loc.to(torch.int64)
        if self._draft_block_cache_loc_buf is None:
            self._draft_block_cache_loc_buf = extra_loc
        else:
            self._draft_block_cache_loc_buf = torch.cat(
                [self._draft_block_cache_loc_buf, extra_loc], dim=0
            )
        self._draft_block_cache_loc_cap = self._draft_block_cache_loc_buf.shape[0]
        try:
            prev_reserved = int(
                (os.environ.get(_DFLASH_RESERVED_ENV_KEY) or "0").strip() or 0
            )
        except Exception:
            prev_reserved = 0
        os.environ[_DFLASH_RESERVED_ENV_KEY] = str(
            max(prev_reserved, self._draft_block_cache_loc_cap)
        )
        return self._draft_block_cache_loc_buf[:needed_tokens]

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(f"DFLASH mask_token must be a non-empty string, got {mask_token!r}.")
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    f"DFLASH mask_token_id={resolved_id} >= vocab_size={vocab_size}. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer when mask_token_id not set (skip_tokenizer_init not supported)."
            )
        resolved_id = tokenizer.get_vocab().get(mask_token)
        if resolved_id is None:
            # Try adding the mask token (safe only if within vocab)
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)
            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, vocab_size=%s",
                    mask_token,
                    resolved_id,
                    vocab_size,
                )
        if resolved_id is None or resolved_id < 0 or resolved_id >= vocab_size:
            raise ValueError(
                f"DFLASH cannot resolve valid mask_token_id for mask_token={mask_token!r}."
            )
        return int(resolved_id)

    # ------------------------------------------------------------------
    #  Core speculative decoding pipeline
    # ------------------------------------------------------------------

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        """Draft a block using the draft model and prepare verify info."""
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError("DFLASH does not support grammar constraints.")
        if dflash_sampling_info_uses_sampled_target(
            getattr(batch, "sampling_info", None),
            reqs=getattr(batch, "reqs", None),
        ):
            if not is_dflash_sampling_verify_available():
                raise RuntimeError(
                    "DFLASH sampled target verification requested but kernels unavailable."
                )

        bs = batch.batch_size()
        linear_mode = self._resolve_linear_sampling_mode(batch)
        if linear_mode != self._last_logged_linear_mode and self.tp_rank == 0:
            logger.info("DFLASH linear sampling mode active: %s", linear_mode)
            self._last_logged_linear_mode = linear_mode

        # 1) Append newly committed tokens to draft KV
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH requires target lm_head with weight and shard_indices."
            )

        # 2) Draft a block (non-causal)
        self._ensure_draft_block_buffers(bs)
        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(self._mask_token_id)
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        input_embeds_3d = self._draft_input_embeds_buf[:bs]
        mask_emb = self._get_mask_token_embedding(embed_module).view(1, 1, -1)
        input_embeds_3d[:] = mask_emb
        verified_embeds = embed_module(draft_input.verified_id.to(torch.long)).to(
            dtype=input_embeds_3d.dtype
        )
        input_embeds_3d[:, 0, :].copy_(verified_embeds)
        input_embeds = input_embeds_3d.view(-1, input_embeds_3d.shape[-1])

        target_prefix_lens = batch.seq_lens  # int32, device
        draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d)
        positions = positions_2d.reshape(-1)

        block_start = draft_prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, self.block_size, out=block_end)

        if not self.use_compact_draft_cache:
            seq_lens_cpu = batch.seq_lens_cpu[:bs]
        else:
            seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = None
        should_free_block_cache_loc = False
        if self.page_size == 1:
            # Allocate per iteration to avoid scheduler leak detection
            block_cache_loc = allocator.alloc(bs * self.block_size)
            should_free_block_cache_loc = True
        else:
            token_to_kv_pool_state_backup = allocator.backup_state()
            block_end_cpu = seq_lens_cpu + self.block_size
            last_loc = get_last_loc(
                self.draft_model_runner.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                block_start,
            )
            block_cache_loc = allocator.alloc_extend(
                block_start,
                seq_lens_cpu,
                block_end,
                block_end_cpu,
                last_loc,
                bs * self.block_size,
            )
        if block_cache_loc is None:
            raise RuntimeError(f"DFLASH draft OOM allocating {bs * self.block_size} block tokens.")
        try:
            if self.page_size == 1 and not self.use_compact_draft_cache:
                draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
                row_ids = batch.req_pool_indices.to(torch.int64).unsqueeze(1).expand(
                    -1, self.block_size
                )
                draft_req_to_token[row_ids, positions_2d.to(torch.int64)] = (
                    block_cache_loc.view(bs, self.block_size).to(draft_req_to_token.dtype)
                )
            else:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    block_start,
                    block_end,
                    block_cache_loc,
                    bs,
                )

            # Run draft forward (TARGET_VERIFY mode)
            draft_spec_info = self._draft_block_spec_info
            seq_lens = draft_prefix_lens
            seq_lens_sum = int(seq_lens_cpu.sum().item())
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.flatten(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,  # draft does not need hidden
            )
            with torch.inference_mode():
                draft_out = self.draft_model_runner.forward(forward_batch)
                draft_logits_output = draft_out.logits_output
        finally:
            if token_to_kv_pool_state_backup is not None:
                allocator.restore_state(token_to_kv_pool_state_backup)
            elif should_free_block_cache_loc:
                # Clean up req->token pointers to avoid dangling references
                try:
                    if self.page_size == 1 and not self.use_compact_draft_cache:
                        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
                        row_ids = batch.req_pool_indices.to(torch.int64).unsqueeze(1).expand(
                            -1, self.block_size
                        )
                        draft_req_to_token[row_ids, positions_2d.to(torch.int64)] = 0
                    else:
                        zeros = torch.zeros_like(block_cache_loc)
                        assign_req_to_token_pool_func(
                            batch.req_pool_indices,
                            self.draft_model_runner.req_to_token_pool.req_to_token,
                            block_start,
                            block_end,
                            zeros,
                            bs,
                        )
                finally:
                    allocator.free(block_cache_loc)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)

        draft_hidden_tail = draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1])

        # Sample draft tokens (greedy or sampled)
        draft_selected_probs = None
        draft_proposal_indices = None
        draft_proposal_probs = None
        if linear_mode == "draft=sampled,target=sampled":
            sampling_info = batch.sampling_info
            assert sampling_info is not None
            row_count = bs * (self.block_size - 1)
            temperatures = torch.repeat_interleave(sampling_info.temperatures, self.block_size - 1, dim=0)
            top_ps = torch.repeat_interleave(sampling_info.top_ps, self.block_size - 1, dim=0)
            top_ks = torch.repeat_interleave(sampling_info.top_ks, self.block_size - 1, dim=0)
            min_ps = torch.repeat_interleave(sampling_info.min_ps, self.block_size - 1, dim=0)
            sampling_seed = getattr(sampling_info, "sampling_seed", None)
            if sampling_seed is not None:
                sampling_seed = torch.repeat_interleave(sampling_seed, self.block_size - 1, dim=0)
            sampled_result = self._sample_from_vocab_parallel_head_tp1(
                hidden_states=draft_hidden_tail,
                lm_head=lm_head,
                temperatures=temperatures[:row_count],
                top_ps=top_ps[:row_count],
                top_ks=top_ks[:row_count],
                min_ps=min_ps[:row_count],
                positions=positions_2d[:, 1:].reshape(-1),
                sampling_seed=sampling_seed[:row_count] if sampling_seed is not None else None,
                return_proposal_support=True,
            )
            draft_next = sampled_result.token_ids.view(bs, self.block_size - 1)
            draft_selected_probs = sampled_result.selected_probs.view(bs, self.block_size - 1)
            support_width = sampled_result.proposal_indices.shape[-1]
            draft_proposal_indices = sampled_result.proposal_indices.view(bs, self.block_size - 1, support_width)
            draft_proposal_probs = sampled_result.proposal_probs.view(bs, self.block_size - 1, support_width)
        else:
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden_tail, lm_head=lm_head
            ).view(bs, self.block_size - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            linear_mode=linear_mode,
            draft_selected_probs=draft_selected_probs,
            draft_proposal_indices=draft_proposal_indices,
            draft_proposal_probs=draft_proposal_probs,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(self.model_runner.attn_backend)
        verify_input.prepare_for_verify(batch, self.page_size, build_custom_mask=build_custom_mask)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False
        draft_input.linear_mode = linear_mode

    # ------------------------------------------------------------------
    #  Greedy / Sampled draft sampling (TP‑safe)
    # ------------------------------------------------------------------

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """TP‑safe greedy argmax over the target LM head."""
        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = tp_group.world_size

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = hidden_states.shape[0]
        out_token_ids = torch.empty((num_tokens,), dtype=torch.long, device=hidden_states.device)

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path: TP=1 and no added vocab
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, chunk_size):
                end = min(num_tokens, start + chunk_size)
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = torch.argmax(logits, dim=-1).to(torch.long) + org_vocab_start
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        # General case with TP>1 or added vocab
        for start in range(0, num_tokens, chunk_size):
            end = min(num_tokens, start + chunk_size)
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = end - start

            # Base vocab logits
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full((chunk_len,), torch.finfo(weight_dtype).min, dtype=weight_dtype, device=hs.device)
                local_arg = torch.zeros((chunk_len,), dtype=torch.int64, device=hs.device)

            # Added vocab logits
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(hs, weight[added_slice_start:added_slice_end].T)
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                local_arg = torch.where(use_added, added_arg + num_org_padded, local_arg)

            # Convert to global token ids
            if num_added == 0:
                global_ids = local_arg + org_vocab_start
            else:
                global_ids = torch.empty_like(local_arg)
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (local_arg[~is_base] - num_org_padded)

            if tp_size == 1:
                out_token_ids[start:end] = global_ids
                continue

            # TP>1: gather per‑rank maxima and select global max
            needed = tp_size * chunk_len
            if (self._draft_greedy_gather_cap < needed or
                self._draft_greedy_gathered_max_buf is None or
                self._draft_greedy_gathered_max_buf.dtype != local_max.dtype):
                self._draft_greedy_gathered_max_buf = torch.empty((needed,), dtype=local_max.dtype, device=hs.device)
                self._draft_greedy_gathered_ids_buf = torch.empty((needed,), dtype=global_ids.dtype, device=hs.device)
                self._draft_greedy_gather_cap = needed
            if (self._draft_greedy_index_cap < chunk_len or
                self._draft_greedy_best_rank_buf is None):
                self._draft_greedy_best_rank_buf = torch.empty((chunk_len,), dtype=torch.int64, device=hs.device)
                self._draft_greedy_rank_index_buf = torch.empty((1, chunk_len), dtype=torch.int64, device=hs.device)
                self._draft_greedy_selected_ids_buf = torch.empty((1, chunk_len), dtype=torch.int64, device=hs.device)
                self._draft_greedy_index_cap = chunk_len

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)
            self._draft_greedy_rank_index_buf[0].copy_(best_rank)
            torch.gather(gathered_ids, 0, self._draft_greedy_rank_index_buf[:, :chunk_len],
                         out=self._draft_greedy_selected_ids_buf[:, :chunk_len])
            out_token_ids[start:end] = self._draft_greedy_selected_ids_buf[0, :chunk_len]

        return out_token_ids

    def _sample_from_vocab_parallel_head_tp1(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        min_ps: torch.Tensor,
        positions: torch.Tensor,
        sampling_seed: Optional[torch.Tensor],
        chunk_size: int = 128,
        return_proposal_support: bool = False,
    ) -> DFlashDraftSamplingResult:
        """Sampled draft (TP=1 only) with optional proposal distribution."""
        if hidden_states.numel() == 0:
            return DFlashDraftSamplingResult(token_ids=torch.empty((0,), dtype=torch.long, device=hidden_states.device))

        tp_group = get_tp_group()
        if tp_group.world_size != 1:
            raise RuntimeError("DFLASH sampled draft requires tp_size == 1.")

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        weight_parts = []
        token_id_parts = []
        if num_org > 0:
            weight_parts.append(weight[:num_org])
            token_id_parts.append(torch.arange(org_vocab_start, org_vocab_start + num_org, dtype=torch.long, device=hidden_states.device))
        if num_added > 0:
            added_slice_start = num_org_padded
            added_slice_end = num_org_padded + num_added
            weight_parts.append(weight[added_slice_start:added_slice_end])
            token_id_parts.append(torch.arange(added_vocab_start, added_vocab_start + num_added, dtype=torch.long, device=hidden_states.device))
        if not weight_parts:
            return DFlashDraftSamplingResult(token_ids=torch.zeros((hidden_states.shape[0],), dtype=torch.long, device=hidden_states.device))

        valid_weight = weight_parts[0] if len(weight_parts) == 1 else torch.cat(weight_parts, dim=0)
        valid_token_ids = token_id_parts[0] if len(token_id_parts) == 1 else torch.cat(token_id_parts, dim=0)

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        num_tokens = hidden_states.shape[0]
        out_token_ids = torch.empty((num_tokens,), dtype=torch.long, device=hidden_states.device)
        out_selected_probs = None
        out_proposal_indices = None
        out_proposal_probs = None
        need_min_p_sampling = torch.any(min_ps > 0).item()
        simple_sampling_case = not (torch.any(top_ps != 1.0).item() or torch.any(top_ks != 1).item() or need_min_p_sampling)
        if return_proposal_support and simple_sampling_case:
            raise RuntimeError("DFLASH sampled draft requires bounded top-k/top-p/min-p support when return_proposal_support=True.")
        if return_proposal_support:
            proposal_width = int(top_ks.to(torch.int64).max().item())
            proposal_width = max(1, min(proposal_width, valid_weight.shape[0]))
            out_selected_probs = torch.empty((num_tokens,), dtype=torch.float32, device=hidden_states.device)
            out_proposal_indices = torch.empty((num_tokens, proposal_width), dtype=torch.long, device=hidden_states.device)
            out_proposal_probs = torch.empty((num_tokens, proposal_width), dtype=torch.float32, device=hidden_states.device)

        for start in range(0, num_tokens, chunk_size):
            end = min(num_tokens, start + chunk_size)
            hs = _cast_hs(hidden_states[start:end])
            logits = torch.matmul(hs, valid_weight.T).float()
            logits.div_(temperatures[start:end].view(-1, 1))
            probs = torch.softmax(logits, dim=-1)

            top_ks_chunk = top_ks[start:end].to(dtype=torch.int32, device=probs.device)
            top_ps_chunk = top_ps[start:end].to(dtype=probs.dtype, device=probs.device)
            min_ps_chunk = min_ps[start:end].to(dtype=probs.dtype, device=probs.device)
            sampling_seed_chunk = sampling_seed[start:end].to(device=probs.device) if sampling_seed is not None else None
            positions_chunk = positions[start:end].to(device=probs.device)

            if return_proposal_support:
                probs_sort, probs_idx = build_dflash_filtered_sampling_distribution_from_probs(
                    probs, top_ks_chunk, top_ps_chunk, min_ps_chunk, need_min_p_sampling
                )
                sampled_local, sampled_prob = sample_dflash_filtered_distribution(
                    probs_sort=probs_sort,
                    probs_idx=probs_idx,
                    sampling_seed=sampling_seed_chunk,
                    positions=positions_chunk,
                )
                out_selected_probs[start:end] = sampled_prob.to(torch.float32)
                out_proposal_probs[start:end].copy_(probs_sort.to(torch.float32))
                out_proposal_indices[start:end].copy_(valid_token_ids.index_select(0, probs_idx.reshape(-1)).view_as(probs_idx))
            else:
                if simple_sampling_case:
                    _sanitize_sampling_probs_for_multinomial_(probs)
                    if sampling_seed_chunk is None:
                        sampled_local = torch.multinomial(probs, num_samples=1).view(-1)
                    else:
                        sampled_local = multinomial_with_seed(torch.log(probs.to(torch.float64)), sampling_seed_chunk, positions_chunk).view(-1)
                else:
                    probs_sort, probs_idx = build_dflash_filtered_sampling_distribution_from_probs(
                        probs, top_ks_chunk, top_ps_chunk, min_ps_chunk, need_min_p_sampling
                    )
                    sampled_local, _ = sample_dflash_filtered_distribution(
                        probs_sort=probs_sort,
                        probs_idx=probs_idx,
                        sampling_seed=sampling_seed_chunk,
                        positions=positions_chunk,
                    )
            out_token_ids[start:end] = valid_token_ids.index_select(0, sampled_local.to(torch.long))

        return DFlashDraftSamplingResult(
            token_ids=out_token_ids,
            selected_probs=out_selected_probs,
            proposal_indices=out_proposal_indices,
            proposal_probs=out_proposal_probs,
        )

    # ------------------------------------------------------------------
    #  KV cache materialization (target -> draft)
    # ------------------------------------------------------------------

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize target hidden states into draft KV cache (before radix updates)."""
        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError("DFLASH draft state missing target_hidden.")
        if draft_input.ctx_lens.numel() != bs or draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError("DFLASH ctx_lens/draft_seq_lens length mismatch.")

        total_ctx = draft_input.target_hidden.shape[0]
        if total_ctx <= 0:
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        is_extend = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        if is_extend:
            ctx_start = batch.seq_lens.to(torch.int64)
            seq_lens_after = (batch.seq_lens.to(torch.int64) + ctx_lens.to(torch.int64)).to(torch.int32)
        else:
            ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)
            seq_lens_after = batch.seq_lens.to(torch.int32)

        # Gather cache locations and positions
        if bs == 1:
            max_ctx = total_ctx
            r = self._block_pos_offsets[:max_ctx] if max_ctx <= self.block_size else torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)
            ctx_positions = pos2d.reshape(-1)
        else:
            max_ctx = int(ctx_lens.max().item()) if is_extend else self.block_size
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")
            r = self._block_pos_offsets[:max_ctx] if max_ctx <= self.block_size else torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]
            pos2d = ctx_start[:, None] + r
            mask = r < ctx_lens[:, None]
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )
            ctx_positions = pos2d[mask]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(draft_input.target_hidden)
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(f"ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}.")

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(ctx_hidden, ctx_positions, ctx_cache_loc)
                except Exception as e:
                    logger.warning("Fused KV append failed, falling back to sequential: %s", e)
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(ctx_hidden, ctx_positions, ctx_cache_loc)
            else:
                self._append_target_hidden_sequential(ctx_hidden, ctx_positions, ctx_cache_loc)

        if self.use_compact_draft_cache:
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(seq_lens_after)
            suffix_start = seq_lens_after.to(torch.int64) - new_draft_seq_lens.to(torch.int64)
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            draft_input.draft_seq_lens = seq_lens_after.to(torch.int32)
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
                attn.attn, ctx_cache_loc, k, v, attn.attn.k_scale, attn.attn.v_scale
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(attn, ctx_cache_loc, cache_k, cache_v, attn.k_scale, attn.v_scale)

        self._fused_kv_helper.materialize(ctx_hidden, ctx_positions, _write_layer_kv)

    def _project_verified_hidden_selected(
        self,
        *,
        hidden_states: torch.Tensor,
        accepted_indices: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.draft_model, "project_target_hidden_selected"):
            return self.draft_model.project_target_hidden_selected(hidden_states, accepted_indices)
        selected_hidden = gather_dflash_committed_hidden(hidden_states, accepted_indices)
        return self.draft_model.project_target_hidden(selected_hidden)

    def _append_verified_hidden_from_cache_plan(
        self,
        *,
        draft_input: DFlashDraftInput,
        verify_positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_plan,
        commit_lens: torch.Tensor,
    ) -> bool:
        return apply_dflash_shared_pool_verify_append(
            draft_input=draft_input,
            verify_positions=verify_positions,
            hidden_states=hidden_states,
            cache_plan=cache_plan,
            commit_lens=commit_lens,
            write_selected_hidden=self._project_and_write_verified_hidden_selected_to_draft_kv,
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
            ctx_hidden = self._project_verified_hidden_selected(
                hidden_states=hidden_states,
                accepted_indices=accepted_indices,
            )
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(f"Selected hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}.")
            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_verified_hidden_selected_fused(ctx_hidden, ctx_positions, ctx_cache_loc)
                    return
                except Exception as e:
                    logger.warning("Fused selected verify append failed, falling back: %s", e)
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
            self._append_target_hidden_sequential(ctx_hidden, ctx_positions, ctx_cache_loc)

    def _append_verified_hidden_selected_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(attn, ctx_cache_loc, cache_k, cache_v, attn.k_scale, attn.v_scale)

        self._fused_kv_helper.materialize(ctx_hidden, ctx_positions, _write_layer_kv)

    # ------------------------------------------------------------------
    #  Batch forward (prefill + decode/verify)
    # ------------------------------------------------------------------

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError("DFLASH does not support return_logprob.")

        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        # ---------- PREFILL / EXTEND ----------
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            # Capture hidden states at the specified layers for prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            model_worker_batch.capture_layer_ids = self.capture_layer_ids

            batch_result = self.target_worker.forward_batch_generation(model_worker_batch, **kwargs)
            logits_output, next_token_ids = batch_result.logits_output, batch_result.next_token_ids
            if logits_output.hidden_states is None:
                raise RuntimeError("DFLASH requires target aux hidden capture for prefill.")

            if model_worker_batch.extend_seq_lens is None or model_worker_batch.extend_prefix_lens is None:
                raise RuntimeError("DFLASH expected extend_seq_lens/extend_prefix_lens in extend mode.")

            device = next_token_ids.device
            def _to_int32_device_tensor(x, device=device):
                if isinstance(x, torch.Tensor):
                    x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(model_worker_batch.extend_seq_lens)
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # ---------- DECODE / VERIFY ----------
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError("DFLASH decode requires DFlashDraftInput on the batch.")

        profile_this_step = (
            self._profile_enabled
            and self.tp_rank == 0
            and (self._profile_step % self._profile_every == 0)
        )
        cprofile_verify_this_step = (
            self.tp_rank == 0
            and _should_cprofile_verify_step(
                _env_enabled("SGLANG_DFLASH_CPROFILE_VERIFY"),
                self._profile_step,
            )
        )
        cuda_event_profile_this_step = (
            profile_this_step
            and self.tp_rank == 0
            and batch.device.type == "cuda"
            and _env_enabled("SGLANG_DFLASH_PROFILE_CUDA_EVENTS")
        )
        self._profile_step += 1

        t0 = time.perf_counter() if profile_this_step else 0.0
        self._prepare_for_speculative_decoding(batch, draft_input)
        t_prepare = time.perf_counter() if profile_this_step else 0.0

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        # Ensure target model captures hidden states for the verify forward
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.capture_layer_ids = self.capture_layer_ids

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend, "update_mamba_state_after_mtp_verify"
        )
        seq_lens_pre_verify = batch.seq_lens.clone() if need_mamba_verify_commit else None

        if cuda_event_profile_this_step:
            cuda_stream = torch.cuda.current_stream(device=batch.device)
            step_cuda_start = torch.cuda.Event(enable_timing=True)
            verify_cuda_start = torch.cuda.Event(enable_timing=True)
            verify_cuda_end = torch.cuda.Event(enable_timing=True)
            step_cuda_end = torch.cuda.Event(enable_timing=True)
            step_cuda_start.record(cuda_stream)
            verify_cuda_start.record(cuda_stream)
        else:
            step_cuda_start = verify_cuda_start = verify_cuda_end = step_cuda_end = None

        t1 = time.perf_counter() if profile_this_step else 0.0
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        t_target = time.perf_counter() if profile_this_step else 0.0
        if cuda_event_profile_this_step and verify_cuda_end is not None:
            verify_cuda_end.record(torch.cuda.current_stream(device=batch.device))
        logits_output, can_run_cuda_graph = batch_result.logits_output, batch_result.can_run_cuda_graph

        t2 = time.perf_counter() if profile_this_step else 0.0
        if cprofile_verify_this_step:
            verify_profile = cProfile.Profile()
            verify_profile.enable()
            try:
                new_verified_id, commit_lens, cache_plan, next_target_hidden, accept_length_per_req_cpu = (
                    verify_input.verify(
                        batch=batch,
                        logits_output=logits_output,
                        page_size=self.page_size,
                        target_next_token_ids=batch_result.next_token_ids,
                    )
                )
            finally:
                verify_profile.disable()
                _log_cprofile_stats(
                    profile=verify_profile,
                    prefix="[DFLASH_CPROFILE_VERIFY]",
                )
        else:
            new_verified_id, commit_lens, cache_plan, next_target_hidden, accept_length_per_req_cpu = (
                verify_input.verify(
                    batch=batch,
                    logits_output=logits_output,
                    page_size=self.page_size,
                    target_next_token_ids=batch_result.next_token_ids,
                )
            )
        
        t_verify = time.perf_counter() if profile_this_step else 0.0

        if need_mamba_verify_commit and seq_lens_pre_verify is not None:
            self._update_target_mamba_state_after_verify(
                batch, seq_lens_pre_verify, commit_lens
            )

        # Update draft state and materialize committed tokens into draft KV
        draft_input.verified_id = new_verified_id
        appended_from_verify = False
        if cache_plan is not None:
            try:
                appended_from_verify = self._append_verified_hidden_from_cache_plan(
                    draft_input=draft_input,
                    verify_positions=verify_input.positions,
                    hidden_states=logits_output.hidden_states,
                    cache_plan=cache_plan,
                    commit_lens=commit_lens,
                )
            except Exception as e:
                logger.warning("Direct verify append failed, falling back to staged path: %s", e)

        if not appended_from_verify:
            draft_input.target_hidden = next_target_hidden
            draft_input.ctx_lens = commit_lens
            self._append_target_hidden_to_draft_kv(batch, draft_input)
        t_append = time.perf_counter() if profile_this_step else 0.0
        if cuda_event_profile_this_step and step_cuda_end is not None:
            step_cuda_end.record(torch.cuda.current_stream(device=batch.device))

        append_path = resolve_dflash_verify_append_path(
            appended_from_verify=appended_from_verify,
            fused_helper_active=bool(self._use_fused_kv_materialize and self._fused_kv_helper is not None),
        )
        update_dflash_req_verify_bookkeeping(
            reqs=list(batch.reqs),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            verify_mode=str(getattr(draft_input, "linear_mode", "target_only")),
            append_path=append_path,
            default_max_steps=self.block_size - 1,
            default_effective_draft_token_num=self.block_size,
            default_effective_step_count=self.block_size - 1,
        )

        logits_output.hidden_states = None
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info("DFLASH verify completed. accept_length_per_req=%s", accept_length_per_req_cpu)
            self._logged_first_verify = True

        if profile_this_step and self.tp_rank == 0:
            verify_cuda_ms = None
            step_cuda_ms = None
            if (
                cuda_event_profile_this_step
                and verify_cuda_start is not None
                and verify_cuda_end is not None
                and step_cuda_start is not None
                and step_cuda_end is not None
            ):
                step_cuda_end.synchronize()
                verify_cuda_ms = float(verify_cuda_start.elapsed_time(verify_cuda_end))
                step_cuda_ms = float(step_cuda_start.elapsed_time(step_cuda_end))
            bs = batch.batch_size()
            logger.info(
                "[DFLASH profile] bs=%d mode=%s can_run_cuda_graph=%s "
                "prepare_ms=%.2f target_verify_ms=%.2f verify_cpu_ms=%.2f verify_wall_ms=%.2f append_ms=%.2f total_ms=%.2f "
                "verify_cuda_ms=%s step_cuda_ms=%s "
                "accept_sum=%d accept_mean=%.2f",
                bs,
                str(getattr(draft_input, "linear_mode", "")),
                can_run_cuda_graph,
                (t_prepare - t0) * 1000.0,
                (t_target - t1) * 1000.0,
                (t_verify - t2) * 1000.0,
                (t_verify - t2) * 1000.0,
                (t_append - t_verify) * 1000.0,
                (t_append - t0) * 1000.0,
                f"{verify_cuda_ms:.2f}" if verify_cuda_ms is not None else "n/a",
                f"{step_cuda_ms:.2f}" if step_cuda_ms is not None else "n/a",
                num_accepted_tokens,
                num_accepted_tokens / max(1, bs),
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    # ------------------------------------------------------------------
    #  Missing helpers (gather, compact seq lens, mamba update)
    # ------------------------------------------------------------------

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(f"{context} expected 2D positions, got {pos2d.shape}.")
        if mask.shape != pos2d.shape:
            raise RuntimeError(f"{context} mask/position shape mismatch.")
        req_pool_indices = req_pool_indices.to(torch.int64)
        mask = mask.to(torch.bool)
        table_width = req_to_token.shape[1]
        if table_width <= 0:
            if mask.any():
                raise RuntimeError(f"{context} req_to_token table empty but mask non-empty.")
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: Optional[torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(max_len, device=self.device, dtype=torch.int64).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(device=self.device, dtype=torch.int32),
            max=self.draft_window_size,
        )
        if self.page_size <= 1:
            return visible_lens
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _update_target_mamba_state_after_verify(
        self,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
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
            tracking_point = (batch.seq_lens // mamba_track_interval) * mamba_track_interval
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (to_track_ith < commit_lens.to(to_track_ith.dtype))
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

    # ------------------------------------------------------------------
    #  Public API passthrough
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        if self._draft_block_cache_loc_buf is not None:
            with torch.inference_mode():
                self.draft_model_runner.token_to_kv_pool_allocator.free(
                    self._draft_block_cache_loc_buf.reshape(-1)
                )
            self._draft_block_cache_loc_buf = None
            self._draft_block_cache_loc_cap = 0
        os.environ[_DFLASH_RESERVED_ENV_KEY] = "0"
