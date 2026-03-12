import logging
import math
import os
import time
from copy import deepcopy
from typing import List, Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_controller import (
    DFlashAdaptivePqConfig,
    DFlashAdaptivePqController,
    DFlashDifficultySignals,
    DFlashReqDifficultyState,
    survival_should_force_target_only,
)
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,
    resolve_dflash_mask_token,
    resolve_dflash_mask_token_id,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


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
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        self._warned_forced_greedy = False
        self._logged_first_verify = False
        self._adaptive_pq = DFlashAdaptivePqController(
            DFlashAdaptivePqConfig.from_env(
                default_temp_mul=float(
                    getattr(server_args, "speculative_dflash_pq_draft_temp_mul", 1.0)
                    or 1.0
                )
            )
        )

        # Draft runner (separate KV cache + attention backend).
        # Share req_to_token_pool + token_to_kv_pool_allocator with the target worker (EAGLE3-style),
        # while keeping a separate draft KV cache pool (the draft model has different KV values).
        shared_req_to_token_pool, shared_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
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
        elif draft_backend not in (
            "flashinfer",
            "fa3",
            "flex_attention",
            "flex_attention2",
            "flex_flash",
            "flex_flash2",
            "flex_flash2_delegate_fa3",
            "flex_flash4",
        ):
            logger.warning(
                "DFLASH draft worker only supports attention_backend in {'flashinfer', 'fa3', 'flex_attention', 'flex_attention2', 'flex_flash', 'flex_flash2', 'flex_flash2_delegate_fa3', 'flex_flash4'} for now, "
                "but got %r. Falling back to 'flashinfer'.",
                draft_backend,
            )
            draft_backend = "flashinfer"

        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        self._draft_attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
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
            memory_pool_config=getattr(target_worker.model_runner, "memory_pool_config", None),
        )
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(getattr(self.draft_model, "block_size", 16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
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

        # The fused KV materialization helper currently uses Triton kernels.
        # If we are experimenting with FlexAttention backends, keep the stack Triton-free by default.
        self._use_fused_kv_materialize = is_cuda() and self._draft_attention_backend not in (
            "flex_attention",
            "flex_attention2",
            "flex_flash",
            "flex_flash2",
        )
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
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
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

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise ValueError(
                "DFLASH does not support grammar-constrained decoding yet."
            )

        batch.maybe_evict_swa()

        bs = batch.batch_size()
        device = self.model_runner.device

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
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the draft model.
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # For spec-v1, the draft KV cache is always materialized to the current target
        # prefix before drafting the next block.
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
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            block_cache_loc = allocator.alloc(bs * self.block_size)
            if block_cache_loc is None:
                raise RuntimeError(
                    f"DFLASH draft OOM when allocating {bs * self.block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode (cuda-graphable) to run a fixed-size draft block.
            # In this mode, `seq_lens` stores the prefix lengths; attention backends
            # derive kv_len by adding `draft_token_num`.
            draft_spec_info = self._draft_block_spec_info
            seq_lens = prefix_lens
            seq_lens_sum = int(batch.seq_lens_sum)
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
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_hidden = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])

        sampling_info = batch.sampling_info
        verify_mode = str(
            getattr(self.server_args, "speculative_dflash_verify_mode", "target_only")
            or "target_only"
        )
        # Survival-weighted scheduling (FailFast/SSD-style): if recent accept lengths are low,
        # avoid spending extra pq bookkeeping and fall back to target_only until we recover.
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
        if verify_mode == "pq" and self._adaptive_pq.should_force_target_only():
            # FailFast: pq was recently deemed pathological; temporarily fall back to
            # target_only verification (still exact target distribution).
            verify_mode = "target_only"
        draft_conf_debug = None
        # NOTE: On some backends/workers, `batch.sampling_info` can be None even when the
        # request uses non-greedy sampling. For DFlash pq verification we must propose
        # tokens by sampling from the draft distribution q, so we reconstruct the minimal
        # sampling tensors from req-level SamplingParams if needed.
        if sampling_info is not None:
            temperatures = sampling_info.temperatures.to(draft_hidden.device)
            top_ps = sampling_info.top_ps.to(draft_hidden.device)
            top_ks = sampling_info.top_ks.to(draft_hidden.device)
            min_ps = sampling_info.min_ps.to(draft_hidden.device)
            need_min_p_sampling = bool(sampling_info.need_min_p_sampling)
            # Do NOT trust sampling_info.is_all_greedy here: the draft worker may force
            # greedy sampling (e.g. vLLM #16899-style) without changing the request's
            # actual sampling params. For pq we care about whether the *request* is
            # greedy, which is equivalent to top_k == 1 for all requests.
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

        if (
            verify_mode == "pq"
            and self.tp_rank == 0
            and not getattr(self, "_logged_pq_draft_gate", False)
        ):
            try:
                topk_dbg = top_ks.detach().to("cpu").to(torch.int64).tolist()
            except Exception:
                topk_dbg = None
            sp0 = None
            try:
                if batch.reqs:
                    sp = batch.reqs[0].sampling_params
                    sp0 = {
                        "temperature": float(getattr(sp, "temperature", 1.0)),
                        "top_k": int(getattr(sp, "top_k", -999)),
                        "top_p": float(getattr(sp, "top_p", 1.0)),
                        "min_p": float(getattr(sp, "min_p", 0.0)),
                    }
            except Exception:
                sp0 = None
            logger.info(
                "DFLASH pq draft gate: sampling_info=%s top_ks=%s is_all_greedy=%s sp0=%s",
                bool(sampling_info is not None),
                topk_dbg,
                bool(is_all_greedy),
                sp0,
            )
            setattr(self, "_logged_pq_draft_gate", True)

        use_pq = verify_mode == "pq" and not is_all_greedy

        draft_topk = 0
        draft_topk_ids = None
        draft_topk_probs = None
        max_steps_per_req = None

        if use_pq:
            # For pq verification we must know q(token), so the draft must propose
            # tokens by sampling from its own distribution (not argmax).
            #
            # IMPORTANT: For acceptance under temperature sampling, we often want to
            # shape the *proposal* distribution q to be closer to the target p
            # (maximize overlap / reduce TV distance). These env vars allow fast
            # experimentation without changing request-level sampling.
            # Prefer ServerArgs knobs (if present), but allow env vars as an override for fast sweeps.
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
            # SGLang uses a huge TOP_K_ALL value for "whole vocab"; that's not feasible here.
            if topk <= 0 or topk >= (1 << 20):
                if self.tp_rank == 0 and not self._warned_forced_greedy:
                    logger.warning(
                        "DFLASH pq verification requires a finite/small top_k (>0), but got topk=%s. "
                        "Falling back to target_only verification (draft uses argmax).",
                        topk,
                    )
                    self._warned_forced_greedy = True
                use_pq = False
                verify_mode = "target_only"
            else:
                step_count = int(self.block_size - 1)
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
                top_ps = torch.repeat_interleave(
                    top_ps, step_count, dim=0
                ).to(topk_p_flat.device)
                top_ks = torch.repeat_interleave(
                    top_ks, step_count, dim=0
                ).to(topk_p_flat.device)
                if draft_topk_cap > 0:
                    cap = torch.tensor(int(draft_topk_cap), device=top_ks.device, dtype=top_ks.dtype)
                    top_ks = torch.minimum(top_ks, cap)
                min_ps = torch.repeat_interleave(
                    min_ps, step_count, dim=0
                ).to(topk_p_flat.device)

                filtered_p = self._filter_topk_probs_for_sampling(
                    topk_p_flat,
                    temperatures=temps,
                    top_ks=top_ks,
                    top_ps=top_ps,
                    min_ps=min_ps,
                    need_min_p_sampling=bool(need_min_p_sampling),
                )
                # Optional invariants to catch silent normalization bugs during development.
                if (os.environ.get("SGLANG_DFLASH_PQ_ASSERTS") or "").strip().lower() not in (
                    "",
                    "0",
                    "false",
                    "off",
                    "no",
                ):
                    with torch.no_grad():
                        if torch.isnan(filtered_p).any() or torch.isinf(filtered_p).any():
                            raise RuntimeError("DFLASH pq: draft filtered_p contains NaN/Inf.")
                        s = filtered_p.sum(dim=1)
                        if not torch.allclose(s, torch.ones_like(s), atol=5e-3, rtol=1e-3):
                            raise RuntimeError(
                                f"DFLASH pq: draft filtered_p rows not normalized: min={float(s.min().item()):.6f} max={float(s.max().item()):.6f}"
                            )

                sampled_col = torch.multinomial(filtered_p, num_samples=1)
                sampled_ids = topk_id_flat.gather(1, sampled_col.to(torch.int64)).view(
                    -1
                )

                draft_tokens[:, 1:].copy_(
                    sampled_ids.view(bs, step_count).to(torch.long)
                )
                draft_topk = int(topk)
                draft_topk_ids = topk_id_flat.view(bs, step_count, int(topk))
                draft_topk_probs = filtered_p.view(bs, step_count, int(topk))

                # DAWN/FailFast-style per-request speculation cap based on draft confidence.
                # If early steps look uncertain (low q_max), cap the number of steps we attempt
                # so we don't waste work speculating deep into a hard/conflict region.
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
                        qmax = q.max(dim=-1).values  # [bs, step_count]
                        min_first = qmax[:, :look].min(dim=1).values
                        hard = min_first < float(qmax_lt)
                        max_steps_per_req = torch.where(
                            hard,
                            torch.full((bs,), int(cap_steps), device=q.device, dtype=torch.int32),
                            torch.full((bs,), int(step_count), device=q.device, dtype=torch.int32),
                        )
                    if self.tp_rank == 0 and not getattr(self, "_logged_dawn_cap", False):
                        try:
                            frac_hard = float((hard.to(torch.float32).mean()).item())
                        except Exception:
                            frac_hard = -1.0
                        logger.info(
                            "DFLASH DAWN cap enabled: qmax_lt=%.4f cap_steps=%d lookahead=%d frac_hard=%.3f",
                            float(qmax_lt),
                            int(cap_steps),
                            int(look),
                            float(frac_hard),
                        )
                        setattr(self, "_logged_dawn_cap", True)
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
                        q_max = q.max(dim=-1).values  # [bs, step_count]
                        q_ent = -(q_safe * torch.log(q_safe)).sum(dim=-1)  # [bs, step_count]
                        first = min(int(step_count), 4)
                        draft_conf_debug = {
                            "q_max_mean_first": float(q_max[:, :first].mean().item()),
                            "q_max_min_first": float(q_max[:, :first].min().item()),
                            "q_ent_mean_first": float(q_ent[:, :first].mean().item()),
                        }

        if not use_pq:
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
                lm_head=lm_head,
            ).view(bs, self.block_size - 1)
            draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            verify_mode=verify_mode,
            draft_topk=draft_topk,
            draft_topk_ids=draft_topk_ids,
            draft_topk_probs=draft_topk_probs,
            max_steps_per_req=max_steps_per_req,
        )
        if draft_conf_debug is not None:
            setattr(verify_input, "draft_conf_debug", draft_conf_debug)
            if self.tp_rank == 0 and not getattr(self, "_logged_draft_conf_debug", False):
                logger.info("DFLASH draft confidence (debug): %s", draft_conf_debug)
                setattr(self, "_logged_draft_conf_debug", True)
        backend_name = type(self.model_runner.attn_backend).__name__
        skip_custom_mask = backend_name in {
            "FlashInferAttnBackend",
            "FlashInferMLAAttnBackend",
            "FlashAttentionBackend",
            "TRTLLMHAAttnBackend",
            "TRTLLMMLABackend",
        }
        build_custom_mask = not skip_custom_mask
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

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

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

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

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

        return out_token_ids

    def _topk_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        topk: int,
        chunk_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k token selection over the target LM head in a TP-safe way.

        Returns:
            topk_p: (num_tokens, topk) float32, normalized over top-k logits.
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
                "DFLASH top-k sampling requires a vocab-parallel head with `weight` and `shard_indices`."
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

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

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
                global_vals, global_sel = torch.topk(gathered_vals, k=int(topk), dim=1)
                global_ids = gathered_ids.gather(1, global_sel.to(torch.int64))
            else:
                global_vals = local_vals.to(torch.float32)
                global_ids = local_ids

            probs = torch.softmax(global_vals, dim=1)
            out_p[start:end].copy_(probs)
            out_ids[start:end].copy_(global_ids)

        return out_p, out_ids

    def _filter_topk_probs_for_sampling(
        self,
        topk_probs: torch.Tensor,
        *,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        need_min_p_sampling: bool,
    ) -> torch.Tensor:
        """Apply SGLang-equivalent sampling filters to a (num_tokens, topk) distribution.

        `topk_probs` is assumed to be sorted (descending) along dim=1.
        """
        from sglang.srt.speculative.pq_filter import filter_topk_probs_like_sglang_sampler

        return filter_topk_probs_like_sglang_sampler(
            topk_probs,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            need_min_p_sampling=bool(need_min_p_sampling),
            # Match the CUDA sampler default: joint top-k/top-p when min_p is off.
            no_min_p_filter_apply_order="joint",
        )

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
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

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
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
            # Fast path for single request.
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

            # Batched gather of cache locations and positions.
            cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [bs, max_ctx]
            ctx_cache_loc = cache2d[mask].to(torch.int64)  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]
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

        draft_input.draft_seq_lens = draft_seq_lens + ctx_lens
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
                attn.attn.k_scale,
                attn.attn.v_scale,
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

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
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
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            for req, draft_len in zip(batch.reqs, batch.seq_lens_cpu, strict=True):
                req.dflash_draft_seq_len = int(draft_len)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
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

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        timing_flag = (os.environ.get("SGLANG_DFLASH_VERIFY_WALL_TIMING") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        t0 = time.perf_counter() if timing_flag else 0.0
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
            dflash_debug,
        ) = verify_input.verify(batch=batch, logits_output=logits_output, page_size=self.page_size)
        if timing_flag and self.tp_rank == 0 and not getattr(self, "_logged_verify_wall", False):
            dt = time.perf_counter() - t0
            logger.info(
                "DFLASH verify wall timing (one-shot): %.6fs bs=%d accept_len_sum=%d verify_mode=%s",
                float(dt),
                int(bs),
                int(sum(accept_length_per_req_cpu)),
                str(getattr(verify_input, "verify_mode", None)),
            )
            setattr(self, "_logged_verify_wall", True)

        # Adaptive pq controller (FailFast/EAFT-inspired): update proposal shaping based
        # on verify-side scalar stats. This never changes verifier correctness; it only
        # changes the draft proposal temperature multiplier for future rounds and may
        # temporarily disable pq if it is clearly counterproductive.
        try:
            sig = DFlashDifficultySignals.from_debug(
                verify_mode=str(getattr(verify_input, "verify_mode", "unknown")),
                dflash_debug=dflash_debug,
                draft_conf_debug=getattr(verify_input, "draft_conf_debug", None),
            )
            new_temp_mul, ctrl_dbg = self._adaptive_pq.on_verify_end(sig)
            self._adaptive_pq.cfg.temp_mul = float(new_temp_mul)
            if self._adaptive_pq.cfg.enabled and self.tp_rank == 0:
                if ctrl_dbg.get("pq_disabled_triggered"):
                    logger.info("DFLASH failfast: pq temporarily disabled: %s", ctrl_dbg)
                if ctrl_dbg.get("eaft_conflict") is True:
                    logger.info("DFLASH EAFT gate: confident conflict detected: %s", ctrl_dbg)
                if not getattr(self, "_logged_adaptive_pq", False):
                    logger.info("DFLASH adaptive_pq (first): %s", ctrl_dbg)
                    setattr(self, "_logged_adaptive_pq", True)
        except Exception:
            # Keep inference robust: never crash the scheduler on controller issues.
            pass

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. verify_mode=%s draft_topk=%s accept_length_per_req=%s",
                getattr(verify_input, "verify_mode", None),
                getattr(verify_input, "draft_topk", None),
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True
        if dflash_debug is not None and self.tp_rank == 0 and not getattr(self, "_logged_pq_step_stats", False):
            logger.info("DFLASH pq step stats (debug): %s", dflash_debug)
            setattr(self, "_logged_pq_step_stats", True)

        # Update per-request rolling difficulty state (used by higher-level schedulers/policies).
        try:
            beta = float(os.environ.get("SGLANG_DFLASH_REQ_EMA_BETA") or "0.9")
        except Exception:
            beta = 0.9
        for req, a_len in zip(batch.reqs, accept_length_per_req_cpu, strict=True):
            st = getattr(req, "dflash_difficulty_state", None)
            if st is None:
                st = DFlashReqDifficultyState()
                setattr(req, "dflash_difficulty_state", st)
            try:
                st.update(accept_len=int(a_len), verify_ct=int(getattr(req, "spec_verify_ct", 0)), ema_beta=float(beta))
            except Exception:
                pass

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
