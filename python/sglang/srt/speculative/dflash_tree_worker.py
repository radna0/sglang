import logging
import math
import copy
from copy import deepcopy
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

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
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,
    resolve_dflash_mask_token,
    resolve_dflash_mask_token_id,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
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
        self._logged_first_verify = False

        # DFLASH_TREE config (validated in ServerArgs).
        self.block_size = int(server_args.speculative_dflash_block_size or 16)
        self.spec_steps = int(
            server_args.speculative_num_steps or (self.block_size - 1)
        )
        self.topk = int(server_args.speculative_eagle_topk or 4)
        self.num_verify_tokens = int(
            server_args.speculative_num_draft_tokens or self.block_size
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
                "DFLASH_TREE draft worker does not support 'trtllm_mha' yet; "
                "falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in ("flashinfer", "fa3"):
            logger.warning(
                "DFLASH_TREE draft worker only supports attention_backend in {'flashinfer', 'fa3'} for now, "
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

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
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
                qkv_weight=first_attn.qkv_proj.weight,
                n_layers=len(layers),
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
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

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # allocator and req_to_token_pool are shared with target worker
        pass

    def on_req_finished(self, req):
        if hasattr(req, "dflash_draft_seq_len"):
            req.dflash_draft_seq_len = 0

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
                global_vals, global_sel = torch.topk(
                    gathered_vals, k=int(topk), dim=1
                )
                global_ids = gathered_ids.gather(1, global_sel.to(torch.int64))
            else:
                global_vals = local_vals.to(torch.float32)
                global_ids = local_ids

            probs = torch.softmax(global_vals, dim=1)
            out_p[start:end].copy_(probs)
            out_ids[start:end].copy_(global_ids)

        return out_p, out_ids

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

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(draft_input.target_hidden)
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH_TREE ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
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

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise ValueError("DFLASH_TREE does not support grammar-constrained decoding yet.")

        bs = batch.batch_size()
        device = self.model_runner.device

        # 1) Append any newly committed tokens into the draft KV cache.
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
                spec_algorithm=SpeculativeAlgorithm.DFLASH_TREE,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_hidden = self.draft_model_runner.forward(
                    forward_batch
                ).logits_output
        finally:
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_hidden.view(bs, self.block_size, -1)

        step_count = int(self.spec_steps)
        if step_count <= 0 or step_count >= int(self.block_size):
            raise RuntimeError(
                f"DFLASH_TREE invalid spec_steps={step_count} for block_size={self.block_size}."
            )

        # 3) Compute per-position top-k candidates for the tree window.
        step_hidden = draft_hidden[:, 1 : 1 + step_count, :].reshape(
            -1, draft_hidden.shape[-1]
        )
        topk_p_flat, topk_index_flat = self._topk_from_vocab_parallel_head(
            hidden_states=step_hidden,
            lm_head=lm_head,
            topk=self.topk,
        )
        topk_p = topk_p_flat.view(bs, step_count, self.topk)
        topk_index = topk_index_flat.view(bs, step_count, self.topk)

        # 4) Build a bounded beam-style tree from per-step top-k candidates.
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        hidden_dummy = torch.empty((0, 0), device=device, dtype=torch.float32)
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

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.num_verify_tokens
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
        )

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
        )
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _verify_tree(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        logits_output,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
        sampling_info = batch.sampling_info
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
        is_all_greedy = sampling_info is None or sampling_info.is_all_greedy

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
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).reshape(
                bs, verify_input.draft_token_num
            )
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
                sampling_info.temperatures, verify_input.draft_token_num, dim=0
            )
            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, verify_input.draft_token_num, dim=0
                ),
            )
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, verify_input.draft_token_num, dim=0
                    ),
                )
            if sampling_info.need_min_p_sampling:
                target_probs = min_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.min_ps, verify_input.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)

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

        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()

        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            proposed: List[int] = []
            for idx in accept_index_cpu[i]:
                if idx == -1:
                    break
                proposed.append(int(predict_cpu[idx]))

            appended = 0
            if (
                req.grammar is None
                and not req.sampling_params.stop_strs
                and not req.sampling_params.stop_regex_strs
            ):
                remaining = int(req.sampling_params.max_new_tokens) - len(req.output_ids)
                if remaining > 0:
                    tokens = proposed[:remaining]
                    if not req.sampling_params.ignore_eos:
                        stop_token_ids = req.sampling_params.stop_token_ids
                        eos_token_ids = req.eos_token_ids
                        tokenizer = req.tokenizer
                        tokenizer_eos = (
                            tokenizer.eos_token_id if tokenizer is not None else None
                        )
                        additional_stop = (
                            tokenizer.additional_stop_token_ids
                            if tokenizer is not None
                            else None
                        )
                        vocab_size = getattr(req, "vocab_size", None)

                        for j, token_id in enumerate(tokens):
                            if vocab_size is not None and (
                                int(token_id) > int(vocab_size) or int(token_id) < 0
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if stop_token_ids and token_id in stop_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if eos_token_ids and token_id in eos_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if tokenizer_eos is not None and int(token_id) == int(
                                tokenizer_eos
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if additional_stop and token_id in additional_stop:
                                tokens = tokens[: j + 1]
                                break

                    req.output_ids.extend(int(tok) for tok in tokens)
                    appended = len(tokens)
                    if appended > 0:
                        req.check_finished(new_accepted_len=appended)
            else:
                for tok in proposed:
                    req.output_ids.append(int(tok))
                    appended += 1
                    req.check_finished()
                    if req.finished():
                        break
                    if req.grammar is not None:
                        req.grammar.accept_token(int(tok))

            if appended < len(proposed):
                accept_index[i, appended:] = -1

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH_TREE verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            if hasattr(req, "update_spec_acceptance_histogram"):
                req.update_spec_acceptance_histogram(accept_length_per_req_cpu[-1])

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(new_verified_list, dtype=torch.int64, device=device)

        if self.page_size != 1:
            raise NotImplementedError("DFLASH_TREE currently requires page_size == 1.")

        accept_index_flat = accept_index[accept_index != -1].to(torch.int64)
        evict_mask = torch.ones_like(verify_input.draft_token, dtype=torch.bool)
        if accept_index_flat.numel() > 0:
            evict_mask[accept_index_flat] = False
        batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        batch.out_cache_loc = batch.out_cache_loc[accept_index_flat]

        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += int(commit_len)
            req.kv_allocated_len = req.kv_committed_len

        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH_TREE verify requires target hidden states, but got None.")
        next_target_hidden = (
            hidden.index_select(0, accept_index_flat)
            if accept_index_flat.numel() > 0
            else hidden[:0]
        )
        logits_output.hidden_states = None

        return new_verified_id, commit_lens, next_target_hidden, accept_length_per_req_cpu

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH_TREE speculative decoding does not support return_logprob yet."
            )

        if isinstance(batch, ModelWorkerBatch):
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
            for req, draft_len in zip(batch.reqs, batch.seq_lens_cpu, strict=True):
                req.dflash_draft_seq_len = int(draft_len)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH_TREE decode requires DFlashDraftInput state on the running batch."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=getattr(batch.spec_info, "seq_lens_cpu", None)
        )
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, EagleVerifyInput)

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = self._verify_tree(
            batch=batch,
            verify_input=verify_input,
            logits_output=logits_output,
        )

        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
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
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
