from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    build_dflash_target_only_cache_plan,
    commit_dflash_proposed_tokens_to_req,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
    materialize_dflash_target_only_commit_metadata,
    pack_dflash_target_only_commits,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_DFLASH_PQ_STATS_LOGGED = False
_DFLASH_VERIFY_TIMING_DETAIL_LOGGED = False

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    Invariant (per request):
      - `draft_seq_len + ctx_len == batch.seq_lens[i]`
        where `ctx_len` is the number of target context-feature tokens carried in
        `target_hidden` for that request.
    """

    # Current token to start the next DFlash block (one per request).
    verified_id: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    ctx_lens: torch.Tensor

    # How many tokens are already in the draft KV cache per request.
    # The next draft step appends ctx_lens[i] tokens starting at draft_seq_lens[i].
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    # Verification rule under non-greedy sampling:
    #   - "target_only": sample from target p and accept only exact matches (current default).
    #   - "pq": true speculative sampling with draft probabilities q (accept with min(1, p/q)).
    verify_mode: str = "target_only"

    # Draft-side distribution (q) for pq verification.
    # Shape: [bs, draft_token_num - 1, draft_topk] where draft_topk matches the draft sampling top-k.
    # Only populated when verify_mode == "pq" and the request uses non-greedy sampling.
    draft_topk: int = 0
    draft_topk_ids: torch.Tensor | None = None
    draft_topk_probs: torch.Tensor | None = None

    # Optional per-request cap on how many speculative steps to attempt (FailFast/DAWN-style).
    # Shape: [bs] int32 on device; each value is in [1, draft_token_num - 1].
    # When provided, pq verification will "stop early" for requests whose cap is reached by
    # sampling the bonus token from the target distribution at that step and ending speculation.
    max_steps_per_req: torch.Tensor | None = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        prefix_lens = batch.seq_lens
        bs = int(prefix_lens.numel())
        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = prefix_lens + self.draft_token_num
        else:
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        # IMPORTANT: TARGET_VERIFY runs attention through the regular paged-KV backend, which
        # consumes `req_to_token_pool.req_to_token` (via metadata.page_table) to resolve KV
        # locations. For DFlash, the verify tokens are uncommitted, so we must temporarily
        # materialize their loc mapping here; later, only committed tokens are kept and the
        # rest are freed/ignored.
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            prefix_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
        debug_verify_mapping = (os.environ.get("SGLANG_DFLASH_DEBUG_VERIFY_MAPPING") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if debug_verify_mapping and not getattr(self, "_logged_verify_mapping_once", False):
            logger.info(
                "DFLASH verify wrote req_to_token mapping for uncommitted tokens (one-shot): "
                "page_size=%d bs=%d draft_token_num=%d",
                int(page_size),
                int(bs),
                int(self.draft_token_num),
            )
            setattr(self, "_logged_verify_mapping_once", True)

        debug_align = (os.environ.get("SGLANG_DFLASH_VERIFY_DEBUG_ALIGN") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if debug_align and int(page_size) > 1:
            try:
                loc2d = batch.out_cache_loc.view(bs, int(self.draft_token_num)).to(torch.int64)
                offs = torch.arange(int(self.draft_token_num), device=loc2d.device, dtype=torch.int64)
                pos_mod = (prefix_lens.to(torch.int64).unsqueeze(1) + offs.unsqueeze(0)) % int(page_size)
                loc_mod = loc2d % int(page_size)
                if not torch.all(loc_mod == pos_mod):
                    mismatch = (loc_mod != pos_mod)
                    mismatch_ct = int(mismatch.sum().item())
                    raise AssertionError(
                        "DFLASH verify paged-KV alignment mismatch: "
                        f"page_size={int(page_size)} mismatch_ct={mismatch_ct} "
                        f"bs={int(bs)} draft_token_num={int(self.draft_token_num)}"
                    )
            except Exception as e:
                # Make it loud and fail-fast for bisects.
                logger.error("DFLASH verify align check failed: %s", e)
                raise
            self.last_loc = last_loc

            debug_align = (os.environ.get("SGLANG_DFLASH_VERIFY_DEBUG_ALIGN") or "").strip().lower() not in (
                "",
                "0",
                "false",
                "off",
                "no",
            )
            if debug_align:
                try:
                    # Basic allocator invariant: (last_loc + 1) mod page_size == prefix_len mod page_size.
                    # If this fails, req_to_token mapping is already inconsistent for paged attention.
                    prefix_mod = (prefix_lens.to(torch.int64) % int(page_size)).to(torch.int64)
                    last_mod = ((last_loc.to(torch.int64) + 1) % int(page_size)).to(torch.int64)
                    if not bool(torch.all(prefix_mod == last_mod).item()):
                        bad = torch.nonzero(prefix_mod != last_mod, as_tuple=False).reshape(-1)
                        bad0 = int(bad[0].item()) if bad.numel() > 0 else -1
                        logger.error(
                            "DFLASH verify paged alloc invariant failed: (last_loc+1)%%page_size != prefix_len%%page_size. "
                            "page_size=%s bad_idx=%s prefix_mod=%s last_mod=%s prefix_len=%s last_loc=%s",
                            int(page_size),
                            bad0,
                            int(prefix_mod[bad0].item()) if bad0 >= 0 else None,
                            int(last_mod[bad0].item()) if bad0 >= 0 else None,
                            int(prefix_lens[bad0].item()) if bad0 >= 0 else None,
                            int(last_loc[bad0].item()) if bad0 >= 0 else None,
                        )

                    # Stronger invariant for paged KV tables used by FA3/FA4: loc%page_size must equal position%page_size
                    # for each appended token slot. This catches misaligned extend allocations that can corrupt verify.
                    bs = int(batch.batch_size())
                    q_len = int(self.draft_token_num)
                    if self.positions is not None and int(self.positions.numel()) == bs * q_len:
                        pos2d = self.positions.view(bs, q_len).to(torch.int64)
                        loc2d = batch.out_cache_loc.view(bs, q_len).to(torch.int64)
                        pos_mod2d = pos2d % int(page_size)
                        loc_mod2d = loc2d % int(page_size)
                        ok = torch.all(pos_mod2d == loc_mod2d)
                        if not bool(ok.item()):
                            bad = torch.nonzero(pos_mod2d != loc_mod2d, as_tuple=False)
                            b0 = bad[0].tolist() if bad.numel() > 0 else None
                            logger.error(
                                "DFLASH verify paged loc misalignment: loc%%page_size != position%%page_size. "
                                "page_size=%s first_bad=%s pos_mod=%s loc_mod=%s pos=%s loc=%s",
                                int(page_size),
                                b0,
                                int(pos_mod2d[b0[0], b0[1]].item()) if b0 is not None else None,
                                int(loc_mod2d[b0[0], b0[1]].item()) if b0 is not None else None,
                                int(pos2d[b0[0], b0[1]].item()) if b0 is not None else None,
                                int(loc2d[b0[0], b0[1]].item()) if b0 is not None else None,
                            )
                except Exception as e:
                    logger.exception("DFLASH verify debug-align failed: %s", e)

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            # Allow attending to the full prefix and to tokens up to (and including) the
            # current query position within the verify block (standard causal masking).
            allow = k_idx <= (prefix_len_i + q_idx)
            mask_chunks.append(allow.flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], Optional[Dict[str, Any]]]:
        """DFlash verification (greedy or sampling).

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, [], None

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        candidates = self.draft_token.view(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        timing_detail = (os.environ.get("SGLANG_DFLASH_VERIFY_TIMING_DETAIL") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        t0 = time.perf_counter() if timing_detail else 0.0
        t_after_logitproc = 0.0
        t_after_accept = 0.0
        t_after_pack = 0.0
        t_after_commit = 0.0
        t_after_kv_free = 0.0
        t_after_mapping = 0.0
        t_after_hidden = 0.0

        if sampling_info is not None:
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required
            ) or sampling_info.logit_bias is not None:
                # Relaxed penalties: treat the bias/penalty as constant within the verify block.
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        linear_penalty, self.draft_token_num, dim=0
                    )
                )

        if timing_detail:
            t_after_logitproc = time.perf_counter()

        use_pq = False
        if sampling_info is not None and not sampling_info.is_all_greedy:
            verify_mode = str(getattr(self, "verify_mode", "target_only"))
            topk_max = int(torch.max(sampling_info.top_ks).item())
            # PQ (p/q + residual sampling) is the distribution-correct speculative sampling
            # path. Enable by default; allow explicit opt-out if needed.
            pq_disable = (os.environ.get("SGLANG_DFLASH_PQ_DISABLE") or "").strip().lower() in (
                "1",
                "true",
                "on",
                "yes",
            )
            pq_enable = not pq_disable
            use_pq = (
                verify_mode == "pq"
                and pq_enable
                and self.draft_topk_ids is not None
                and self.draft_topk_probs is not None
                and int(self.draft_topk) > 0
                and topk_max > 0
                # SGLang uses a huge TOP_K_ALL value for "whole vocab"; pq needs a small finite top_k.
                and topk_max < (1 << 20)
            )
            if verify_mode == "pq" and not pq_enable and not getattr(self, "_warned_pq_enable", False):
                logger.warning(
                    "DFLASH pq requested but disabled (unset SGLANG_DFLASH_PQ_DISABLE to allow pq). Falling back to target_only."
                )
                setattr(self, "_warned_pq_enable", True)
            if (
                verify_mode == "pq"
                and pq_enable
                and not use_pq
                and not getattr(self, "_warned_pq_disabled", False)
            ):
                logger.warning(
                    "DFLASH pq requested but disabled: draft_topk=%s topk_max=%s has_q=%s",
                    getattr(self, "draft_topk", None),
                    topk_max,
                    bool(self.draft_topk_ids is not None and self.draft_topk_probs is not None),
                )
                setattr(self, "_warned_pq_disabled", True)

        dflash_debug: Optional[Dict[str, Any]] = None
        # Lightweight diagnostics for paged-KV bring-up: acceptance collapse often correlates with
        # verify blocks crossing page boundaries (prefix_len % page_size near the end of a page).
        # Keep this cheap (scalar stats only) so it can be enabled in production benchmarks.
        try:
            verify_mode_str = str(getattr(self, "verify_mode", "target_only") or "target_only")
        except Exception:
            verify_mode_str = "target_only"
        if int(page_size) > 1:
            try:
                prefix_lens_i64 = batch.seq_lens.to(torch.int64)
                prefix_mod = (prefix_lens_i64 % int(page_size)).to(torch.int64)
                cross_page = (prefix_mod + int(self.draft_token_num)) > int(page_size)
                dbg_base = {
                    "verify_mode": verify_mode_str,
                    "page_size": int(page_size),
                    "draft_token_num": int(self.draft_token_num),
                    "prefix_len_mean": float(prefix_lens_i64.float().mean().detach().cpu().item())
                    if int(prefix_lens_i64.numel()) > 0
                    else 0.0,
                    "prefix_mod_mean": float(prefix_mod.float().mean().detach().cpu().item())
                    if int(prefix_mod.numel()) > 0
                    else 0.0,
                    "prefix_mod_max": int(prefix_mod.max().detach().cpu().item())
                    if int(prefix_mod.numel()) > 0
                    else 0,
                    "cross_page_frac": float(cross_page.float().mean().detach().cpu().item())
                    if int(cross_page.numel()) > 0
                    else 0.0,
                }
            except Exception:
                dbg_base = {
                    "verify_mode": verify_mode_str,
                    "page_size": int(page_size),
                    "draft_token_num": int(self.draft_token_num),
                }
        else:
            dbg_base = {"verify_mode": verify_mode_str, "page_size": int(page_size), "draft_token_num": int(self.draft_token_num)}

        log_paged_stats = (os.environ.get("SGLANG_DFLASH_LOG_PAGED_VERIFY_STATS") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        if log_paged_stats and int(page_size) > 1 and not getattr(self, "_logged_paged_verify_stats", False):
            try:
                from sglang.srt.distributed import get_tp_group

                tp_rank = int(get_tp_group().rank)
            except Exception:
                tp_rank = 0
            if tp_rank == 0:
                logger.warning("DFLASH paged verify geometry (first): %s", dbg_base)
                print(f"[DFLASH] paged verify geometry (first): {dbg_base}", flush=True)
                setattr(self, "_logged_paged_verify_stats", True)

        if sampling_info is None or sampling_info.is_all_greedy or not use_pq:
            # Adaptive cap currently uses accept EMA plus draft-side q stats. Computing
            # full target-side entropy/max over the whole vocab on every greedy verify
            # round is much more expensive and should stay opt-in.
            targetonly_scalar_stats = (
                (os.environ.get("SGLANG_DFLASH_TARGETONLY_SCALAR_STATS") or "")
                .strip()
                .lower()
                not in ("", "0", "false", "off", "no")
            )
            if sampling_info is None or sampling_info.is_all_greedy:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, self.draft_token_num)
            else:
                sampled_helper_disable = (
                    (os.environ.get("SGLANG_DFLASH_DISABLE_TARGETONLY_SAMPLED_HELPER") or "")
                    .strip()
                    .lower()
                    not in ("", "0", "false", "off", "no")
                )
                sampled_helper_used = False
                if not sampled_helper_disable and is_dflash_sampling_verify_available():
                    try:
                        accept_len, bonus, target_predict = (
                            compute_dflash_sampling_accept_len_and_bonus(
                                candidates=candidates,
                                next_token_logits=logits_output.next_token_logits,
                                sampling_info=sampling_info,
                                max_steps_per_req=self.max_steps_per_req,
                                return_proposed_tokens=True,
                            )
                        )
                        sampled_helper_used = True
                    except Exception as e:
                        if not getattr(
                            self, "_warned_targetonly_sampled_helper_fallback", False
                        ):
                            logger.warning(
                                "DFLASH target_only sampled helper failed, falling back to inline path: %s",
                                e,
                            )
                            setattr(
                                self,
                                "_warned_targetonly_sampled_helper_fallback",
                                True,
                            )

                if not sampled_helper_used:
                    # Target-only speculative sampling: sample target tokens for each verify position
                    # and accept draft tokens only while they match the sampled target tokens.
                    #
                    # This produces the exact target distribution (same as baseline) without requiring
                    # draft probabilities q(token).
                    logits = logits_output.next_token_logits
                    expanded_temperature = torch.repeat_interleave(
                        sampling_info.temperatures, self.draft_token_num, dim=0
                    )
                    probs = torch.softmax(
                        logits / expanded_temperature.view(-1, 1), dim=-1
                    )

                    expanded_top_ks = torch.repeat_interleave(
                        sampling_info.top_ks, self.draft_token_num, dim=0
                    )
                    expanded_top_ps = torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    )

                    if sampling_info.need_min_p_sampling:
                        expanded_min_ps = torch.repeat_interleave(
                            sampling_info.min_ps, self.draft_token_num, dim=0
                        )
                        probs = top_k_renorm_prob(probs, expanded_top_ks)
                        probs = top_p_renorm_prob(probs, expanded_top_ps)
                        sampled = min_p_sampling_from_probs(probs, expanded_min_ps)
                    else:
                        sampled = top_k_top_p_sampling_from_probs(
                            probs.contiguous(),
                            expanded_top_ks,
                            expanded_top_ps,
                            filter_apply_order="joint",
                            check_nan=False,
                        )

                    target_predict = sampled.view(bs, self.draft_token_num)
            # Ensure we always return basic debug scalars so harnesses can correlate
            # accept collapse with paged-KV geometry.
            dflash_debug = dict(dbg_base)
            dflash_debug["targetonly_sampled_helper"] = (
                0
                if (sampling_info is None or sampling_info.is_all_greedy)
                else int(sampled_helper_used)
            )
            if targetonly_scalar_stats and int(self.draft_token_num) > 1:
                try:
                    step_count = int(self.draft_token_num - 1)
                    target_logits = logits_output.next_token_logits.view(
                        bs, self.draft_token_num, -1
                    )[:, :step_count, :].reshape(
                        -1, logits_output.next_token_logits.shape[-1]
                    )
                    target_log_probs = torch.log_softmax(
                        target_logits.to(torch.float32), dim=-1
                    )
                    target_probs = torch.exp(target_log_probs)
                    p_entropy = -(target_probs * target_log_probs).sum(dim=-1)
                    p_max = target_probs.max(dim=-1).values
                    dflash_debug["p_entropy_mean"] = float(p_entropy.mean().item())
                    dflash_debug["p_max_mean"] = float(p_max.mean().item())
                except Exception:
                    pass

            if sampling_info is None or sampling_info.is_all_greedy or not sampled_helper_used:
                accept_len, bonus = compute_dflash_accept_len_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                    max_steps_per_req=self.max_steps_per_req,
                )
            if timing_detail:
                t_after_accept = time.perf_counter()
        else:
            # True speculative sampling (maximal coupling) using draft probabilities q:
            # accept a draft token y with prob min(1, p(y)/q(y)); on rejection, sample
            # from the residual (p - q)+. This preserves the exact target distribution
            # while avoiding the entropy ceiling of target-only exact-match sampling.
            if (
                self.draft_topk_ids is None
                or self.draft_topk_probs is None
                or int(self.draft_topk) <= 0
            ):
                raise RuntimeError(
                    "DFLASH pq verification requires draft_topk_ids/draft_topk_probs."
                )

            step_count = int(self.draft_token_num - 1)
            num_rows = int(bs * self.draft_token_num)
            device = logits_output.next_token_logits.device

            # Build per-position target distributions p over top-k.
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            ).to(device)
            expanded_top_ks = torch.repeat_interleave(
                sampling_info.top_ks, self.draft_token_num, dim=0
            ).to(device)
            expanded_top_ps = torch.repeat_interleave(
                sampling_info.top_ps, self.draft_token_num, dim=0
            ).to(device)
            expanded_min_ps = torch.repeat_interleave(
                sampling_info.min_ps, self.draft_token_num, dim=0
            ).to(device)

            topk = int(torch.max(expanded_top_ks).item())
            if topk <= 0 or topk == -1:
                raise RuntimeError(
                    f"DFLASH pq verification requires finite top_k (>0), got topk={topk}."
                )

            logits = logits_output.next_token_logits
            if int(logits.shape[0]) != num_rows:
                raise RuntimeError(
                    f"DFLASH verify logits shape mismatch: logits.shape[0]={int(logits.shape[0])} vs expected {num_rows}."
                )
            p_vals, p_ids = torch.topk(logits, k=topk, dim=-1)
            p_vals = p_vals.to(torch.float32) / expanded_temperature.to(torch.float32).view(
                -1, 1
            )
            p_probs = torch.softmax(p_vals, dim=1)

            # Apply the same filtering semantics as SGLang's CUDA sampler:
            # - when min_p is OFF: joint top-k/top-p
            # - when min_p is ON:  top-k -> top-p -> min-p
            from sglang.srt.speculative.pq_filter import (
                filter_topk_probs_like_sglang_sampler,
            )

            p_probs = filter_topk_probs_like_sglang_sampler(
                p_probs,
                temperatures=torch.ones_like(expanded_temperature, device=device),
                top_ks=expanded_top_ks,
                top_ps=expanded_top_ps,
                min_ps=expanded_min_ps,
                need_min_p_sampling=bool(sampling_info.need_min_p_sampling),
                no_min_p_filter_apply_order="joint",
            )
            # Safety: multinomial asserts if probs contain NaN/Inf/negative or if rows are all-zero.
            # Keep verification robust even if upstream kernels produce transient NaNs.
            p_probs = torch.where(torch.isfinite(p_probs), p_probs, torch.zeros_like(p_probs)).clamp_min(0.0)
            p_denom = p_probs.sum(dim=-1, keepdim=True)
            p_fallback = torch.zeros_like(p_probs)
            p_fallback[:, :1] = 1.0
            p_probs = torch.where(p_denom > 0, p_probs / p_denom.clamp_min(1e-20), p_fallback)

            # Draft (q) distribution over top-k, provided by the draft worker.
            q_ids = self.draft_topk_ids.to(device)
            q_probs = self.draft_topk_probs.to(torch.float32).to(device)
            q_probs = torch.where(torch.isfinite(q_probs), q_probs, torch.zeros_like(q_probs)).clamp_min(0.0)
            q_denom = q_probs.sum(dim=-1, keepdim=True)
            q_fallback = torch.zeros_like(q_probs)
            q_fallback[:, :, :1] = 1.0
            q_probs = torch.where(q_denom > 0, q_probs / q_denom.clamp_min(1e-20), q_fallback)

            def _env_flag(name: str) -> bool:
                v = (os.environ.get(name) or "").strip().lower()
                return v not in ("", "0", "false", "off", "no")

            # Optional invariants to catch silent coupling/normalization bugs during development.
            asserts_flag = _env_flag("SGLANG_DFLASH_PQ_ASSERTS")
            if asserts_flag:
                with torch.no_grad():
                    if torch.isnan(q_probs).any() or torch.isinf(q_probs).any():
                        raise RuntimeError("DFLASH pq: q_probs contains NaN/Inf.")
                    q_s = q_probs.sum(dim=-1)  # [bs, step_count]
                    if not torch.allclose(
                        q_s, torch.ones_like(q_s), atol=5e-3, rtol=1e-3
                    ):
                        raise RuntimeError(
                            f"DFLASH pq: q_probs rows not normalized: min={float(q_s.min().item()):.6f} max={float(q_s.max().item()):.6f}"
                        )
                    if torch.isnan(p_probs).any() or torch.isinf(p_probs).any():
                        raise RuntimeError("DFLASH pq: p_probs contains NaN/Inf.")
                    p_s = p_probs.sum(dim=-1)  # [bs*block]
                    # Allow all-zero rows in extreme filter cases.
                    mask = p_s > 0
                    if bool(mask.any()):
                        if not torch.allclose(
                            p_s[mask],
                            torch.ones_like(p_s[mask]),
                            atol=5e-3,
                            rtol=1e-3,
                        ):
                            raise RuntimeError(
                                f"DFLASH pq: p_probs rows not normalized: min={float(p_s[mask].min().item()):.6f} max={float(p_s[mask].max().item()):.6f}"
                            )

            # candidates[:, 1:] are the proposed draft tokens y_1..y_K.
            proposed = candidates[:, 1:]
            accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)
            bonus = torch.empty((bs,), dtype=torch.int64, device=device)
            alive = torch.ones((bs,), dtype=torch.bool, device=device)

            # Step stats are expensive only in terms of extra storage/logging; the pq loop
            # already computes p/q terms. Scalar summaries are cheap and can be used for
            # adaptive controllers.
            collect_step_stats = _env_flag("SGLANG_DFLASH_PQ_STEP_STATS")
            collect_scalar_stats = (
                collect_step_stats
                or _env_flag("SGLANG_DFLASH_PQ_SCALAR_STATS")
                or _env_flag("SGLANG_DFLASH_ADAPTIVE_PQ")
            )
            # Diagnostics (TV/entropy/top-k overlap) add meaningful extra compute. Keep them
            # behind an explicit flag so we can always emit scalar acceptance stats without
            # perturbing throughput-critical benchmarks.
            collect_diag_stats = bool(collect_step_stats or _env_flag("SGLANG_DFLASH_PQ_DIAG_STATS"))
            need_diag = bool(collect_diag_stats)

            # Debug stats to understand acceptance under high-entropy sampling.
            # These help distinguish "algorithm bug" (p/q wrong) vs "draft distribution mismatch" (q far from p).
            p_y_sum = torch.zeros((), dtype=torch.float32, device=device)
            q_y_sum = torch.zeros((), dtype=torch.float32, device=device)
            a_sum = torch.zeros((), dtype=torch.float32, device=device)
            tv_sum = torch.zeros((), dtype=torch.float32, device=device)
            q_mass_in_p_sum = torch.zeros((), dtype=torch.float32, device=device)
            topk_overlap_sum = torch.zeros((), dtype=torch.float32, device=device)
            p_y_zero = torch.zeros((), dtype=torch.float32, device=device)
            q_y_zero = torch.zeros((), dtype=torch.float32, device=device)
            p_ent_sum = torch.zeros((), dtype=torch.float32, device=device)
            q_ent_sum = torch.zeros((), dtype=torch.float32, device=device)
            p_max_sum = torch.zeros((), dtype=torch.float32, device=device)
            q_max_sum = torch.zeros((), dtype=torch.float32, device=device)
            stat_count = torch.zeros((), dtype=torch.float32, device=device)

            step_stat_count = None
            step_a_sum = None
            step_tv_sum = None
            step_p_y_zero = None
            step_q_y_zero = None
            step_p_ent_sum = None
            step_q_ent_sum = None
            step_p_max_sum = None
            step_q_max_sum = None
            if collect_step_stats:
                step_stat_count = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_a_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_tv_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_p_y_zero = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_q_y_zero = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_p_ent_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_q_ent_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_p_max_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)
                step_q_max_sum = torch.zeros((step_count,), dtype=torch.float32, device=device)

            # Precompute row indices for each step: row = req_idx * block + step
            req_rows = (
                torch.arange(bs, device=device, dtype=torch.int64) * self.draft_token_num
            )

            for step in range(step_count):
                if not bool(alive.any()):
                    break
                row = req_rows + int(step)
                y = proposed[:, step]

                p_ids_step = p_ids[row]
                p_probs_step = p_probs[row]

                # DAWN/FailFast: stop early for some requests and sample the "bonus" token
                # immediately from the target distribution at this step. This keeps the
                # overall generation distribution exact, while avoiding wasted speculation
                # in hard/conflict regions.
                if self.max_steps_per_req is not None:
                    caps = self.max_steps_per_req.to(device)
                    stop_mask = alive & (int(step) >= caps.to(torch.int64))
                    if bool(stop_mask.any()):
                        stop_idx = torch.nonzero(stop_mask, as_tuple=False).view(-1)
                        p_ids_stop = p_ids_step[stop_idx]
                        p_probs_stop = p_probs_step[stop_idx]
                        sampled_col = torch.multinomial(p_probs_stop, num_samples=1).view(-1)
                        sampled_tok = p_ids_stop.gather(
                            1, sampled_col.to(torch.int64).view(-1, 1)
                        ).view(-1)
                        bonus[stop_idx] = sampled_tok
                        alive[stop_idx] = False
                        if not bool(alive.any()):
                            break
                p_y = (p_probs_step * (p_ids_step == y[:, None])).sum(dim=1)

                q_ids_step = q_ids[:, step, :]
                q_probs_step = q_probs[:, step, :]
                q_y = (q_probs_step * (q_ids_step == y[:, None])).sum(dim=1)

                ratio = torch.where(q_y > 0, p_y / q_y, torch.zeros_like(p_y))
                a = torch.minimum(ratio, torch.ones_like(ratio))

                tv = None
                p_ent = None
                q_ent = None
                p_max = None
                q_max = None
                q_on_p_sum = None
                topk_inter = None
                if need_diag:
                    # Estimate TV(p, q) for this step (both after filters).
                    # Achievable maximal-coupling match probability is 1 - TV.
                    q_on_p = (
                        q_probs_step[:, None, :]
                        * (q_ids_step[:, None, :] == p_ids_step[:, :, None])
                    ).sum(dim=2)  # [bs, topk]
                    q_on_p_sum = q_on_p.sum(dim=1)  # [bs]
                    # Intersection size between p top-k ids and q top-k ids.
                    # (Counts how many draft top-k tokens are also in target top-k.)
                    topk_inter = (
                        (q_ids_step[:, :, None] == p_ids_step[:, None, :])
                        .any(dim=2)
                        .sum(dim=1)
                        .to(torch.float32)
                    )
                    l1 = (p_probs_step - q_on_p).abs().sum(dim=1) + (
                        (1.0 - q_on_p_sum).clamp_min(0.0)
                    )
                    tv = 0.5 * l1  # [bs]

                    # Entropy / confidence diagnostics (computed on the *filtered/renormalized* supports).
                    p_probs_safe = p_probs_step.clamp_min(1e-20)
                    q_probs_safe = q_probs_step.clamp_min(1e-20)
                    p_ent = -(p_probs_safe * torch.log(p_probs_safe)).sum(dim=1)  # [bs]
                    q_ent = -(q_probs_safe * torch.log(q_probs_safe)).sum(dim=1)  # [bs]
                    p_max = p_probs_step.max(dim=1).values  # [bs]
                    q_max = q_probs_step.max(dim=1).values  # [bs]

                # Accumulate stats only for still-alive requests.
                mask_f = alive.to(torch.float32)
                stat_count = stat_count + mask_f.sum().to(torch.float32)
                p_y_sum = p_y_sum + (p_y.to(torch.float32) * mask_f).sum().to(torch.float32)
                q_y_sum = q_y_sum + (q_y.to(torch.float32) * mask_f).sum().to(torch.float32)
                a_sum = a_sum + (a.to(torch.float32) * mask_f).sum().to(torch.float32)
                p_y_zero = p_y_zero + ((p_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)
                q_y_zero = q_y_zero + ((q_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)
                if need_diag and tv is not None and q_on_p_sum is not None and topk_inter is not None:
                    tv_sum = tv_sum + (tv.to(torch.float32) * mask_f).sum().to(torch.float32)
                    q_mass_in_p_sum = q_mass_in_p_sum + (q_on_p_sum.to(torch.float32) * mask_f).sum().to(torch.float32)
                    topk_overlap_sum = topk_overlap_sum + (topk_inter.to(torch.float32) * mask_f).sum().to(torch.float32)
                if need_diag and p_ent is not None and q_ent is not None and p_max is not None and q_max is not None:
                    p_ent_sum = p_ent_sum + (p_ent.to(torch.float32) * mask_f).sum().to(torch.float32)
                    q_ent_sum = q_ent_sum + (q_ent.to(torch.float32) * mask_f).sum().to(torch.float32)
                    p_max_sum = p_max_sum + (p_max.to(torch.float32) * mask_f).sum().to(torch.float32)
                    q_max_sum = q_max_sum + (q_max.to(torch.float32) * mask_f).sum().to(torch.float32)

                if collect_step_stats and step_stat_count is not None:
                    step_stat_count[step] = step_stat_count[step] + mask_f.sum().to(torch.float32)
                    step_a_sum[step] = step_a_sum[step] + (a.to(torch.float32) * mask_f).sum().to(torch.float32)
                    step_p_y_zero[step] = step_p_y_zero[step] + ((p_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)
                    step_q_y_zero[step] = step_q_y_zero[step] + ((q_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)
                    if tv is not None:
                        step_tv_sum[step] = step_tv_sum[step] + (tv.to(torch.float32) * mask_f).sum().to(torch.float32)
                    if p_ent is not None and q_ent is not None and p_max is not None and q_max is not None:
                        step_p_ent_sum[step] = step_p_ent_sum[step] + (p_ent.to(torch.float32) * mask_f).sum().to(torch.float32)
                        step_q_ent_sum[step] = step_q_ent_sum[step] + (q_ent.to(torch.float32) * mask_f).sum().to(torch.float32)
                        step_p_max_sum[step] = step_p_max_sum[step] + (p_max.to(torch.float32) * mask_f).sum().to(torch.float32)
                        step_q_max_sum[step] = step_q_max_sum[step] + (q_max.to(torch.float32) * mask_f).sum().to(torch.float32)

                u = torch.rand_like(a)
                accept = (u < a) & alive
                reject = (~accept) & alive
                accept_len = accept_len + accept.to(torch.int32)

                if bool(reject.any()):
                    # Residual distribution on p's support: (p - q)+.
                    #
                    # Performance: compute the residual only for rejected rows, not the full batch.
                    rej_idx = torch.nonzero(reject, as_tuple=False).view(-1)
                    p_ids_rej = p_ids_step[rej_idx]
                    p_probs_rej = p_probs_step[rej_idx]
                    q_ids_rej = q_ids_step[rej_idx]
                    q_probs_rej = q_probs_step[rej_idx]

                    q_on_p_rej = (
                        q_probs_rej[:, None, :]
                        * (q_ids_rej[:, None, :] == p_ids_rej[:, :, None])
                    ).sum(dim=2)
                    resid = (p_probs_rej - q_on_p_rej).clamp_min(0.0)
                    resid = torch.where(
                        torch.isfinite(resid), resid, torch.zeros_like(resid)
                    ).clamp_min(0.0)
                    resid_sum = resid.sum(dim=1, keepdim=True)
                    resid = torch.where(
                        resid_sum > 0,
                        resid / resid_sum.clamp_min(1e-20),
                        p_probs_rej,
                    )
                    resid = torch.where(
                        torch.isfinite(resid), resid, torch.zeros_like(resid)
                    ).clamp_min(0.0)
                    resid_sum2 = resid.sum(dim=1, keepdim=True)
                    resid = torch.where(
                        resid_sum2 > 0,
                        resid / resid_sum2.clamp_min(1e-20),
                        p_probs_rej,
                    )
                    sampled_col = torch.multinomial(resid, num_samples=1).view(-1)
                    sampled_tok = p_ids_rej.gather(
                        1, sampled_col.to(torch.int64).view(-1, 1)
                    ).view(-1)
                    bonus[rej_idx] = sampled_tok
                    alive[rej_idx] = False

            if bool(alive.any()):
                # If all proposed draft tokens were accepted, sample the bonus token from
                # the next target distribution p_K (row = K).
                row = req_rows + int(step_count)
                p_ids_step = p_ids[row]
                p_probs_step = p_probs[row]
                p_probs_step = torch.where(
                    torch.isfinite(p_probs_step),
                    p_probs_step,
                    torch.zeros_like(p_probs_step),
                ).clamp_min(0.0)
                denom = p_probs_step.sum(dim=1, keepdim=True)
                p_probs_step = torch.where(
                    denom > 0, p_probs_step / denom.clamp_min(1e-20), p_probs_step
                )
                sampled_col = torch.multinomial(p_probs_step, num_samples=1).view(-1)
                sampled_tok = p_ids_step.gather(
                    1, sampled_col.to(torch.int64).view(-1, 1)
                ).view(-1)
                bonus[alive] = sampled_tok[alive]

            # Log stats once per process (tp-rank 0 only), to keep logs readable.
            global _DFLASH_PQ_STATS_LOGGED
            if need_diag and not _DFLASH_PQ_STATS_LOGGED:
                try:
                    from sglang.srt.distributed import get_tp_group

                    tp_rank = int(get_tp_group().rank)
                except Exception:
                    tp_rank = 0
                if tp_rank == 0:
                    denom = float(stat_count.detach().cpu().item()) + 1e-6
                    logger.info(
                        "DFLASH pq stats: count=%.0f mean_p_y=%.6f mean_q_y=%.6f mean_a=%.6f mean_tv=%.6f mean_match=%.6f mean_q_mass_in_p=%.6f mean_topk_overlap=%.2f frac_p_y_zero=%.4f frac_q_y_zero=%.4f",
                        denom,
                        float(p_y_sum.detach().cpu().item()) / denom,
                        float(q_y_sum.detach().cpu().item()) / denom,
                        float(a_sum.detach().cpu().item()) / denom,
                        float(tv_sum.detach().cpu().item()) / denom,
                        1.0 - (float(tv_sum.detach().cpu().item()) / denom),
                        float(q_mass_in_p_sum.detach().cpu().item()) / denom,
                        float(topk_overlap_sum.detach().cpu().item()) / denom,
                        float(p_y_zero.detach().cpu().item()) / denom,
                        float(q_y_zero.detach().cpu().item()) / denom,
                    )
                    _DFLASH_PQ_STATS_LOGGED = True

            denom_global = float(stat_count.detach().cpu().item()) + 1e-6
            if collect_scalar_stats and denom_global > 0:
                sample_mode = (os.environ.get("SGLANG_DFLASH_DRAFT_SAMPLE_MODE") or "").strip().lower()
                dflash_debug = {
                    "verify_mode": "pq",
                    "draft_topk": int(self.draft_topk),
                    "draft_sample_mode": sample_mode or "multinomial",
                    # When the draft uses argmax proposals, the verifier must treat q as a delta at argmax
                    # for correctness (see DFlashWorker). This flag lets benchmarks detect that regime.
                    "draft_q_is_delta": bool(sample_mode not in ("", "multinomial")),
                    # Accept-ratio stats: per-step acceptance probability min(1, p(y)/q(y)).
                    # NOTE: This is NOT the same as accept_length_mean (K), which is reported
                    # separately by the benchmark harness as spec_accept_length_mean.
                    "accept_ratio_mean": float(a_sum.detach().cpu().item()) / denom_global,
                    "p_y_mean": float(p_y_sum.detach().cpu().item()) / denom_global,
                    "q_y_mean": float(q_y_sum.detach().cpu().item()) / denom_global,
                    "frac_p_y_zero": float(p_y_zero.detach().cpu().item()) / denom_global,
                    "frac_q_y_zero": float(q_y_zero.detach().cpu().item()) / denom_global,
                    # Backward-compat aliases.
                    "a_mean": float(a_sum.detach().cpu().item()) / denom_global,
                }
                dflash_debug.update(dbg_base)
                if collect_diag_stats:
                    dflash_debug.update(
                        {
                            "tv_mean": float(tv_sum.detach().cpu().item()) / denom_global,
                            "q_mass_in_p_mean": float(q_mass_in_p_sum.detach().cpu().item()) / denom_global,
                            "topk_overlap_mean": float(topk_overlap_sum.detach().cpu().item()) / denom_global,
                            "p_entropy_mean": float(p_ent_sum.detach().cpu().item()) / denom_global,
                            "q_entropy_mean": float(q_ent_sum.detach().cpu().item()) / denom_global,
                            "p_max_mean": float(p_max_sum.detach().cpu().item()) / denom_global,
                            "q_max_mean": float(q_max_sum.detach().cpu().item()) / denom_global,
                        }
                    )
            if collect_step_stats and step_stat_count is not None and dflash_debug is not None:
                denom = step_stat_count.detach().cpu() + 1e-6
                dflash_debug.update(
                    {
                        "step_count": int(step_count),
                        "alive_frac_by_step": (step_stat_count.detach().cpu() / float(bs)).tolist(),
                        "accept_ratio_mean_by_step": (step_a_sum.detach().cpu() / denom).tolist(),
                        "tv_mean_by_step": (step_tv_sum.detach().cpu() / denom).tolist(),
                        "frac_p_y_zero_by_step": (step_p_y_zero.detach().cpu() / denom).tolist(),
                        "frac_q_y_zero_by_step": (step_q_y_zero.detach().cpu() / denom).tolist(),
                        "p_entropy_mean_by_step": (step_p_ent_sum.detach().cpu() / denom).tolist(),
                        "q_entropy_mean_by_step": (step_q_ent_sum.detach().cpu() / denom).tolist(),
                        "p_max_mean_by_step": (step_p_max_sum.detach().cpu() / denom).tolist(),
                        "q_max_mean_by_step": (step_q_max_sum.detach().cpu() / denom).tolist(),
                        # Backward-compat alias.
                        "a_mean_by_step": (step_a_sum.detach().cpu() / denom).tolist(),
                    }
                )

        if timing_detail and t_after_accept == 0.0:
            t_after_accept = time.perf_counter()

        # Compact D2H transfer for CPU-side commit logic.
        #
        # For `verify_mode=target_only`, commit only the accepted target prefix plus
        # bonus token per request rather than the full verify block.
        if use_pq:
            # PQ needs candidates (draft proposals) + accept_len + bonus.
            packed = torch.cat(
                [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
            ).cpu()
        else:
            packed_target_only = pack_dflash_target_only_commits(
                target_predict=target_predict,
                accept_len=accept_len,
            )
            packed = packed_target_only.proposed_flat.cpu()
            target_commit_offsets_cpu = packed_target_only.commit_offsets.cpu()
        if timing_detail:
            t_after_pack = time.perf_counter()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_results = []

        for i, req in enumerate(batch.reqs):
            if use_pq:
                # Layout: [candidates[:, 1:], accept_len, bonus]
                acc_len = int(packed[i, max_acc].item())
                proposed = packed[i, :acc_len].tolist() + [
                    int(packed[i, max_acc + 1].item())
                ]
            else:
                # Compact target_only layout: flattened committed prefixes plus per-req commit lens.
                start_offset = int(target_commit_offsets_cpu[i].item())
                end_offset = int(target_commit_offsets_cpu[i + 1].item())
                proposed = packed[start_offset:end_offset].tolist()

            outcome = commit_dflash_proposed_tokens_to_req(
                req=req,
                proposed=proposed,
                empty_error_prefix="DFLASH verify",
            )
            commit_results.append(outcome)
            accept_length_per_req_cpu.append(outcome.accepted_draft_tokens)

        if timing_detail:
            t_after_commit = time.perf_counter()

        commit_lens_cpu = [result.commit_len for result in commit_results]
        commit_metadata = materialize_dflash_target_only_commit_metadata(
            commit_results=commit_results,
            device=device,
        )
        commit_lens = commit_metadata.commit_lens
        new_verified_id = commit_metadata.new_verified_id
        cache_plan = build_dflash_target_only_cache_plan(
            out_cache_loc=batch.out_cache_loc,
            commit_lens=commit_lens,
            seq_lens=batch.seq_lens,
            draft_token_num=self.draft_token_num,
            page_size=page_size,
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        if page_size == 1:
            batch.token_to_kv_pool_allocator.free(cache_plan.evicted_slots)
            batch.out_cache_loc = cache_plan.compact_out_cache_loc
        else:
            # Page-size > 1: the allocator frees *pages* (not individual token slots),
            # so we must only evict full pages and never free a page that contains any
            # committed token (or any prefix token).
            #
            # DFlash always commits a prefix of the verify block (first `commit_len`
            # tokens), so there is at most one "mixed" page that contains both kept
            # and evicted tokens. We keep that whole page and only free pages fully
            # beyond the committed prefix (page-aligned).
            if int(cache_plan.evicted_slots.numel()) > 0:
                fast_free_used = False
                if (
                    hasattr(batch.token_to_kv_pool_allocator, "free_page_indices")
                    and cache_plan.evicted_pages is not None
                ):
                    if bool(
                        os.environ.get("SGLANG_DFLASH_DEBUG_PAGE_FREE", "")
                        .strip()
                        .lower()
                        not in ("", "0", "false", "off", "no")
                    ):
                        page_rows = cache_plan.evicted_slots.view(-1, int(page_size))
                        same_page = (
                            page_rows // int(page_size)
                        ) == cache_plan.evicted_pages[:, None]
                        if not bool(torch.all(same_page)):
                            raise RuntimeError(
                                "DFLASH paged fast-free invariant failed: evicted slots are not page-aligned."
                            )
                    batch.token_to_kv_pool_allocator.free_page_indices(
                        cache_plan.evicted_pages
                    )
                    fast_free_used = True
                if not fast_free_used:
                    batch.token_to_kv_pool_allocator.free(cache_plan.evicted_slots)

            # Compact the committed token slots for downstream mapping updates.
            batch.out_cache_loc = cache_plan.compact_out_cache_loc

        if timing_detail:
            t_after_kv_free = time.perf_counter()

        # Update req-level KV cache accounting.
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.decode_batch_idx += commit_len
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool mapping for newly committed tokens.
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
        if timing_detail:
            t_after_mapping = time.perf_counter()

        # Paged-KV safety: we temporarily wrote req_to_token mappings for the *entire*
        # verify block (including uncommitted speculative slots) so TARGET_VERIFY can
        # read them via metadata.page_table. After commit, some backends can still
        # consult page-aligned tables, so we must explicitly clear the uncommitted
        # tail mappings to a safe pad slot (0) to avoid any accidental visibility.
        clear_start = cache_plan.clear_start
        clear_end = cache_plan.clear_end
        if cache_plan.clear_token_count > 0:
            pad_locs = torch.zeros(
                (cache_plan.clear_token_count,), dtype=torch.int64, device=device
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                clear_start,
                clear_end,
                pad_locs,
                bs,
            )

        # Update batch seq lens.
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]
        if timing_detail:
            t_after_hidden = time.perf_counter()

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        if timing_detail:
            global _DFLASH_VERIFY_TIMING_DETAIL_LOGGED
            if not _DFLASH_VERIFY_TIMING_DETAIL_LOGGED:
                _DFLASH_VERIFY_TIMING_DETAIL_LOGGED = True
                verify_mode = str(getattr(self, "verify_mode", "target_only"))
                logger.info(
                    "DFLASH verify timing detail: mode=%s bs=%d tokens_committed=%d "
                    "t_logitsproc=%.6fs t_accept=%.6fs t_pack_d2h=%.6fs "
                    "t_commit_cpu=%.6fs t_kv_free=%.6fs t_mapping=%.6fs t_hidden=%.6fs total=%.6fs",
                    verify_mode,
                    int(bs),
                    int(sum(commit_lens_cpu)),
                    float(t_after_logitproc - t0),
                    float(t_after_accept - t_after_logitproc),
                    float(t_after_pack - t_after_accept),
                    float(t_after_commit - t_after_pack),
                    float(t_after_kv_free - t_after_commit),
                    float(t_after_mapping - t_after_kv_free),
                    float(t_after_hidden - t_after_mapping),
                    float(t_after_hidden - t0),
                )

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
            dflash_debug,
        )
