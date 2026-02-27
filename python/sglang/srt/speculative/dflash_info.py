from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

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
from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_DFLASH_PQ_STATS_LOGGED = False

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

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
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
            self.last_loc = last_loc

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification (greedy or sampling).

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        candidates = self.draft_token.view(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

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

        use_pq = False
        if sampling_info is not None and not sampling_info.is_all_greedy:
            verify_mode = str(getattr(self, "verify_mode", "target_only"))
            topk_max = int(torch.max(sampling_info.top_ks).item())
            use_pq = (
                verify_mode == "pq"
                and self.draft_topk_ids is not None
                and self.draft_topk_probs is not None
                and int(self.draft_topk) > 0
                and topk_max > 0
                # SGLang uses a huge TOP_K_ALL value for "whole vocab"; pq needs a small finite top_k.
                and topk_max < (1 << 20)
            )
            if verify_mode == "pq" and not use_pq and not getattr(self, "_warned_pq_disabled", False):
                logger.warning(
                    "DFLASH pq requested but disabled: draft_topk=%s topk_max=%s has_q=%s",
                    getattr(self, "draft_topk", None),
                    topk_max,
                    bool(self.draft_topk_ids is not None and self.draft_topk_probs is not None),
                )
                setattr(self, "_warned_pq_disabled", True)

        if sampling_info is None or sampling_info.is_all_greedy or not use_pq:
            if sampling_info is None or sampling_info.is_all_greedy:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, self.draft_token_num)
            else:
                # Target-only speculative sampling: sample target tokens for each verify position
                # and accept draft tokens only while they match the sampled target tokens.
                #
                # This produces the exact target distribution (same as baseline) without requiring
                # draft probabilities q(token).
                logits = logits_output.next_token_logits
                expanded_temperature = torch.repeat_interleave(
                    sampling_info.temperatures, self.draft_token_num, dim=0
                )
                logits.div_(expanded_temperature)
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

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

            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
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

            # Per-row top-k (some requests may use smaller top_k than `topk`).
            top_ks_i = expanded_top_ks.to(torch.int64).clamp(min=1, max=topk).view(
                -1, 1
            )
            ar = torch.arange(topk, device=device, dtype=torch.int64).view(1, -1)
            p_probs = p_probs.masked_fill(ar >= top_ks_i, 0.0)
            p_probs = p_probs / p_probs.sum(dim=1, keepdim=True).clamp_min(1e-20)

            # Top-p filtering.
            tp = expanded_top_ps.to(torch.float32).clamp(min=0.0, max=1.0).view(-1, 1)
            cumsum = torch.cumsum(p_probs, dim=1)
            p_probs = p_probs.masked_fill((cumsum - p_probs) > tp, 0.0)

            # Min-p filtering (relative to the max prob).
            if sampling_info.need_min_p_sampling:
                mp = (
                    expanded_min_ps.to(torch.float32)
                    .clamp(min=0.0, max=1.0)
                    .view(-1, 1)
                )
                thresh = p_probs[:, :1] * mp
                p_probs = p_probs.masked_fill(p_probs < thresh, 0.0)

            p_denom = p_probs.sum(dim=1, keepdim=True)
            p_probs = torch.where(
                p_denom > 0, p_probs / p_denom.clamp_min(1e-20), p_probs
            )

            # Draft (q) distribution over top-k, provided by the draft worker.
            q_ids = self.draft_topk_ids.to(device)
            q_probs = self.draft_topk_probs.to(torch.float32).to(device)

            # candidates[:, 1:] are the proposed draft tokens y_1..y_K.
            proposed = candidates[:, 1:]
            accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)
            bonus = torch.empty((bs,), dtype=torch.int64, device=device)
            alive = torch.ones((bs,), dtype=torch.bool, device=device)

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
            stat_count = torch.zeros((), dtype=torch.float32, device=device)

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
                p_y = (p_probs_step * (p_ids_step == y[:, None])).sum(dim=1)

                q_ids_step = q_ids[:, step, :]
                q_probs_step = q_probs[:, step, :]
                q_y = (q_probs_step * (q_ids_step == y[:, None])).sum(dim=1)

                ratio = torch.where(q_y > 0, p_y / q_y, torch.zeros_like(p_y))
                a = torch.minimum(ratio, torch.ones_like(ratio))

                # Estimate TV(p, q) for this step (both after filters).
                # Achievable maximal-coupling match probability is 1 - TV.
                q_on_p = (
                    q_probs_step[:, None, :]
                    * (q_ids_step[:, None, :] == p_ids_step[:, :, None])
                ).sum(dim=2)  # [bs, topk]
                q_on_p_sum = q_on_p.sum(dim=1)  # [bs]
                # Intersection size between p top-k ids and q top-k ids.
                # (Counts how many draft top-k tokens are also in target top-k.)
                topk_inter = (q_ids_step[:, :, None] == p_ids_step[:, None, :]).any(dim=2).sum(dim=1).to(torch.float32)
                l1 = (p_probs_step - q_on_p).abs().sum(dim=1) + (1.0 - q_on_p_sum).clamp_min(0.0)
                tv = 0.5 * l1  # [bs]

                # Accumulate stats only for still-alive requests.
                mask_f = alive.to(torch.float32)
                stat_count = stat_count + mask_f.sum().to(torch.float32)
                p_y_sum = p_y_sum + (p_y.to(torch.float32) * mask_f).sum().to(torch.float32)
                q_y_sum = q_y_sum + (q_y.to(torch.float32) * mask_f).sum().to(torch.float32)
                a_sum = a_sum + (a.to(torch.float32) * mask_f).sum().to(torch.float32)
                tv_sum = tv_sum + (tv.to(torch.float32) * mask_f).sum().to(torch.float32)
                q_mass_in_p_sum = q_mass_in_p_sum + (q_on_p_sum.to(torch.float32) * mask_f).sum().to(torch.float32)
                topk_overlap_sum = topk_overlap_sum + (topk_inter.to(torch.float32) * mask_f).sum().to(torch.float32)
                p_y_zero = p_y_zero + ((p_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)
                q_y_zero = q_y_zero + ((q_y <= 0).to(torch.float32) * mask_f).sum().to(torch.float32)

                u = torch.rand_like(a)
                accept = (u < a) & alive
                reject = (~accept) & alive
                accept_len = accept_len + accept.to(torch.int32)

                if bool(reject.any()):
                    # Residual distribution on p's support: (p - q)+.
                    q_on_p = (
                        q_probs_step[:, None, :]
                        * (q_ids_step[:, None, :] == p_ids_step[:, :, None])
                    ).sum(dim=2)
                    resid = (p_probs_step - q_on_p).clamp_min(0.0)
                    resid_sum = resid.sum(dim=1, keepdim=True)
                    resid = torch.where(
                        resid_sum > 0,
                        resid / resid_sum.clamp_min(1e-20),
                        p_probs_step,
                    )
                    sampled_col = torch.multinomial(resid, num_samples=1).view(-1)
                    sampled_tok = p_ids_step.gather(
                        1, sampled_col.to(torch.int64).view(-1, 1)
                    ).view(-1)
                    bonus[reject] = sampled_tok[reject]
                    alive[reject] = False

            if bool(alive.any()):
                # If all proposed draft tokens were accepted, sample the bonus token from
                # the next target distribution p_K (row = K).
                row = req_rows + int(step_count)
                p_ids_step = p_ids[row]
                p_probs_step = p_probs[row]
                sampled_col = torch.multinomial(p_probs_step, num_samples=1).view(-1)
                sampled_tok = p_ids_step.gather(
                    1, sampled_col.to(torch.int64).view(-1, 1)
                ).view(-1)
                bonus[alive] = sampled_tok[alive]

            # Log stats once per process (tp-rank 0 only), to keep logs readable.
            global _DFLASH_PQ_STATS_LOGGED
            if not _DFLASH_PQ_STATS_LOGGED:
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

        # Single D2H transfer: candidates[1:] + accept_len + bonus
        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed[i, max_acc].item())
            proposed = packed[i, :acc_len].tolist() + [
                int(packed[i, max_acc + 1].item())
            ]

            appended = 0
            if (
                req.grammar is None
                and not req.sampling_params.stop_strs
                and not req.sampling_params.stop_regex_strs
            ):
                remaining = int(req.sampling_params.max_new_tokens) - len(
                    req.output_ids
                )
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

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # If no token was appended in this verify step, keep the current token unchanged.
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        if page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            # Page-size > 1 is not supported in the initial DFlash implementation.
            raise NotImplementedError(
                "DFLASH verify with page_size > 1 is not supported yet."
            )

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

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
