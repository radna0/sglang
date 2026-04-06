from __future__ import annotations

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
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_target_only_cache_plan,
    apply_dflash_target_only_mapping_updates,
    apply_dflash_target_only_req_kv_accounting,
    build_dflash_target_only_cache_plan,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    commit_dflash_target_only_batch,
    dflash_sampling_info_uses_sampled_target,
    gather_dflash_committed_hidden,
    is_dflash_sampling_verify_available,
    materialize_dflash_target_only_commit_metadata,
    pack_dflash_target_only_commits,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + int(draft_token_num)
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


def _ensure_dflash_cpu_stage_buffer(
    batch: ScheduleBatch,
    *,
    attr_name: str,
    needed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    needed = int(needed)
    if needed < 0:
        raise ValueError(f"needed must be non-negative, got {needed}.")
    existing = getattr(batch, attr_name, None)
    cap = int(existing.numel()) if isinstance(existing, torch.Tensor) else 0
    if (
        not isinstance(existing, torch.Tensor)
        or existing.device.type != "cpu"
        or existing.dtype != dtype
        or cap < needed
    ):
        new_cap = max(needed, cap * 2 if cap > 0 else needed)
        existing = torch.empty(
            (new_cap,),
            dtype=dtype,
            device="cpu",
            pin_memory=bool(torch.cuda.is_available()),
        )
        setattr(batch, attr_name, existing)
    return existing[:needed]


def _assign_fixed_width_req_to_token_direct(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    positions_2d: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    if positions_2d.ndim != 2:
        raise ValueError(
            f"positions_2d must be 2D, got shape={tuple(positions_2d.shape)}."
        )
    bs, width = positions_2d.shape
    if int(out_cache_loc.numel()) != int(bs * width):
        raise ValueError(
            "Fixed-width req_to_token write size mismatch: "
            f"expected {bs * width} values, got {int(out_cache_loc.numel())}."
        )
    row_ids = req_pool_indices.to(torch.int64).unsqueeze(1).expand(-1, width)
    req_to_token[row_ids, positions_2d.to(torch.int64)] = out_cache_loc.view(
        bs, width
    ).to(dtype=req_to_token.dtype)


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
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

    # How many committed tokens are visible to the draft worker per request.
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
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    linear_mode: str = "draft=greedy,target=greedy"
    draft_selected_probs: torch.Tensor | None = None
    draft_proposal_indices: torch.Tensor | None = None
    draft_proposal_probs: torch.Tensor | None = None

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

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
        if int(page_size) == 1:
            positions_2d = self.positions.view(bs, self.draft_token_num)
            _assign_fixed_width_req_to_token_direct(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                positions_2d=positions_2d,
                out_cache_loc=batch.out_cache_loc,
            )
        else:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, object, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), None, empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # Keep speculative verify semantics consistent with normal sampling path.
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        candidates = self.draft_token.view(bs, self.draft_token_num)
        if dflash_sampling_info_uses_sampled_target(
            sampling_info, reqs=batch.reqs
        ):
            if not is_dflash_sampling_verify_available():
                raise RuntimeError(
                    "DFLASH sampled verification was requested, but the exact sampled verify "
                    "path is unavailable on this build/device."
                )
            accept_len, bonus, proposed_tokens = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                linear_mode=self.linear_mode,
                draft_selected_probs=self.draft_selected_probs,
                draft_proposal_indices=self.draft_proposal_indices,
                draft_proposal_probs=self.draft_proposal_probs,
                return_proposed_tokens=True,
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
            proposed_tokens = torch.zeros(
                (bs, self.draft_token_num), dtype=torch.int64, device=device
            )
            if self.draft_token_num > 1:
                accepted_prefix = candidates[:, 1:].to(torch.int64)
                prefix_cols = torch.arange(
                    self.draft_token_num - 1, device=device, dtype=torch.int32
                )[None, :]
                prefix_mask = prefix_cols < accept_len.unsqueeze(1)
                proposed_tokens[:, :-1] = torch.where(
                    prefix_mask,
                    accepted_prefix,
                    torch.zeros_like(accepted_prefix),
                )
            bonus_pos = accept_len.to(torch.long).clamp(
                min=0, max=int(self.draft_token_num - 1)
            )
            proposed_tokens.scatter_(1, bonus_pos.unsqueeze(1), bonus.unsqueeze(1))

        packed_commits = pack_dflash_target_only_commits(
            target_predict=proposed_tokens,
            accept_len=accept_len,
        )
        proposed_flat_cpu = _ensure_dflash_cpu_stage_buffer(
            batch,
            attr_name="_dflash_proposed_flat_cpu_stage",
            needed=int(packed_commits.proposed_flat.numel()),
            dtype=torch.int64,
        )
        commit_lens_cpu_t = _ensure_dflash_cpu_stage_buffer(
            batch,
            attr_name="_dflash_commit_lens_cpu_stage",
            needed=int(packed_commits.commit_lens.numel()),
            dtype=torch.int32,
        )
        proposed_flat_cpu.copy_(
            packed_commits.proposed_flat.to(dtype=torch.int64),
            non_blocking=bool(
                packed_commits.proposed_flat.is_cuda and proposed_flat_cpu.is_pinned()
            ),
        )
        commit_lens_cpu_t.copy_(
            packed_commits.commit_lens.to(dtype=torch.int32),
            non_blocking=bool(
                packed_commits.commit_lens.is_cuda and commit_lens_cpu_t.is_pinned()
            ),
        )
        if packed_commits.proposed_flat.is_cuda:
            torch.cuda.current_stream(device=device).synchronize()
        commit_results = commit_dflash_target_only_batch(
            reqs=batch.reqs,
            proposed_flat_cpu=proposed_flat_cpu,
            commit_lens_cpu=commit_lens_cpu_t,
            empty_error_prefix="DFLASH verify",
        )
        accept_length_per_req_cpu = [
            int(result.accepted_draft_tokens) for result in commit_results
        ]
        commit_lens_cpu = [int(result.commit_len) for result in commit_results]
        commit_metadata = materialize_dflash_target_only_commit_metadata(
            commit_results=commit_results,
            device=device,
            default_commit_lens=packed_commits.commit_lens,
            default_new_verified_id=packed_commits.default_new_verified_id,
        )
        commit_lens = commit_metadata.commit_lens
        new_verified_id = commit_metadata.new_verified_id

        cache_plan = build_dflash_target_only_cache_plan(
            out_cache_loc=batch.out_cache_loc,
            commit_lens=commit_lens,
            seq_lens=batch.seq_lens,
            draft_token_num=int(self.draft_token_num),
            page_size=int(page_size),
        )
        apply_dflash_target_only_cache_plan(
            batch=batch,
            cache_plan=cache_plan,
            page_size=int(page_size),
        )
        apply_dflash_target_only_req_kv_accounting(
            reqs=batch.reqs,
            commit_lens_cpu=commit_lens_cpu,
        )
        apply_dflash_target_only_mapping_updates(
            batch=batch,
            commit_lens=commit_lens,
            commit_lens_cpu=commit_lens_cpu,
            commit_lens_cpu_tensor=commit_lens_cpu_t,
            cache_plan=cache_plan,
        )

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        next_target_hidden = gather_dflash_committed_hidden(
            hidden_states=hidden,
            accepted_indices=cache_plan.accepted_indices,
        )

        return (
            new_verified_id,
            commit_lens,
            cache_plan,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
