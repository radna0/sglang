from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import os
import time
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
    apply_dflash_target_only_plain_device_defaults,
    build_dflash_target_only_cache_plan,
    can_dflash_target_only_use_plain_device_defaults,
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

import logging

logger = logging.getLogger(__name__)
_DFLASH_GREEDY_VERIFY_DEBUG_CT: int = 0
_DFLASH_GREEDY_VERIFY_LONGCTX_DEBUG_CT: int = 0
_DFLASH_VERIFY_PROFILE_CT: int = 0


def _should_profile_verify_details() -> bool:
    raw = (os.environ.get("SGLANG_DFLASH_PROFILE_VERIFY_DETAILS") or "").strip().lower()
    if raw not in ("1", "true", "yes", "y", "on"):
        return False

    try:
        every = max(
            1,
            int(
                (os.environ.get("SGLANG_DFLASH_PROFILE_VERIFY_DETAILS_EVERY") or "25").strip()
            ),
        )
    except Exception:
        every = 25

    global _DFLASH_VERIFY_PROFILE_CT
    _DFLASH_VERIFY_PROFILE_CT += 1
    return (_DFLASH_VERIFY_PROFILE_CT % every) == 0


def _get_dflash_target_only_small_cpu_pack_max() -> int:
    try:
        return max(
            0,
            int(
                (
                    os.environ.get("SGLANG_DFLASH_TARGET_ONLY_SMALL_CPU_PACK_MAX")
                    or "512"
                ).strip()
            ),
        )
    except Exception:
        return 512


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


def _get_dflash_cpu_copy_stream(
    batch: ScheduleBatch,
    *,
    device: torch.device,
) -> torch.cuda.Stream:
    stream = getattr(batch, "_dflash_cpu_copy_stream", None)
    stream_device_index = getattr(batch, "_dflash_cpu_copy_stream_device_index", None)
    device_index = int(device.index) if device.index is not None else torch.cuda.current_device()
    if (
        not isinstance(stream, torch.cuda.Stream)
        or stream_device_index != device_index
    ):
        stream = torch.cuda.Stream(device=device)
        setattr(batch, "_dflash_cpu_copy_stream", stream)
        setattr(batch, "_dflash_cpu_copy_stream_device_index", device_index)
    return stream


def _get_dflash_cpu_copy_event(
    batch: ScheduleBatch,
    *,
    device: torch.device,
) -> torch.cuda.Event:
    event = getattr(batch, "_dflash_cpu_copy_event", None)
    event_device_index = getattr(batch, "_dflash_cpu_copy_event_device_index", None)
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    if not isinstance(event, torch.cuda.Event) or event_device_index != device_index:
        event = torch.cuda.Event()
        setattr(batch, "_dflash_cpu_copy_event", event)
        setattr(batch, "_dflash_cpu_copy_event_device_index", device_index)
    return event


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
        target_next_token_ids: torch.Tensor | None = None,
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
        next_token_logits = logits_output.next_token_logits
        if isinstance(next_token_logits, torch.Tensor):
            device = next_token_logits.device
        elif isinstance(target_next_token_ids, torch.Tensor):
            device = target_next_token_ids.device
        else:
            device = self.draft_token.device
        profile_details = _should_profile_verify_details()
        verify_profile_ms: dict[str, float] = {}

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # Keep speculative verify semantics consistent with normal sampling path.
            if sampling_info.has_custom_logit_processor:
                if next_token_logits is None:
                    raise RuntimeError(
                        "DFLASH verify requires next_token_logits when custom logit processors are active."
                    )
                apply_custom_logit_processor(
                    next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                if next_token_logits is None:
                    raise RuntimeError(
                        "DFLASH verify requires next_token_logits when logit bias or penalizers are active."
                    )
                linear_penalty = torch.zeros(
                    (bs, next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                next_token_logits.add_(
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
            commit_source_tokens = proposed_tokens
        else:
            if target_next_token_ids is not None:
                if int(target_next_token_ids.numel()) != int(bs * self.draft_token_num):
                    raise RuntimeError(
                        "DFLASH greedy verify target_next_token_ids size mismatch: "
                        f"got {int(target_next_token_ids.numel())}, "
                        f"expected {int(bs * self.draft_token_num)}."
                    )
                target_predict = target_next_token_ids.to(
                    device=device, dtype=torch.int64, non_blocking=True
                ).view(bs, self.draft_token_num)
            else:
                if next_token_logits is None:
                    raise RuntimeError(
                        "DFLASH greedy verify needs either target_next_token_ids or next_token_logits."
                    )
                target_predict = torch.argmax(
                    next_token_logits, dim=-1
                ).view(bs, self.draft_token_num)
            accept_len, _bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
            global _DFLASH_GREEDY_VERIFY_DEBUG_CT
            global _DFLASH_GREEDY_VERIFY_LONGCTX_DEBUG_CT
            greedy_verify_debug_enabled = (
                (os.environ.get("SGLANG_DFLASH_DEBUG_GREEDY_VERIFY") or "")
                .strip()
                .lower()
                not in ("", "0", "false", "off", "no")
            ) or (
                (os.environ.get("SGLANG_DFLASH_PROFILE_VERIFY_DETAILS") or "")
                .strip()
                .lower()
                not in ("", "0", "false", "off", "no")
            )
            first_pos_dbg = None
            longctx_debug = False
            should_log_greedy_debug = False
            if greedy_verify_debug_enabled:
                try:
                    first_pos_dbg = int(
                        self.positions.view(bs, self.draft_token_num)[0, 0]
                        .detach()
                        .to("cpu")
                        .item()
                    )
                    longctx_debug = bool(first_pos_dbg >= 512)
                except Exception:
                    first_pos_dbg = None
                    longctx_debug = False
                if longctx_debug:
                    if _DFLASH_GREEDY_VERIFY_LONGCTX_DEBUG_CT < 10:
                        _DFLASH_GREEDY_VERIFY_LONGCTX_DEBUG_CT += 1
                        should_log_greedy_debug = True
                elif _DFLASH_GREEDY_VERIFY_DEBUG_CT < 10:
                    _DFLASH_GREEDY_VERIFY_DEBUG_CT += 1
                    should_log_greedy_debug = True
            if should_log_greedy_debug:
                with torch.no_grad():
                    try:
                        step_count = int(max(0, self.draft_token_num - 1))
                        if step_count > 0:
                            first_match = (candidates[:, 1] == target_predict[:, 0]).to(
                                torch.float32
                            )
                            first_match_frac = float(first_match.mean().item())
                        else:
                            first_match_frac = 0.0
                        accept_mean = float(accept_len.to(torch.float32).mean().item())
                        commit_lens_dbg = (
                            accept_len.to(device=device, dtype=torch.int32).clamp(
                                min=0, max=int(self.draft_token_num - 1)
                            )
                            + 1
                        )
                        # Log a tiny sample for quick diagnosis (token ids only).
                        samp_n = min(2, bs)
                        trace_width = min(6, self.draft_token_num)
                        cand0 = (
                            candidates[:samp_n, :trace_width].detach().to("cpu").tolist()
                        )
                        pred0 = (
                            target_predict[:samp_n, :trace_width].detach().to("cpu").tolist()
                        )
                        commit0 = (
                            commit_lens_dbg[:samp_n].detach().to("cpu").tolist()
                        )
                        alt0 = getattr(self, "_debug_alt_next0", None)
                        if isinstance(alt0, torch.Tensor):
                            alt0 = alt0[:samp_n].detach().to("cpu").tolist()
                        logger.info(
                            "DFLASH greedy-verify debug: bs=%d block=%d first_match_frac=%.3f accept_mean=%.3f "
                            "pos0=%s cand[:%d,:6]=%s target_pred[:%d,:6]=%s commit_lens=%s alt_next0=%s",
                            int(bs),
                            int(self.draft_token_num),
                            float(first_match_frac),
                            float(accept_mean),
                            "?" if first_pos_dbg is None else int(first_pos_dbg),
                            int(samp_n),
                            cand0,
                            int(samp_n),
                            pred0,
                            commit0,
                            alt0,
                        )
                    except Exception as e:
                        logger.info("DFLASH greedy-verify debug failed: %s", e)
            # In greedy mode, the committed verify sequence is exactly the target
            # prefix `target_predict[:, :accept_len+1]`. We do not need to
            # materialize a separate proposed_tokens buffer.
            commit_source_tokens = target_predict.to(torch.int64)

        # Dense CPU commit copies the full [bs, block] token matrix to host every step.
        # This is often slower than the packed path when acceptance is low (common in GPT-OSS
        # DFlash). Default to packed commits unless explicitly enabled.
        dense_enabled = (os.environ.get("SGLANG_DFLASH_TARGET_ONLY_DENSE_CPU_COMMIT") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        use_dense_cpu_commit = bool(
            dense_enabled
            and commit_source_tokens.is_cuda
            and int(commit_source_tokens.numel())
            <= _get_dflash_target_only_small_cpu_pack_max()
        )
        can_use_plain_device_defaults = can_dflash_target_only_use_plain_device_defaults(
            reqs=batch.reqs,
            max_commit_len=int(self.draft_token_num),
        )
        packed_token_count = 0
        if use_dense_cpu_commit:
            t_sub = time.perf_counter() if profile_details else 0.0
            default_commit_lens = accept_len.to(device=device, dtype=torch.int32).clamp(
                min=0, max=int(self.draft_token_num - 1)
            )
            default_commit_lens = default_commit_lens + 1
            default_new_verified_id = commit_source_tokens[
                torch.arange(bs, device=device),
                default_commit_lens.to(torch.int64) - 1,
            ].to(torch.int64)
            if profile_details:
                verify_profile_ms["pack_commits"] = (
                    time.perf_counter() - t_sub
                ) * 1000.0

            t_sub = time.perf_counter() if profile_details else 0.0
            proposed_dense_cpu = _ensure_dflash_cpu_stage_buffer(
                batch,
                attr_name="_dflash_proposed_dense_cpu_stage",
                needed=int(bs * self.draft_token_num),
                dtype=torch.int64,
            ).view(bs, self.draft_token_num)
            commit_lens_cpu_t = _ensure_dflash_cpu_stage_buffer(
                batch,
                attr_name="_dflash_commit_lens_cpu_stage",
                needed=int(default_commit_lens.numel()),
                dtype=torch.int32,
            )
            if profile_details:
                verify_profile_ms["stage_buffer"] = (
                    time.perf_counter() - t_sub
                ) * 1000.0

            t_sub = time.perf_counter() if profile_details else 0.0
            proposed_dense_src = commit_source_tokens
            if proposed_dense_src.dtype != torch.int64:
                proposed_dense_src = proposed_dense_src.to(torch.int64)
            copy_stream = _get_dflash_cpu_copy_stream(batch, device=device)
            current_stream = torch.cuda.current_stream(device=device)
            copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(copy_stream):
                proposed_dense_cpu.copy_(
                    proposed_dense_src.view(bs, self.draft_token_num),
                    non_blocking=bool(
                        commit_source_tokens.is_cuda and proposed_dense_cpu.is_pinned()
                    ),
                )
                commit_lens_cpu_t.copy_(
                    default_commit_lens,
                    non_blocking=bool(
                        default_commit_lens.is_cuda and commit_lens_cpu_t.is_pinned()
                    ),
                )
            if profile_details:
                verify_profile_ms["stage_copy"] = (time.perf_counter() - t_sub) * 1000.0

            t_sync_0 = time.perf_counter()
            # Prefer an event-based wait (waits only for the D2H copies we recorded) instead of
            # synchronizing the entire stream. This is still a host wait, but avoids an extra
            # full stream sync point in the profiler hot path.
            copy_done = _get_dflash_cpu_copy_event(batch, device=device)
            copy_done.record(copy_stream)
            if can_use_plain_device_defaults:
                if profile_details:
                    verify_profile_ms["verify_sync"] = 0.0
                    verify_profile_ms["commit_batch"] = 0.0
                    verify_profile_ms["commit_lists"] = 0.0

                t_sub = time.perf_counter() if profile_details else 0.0
                commit_metadata = materialize_dflash_target_only_commit_metadata(
                    commit_results=[],
                    device=device,
                    default_commit_lens=default_commit_lens,
                    default_new_verified_id=default_new_verified_id,
                )
                if profile_details:
                    verify_profile_ms["materialize_metadata"] = (
                        time.perf_counter() - t_sub
                    ) * 1000.0
                commit_lens_cpu = None
                packed_token_count = int(default_commit_lens.to(torch.int64).sum().item())
            else:
                copy_done.synchronize()
                if os.getenv("SGLANG_DFLASH_PROFILE"):
                    dt_sync = (time.perf_counter() - t_sync_0) * 1000
                    logger.info(f"[DFLASH_PROF] verify_synchronize: {dt_sync:.3f}ms")
                if profile_details:
                    verify_profile_ms["verify_sync"] = (time.perf_counter() - t_sync_0) * 1000.0

                t_commit_0 = time.perf_counter()
                commit_results = commit_dflash_target_only_batch(
                    reqs=batch.reqs,
                    proposed_dense_cpu=proposed_dense_cpu,
                    commit_lens_cpu=commit_lens_cpu_t,
                    empty_error_prefix="DFLASH verify",
                )
                if os.getenv("SGLANG_DFLASH_PROFILE"):
                    dt_commit = (time.perf_counter() - t_commit_0) * 1000
                    logger.info(f"[DFLASH_PROF] commit_dflash_batch: {dt_commit:.3f}ms")
                if profile_details:
                    verify_profile_ms["commit_batch"] = (
                        time.perf_counter() - t_commit_0
                    ) * 1000.0

                t_sub = time.perf_counter() if profile_details else 0.0
                accept_length_per_req_cpu = [
                    int(result.accepted_draft_tokens) for result in commit_results
                ]
                commit_lens_cpu = [int(result.commit_len) for result in commit_results]
                packed_token_count = int(sum(commit_lens_cpu))
                if profile_details:
                    verify_profile_ms["commit_lists"] = (
                        time.perf_counter() - t_sub
                    ) * 1000.0

                t_sub = time.perf_counter() if profile_details else 0.0
                commit_metadata = materialize_dflash_target_only_commit_metadata(
                    commit_results=commit_results,
                    device=device,
                    default_commit_lens=default_commit_lens,
                    default_new_verified_id=default_new_verified_id,
                )
                if profile_details:
                    verify_profile_ms["materialize_metadata"] = (
                        time.perf_counter() - t_sub
                    ) * 1000.0
        else:
            t_sub = time.perf_counter() if profile_details else 0.0
            packed_commits = pack_dflash_target_only_commits(
                target_predict=commit_source_tokens,
                accept_len=accept_len,
            )
            if profile_details:
                verify_profile_ms["pack_commits"] = (time.perf_counter() - t_sub) * 1000.0

            t_sub = time.perf_counter() if profile_details else 0.0
            packed_commits_are_cpu = (
                packed_commits.proposed_flat.device.type == "cpu"
                and packed_commits.commit_lens.device.type == "cpu"
            )
            if packed_commits_are_cpu:
                proposed_flat_cpu = packed_commits.proposed_flat.to(dtype=torch.int64)
                commit_lens_cpu_t = packed_commits.commit_lens.to(dtype=torch.int32)
            else:
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
            if profile_details:
                verify_profile_ms["stage_buffer"] = (time.perf_counter() - t_sub) * 1000.0

            t_sub = time.perf_counter() if profile_details else 0.0
            if not packed_commits_are_cpu:
                copy_stream = _get_dflash_cpu_copy_stream(batch, device=device)
                current_stream = torch.cuda.current_stream(device=device)
                copy_stream.wait_stream(current_stream)
                with torch.cuda.stream(copy_stream):
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
            if profile_details:
                verify_profile_ms["stage_copy"] = (time.perf_counter() - t_sub) * 1000.0
            need_copy_wait = bool(packed_commits.proposed_flat.is_cuda)
            if can_use_plain_device_defaults:
                if profile_details:
                    verify_profile_ms["verify_sync"] = 0.0
                    verify_profile_ms["commit_batch"] = 0.0
                    verify_profile_ms["commit_lists"] = 0.0

                t_sub = time.perf_counter() if profile_details else 0.0
                commit_metadata = materialize_dflash_target_only_commit_metadata(
                    commit_results=[],
                    device=device,
                    default_commit_lens=packed_commits.commit_lens,
                    default_new_verified_id=packed_commits.default_new_verified_id,
                )
                if profile_details:
                    verify_profile_ms["materialize_metadata"] = (
                        time.perf_counter() - t_sub
                    ) * 1000.0
                commit_lens_cpu = None
                packed_token_count = int(packed_commits.proposed_flat.numel())
            else:
                if need_copy_wait:
                    t_sync_0 = time.perf_counter()
                    copy_done = _get_dflash_cpu_copy_event(batch, device=device)
                    copy_done.record(copy_stream)
                    copy_done.synchronize()
                    if os.getenv("SGLANG_DFLASH_PROFILE"):
                        dt_sync = (time.perf_counter() - t_sync_0) * 1000
                        logger.info(f"[DFLASH_PROF] verify_synchronize: {dt_sync:.3f}ms")
                    if profile_details:
                        verify_profile_ms["verify_sync"] = (time.perf_counter() - t_sync_0) * 1000.0
                elif profile_details:
                    verify_profile_ms["verify_sync"] = 0.0

                t_commit_0 = time.perf_counter()
                commit_results = commit_dflash_target_only_batch(
                    reqs=batch.reqs,
                    proposed_flat_cpu=proposed_flat_cpu,
                    commit_lens_cpu=commit_lens_cpu_t,
                    empty_error_prefix="DFLASH verify",
                )
                if os.getenv("SGLANG_DFLASH_PROFILE"):
                    dt_commit = (time.perf_counter() - t_commit_0) * 1000
                    logger.info(f"[DFLASH_PROF] commit_dflash_batch: {dt_commit:.3f}ms")
                if profile_details:
                    verify_profile_ms["commit_batch"] = (time.perf_counter() - t_commit_0) * 1000.0

                t_sub = time.perf_counter() if profile_details else 0.0
                accept_length_per_req_cpu = [
                    int(result.accepted_draft_tokens) for result in commit_results
                ]
                commit_lens_cpu = [int(result.commit_len) for result in commit_results]
                packed_token_count = int(packed_commits.proposed_flat.numel())
                if profile_details:
                    verify_profile_ms["commit_lists"] = (time.perf_counter() - t_sub) * 1000.0

                t_sub = time.perf_counter() if profile_details else 0.0
                commit_metadata = materialize_dflash_target_only_commit_metadata(
                    commit_results=commit_results,
                    device=device,
                    default_commit_lens=packed_commits.commit_lens,
                    default_new_verified_id=packed_commits.default_new_verified_id,
                )
                if profile_details:
                    verify_profile_ms["materialize_metadata"] = (
                        time.perf_counter() - t_sub
                    ) * 1000.0
        commit_lens = commit_metadata.commit_lens
        new_verified_id = commit_metadata.new_verified_id
        mapping_commit_lens_cpu_tensor = (
            None if can_use_plain_device_defaults else commit_lens_cpu_t
        )

        t_sub = time.perf_counter() if profile_details else 0.0
        cache_plan = build_dflash_target_only_cache_plan(
            out_cache_loc=batch.out_cache_loc,
            commit_lens=commit_lens,
            commit_lens_cpu=commit_lens_cpu,
            seq_lens=batch.seq_lens,
            draft_token_num=int(self.draft_token_num),
            page_size=int(page_size),
        )
        if profile_details:
            verify_profile_ms["build_cache_plan"] = (time.perf_counter() - t_sub) * 1000.0

        t_sub = time.perf_counter() if profile_details else 0.0
        apply_dflash_target_only_cache_plan(
            batch=batch,
            cache_plan=cache_plan,
            page_size=int(page_size),
        )
        if profile_details:
            verify_profile_ms["apply_cache_plan"] = (time.perf_counter() - t_sub) * 1000.0

        t_sub = time.perf_counter() if profile_details else 0.0
        apply_dflash_target_only_req_kv_accounting(
            reqs=batch.reqs,
            commit_lens_cpu=(
                commit_lens_cpu if commit_lens_cpu is not None else []
            ),
        )
        if profile_details:
            verify_profile_ms["req_kv_accounting"] = (time.perf_counter() - t_sub) * 1000.0

        t_sub = time.perf_counter() if profile_details else 0.0
        apply_dflash_target_only_mapping_updates(
            batch=batch,
            commit_lens=commit_lens,
            commit_lens_cpu=commit_lens_cpu,
            commit_lens_cpu_tensor=mapping_commit_lens_cpu_tensor,
            cache_plan=cache_plan,
            draft_token_num=int(self.draft_token_num),
        )
        if profile_details:
            verify_profile_ms["mapping_updates"] = (time.perf_counter() - t_sub) * 1000.0

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        t_sub = time.perf_counter() if profile_details else 0.0
        next_target_hidden = gather_dflash_committed_hidden(
            hidden_states=hidden,
            accepted_indices=cache_plan.accepted_indices,
        )
        if can_use_plain_device_defaults:
            t_sync_0 = time.perf_counter()
            if use_dense_cpu_commit or packed_commits.proposed_flat.is_cuda:
                copy_done = _get_dflash_cpu_copy_event(batch, device=device)
                if not use_dense_cpu_commit:
                    copy_done.record(copy_stream)
                copy_done.synchronize()
            if os.getenv("SGLANG_DFLASH_PROFILE"):
                dt_sync = (time.perf_counter() - t_sync_0) * 1000
                logger.info(f"[DFLASH_PROF] verify_synchronize: {dt_sync:.3f}ms")
            if profile_details:
                verify_profile_ms["verify_sync"] = verify_profile_ms.get(
                    "verify_sync", 0.0
                ) + (time.perf_counter() - t_sync_0) * 1000.0

            t_commit_0 = time.perf_counter()
            if use_dense_cpu_commit:
                accept_length_per_req_cpu, commit_lens_cpu = (
                    apply_dflash_target_only_plain_device_defaults(
                        reqs=batch.reqs,
                        proposed_dense_cpu=proposed_dense_cpu,
                        commit_lens_cpu=commit_lens_cpu_t,
                    )
                )
            else:
                accept_length_per_req_cpu, commit_lens_cpu = (
                    apply_dflash_target_only_plain_device_defaults(
                        reqs=batch.reqs,
                        proposed_flat_cpu=proposed_flat_cpu,
                        commit_lens_cpu=commit_lens_cpu_t,
                    )
                )
            apply_dflash_target_only_req_kv_accounting(
                reqs=batch.reqs,
                commit_lens_cpu=commit_lens_cpu,
            )
            batch.seq_lens_cpu.add_(commit_lens_cpu_t.to(dtype=batch.seq_lens_cpu.dtype))
            batch.seq_lens_sum += int(commit_lens_cpu_t.sum().item())
            if os.getenv("SGLANG_DFLASH_PROFILE"):
                dt_commit = (time.perf_counter() - t_commit_0) * 1000
                logger.info(f"[DFLASH_PROF] commit_dflash_batch: {dt_commit:.3f}ms")
            if profile_details:
                verify_profile_ms["commit_batch"] = verify_profile_ms.get(
                    "commit_batch", 0.0
                ) + (time.perf_counter() - t_commit_0) * 1000.0
        if profile_details:
            verify_profile_ms["gather_hidden"] = (time.perf_counter() - t_sub) * 1000.0
            verify_profile_ms["verify_total"] = sum(verify_profile_ms.values())
            logger.info(
                "[DFLASH_VERIFY_PROF] bs=%d block=%d page=%d commit_sum=%d accept_sum=%d "
                "packed_tokens=%d accepted_indices=%d clear_tokens=%d "
                "pack_ms=%.3f stage_buf_ms=%.3f stage_copy_ms=%.3f sync_ms=%.3f "
                "commit_batch_ms=%.3f commit_lists_ms=%.3f metadata_ms=%.3f "
                "build_cache_ms=%.3f apply_cache_ms=%.3f req_kv_ms=%.3f mapping_ms=%.3f "
                "gather_hidden_ms=%.3f total_ms=%.3f",
                int(bs),
                int(self.draft_token_num),
                int(page_size),
                int(sum(commit_lens_cpu)),
                int(sum(accept_length_per_req_cpu)),
                int(packed_token_count),
                int(cache_plan.accepted_indices.numel()),
                int(cache_plan.clear_token_count),
                float(verify_profile_ms.get("pack_commits", 0.0)),
                float(verify_profile_ms.get("stage_buffer", 0.0)),
                float(verify_profile_ms.get("stage_copy", 0.0)),
                float(verify_profile_ms.get("verify_sync", 0.0)),
                float(verify_profile_ms.get("commit_batch", 0.0)),
                float(verify_profile_ms.get("commit_lists", 0.0)),
                float(verify_profile_ms.get("materialize_metadata", 0.0)),
                float(verify_profile_ms.get("build_cache_plan", 0.0)),
                float(verify_profile_ms.get("apply_cache_plan", 0.0)),
                float(verify_profile_ms.get("req_kv_accounting", 0.0)),
                float(verify_profile_ms.get("mapping_updates", 0.0)),
                float(verify_profile_ms.get("gather_hidden", 0.0)),
                float(verify_profile_ms.get("verify_total", 0.0)),
            )
        setattr(batch, "_dflash_verify_profile_ms", dict(verify_profile_ms))

        return (
            new_verified_id,
            commit_lens,
            cache_plan,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
