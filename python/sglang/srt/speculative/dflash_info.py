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
from sglang.srt.managers.overlap_utils import FutureIndices
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
    commit_dflash_target_only_batch,
    commit_dflash_proposed_tokens_to_req,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    gather_dflash_committed_hidden,
    is_dflash_sampling_verify_available,
    materialize_dflash_target_only_commit_metadata,
    pack_dflash_target_only_commits,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)
_DFLASH_VERIFY_TIMING_DETAIL_LOGGED = False

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


def _top_p_is_effectively_disabled(top_ps: torch.Tensor) -> bool:
    if top_ps.numel() == 0:
        return True
    return bool(torch.all(top_ps >= 1.0).item())


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

    # Inputs for a future DFLASH-specific spec-v2 overlap payload.
    # This stores the post-append draft state, not the transient pre-append
    # target_hidden bundle.
    future_indices: Optional[FutureIndices] = None
    new_seq_lens: torch.Tensor | None = None
    verify_done: Optional[torch.cuda.Event] = None
    plain_decode_only: bool = False

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)
        if (
            self.new_seq_lens is None
            and self.draft_seq_lens is not None
            and self.ctx_lens is not None
            and self.draft_seq_lens.numel() == self.ctx_lens.numel()
        ):
            self.new_seq_lens = self.draft_seq_lens + self.ctx_lens

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    @classmethod
    def create_idle_input(cls, device: torch.device):
        return cls(
            verified_id=torch.empty((0,), dtype=torch.int64, device=device),
            target_hidden=torch.empty((0,), dtype=torch.float32, device=device),
            ctx_lens=torch.empty((0,), dtype=torch.int32, device=device),
            draft_seq_lens=torch.empty((0,), dtype=torch.int32, device=device),
            new_seq_lens=torch.empty((0,), dtype=torch.int32, device=device),
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        if batch.forward_mode.is_idle():
            return

        batch.maybe_evict_swa()
        batch.maybe_wait_verify_done()

        batch.input_ids = self.verified_id
        if self.new_seq_lens is not None:
            batch.seq_lens = self.new_seq_lens
            batch.seq_lens_cpu = self.new_seq_lens.to(
                dtype=torch.int32, device="cpu"
            )
            batch.seq_lens_sum = int(self.new_seq_lens.sum().item())

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            self.verified_id = self.verified_id[new_indices]
            self.draft_seq_lens = self.draft_seq_lens[new_indices]
            self.ctx_lens = self.ctx_lens[new_indices]
            if self.new_seq_lens is not None:
                self.new_seq_lens = self.new_seq_lens[new_indices]
            if self.target_hidden is not None and self.target_hidden.numel() > 0:
                self.target_hidden = self.target_hidden[:0]
            return

        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]
        if self.new_seq_lens is not None:
            self.new_seq_lens = self.new_seq_lens[new_indices]

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
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices],
                    dim=0,
                )
            )
            self.verified_id = torch.cat(
                [self.verified_id, spec_info.verified_id], dim=0
            )
            self.draft_seq_lens = torch.cat(
                [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
            )
            self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
            if self.new_seq_lens is None:
                self.new_seq_lens = spec_info.new_seq_lens
            elif spec_info.new_seq_lens is not None:
                self.new_seq_lens = torch.cat(
                    [self.new_seq_lens, spec_info.new_seq_lens], dim=0
                )
            if self.target_hidden is None or self.target_hidden.numel() == 0:
                self.target_hidden = spec_info.target_hidden
            elif (
                spec_info.target_hidden is not None
                and spec_info.target_hidden.numel() > 0
            ):
                self.target_hidden = torch.cat(
                    [self.target_hidden, spec_info.target_hidden], dim=0
                )
            self.plain_decode_only = bool(
                self.plain_decode_only or spec_info.plain_decode_only
            )
            return

        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.new_seq_lens is None:
            self.new_seq_lens = spec_info.new_seq_lens
        elif spec_info.new_seq_lens is not None:
            self.new_seq_lens = torch.cat(
                [self.new_seq_lens, spec_info.new_seq_lens], dim=0
            )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )
        self.plain_decode_only = bool(
            self.plain_decode_only or spec_info.plain_decode_only
        )


@dataclass
class DFlashVerifyTimingStats:
    tokens_committed: int
    logitsproc_s: float
    accept_s: float
    pack_gpu_s: float
    pack_d2h_s: float
    commit_cpu_s: float
    kv_free_s: float
    mapping_s: float
    hidden_s: float
    total_s: float


@dataclass
class DFlashVerifyResult:
    new_verified_id: torch.Tensor
    commit_lens: torch.Tensor
    next_target_hidden: torch.Tensor | None
    accept_length_per_req_cpu: List[int]
    dflash_debug: Optional[Dict[str, Any]]
    cache_plan: Any
    timing_stats: DFlashVerifyTimingStats | None = None


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

    # Verification rule under non-greedy sampling.
    # DFLASH now uses a single target-only contract:
    # sample from target p and accept only exact matches.
    verify_mode: str = "target_only"

    # Legacy draft-side proposal distribution storage.
    # Kept for compatibility with older checkpoints / payloads but unused by the
    # target-only verify contract.
    # Shape: [bs, draft_token_num - 1, draft_topk] where draft_topk matches the draft sampling top-k.
    draft_topk: int = 0
    draft_topk_ids: torch.Tensor | None = None
    draft_topk_probs: torch.Tensor | None = None

    # Optional per-request cap on how many speculative steps to attempt (FailFast/DAWN-style).
    # Shape: [bs] int32 on device; each value is in [0, draft_token_num - 1].
    # When provided, target-only verification will "stop early" for requests whose cap is reached by
    # sampling the bonus token from the target distribution at that step and ending speculation.
    max_steps_per_req: torch.Tensor | None = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    @classmethod
    def create_idle_input(
        cls,
        *,
        device: torch.device,
        draft_token_num: int,
        custom_mask: torch.Tensor | None,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            draft_token_num=int(draft_token_num),
            custom_mask=custom_mask,
            capture_hidden_mode=capture_hidden_mode,
        )

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
        need_target_hidden: bool = True,
    ) -> DFlashVerifyResult:
        """DFlash verification (greedy or sampling).

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim] when requested
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty_i64 = torch.empty((0,), dtype=torch.int64, device=batch.device)
            empty_i32 = torch.empty((0,), dtype=torch.int32, device=batch.device)
            return DFlashVerifyResult(
                new_verified_id=empty_i64,
                commit_lens=empty_i32,
                next_target_hidden=None,
                accept_length_per_req_cpu=[],
                dflash_debug=None,
                cache_plan=None,
                timing_stats=None,
            )

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
        profile_timing = timing_detail or (os.environ.get("SGLANG_DFLASH_PROFILE") or "").strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        )
        profile_sync = profile_timing and device.type != "cpu" and (
            os.environ.get("SGLANG_DFLASH_PROFILE_SYNC") or ""
        ).strip().lower() not in (
            "0",
            "false",
            "off",
            "no",
        )

        def _profile_sync() -> None:
            if not profile_sync:
                return
            torch.get_device_module(device).synchronize()

        if profile_timing:
            _profile_sync()
        t0 = time.perf_counter() if profile_timing else 0.0
        t_after_logitproc = 0.0
        t_after_accept = 0.0
        t_after_pack = 0.0
        t_after_d2h = 0.0
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

        if profile_timing:
            _profile_sync()
            t_after_logitproc = time.perf_counter()

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

        # Adaptive cap currently uses accept EMA plus draft-side q stats. Computing
        # full target-side entropy/max over the whole vocab on every greedy verify
        # round is much more expensive and should stay opt-in.
        targetonly_scalar_stats = (
            (os.environ.get("SGLANG_DFLASH_TARGETONLY_SCALAR_STATS") or "")
            .strip()
            .lower()
            not in ("", "0", "false", "off", "no")
        )
        sampled_helper_used = False
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
                # This preserves the exact target distribution without requiring draft q bookkeeping.
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
                    if not _top_p_is_effectively_disabled(expanded_top_ps):
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

        dflash_debug = dict(dbg_base)
        dflash_debug["targetonly_sampled_helper"] = (
            0 if (sampling_info is None or sampling_info.is_all_greedy) else int(sampled_helper_used)
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
        if profile_timing:
            _profile_sync()
            t_after_accept = time.perf_counter()

        if profile_timing and t_after_accept == 0.0:
            t_after_accept = time.perf_counter()

        # Single D2H transfer for CPU-side commit logic.
        #
        # For `verify_mode=target_only`, only the committed prefix needs to cross the
        # device boundary. Reuse the compact batch packer / committer already used by
        # the indexed tree path so linear DFLASH does not ship the full verify block.
        packed_target_only = pack_dflash_target_only_commits(
            target_predict=target_predict.to(torch.int64),
            accept_len=accept_len,
        )
        if profile_timing:
            _profile_sync()
            t_after_pack = time.perf_counter()

        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        proposed_flat_cpu = packed_target_only.proposed_flat.cpu()
        commit_offsets_cpu = packed_target_only.commit_offsets.cpu()
        if profile_timing:
            t_after_d2h = time.perf_counter()
        commit_results = commit_dflash_target_only_batch(
            reqs=batch.reqs,
            proposed_flat_cpu=proposed_flat_cpu,
            commit_offsets_cpu=commit_offsets_cpu,
            empty_error_prefix="DFLASH verify",
        )
        accept_length_per_req_cpu.extend(
            result.accepted_draft_tokens for result in commit_results
        )
        commit_lens_cpu.extend(result.commit_len for result in commit_results)
        commit_metadata = materialize_dflash_target_only_commit_metadata(
            commit_results=commit_results,
            device=device,
            default_commit_lens=packed_target_only.commit_lens,
            default_new_verified_id=packed_target_only.default_new_verified_id,
        )
        commit_lens = commit_metadata.commit_lens
        new_verified_id = commit_metadata.new_verified_id

        if profile_timing:
            _profile_sync()
            t_after_commit = time.perf_counter()
        cache_plan = build_dflash_target_only_cache_plan(
            out_cache_loc=batch.out_cache_loc,
            commit_lens=commit_lens,
            seq_lens=batch.seq_lens,
            draft_token_num=self.draft_token_num,
            page_size=page_size,
        )

        apply_dflash_target_only_cache_plan(
            batch=batch,
            cache_plan=cache_plan,
            page_size=page_size,
            debug_page_free=bool(
                os.environ.get("SGLANG_DFLASH_DEBUG_PAGE_FREE", "")
                .strip()
                .lower()
                not in ("", "0", "false", "off", "no")
            ),
        )

        if profile_timing:
            _profile_sync()
            t_after_kv_free = time.perf_counter()

        apply_dflash_target_only_req_kv_accounting(
            reqs=batch.reqs,
            commit_lens_cpu=commit_lens_cpu,
        )
        apply_dflash_target_only_mapping_updates(
            batch=batch,
            commit_lens=commit_lens,
            commit_lens_cpu=commit_lens_cpu,
            cache_plan=cache_plan,
        )
        if profile_timing:
            _profile_sync()
            t_after_mapping = time.perf_counter()

        next_target_hidden = None
        if need_target_hidden:
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "DFLASH verify requires target hidden states, but got None."
                )
            keep_mask = torch.arange(
                self.draft_token_num, device=device, dtype=torch.int32
            )[None, :] < commit_lens.unsqueeze(1)
            next_target_hidden = gather_dflash_committed_hidden(
                hidden_states=hidden,
                keep_mask=keep_mask,
                draft_token_num=self.draft_token_num,
            )
        if profile_timing:
            _profile_sync()
            t_after_hidden = time.perf_counter()

        logits_output.hidden_states = None

        timing_stats = None
        if profile_timing:
            timing_stats = DFlashVerifyTimingStats(
                tokens_committed=int(sum(commit_lens_cpu)),
                logitsproc_s=float(t_after_logitproc - t0),
                accept_s=float(t_after_accept - t_after_logitproc),
                pack_gpu_s=float(t_after_pack - t_after_accept),
                pack_d2h_s=float(t_after_d2h - t_after_pack),
                commit_cpu_s=float(t_after_commit - t_after_d2h),
                kv_free_s=float(t_after_kv_free - t_after_commit),
                mapping_s=float(t_after_mapping - t_after_kv_free),
                hidden_s=float(t_after_hidden - t_after_mapping),
                total_s=float(t_after_hidden - t0),
            )

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

        return DFlashVerifyResult(
            new_verified_id=new_verified_id,
            commit_lens=commit_lens,
            next_target_hidden=next_target_hidden,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            dflash_debug=dflash_debug,
            cache_plan=cache_plan,
            timing_stats=timing_stats,
        )
