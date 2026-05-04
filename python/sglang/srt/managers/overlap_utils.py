from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import get_compiler_backend, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.dflash_info import DFlashDraftInput
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: Optional[SpeculativeAlgorithm] = None,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

    def _lazy_init_dflash_buf(self, draft_input: DFlashDraftInput):
        self.buf_initialized = True

        verified_id0 = draft_input.verified_id[0]
        draft_seq_lens0 = draft_input.draft_seq_lens[0]
        new_seq_lens0 = (
            draft_input.new_seq_lens[0]
            if draft_input.new_seq_lens is not None
            else draft_input.draft_seq_lens[0]
        )

        self.dflash_verified_id_buf = torch.empty(
            (self.future_buffer_len, *verified_id0.shape),
            dtype=verified_id0.dtype,
            device=self.device,
        )
        self.dflash_draft_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *draft_seq_lens0.shape),
            dtype=draft_seq_lens0.dtype,
            device=self.device,
        )
        self.dflash_new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )
        self.dflash_verify_done_buf = [None] * self.future_buffer_len

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        if self.spec_algo.is_dflash_family():
            self._lazy_init_dflash_buf(draft_input)
            return

        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        verified_id0 = draft_input.verified_id[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.verified_id_buf = torch.empty(
            (self.future_buffer_len, *verified_id0.shape),
            dtype=verified_id0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            if torch.device(self.device).type != "cpu":
                # The indices tensor was allocated on the default stream but is
                # used here on the forward stream. Meanwhile, the old spec_info
                # holding this tensor will lose all Python references (replaced at
                # model_worker_batch.spec_info and batch.spec_info), so the
                # caching allocator (torch GC) could reclaim the memory before
                # the GPU finishes reading it.
                indices.record_stream(
                    torch.get_device_module(self.device).current_stream()
                )
            if self.spec_algo.is_dflash_family():
                draft_input.verified_id = self.dflash_verified_id_buf[indices]
                draft_input.draft_seq_lens = self.dflash_draft_seq_lens_buf[indices]
                draft_input.new_seq_lens = self.dflash_new_seq_lens_buf[indices]
                draft_input.ctx_lens = torch.zeros_like(draft_input.draft_seq_lens)
                draft_input.target_hidden = torch.empty(
                    (0,), dtype=torch.float32, device=self.device
                )
                if getattr(self, "dflash_verify_done_buf", None) is not None:
                    draft_input.verify_done = self.dflash_verify_done_buf[
                        int(indices[0].item())
                    ]
                return

            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.verified_id = self.verified_id_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        if self.spec_algo.is_dflash_family():
            self.store_to_map_for_new_dflash_batch(future_indices, draft_input)
            return

        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.verified_id_buf[intv] = draft_input.verified_id
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states

    def store_to_map_for_new_dflash_batch(
        self, future_indices: FutureIndices, draft_input: DFlashDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            return

        if not self.buf_initialized:
            self._lazy_init_dflash_buf(draft_input)

        if draft_input.new_seq_lens is None:
            raise ValueError(
                "DFLASH overlap payload requires explicit new_seq_lens."
            )
        if draft_input.target_hidden is not None and draft_input.target_hidden.numel() > 0:
            raise ValueError(
                "DFLASH overlap payload must store post-append draft state with empty target_hidden."
            )
        if draft_input.ctx_lens is not None and draft_input.ctx_lens.numel() > 0:
            if int(draft_input.ctx_lens.sum().item()) != 0:
                raise ValueError(
                    "DFLASH overlap payload must store post-append draft state with zero ctx_lens."
                )

        self.dflash_verified_id_buf[intv] = draft_input.verified_id
        self.dflash_draft_seq_lens_buf[intv] = draft_input.draft_seq_lens
        self.dflash_new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if getattr(self, "dflash_verify_done_buf", None) is not None:
            verify_done = getattr(draft_input, "verify_done", None)
            for idx in range(int(intv.start), int(intv.stop)):
                self.dflash_verify_done_buf[idx] = verify_done
