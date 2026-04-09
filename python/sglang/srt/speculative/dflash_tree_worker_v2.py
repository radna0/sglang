from __future__ import annotations

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.speculative.dflash_tree_worker import DFlashTreeWorker


class DFlashTreeWorkerV2(DFlashTreeWorker):
    """DFLASH_TREE-native worker for the overlap/spec-v2 path."""

    def _forward_prefill_v2(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ):
        return self._forward_batch_generation_impl(
            model_worker_batch, overlap_v2=True, **kwargs
        )

    def _forward_decode_v2(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ):
        return self._forward_batch_generation_impl(
            model_worker_batch, overlap_v2=True, **kwargs
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ):
        if not isinstance(model_worker_batch, ModelWorkerBatch):
            raise TypeError(
                "DFLASH_TREE overlap-v2 worker expects ModelWorkerBatch at entry."
            )
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            return self._forward_prefill_v2(model_worker_batch, **kwargs)
        return self._forward_decode_v2(model_worker_batch, **kwargs)
