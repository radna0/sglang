from __future__ import annotations


class DFlashTreeDraftCudaGraphRunner:
    """Deferred on the clean tree baseline until DFLASH_TREE correctness is grounded."""

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "DFLASH_TREE draft CUDA graph runner is not enabled on the clean baseline yet."
        )
