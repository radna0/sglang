from __future__ import annotations

from sglang.srt.speculative.dflash_worker import DFlashWorker


class DFlashWorkerV2(DFlashWorker):
    """DFlash-native worker for the future spec-v2 / overlap path.

    The overlap gate remains disabled until the DFlash-specific scheduler contract
    is fully validated end to end. This class exists so the v2 path can evolve
    without depending on Eagle worker semantics.
    """

    pass
