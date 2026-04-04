import unittest
import threading
from types import SimpleNamespace

import torch

TRITON_AVAILABLE = True
try:
    import triton  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.dflash_worker import DFlashWorker


@unittest.skipUnless(TRITON_AVAILABLE, "triton is not installed")
class TestDFlashSsdShadowPrep(unittest.TestCase):
    def _make_worker(self, *, shadow_bs: int = 4, ctx: int = 16, block_size: int = 4):
        w = DFlashWorker.__new__(DFlashWorker)
        w.block_size = int(block_size)
        w.device = "cpu"
        w._draft_worker_lock = threading.Lock()

        shadow_pool = ReqToTokenPool(
            size=int(shadow_bs),
            max_context_len=int(ctx),
            device="cpu",
            enable_memory_saver=False,
        )
        shadow_pool.req_to_token.fill_(999)
        w._ssd_shadow_req_to_token_pool = shadow_pool
        w._ssd_shadow_req_pool_indices = torch.arange(
            int(shadow_bs), dtype=torch.int32, device="cpu"
        )
        w._ssd_shadow_block_cache_loc = torch.zeros(
            (int(shadow_bs) * int(block_size),), dtype=torch.int64, device="cpu"
        )

        real_pool = ReqToTokenPool(
            size=1, max_context_len=int(ctx), device="cpu", enable_memory_saver=False
        )
        real_pool.req_to_token[0].copy_(
            torch.arange(int(ctx), dtype=torch.int32, device="cpu") + 1000
        )
        w.draft_model_runner = SimpleNamespace(req_to_token_pool=real_pool)
        return w

    def test_prepare_shadow_copies_prefix_and_zeros_block(self):
        w = self._make_worker(shadow_bs=4, ctx=16, block_size=4)
        w._prepare_ssd_shadow_req_to_token(
            real_req_pool_index=torch.tensor([0], dtype=torch.int32),
            prefix_len=torch.tensor([5], dtype=torch.int32),
            rows=3,
        )

        shadow = w._ssd_shadow_req_to_token_pool.req_to_token
        real = w.draft_model_runner.req_to_token_pool.req_to_token[0]
        block_end = 5 + int(w.block_size)

        for r in range(3):
            self.assertTrue(torch.equal(shadow[r, :5], real[:5]))
            self.assertTrue(torch.equal(shadow[r, 5:block_end], torch.zeros(4, dtype=torch.int32)))
            self.assertTrue(torch.equal(shadow[r, block_end:], torch.full((16 - block_end,), 999, dtype=torch.int32)))

        # Unused row stays untouched.
        self.assertTrue(torch.equal(shadow[3], torch.full((16,), 999, dtype=torch.int32)))

    def test_prepare_shadow_rows_clamped_to_pool_size(self):
        w = self._make_worker(shadow_bs=2, ctx=12, block_size=4)
        w._prepare_ssd_shadow_req_to_token(
            real_req_pool_index=torch.tensor([0], dtype=torch.int32),
            prefix_len=torch.tensor([0], dtype=torch.int32),
            rows=99,
        )
        shadow = w._ssd_shadow_req_to_token_pool.req_to_token
        # Both rows should have the first block_size tokens zeroed; rest untouched.
        for r in range(2):
            self.assertTrue(torch.equal(shadow[r, :4], torch.zeros(4, dtype=torch.int32)))
            self.assertTrue(torch.equal(shadow[r, 4:], torch.full((8,), 999, dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
