import unittest

import torch

from sglang.srt.layers.attention.flex_flash4_sparse_utils import (
    fill_decode_sparse_buffers,
)


class FlexFlash4SparseUtilsTest(unittest.TestCase):
    def _alloc(self, bsz: int, num_blocks: int):
        return {
            "full_cnt": torch.zeros((bsz, 1, 1), dtype=torch.int32),
            "full_idx": torch.zeros((bsz, 1, 1, num_blocks), dtype=torch.int32),
            "mask_cnt": torch.zeros((bsz, 1, 1), dtype=torch.int32),
            "mask_idx": torch.zeros((bsz, 1, 1, num_blocks), dtype=torch.int32),
            "block_arange": torch.arange(num_blocks, dtype=torch.int32).view(1, -1),
        }

    def test_single_partial_block_becomes_one_mask_block(self):
        bufs = self._alloc(bsz=1, num_blocks=4)
        fill_decode_sparse_buffers(
            cache_seqlens=torch.tensor([65], dtype=torch.int32),
            window_left=-1,
            block_size=128,
            full_cnt=bufs["full_cnt"],
            full_idx=bufs["full_idx"],
            mask_cnt=bufs["mask_cnt"],
            mask_idx=bufs["mask_idx"],
            block_arange=bufs["block_arange"],
        )
        self.assertEqual(int(bufs["mask_cnt"][0, 0, 0].item()), 1)
        self.assertEqual(int(bufs["mask_idx"][0, 0, 0, 0].item()), 0)
        self.assertEqual(int(bufs["full_cnt"][0, 0, 0].item()), 0)

    def test_window_with_full_middle_and_partial_edges(self):
        bufs = self._alloc(bsz=1, num_blocks=8)
        # seqlen=500, window_left=300 -> attend [199, 499]
        # block size 128 => start block 1 (partial), end block 3 (partial), full block 2.
        fill_decode_sparse_buffers(
            cache_seqlens=torch.tensor([500], dtype=torch.int32),
            window_left=300,
            block_size=128,
            full_cnt=bufs["full_cnt"],
            full_idx=bufs["full_idx"],
            mask_cnt=bufs["mask_cnt"],
            mask_idx=bufs["mask_idx"],
            block_arange=bufs["block_arange"],
        )
        self.assertEqual(int(bufs["mask_cnt"][0, 0, 0].item()), 2)
        self.assertEqual(int(bufs["mask_idx"][0, 0, 0, 0].item()), 1)
        self.assertEqual(int(bufs["mask_idx"][0, 0, 0, 1].item()), 3)
        self.assertEqual(int(bufs["full_cnt"][0, 0, 0].item()), 1)
        self.assertEqual(int(bufs["full_idx"][0, 0, 0, 0].item()), 2)


if __name__ == "__main__":
    unittest.main()
