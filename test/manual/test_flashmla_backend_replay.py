from types import SimpleNamespace

import torch
import unittest

from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend


class _LayerStub:
    def __init__(self, sliding_window_size: int, tp_q_head_num: int = 2, head_dim: int = 4):
        self.sliding_window_size = sliding_window_size
        self.tp_q_head_num = tp_q_head_num
        self.head_dim = head_dim
        self.v_head_dim = 2


class TestFlashMLABackendMetadata(unittest.TestCase):
    def _make_backend(self) -> FlashMLABackend:
        backend = object.__new__(FlashMLABackend)
        import logging

        backend._logger = logging.getLogger(__name__)
        backend.cuda_graph_mla_metadata = torch.zeros((4, 8), dtype=torch.int32, device="cpu")
        backend.cuda_graph_num_splits = torch.zeros((5,), dtype=torch.int32, device="cpu")
        backend.cuda_graph_kv_indices = torch.arange(4 * 6, dtype=torch.int32).view(4, 6)
        backend.cuda_graph_mla_metadata_view = None
        backend.cuda_graph_num_splits_view = None
        backend.forward_metadata = None
        return backend

    def test_bind_cuda_graph_decode_metadata_refresh(self):
        backend = self._make_backend()
        backend._bind_cuda_graph_decode_metadata(
            bs=3,
            max_seqlen_pad=4,
            actual_num_sm_parts=2,
            mla_metadata=torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
                dtype=torch.int32,
            ),
            num_splits=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )
        self.assertIsNotNone(backend.forward_metadata)
        self.assertEqual(tuple(backend.forward_metadata.block_kv_indices.shape), (3, 4))
        self.assertEqual(backend.cuda_graph_mla_metadata_view.shape[0], 2)
        self.assertTrue(torch.equal(backend.cuda_graph_kv_indices[:3, :4], backend.forward_metadata.block_kv_indices))

    def test_bind_cuda_graph_decode_metadata_overflow(self):
        backend = self._make_backend()
        with self.assertRaises(RuntimeError):
            backend._bind_cuda_graph_decode_metadata(
                bs=3,
                max_seqlen_pad=2,
                actual_num_sm_parts=10,
                mla_metadata=torch.zeros((10, 8), dtype=torch.int32),
                num_splits=torch.zeros((4,), dtype=torch.int32),
            )


class TestFlashMLABackendSparseIndexing(unittest.TestCase):
    def _make_backend(self):
        backend = object.__new__(FlashMLABackend)
        backend.req_to_token = torch.tensor(
            [
                [100, 101, 102, 103],
                [102, 103, 104, 105],
            ],
            dtype=torch.int64,
        )
        return backend

    def test_decode_sparse_indices_are_request_local_offsets(self):
        backend = self._make_backend()
        k_cache = torch.arange(0, 200, dtype=torch.float32).view(200, 1, 1)
        layer = _LayerStub(sliding_window_size=-1, tp_q_head_num=1, head_dim=4)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([2, 1], dtype=torch.int64),
            token_to_kv_pool=SimpleNamespace(),
        )

        kv_dense, indices = backend._build_decode_dense_kv_and_indices(
            k_cache,
            layer,
            forward_batch,
        )

        self.assertEqual(tuple(kv_dense.shape), (3, 1, 1))
        self.assertEqual(tuple(indices.shape), (2, 1, 128))
        self.assertTrue(
            torch.equal(indices[0, 0, :2], torch.tensor([0, 1], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(indices[1, 0, :1], torch.tensor([2], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(
                kv_dense[:, 0, 0],
                torch.tensor([100.0, 101.0, 102.0]),
            )
        )

    def test_decode_sparse_sliding_window_offset(self):
        backend = self._make_backend()
        k_cache = torch.arange(0, 20, dtype=torch.float32).view(20, 1, 1)
        layer = _LayerStub(sliding_window_size=0, tp_q_head_num=1, head_dim=4)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
            token_to_kv_pool=SimpleNamespace(
                translate_loc_from_full_to_swa=lambda idx: idx - 99
            ),
        )

        _, indices = backend._build_decode_dense_kv_and_indices(k_cache, layer, forward_batch)
        self.assertTrue(
            torch.equal(indices[0, 0, :1], torch.tensor([2], dtype=torch.int32))
        )

    def test_extend_sparse_sliding_window_rebases_visible_suffix(self):
        backend = self._make_backend()
        k_cache = torch.arange(0, 40, dtype=torch.float32).view(40, 1, 1)
        layer = _LayerStub(sliding_window_size=1, tp_q_head_num=1, head_dim=4)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
            extend_prefix_lens_cpu=[3],
            extend_seq_lens_cpu=[1],
            token_to_kv_pool=SimpleNamespace(
                translate_loc_from_full_to_swa=lambda idx: idx - 99
            ),
        )

        kv_dense, indices = backend._build_extend_dense_kv_and_indices(
            k_cache,
            layer,
            forward_batch,
        )

        self.assertEqual(tuple(kv_dense.shape), (2, 1, 1))
        self.assertTrue(
            torch.equal(
                kv_dense[:, 0, 0],
                torch.tensor([3.0, 4.0]),
            )
        )
        self.assertTrue(
            torch.equal(indices[0, 0, :2], torch.tensor([0, 1], dtype=torch.int32))
        )


class TestFlashMLABackendAlignment(unittest.TestCase):
    def test_align_sparse_topk_rounds_up_to_aligned_boundary(self):
        backend = object.__new__(FlashMLABackend)
        self.assertEqual(backend._align_sparse_topk(0), 128)
        self.assertEqual(backend._align_sparse_topk(1), 128)
        self.assertEqual(backend._align_sparse_topk(129), 256)


if __name__ == "__main__":
    unittest.main()
