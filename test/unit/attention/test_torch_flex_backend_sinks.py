import math
import unittest

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import sys
from pathlib import Path

# Ensure the repo's `python/` is on sys.path when running this test directly.
repo_root = Path(__file__).resolve().parents[3]
repo_python = str(repo_root / "python")
if repo_python not in sys.path:
    sys.path.insert(0, repo_python)

from sglang.srt.layers.attention.flex_utils import (  # noqa: E402
    apply_attention_sinks,
    convert_logical_block_mask_to_physical_pages,
    make_extend_causal_mask_mod,
)


def _naive_attention_with_sink(q, k, v, *, causal: bool, scale: float, sinks: torch.Tensor):
    # q,k,v: [B, H, Q, D], [B, H, K, D], [B, H, K, D]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if causal:
        q_len = scores.shape[-2]
        k_len = scores.shape[-1]
        causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    # softmax in exp-domain; keep sink consistent with the max-subtraction.
    m = scores.max(dim=-1, keepdim=True).values
    m = torch.nan_to_num(m, nan=0.0, neginf=0.0, posinf=0.0)
    probs_unnorm = torch.exp(scores - m)
    # denom = sum(exp(score)) + sink
    #       = exp(m) * sum(exp(score-m)) + sink
    # => in shifted domain: denom_shifted = sum(exp(score-m)) + sink * exp(-m)
    sink_shift = sinks.view(1, -1, 1, 1).to(probs_unnorm.dtype) * torch.exp(-m)
    denom = probs_unnorm.sum(dim=-1, keepdim=True) + sink_shift
    probs = probs_unnorm / denom
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v)
    return out


class TestTorchFlexBackendSinks(unittest.TestCase):
    def test_blockmask_convert_logical_to_physical_pages(self):
        torch.manual_seed(0)
        B, H, Q, D = 1, 1, 4, 4
        page_size = 2
        num_logical_pages = 4  # KV_LEN = 8
        kv_len_logical = num_logical_pages * page_size
        num_physical_pages = 6  # KV_LEN_PHYS = 12
        kv_len_physical = num_physical_pages * page_size
        scale = 1.0 / math.sqrt(D)

        # logical page -> physical page mapping for this single batch item
        page_table = torch.tensor([[2, 0, 5, 1]], dtype=torch.int64)

        # Logical K/V sequence (8 tokens).
        q = torch.randn(B, H, Q, D)
        k_log = torch.randn(B, H, kv_len_logical, D)
        v_log = torch.randn(B, H, kv_len_logical, D)

        # Place logical pages into physical global pages (12 tokens) by page_table.
        k_phys = torch.zeros(B, H, kv_len_physical, D)
        v_phys = torch.zeros(B, H, kv_len_physical, D)
        for lp in range(num_logical_pages):
            pp = int(page_table[0, lp])
            k_phys[:, :, pp * page_size : (pp + 1) * page_size, :] = k_log[
                :, :, lp * page_size : (lp + 1) * page_size, :
            ]
            v_phys[:, :, pp * page_size : (pp + 1) * page_size, :] = v_log[
                :, :, lp * page_size : (lp + 1) * page_size, :
            ]

        # Causal mask in logical space (kv_idx <= q_idx).
        logical_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
            B,
            H,
            Q,
            kv_len_logical,
            device="cpu",
            BLOCK_SIZE=(page_size, page_size),
            _compile=False,
        )

        physical_mask = convert_logical_block_mask_to_physical_pages(
            logical_mask,
            page_table=page_table,
            num_physical_pages=num_physical_pages,
        )

        out_logical = flex_attention(q, k_log, v_log, block_mask=logical_mask, scale=scale)
        out_physical = flex_attention(q, k_phys, v_phys, block_mask=physical_mask, scale=scale)

        self.assertTrue(torch.allclose(out_logical, out_physical, rtol=1e-4, atol=1e-4))

    def test_blockmask_physical_pages_with_window(self):
        torch.manual_seed(0)
        B, H, Q, D = 1, 2, 8, 8
        page_size = 4
        num_logical_pages = 4  # KV_LEN = 16
        kv_len_logical = num_logical_pages * page_size
        num_physical_pages = 6  # KV_LEN_PHYS = 24
        kv_len_physical = num_physical_pages * page_size
        scale = 1.0 / math.sqrt(D)

        # logical page -> physical page mapping for this single batch item
        page_table = torch.tensor([[3, 0, 5, 1]], dtype=torch.int64)

        q = torch.randn(B, H, Q, D)
        k_log = torch.randn(B, H, kv_len_logical, D)
        v_log = torch.randn(B, H, kv_len_logical, D)

        k_phys = torch.zeros(B, H, kv_len_physical, D)
        v_phys = torch.zeros(B, H, kv_len_physical, D)
        for lp in range(num_logical_pages):
            pp = int(page_table[0, lp])
            k_phys[:, :, pp * page_size : (pp + 1) * page_size, :] = k_log[
                :, :, lp * page_size : (lp + 1) * page_size, :
            ]
            v_phys[:, :, pp * page_size : (pp + 1) * page_size, :] = v_log[
                :, :, lp * page_size : (lp + 1) * page_size, :
            ]

        # Sliding window: only attend to last 8 tokens (logical indices [8..15]).
        window_start = kv_len_logical - 8

        logical_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: (kv_idx >= window_start) & (q_idx >= kv_idx),
            B,
            H,
            Q,
            kv_len_logical,
            device="cpu",
            BLOCK_SIZE=(page_size, page_size),
            _compile=False,
        )
        physical_mask = convert_logical_block_mask_to_physical_pages(
            logical_mask,
            page_table=page_table,
            num_physical_pages=num_physical_pages,
        )

        out_logical = flex_attention(q, k_log, v_log, block_mask=logical_mask, scale=scale)
        out_physical = flex_attention(
            q, k_phys, v_phys, block_mask=physical_mask, scale=scale
        )
        self.assertTrue(torch.allclose(out_logical, out_physical, rtol=1e-4, atol=1e-4))

    def test_decode_window_truncation_matches_full_mask(self):
        torch.manual_seed(0)
        B, H, D = 1, 2, 8
        seq_len = 10
        window_left = 3
        abs_qpos = seq_len - 1
        kv_abs_start = max(0, abs_qpos - window_left)
        kv_abs_end = abs_qpos
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, 1, D)
        k_full = torch.randn(B, H, seq_len, D)
        v_full = torch.randn(B, H, seq_len, D)

        def full_mask(_b, _h, q_idx, kv_idx):
            # Allow only the sliding-window range [kv_abs_start, kv_abs_end].
            return (kv_idx >= kv_abs_start) & (kv_idx <= kv_abs_end)

        block_mask_full = create_block_mask(
            full_mask,
            B,
            H,
            1,
            seq_len,
            device="cpu",
            _compile=False,
        )
        out_full = flex_attention(q, k_full, v_full, block_mask=block_mask_full, scale=scale)

        k_slice = k_full[:, :, kv_abs_start : kv_abs_end + 1, :]
        v_slice = v_full[:, :, kv_abs_start : kv_abs_end + 1, :]
        out_slice = flex_attention(q, k_slice, v_slice, block_mask=None, scale=scale)

        self.assertTrue(torch.allclose(out_full, out_slice, rtol=1e-4, atol=1e-4))

    def test_extend_window_truncation_matches_full_mask(self):
        torch.manual_seed(0)
        B, H, D = 1, 2, 8
        seq_len = 10
        prefix_len = 6
        extend_len = seq_len - prefix_len
        window_left = 3
        abs_q0 = prefix_len
        abs_q_last = seq_len - 1
        kv_abs_start = max(0, abs_q_last - window_left)
        kv_abs_end = abs_q_last
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H, extend_len, D)
        k_full = torch.randn(B, H, seq_len, D)
        v_full = torch.randn(B, H, seq_len, D)

        def full_mask(_b, _h, q_idx, kv_idx):
            # Causal for extend tokens: kv_idx <= abs_q0 + q_idx
            # Sliding window: kv_idx >= kv_abs_start
            return (kv_idx >= kv_abs_start) & (kv_idx <= (abs_q0 + q_idx))

        block_mask_full = create_block_mask(
            full_mask,
            B,
            H,
            extend_len,
            seq_len,
            device="cpu",
            _compile=False,
        )
        out_full = flex_attention(q, k_full, v_full, block_mask=block_mask_full, scale=scale)

        k_slice = k_full[:, :, kv_abs_start : kv_abs_end + 1, :]
        v_slice = v_full[:, :, kv_abs_start : kv_abs_end + 1, :]

        q_kv_offset = abs_q0 - kv_abs_start
        block_mask_slice = create_block_mask(
            make_extend_causal_mask_mod(q_kv_offset=q_kv_offset),
            B,
            H,
            extend_len,
            (kv_abs_end - kv_abs_start + 1),
            device="cpu",
            _compile=False,
        )
        out_slice = flex_attention(q, k_slice, v_slice, block_mask=block_mask_slice, scale=scale)

        self.assertTrue(torch.allclose(out_full, out_slice, rtol=1e-4, atol=1e-4))

    def test_sink_scaling_matches_naive(self):
        torch.manual_seed(0)
        B, H, Q, K, D = 1, 2, 3, 5, 8
        q = torch.randn(B, H, Q, D)
        k = torch.randn(B, H, K, D)
        v = torch.randn(B, H, K, D)
        scale = 1.0 / math.sqrt(D)
        sinks = torch.tensor([0.25, 1.75], dtype=torch.float32)

        causal_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
            B,
            H,
            Q,
            K,
            device="cpu",
            _compile=False,
        )
        out, lse = flex_attention(q, k, v, block_mask=causal_mask, scale=scale, return_lse=True)

        out_sink = apply_attention_sinks(out, lse, sinks)
        out_naive = _naive_attention_with_sink(
            q, k, v, causal=True, scale=scale, sinks=sinks
        )

        self.assertTrue(torch.allclose(out_sink, out_naive, rtol=1e-4, atol=1e-4))

    def test_extend_offset_mask_matches_naive(self):
        torch.manual_seed(0)
        B, H, Q, K, D = 1, 2, 3, 5, 8
        q = torch.randn(B, H, Q, D)
        k = torch.randn(B, H, K, D)
        v = torch.randn(B, H, K, D)
        scale = 1.0 / math.sqrt(D)

        # q_kv_offset=2 => q_idx 0 attends up to kv_idx 2, q_idx 1 up to 3, q_idx 2 up to 4.
        q_kv_offset = 2

        block_mask = create_block_mask(
            make_extend_causal_mask_mod(q_kv_offset=q_kv_offset),
            B,
            H,
            Q,
            K,
            device="cpu",
            _compile=False,
        )
        out = flex_attention(q, k, v, block_mask=block_mask, scale=scale)

        # Naive mask: allow kv_idx <= q_kv_offset + q_idx
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        q_idx = torch.arange(Q).view(1, 1, Q, 1)
        kv_idx = torch.arange(K).view(1, 1, 1, K)
        allowed = kv_idx <= (q_kv_offset + q_idx)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_naive = torch.einsum("bhqk,bhkd->bhqd", probs, v)

        self.assertTrue(torch.allclose(out, out_naive, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
