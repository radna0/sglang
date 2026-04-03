import unittest

import torch


def _renorm(x: torch.Tensor) -> torch.Tensor:
    d = x.sum(dim=-1, keepdim=True)
    return torch.where(d > 0, x / d.clamp_min(1e-20), x)


def _ref_joint_filter(
    probs: torch.Tensor, *, top_k: torch.Tensor, top_p: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    probs = probs.to(torch.float32)
    probs = _renorm(probs)

    # top-p mask (match sgl-kernel test: ascending sort, keep if cdf > (1-p)-eps)
    sorted_prob, indices = torch.sort(probs, descending=False, dim=-1)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    keep_p = cdf > ((1.0 - top_p.to(torch.float32).view(-1, 1)) - eps)
    mask_p = torch.zeros_like(probs, dtype=torch.bool)
    mask_p.scatter_(1, indices, keep_p)

    # top-k mask (pivot + ties)
    sorted_desc, _ = torch.sort(probs, descending=True, dim=-1)
    pivot = sorted_desc.gather(1, (top_k.to(torch.int64).view(-1, 1) - 1).clamp_min(0))
    mask_k = probs >= pivot

    mask = mask_p & mask_k
    out = probs.masked_fill(~mask, 0.0)
    return _renorm(out)


def _ref_top_p_renorm(probs: torch.Tensor, *, top_p: torch.Tensor) -> torch.Tensor:
    probs = probs.to(torch.float32)
    probs = _renorm(probs)
    sorted_prob, indices = torch.sort(probs, descending=False, dim=-1)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    keep = cdf >= (1.0 - top_p.to(torch.float32).view(-1, 1))
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, indices, keep)
    out = probs.masked_fill(~mask, 0.0)
    return _renorm(out)


def _ref_top_k_mask_pivot(probs: torch.Tensor, *, top_k: torch.Tensor) -> torch.Tensor:
    probs = probs.to(torch.float32)
    sorted_desc, _ = torch.sort(probs, descending=True, dim=-1)
    pivot = sorted_desc.gather(
        1, (top_k.to(torch.int64).view(-1, 1) - 1).clamp_min(0)
    )
    return probs.masked_fill(probs < pivot, 0.0)


class TestDflashPQFilterSemantics(unittest.TestCase):
    def setUp(self):
        import sys

        if sys.platform == "win32":
            raise unittest.SkipTest("SGLang runtime is not supported on Windows.")

    def test_joint_matches_reference(self):
        from sglang.srt.speculative.pq_filter import filter_topk_probs_like_sglang_sampler

        torch.manual_seed(0)
        b, k = 32, 64
        raw = torch.rand(b, k, dtype=torch.float32)
        probs = raw / raw.sum(dim=-1, keepdim=True)

        top_ks = torch.randint(low=1, high=k + 1, size=(b,), dtype=torch.int32)
        top_ps = torch.rand(b, dtype=torch.float32) * 0.9 + 0.05
        temps = torch.rand(b, dtype=torch.float32) * 1.5 + 0.5

        out = filter_topk_probs_like_sglang_sampler(
            probs,
            temperatures=temps,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=torch.zeros_like(top_ps),
            need_min_p_sampling=False,
            no_min_p_filter_apply_order="joint",
        )

        # Reference: temp scaling then joint mask intersection then renorm.
        inv_t = (1.0 / temps.view(-1, 1)).clamp_max(1e6)
        p = probs.clamp_min(1e-20).pow(inv_t)
        p = _renorm(p)
        ref = _ref_joint_filter(p, top_k=top_ks, top_p=top_ps)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    def test_min_p_path_matches_reference(self):
        from sglang.srt.speculative.pq_filter import filter_topk_probs_like_sglang_sampler

        torch.manual_seed(1)
        b, k = 16, 50
        raw = torch.rand(b, k, dtype=torch.float32)
        probs = raw / raw.sum(dim=-1, keepdim=True)

        top_ks = torch.randint(low=1, high=k + 1, size=(b,), dtype=torch.int32)
        top_ps = torch.rand(b, dtype=torch.float32) * 0.9 + 0.05
        min_ps = torch.rand(b, dtype=torch.float32) * 0.2  # typical small min_p
        temps = torch.rand(b, dtype=torch.float32) * 1.2 + 0.6

        out = filter_topk_probs_like_sglang_sampler(
            probs,
            temperatures=temps,
            top_ks=top_ks,
            top_ps=top_ps,
            min_ps=min_ps,
            need_min_p_sampling=True,
            no_min_p_filter_apply_order="joint",
        )

        # Reference: temp scaling, top-k pivot mask + renorm, top-p renorm, min-p threshold, renorm.
        inv_t = (1.0 / temps.view(-1, 1)).clamp_max(1e6)
        p = probs.clamp_min(1e-20).pow(inv_t)
        p = _renorm(p)
        p = _ref_top_k_mask_pivot(p, top_k=top_ks)
        p = _renorm(p)
        p = _ref_top_p_renorm(p, top_p=top_ps)
        p = _renorm(p)
        thresh = p.max(dim=-1, keepdim=True).values * min_ps.view(-1, 1)
        p = p.masked_fill(p < thresh, 0.0)
        ref = _renorm(p)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
