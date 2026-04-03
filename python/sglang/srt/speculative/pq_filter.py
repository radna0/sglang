from __future__ import annotations

import os

import torch


def _renorm_rows(probs: torch.Tensor) -> torch.Tensor:
    denom = probs.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, probs / denom.clamp_min(1e-20), probs)


def _apply_top_k_mask_pivot(
    probs: torch.Tensor, top_ks: torch.Tensor
) -> torch.Tensor:
    """Kernel-equivalent top-k masking (ties included via pivot)."""
    k = int(probs.shape[-1])
    top_ks = top_ks.to(torch.int64).clamp(min=1, max=k)
    sorted_desc, _ = torch.sort(probs, descending=True, dim=-1)
    pivot = sorted_desc.gather(1, (top_ks - 1).view(-1, 1))
    return probs.masked_fill(probs < pivot, 0.0)


def _apply_top_p_mask(
    probs: torch.Tensor, top_ps: torch.Tensor, *, joint_eps: float | None
) -> torch.Tensor:
    """Kernel-equivalent top-p masking (ties handled by scatter mask).

    When `joint_eps` is provided, match the joint top-k+top-p kernel's tolerance:
      keep if cdf > (1 - p) - eps
    Otherwise, match `top_p_renorm_prob` semantics:
      keep if cdf >= (1 - p)
    """
    top_ps = top_ps.to(torch.float32).clamp(min=0.0, max=1.0).view(-1, 1)
    probs_sort, idx = torch.sort(probs, descending=False, dim=-1)
    cdf = torch.cumsum(probs_sort, dim=-1)
    thresh = (1.0 - top_ps).to(cdf.dtype)
    if joint_eps is None:
        keep = cdf >= thresh
    else:
        keep = cdf > (thresh - float(joint_eps))
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, idx, keep)
    return probs.masked_fill(~mask, 0.0)


def filter_topk_probs_like_sglang_sampler(
    topk_probs_desc: torch.Tensor,
    *,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    no_min_p_filter_apply_order: str = "joint",
    joint_eps: float = 1e-4,
) -> torch.Tensor:
    """Apply SGLang-equivalent sampling filters to a (num_tokens, topk) distribution.

    This is intended for DFlash pq-style speculative sampling where we only have a
    truncated top-k distribution.

    Notes:
    - SGLang's CUDA sampler uses `filter_apply_order="joint"` for top-k/top-p when
      min_p is disabled, and uses sequential top-k -> top-p -> min-p when min_p is enabled.
    - Input is expected to be sorted in descending order along dim=-1.
    """
    if topk_probs_desc.numel() == 0:
        return topk_probs_desc

    probs = topk_probs_desc.to(torch.float32)

    # Temperature scaling on probabilities (equivalent to logits / T).
    temps = temperatures.to(torch.float32).clamp_min(1e-6).view(-1, 1)
    inv_t = 1.0 / temps
    probs = probs.clamp_min(1e-20).pow(inv_t)
    probs = _renorm_rows(probs)
    base_probs = probs

    asserts_flag = (os.environ.get("SGLANG_DFLASH_PQ_ASSERTS") or "").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )

    if need_min_p_sampling:
        # SGLang: top_k_renorm_prob -> top_p_renorm_prob -> min_p_sampling_from_probs
        probs = _apply_top_k_mask_pivot(probs, top_ks)
        probs = _renorm_rows(probs)
        probs = _apply_top_p_mask(probs, top_ps, joint_eps=None)
        probs = _renorm_rows(probs)
        mp = min_ps.to(torch.float32).clamp(min=0.0, max=1.0).view(-1, 1)
        thresh = probs.max(dim=-1, keepdim=True).values * mp
        probs = probs.masked_fill(probs < thresh, 0.0)
        probs = _renorm_rows(probs)
        denom = probs.sum(dim=-1, keepdim=True)
        out = torch.where(denom > 0, probs, base_probs)
        if asserts_flag:
            with torch.no_grad():
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError("DFLASH pq_filter: output contains NaN/Inf.")
                s = out.sum(dim=-1)
                if not torch.allclose(s, torch.ones_like(s), atol=5e-3, rtol=1e-3):
                    raise RuntimeError(
                        f"DFLASH pq_filter: rows not normalized: min={float(s.min().item()):.6f} max={float(s.max().item()):.6f}"
                    )
        return out

    # No min_p: SGLang uses `top_k_top_p_sampling_from_probs(..., filter_apply_order="joint")`.
    if no_min_p_filter_apply_order not in ("joint", "top_k_first"):
        raise ValueError(
            f"Invalid no_min_p_filter_apply_order={no_min_p_filter_apply_order!r}"
        )

    if no_min_p_filter_apply_order == "top_k_first":
        probs = _apply_top_k_mask_pivot(probs, top_ks)
        probs = _renorm_rows(probs)
        probs = _apply_top_p_mask(probs, top_ps, joint_eps=None)
        probs = _renorm_rows(probs)
        denom = probs.sum(dim=-1, keepdim=True)
        out = torch.where(denom > 0, probs, base_probs)
        if asserts_flag:
            with torch.no_grad():
                if torch.isnan(out).any() or torch.isinf(out).any():
                    raise RuntimeError("DFLASH pq_filter: output contains NaN/Inf.")
                s = out.sum(dim=-1)
                if not torch.allclose(s, torch.ones_like(s), atol=5e-3, rtol=1e-3):
                    raise RuntimeError(
                        f"DFLASH pq_filter: rows not normalized: min={float(s.min().item()):.6f} max={float(s.max().item()):.6f}"
                    )
        return out

    # joint: intersect top-k and top-p masks on the same distribution.
    probs_k = _apply_top_k_mask_pivot(probs, top_ks)
    probs_p = _apply_top_p_mask(probs, top_ps, joint_eps=joint_eps)
    keep = (probs_k > 0) & (probs_p > 0)
    probs = probs.masked_fill(~keep, 0.0)
    probs = _renorm_rows(probs)
    denom = probs.sum(dim=-1, keepdim=True)
    out = torch.where(denom > 0, probs, base_probs)
    if asserts_flag:
        with torch.no_grad():
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError("DFLASH pq_filter: output contains NaN/Inf.")
            s = out.sum(dim=-1)
            if not torch.allclose(s, torch.ones_like(s), atol=5e-3, rtol=1e-3):
                raise RuntimeError(
                    f"DFLASH pq_filter: rows not normalized: min={float(s.min().item()):.6f} max={float(s.max().item()):.6f}"
                )
    return out
