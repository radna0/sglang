from __future__ import annotations

import torch


def min_p_renorm_prob(probs: torch.Tensor, min_ps: torch.Tensor) -> torch.Tensor:
    """Apply min-p filtering (relative-to-max) and renormalize.

    Keep tokens with `p >= max(p) * min_p`, then renormalize.

    This matches the behavior used by SGLang's standard sampling path.
    """
    if probs.numel() == 0 or min_ps.numel() == 0:
        return probs

    if min_ps.dim() != 1:
        min_ps = min_ps.view(-1)

    max_probs = probs.max(dim=-1).values
    thresholds = max_probs * min_ps.to(dtype=probs.dtype)
    probs = probs.masked_fill(probs < thresholds.unsqueeze(1), 0.0)

    denom = probs.sum(dim=-1, keepdim=True)
    return probs / denom.clamp_min(1e-20)

