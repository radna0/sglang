from __future__ import annotations

import torch
import math
from torch.nn.attention.flex_attention import BlockMask


def apply_attention_sinks(
    out: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor | None,
    base: float = math.e,
) -> torch.Tensor:
    """
    Apply GPT-OSS attention sinks to a FlexAttention output.

    In SGLang, `sinks` is defined as an additional per-head value in the *exp-domain*
    added to the softmax denominator:

        w_i = exp(score_i) / (sum_j exp(score_j) + sinks[h])

    If `out` was computed with the standard softmax denominator `sum_j exp(score_j)`,
    the adjusted output is:

        out_sink = out * 1 / (1 + sinks[h] * exp(-lse))

    where `lse = log(sum_j exp(score_j))`.

    For backends like FlashMLA that use 2-based log-sum-exp, passing `base=2.0`
    will use `2**(-lse)` instead of `exp(-lse)`.
    """
    if sinks is None:
        return out
    if sinks.numel() == 0:
        return out

    # sinks: [num_q_heads] (or broadcastable). Keep math in float32 for stability,
    # but return in `out.dtype` to avoid inflating activations.
    sinks_f = sinks.to(dtype=torch.float32).clamp_min(0.0)

    lse_f = lse.to(dtype=torch.float32)
    if sinks_f.ndim == 1:
        sinks_f = sinks_f.view(*([1] * max(lse_f.ndim - 1, 0)), sinks_f.shape[0])
    elif sinks_f.ndim == 2 and lse_f.ndim >= 2:
        sinks_f = sinks_f.view(*([1] * max(lse_f.ndim - 2, 0)), *sinks_f.shape)

    if base == math.e:
        scale = torch.reciprocal(1.0 + sinks_f * torch.exp(-lse_f)).to(dtype=out.dtype)
    elif base == 2.0:
        scale = torch.reciprocal(1.0 + sinks_f * torch.exp2(-lse_f)).to(dtype=out.dtype)
    else:
        scale = torch.reciprocal(
            1.0 + sinks_f * torch.pow(base, -lse_f)
        ).to(dtype=out.dtype)

    if scale.ndim == out.ndim - 1:
        scale = scale.unsqueeze(-1)
    return out * scale


def make_extend_causal_mask_mod(*, q_kv_offset: int):
    """
    Mask mod for extend-mode causal attention with a KV prefix already present.

    Let abs_q0 be the absolute position of q_idx=0 in the full sequence, and abs_kv0 be
    the absolute position of kv_idx=0 in the (possibly window-truncated) KV tensor.
    Define:
        q_kv_offset = abs_q0 - abs_kv0

    Then the causal constraint is:
        abs_kv <= abs_q
      <=> (abs_kv0 + kv_idx) <= (abs_q0 + q_idx)
      <=> kv_idx <= q_kv_offset + q_idx
    """
    offset = int(q_kv_offset)

    def _mask(_b, _h, q_idx, kv_idx):
        return kv_idx <= (offset + q_idx)

    return _mask


def convert_logical_block_mask_to_physical_pages(
    block_mask: BlockMask,
    *,
    page_table: torch.Tensor,
    num_physical_pages: int | None = None,
) -> BlockMask:
    """
    Convert a logical BlockMask (kv blocks are logical pages) to a physical BlockMask
    (kv blocks are physical pages) using a page table.

    Args:
        block_mask: BlockMask whose kv_indices are logical page indices.
        page_table: int64 tensor of shape [B, N_LOGICAL_PAGES] mapping each logical page
            to a physical page index in the global paged KV cache.

    Returns:
        A new BlockMask with kv_indices mapped to physical page indices.

    Notes:
        - This mirrors the flex-nano-vllm approach for FlexAttention-II style paged KV.
        - BLOCK_SIZE[1] should match the KV page_size for correctness.
    """
    if page_table.ndim != 2:
        raise ValueError(
            f"page_table must have shape [B, N_LOGICAL_PAGES], got {tuple(page_table.shape)}"
        )

    kv_indices = block_mask.kv_indices.to(torch.int64)
    bsz = int(kv_indices.shape[0])
    if bsz != int(page_table.shape[0]):
        raise ValueError(
            f"Batch mismatch: block_mask B={bsz} but page_table B={int(page_table.shape[0])}"
        )

    if num_physical_pages is None:
        # Best-effort inference. Assumes unused entries are negative or small.
        max_phys = int(page_table.clamp_min(0).max().item()) if page_table.numel() else 0
        num_physical_pages = max_phys + 1

    num_physical_pages = int(num_physical_pages)
    if num_physical_pages <= 0:
        raise ValueError(f"num_physical_pages must be positive, got {num_physical_pages}")

    def _map_indices(indices: torch.Tensor | None) -> torch.Tensor | None:
        if indices is None:
            return None
        flat = indices.to(torch.int64).reshape(bsz, -1).clamp_min(0)
        mapped = page_table.gather(1, flat).reshape_as(indices).to(torch.int32)
        return mapped

    mapped_kv_indices = _map_indices(block_mask.kv_indices)
    mapped_full_kv_indices = _map_indices(getattr(block_mask, "full_kv_indices", None))
    full_kv_num_blocks = getattr(block_mask, "full_kv_num_blocks", None)

    # Expand the last dim so mapped physical page indices are always in range for BlockMask internals.
    # See torch.nn.attention.flex_attention._ordered_to_dense: it assumes indices < num_cols+1.
    orig_cols = int(block_mask.kv_indices.shape[-1])
    expanded_shape = (*block_mask.kv_indices.shape[:-1], num_physical_pages)

    def _expand_cols(mapped: torch.Tensor | None) -> torch.Tensor | None:
        if mapped is None:
            return None
        out = torch.full(
            expanded_shape,
            fill_value=num_physical_pages,
            dtype=torch.int32,
            device=mapped.device,
        )
        out[..., :orig_cols] = mapped.to(torch.int32)
        return out

    mapped_kv_indices_exp = _expand_cols(mapped_kv_indices)
    mapped_full_kv_indices_exp = _expand_cols(mapped_full_kv_indices)

    # Wrap the original logical mask_mod so it is evaluated on logical kv indices.
    # This mirrors flex-nano-vllm's PageTable.get_mask_mod mapping.
    page_size = int(block_mask.BLOCK_SIZE[1])
    physical_to_logical = torch.full(
        (bsz, num_physical_pages),
        fill_value=-1,
        dtype=torch.int64,
        device=page_table.device,
    )
    logical_pages = torch.arange(page_table.shape[1], device=page_table.device, dtype=torch.int64)
    valid = page_table >= 0
    if valid.any():
        b_idx = torch.arange(bsz, device=page_table.device, dtype=torch.int64).view(-1, 1).expand_as(page_table)
        logical_pages_exp = logical_pages.view(1, -1).expand_as(page_table)
        physical_to_logical[b_idx[valid], page_table[valid].to(torch.int64)] = logical_pages_exp[valid]

    logical_mask_mod = getattr(block_mask, "mask_mod", None)
    if logical_mask_mod is None:
        # BlockMask.from_kv_blocks will default to noop_mask, but keep it explicit here.
        from torch.nn.attention.flex_attention import noop_mask as logical_mask_mod  # type: ignore

    def physical_mask_mod(b, h, q_idx, physical_kv_idx):
        physical_page = torch.div(physical_kv_idx, page_size, rounding_mode="floor").to(torch.int64)
        offset = (physical_kv_idx - physical_page.to(physical_kv_idx.dtype) * page_size).to(torch.int64)
        logical_page = physical_to_logical[b.to(torch.int64), physical_page]
        is_valid = logical_page >= 0
        logical_kv_idx = logical_page * page_size + offset
        safe_logical_kv_idx = logical_kv_idx.clamp_min(0)
        return torch.where(is_valid, logical_mask_mod(b, h, q_idx, safe_logical_kv_idx), False)

    seq_lengths = (int(block_mask.seq_lengths[0]), num_physical_pages * page_size)
    return BlockMask.from_kv_blocks(
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=mapped_kv_indices_exp if mapped_kv_indices_exp is not None else block_mask.kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=mapped_full_kv_indices_exp,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=physical_mask_mod,
        seq_lengths=seq_lengths,
    )
