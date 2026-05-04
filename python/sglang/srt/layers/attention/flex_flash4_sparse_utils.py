from __future__ import annotations

import torch


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def fill_decode_sparse_buffers(
    *,
    cache_seqlens: torch.Tensor,
    window_left: int,
    block_size: int,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    mask_cnt: torch.Tensor,
    mask_idx: torch.Tensor,
    block_arange: torch.Tensor,
) -> None:
    """
    Fill preallocated decode sparse buffers in-place.

    Shapes:
    - full_cnt/mask_cnt: [B, 1, 1]
    - full_idx/mask_idx: [B, 1, 1, N]
    - block_arange: [1, N]
    """
    seq = cache_seqlens.to(dtype=torch.int32)
    device = seq.device
    bsz = int(seq.shape[0])
    if bsz == 0:
        return

    valid = seq > 0
    end_tok = torch.clamp(seq - 1, min=0)
    if int(window_left) < 0:
        start_tok = torch.zeros_like(end_tok)
    else:
        start_tok = torch.clamp(seq - (int(window_left) + 1), min=0)

    start_block = torch.div(start_tok, int(block_size), rounding_mode="floor")
    end_block = torch.div(end_tok, int(block_size), rounding_mode="floor")

    same_block = valid & (start_block == end_block)
    start_partial = valid & ((start_tok % int(block_size)) != 0)
    end_partial = valid & ((end_tok % int(block_size)) != (int(block_size) - 1))

    # Boundary blocks are handled via mask_mod for exact token semantics.
    mask_has_first = same_block | (~same_block & start_partial)
    mask_has_second = (~same_block) & end_partial
    mask_count = mask_has_first.to(torch.int32) + mask_has_second.to(torch.int32)

    full_start = start_block + start_partial.to(torch.int32)
    full_end = end_block - end_partial.to(torch.int32)
    full_valid = valid & (~same_block) & (full_start <= full_end)
    full_count = torch.where(full_valid, full_end - full_start + 1, 0).to(torch.int32)

    full_cnt_view = full_cnt[:, 0, 0]
    mask_cnt_view = mask_cnt[:, 0, 0]
    full_idx_view = full_idx[:, 0, 0, :]
    mask_idx_view = mask_idx[:, 0, 0, :]

    full_cnt_view.copy_(full_count)
    mask_cnt_view.copy_(mask_count)

    # Fill buffers in-place so shapes/storage are stable for graph/compile.
    full_idx_view.zero_()
    mask_idx_view.zero_()

    if full_idx_view.numel() > 0:
        dense_full = block_arange + full_start.view(-1, 1)
        dense_full.clamp_(min=0, max=max(int(block_arange.shape[1]) - 1, 0))
        full_idx_view.copy_(dense_full.to(torch.int32))

    if mask_idx_view.shape[1] > 0:
        first_mask = torch.where(mask_has_first, start_block, end_block).to(torch.int32)
        mask_idx_view[:, 0].copy_(first_mask)
    if mask_idx_view.shape[1] > 1:
        mask_idx_view[:, 1].copy_(end_block.to(torch.int32))

    # Zero out rows that are actually invalid so stale data cannot leak if counts are ignored.
    invalid_rows = ~valid
    if invalid_rows.any():
        full_cnt_view[invalid_rows] = 0
        mask_cnt_view[invalid_rows] = 0
        full_idx_view[invalid_rows] = 0
        mask_idx_view[invalid_rows] = 0
