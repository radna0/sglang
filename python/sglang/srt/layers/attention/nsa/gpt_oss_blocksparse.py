from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.flash_attention.cute.block_sparsity import BlockSparseTensorsTorch


@dataclass(frozen=True)
class GptOssBlockSparseKV:
    k: torch.Tensor
    v: torch.Tensor
    block_sparse: BlockSparseTensorsTorch
    seqused_k: torch.Tensor


def _unique_sorted_int32(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    if x.numel() == 0:
        return x
    return torch.unique(x, sorted=True)


def build_blocksparse_kv_from_topk(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    key_buf: torch.Tensor,
    value_buf: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    n_block_size: int = 128,
    num_heads_for_blocksparse: int = 1,
    device: Optional[torch.device] = None,
) -> GptOssBlockSparseKV:
    """
    Convert per-request token topk indices into a block-sparse layout by selecting whole KV blocks.

    Notes:
    - This is an approximation: it upgrades token-level topk into block-level selection.
    - It compacts selected blocks into a dense K/V tensor (no paged-KV support in SM90 CUTE path yet).
    """
    if device is None:
        device = topk_indices.device

    assert req_to_token.is_cuda and req_pool_indices.is_cuda
    assert key_buf.is_cuda and value_buf.is_cuda
    assert topk_indices.is_cuda and seq_lens.is_cuda
    assert topk_indices.shape[0] == seq_lens.shape[0]

    bs = int(topk_indices.shape[0])
    assert bs > 0

    seq_lens_i64 = seq_lens.to(torch.int64)
    topk_i64 = topk_indices.to(torch.int64)

    kv_heads = int(key_buf.shape[1])
    head_dim = int(key_buf.shape[2])
    v_head_dim = int(value_buf.shape[2])
    assert key_buf.shape[0] == value_buf.shape[0]
    assert key_buf.dtype == value_buf.dtype

    # Per-request block lists. Keep it simple/robust: loop over batch items.
    block_lists: list[torch.Tensor] = []
    max_blocks = 0
    for i in range(bs):
        seq_len = int(seq_lens_i64[i].item())
        idx = topk_i64[i]
        valid = (idx >= 0) & (idx < seq_len)
        if torch.any(valid):
            blocks = _unique_sorted_int32(idx[valid] // n_block_size)
        else:
            blocks = torch.empty((0,), dtype=torch.int32, device=device)
        block_lists.append(blocks)
        max_blocks = max(max_blocks, int(blocks.numel()))

    if max_blocks == 0:
        # No valid blocks; return empty KV. Caller should treat seqused_k==0 as no-op.
        empty_k = key_buf.new_empty((bs, 0, kv_heads, head_dim))
        empty_v = value_buf.new_empty((bs, 0, kv_heads, v_head_dim))
        zeros = torch.zeros((bs,), dtype=torch.int32, device=device)
        block_sparse = BlockSparseTensorsTorch(
            mask_block_cnt=torch.zeros((bs, 1, 1), dtype=torch.int32, device=device),
            mask_block_idx=torch.zeros((bs, 1, 1, 1), dtype=torch.int32, device=device),
            full_block_cnt=zeros.view(bs, 1, 1),
            full_block_idx=torch.zeros((bs, 1, 1, 1), dtype=torch.int32, device=device),
        )
        return GptOssBlockSparseKV(
            k=empty_k, v=empty_v, block_sparse=block_sparse, seqused_k=zeros
        )

    # Compact K/V buffers: (bs, max_blocks * n_block_size, kv_heads, d)
    max_k_len = max_blocks * n_block_size
    k_out = key_buf.new_empty((bs, max_k_len, kv_heads, head_dim))
    v_out = value_buf.new_empty((bs, max_k_len, kv_heads, v_head_dim))
    seqused_k = torch.zeros((bs,), dtype=torch.int32, device=device)

    full_block_cnt = torch.zeros((bs, 1, 1), dtype=torch.int32, device=device)
    full_block_idx = torch.zeros((bs, 1, 1, max_blocks), dtype=torch.int32, device=device)

    arange_block = torch.arange(n_block_size, device=device, dtype=torch.int64)

    for i in range(bs):
        seq_len = int(seq_lens_i64[i].item())
        blocks = block_lists[i]
        num_blocks = int(blocks.numel())
        if num_blocks == 0:
            seqused_k[i] = 0
            full_block_cnt[i, 0, 0] = 0
            continue

        # Remap to compact block indices: 0..num_blocks-1
        full_block_cnt[i, 0, 0] = num_blocks
        full_block_idx[i, 0, 0, :num_blocks] = torch.arange(
            num_blocks, device=device, dtype=torch.int32
        )

        # Materialize K/V for selected blocks.
        # positions: (num_blocks, n_block_size)
        positions = blocks.to(torch.int64).unsqueeze(1) * n_block_size + arange_block.view(
            1, n_block_size
        )
        valid_pos = positions < seq_len
        positions = torch.where(
            valid_pos, positions, torch.zeros_like(positions)
        )

        req_idx = req_pool_indices[i]
        loc = req_to_token[req_idx, positions]  # (num_blocks, n_block_size)
        loc = loc.reshape(-1).to(torch.int64)  # (num_blocks*n_block_size,)

        k_sel = key_buf[loc]
        v_sel = value_buf[loc]
        k_sel = k_sel.view(num_blocks * n_block_size, kv_heads, head_dim)
        v_sel = v_sel.view(num_blocks * n_block_size, kv_heads, v_head_dim)

        # Zero out invalid tail positions in the last block.
        if not torch.all(valid_pos):
            mask_flat = valid_pos.reshape(-1)
            k_sel = torch.where(
                mask_flat.view(-1, 1, 1), k_sel, torch.zeros_like(k_sel)
            )
            v_sel = torch.where(
                mask_flat.view(-1, 1, 1), v_sel, torch.zeros_like(v_sel)
            )

        k_out[i, : num_blocks * n_block_size] = k_sel
        v_out[i, : num_blocks * n_block_size] = v_sel
        seqused_k[i] = seq_len if seq_len < num_blocks * n_block_size else num_blocks * n_block_size

    # Provide empty masked-block list and full-block list for all heads.
    mask_block_cnt = torch.zeros((bs, 1, 1), dtype=torch.int32, device=device)
    mask_block_idx = torch.zeros((bs, 1, 1, max_blocks), dtype=torch.int32, device=device)

    block_sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )

    return GptOssBlockSparseKV(
        k=k_out,
        v=v_out,
        block_sparse=block_sparse,
        seqused_k=seqused_k,
    )

