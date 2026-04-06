from typing import List, Optional

import torch
import triton
import triton.language as tl

_INT32_MAX: int = 2**31 - 1


def transform_index_page_table_prefill(**kwargs):
    page_table = kwargs["page_table"]
    topk_indices = kwargs["topk_indices"]
    page_size = kwargs.get("page_size", 1)
    if (
        page_size == 1
        and page_table.is_cuda
        and topk_indices.is_cuda
        and topk_indices.shape[1] == 2048
    ):
        return transform_index_page_table_prefill_fast(**kwargs)
    return transform_index_page_table_prefill_ref(**kwargs)


def transform_index_page_table_decode(**kwargs):
    page_table = kwargs["page_table"]
    topk_indices = kwargs["topk_indices"]
    page_size = kwargs.get("page_size", 1)
    if (
        page_size == 1
        and page_table.is_cuda
        and topk_indices.is_cuda
        and topk_indices.shape[1] == 2048
    ):
        return transform_index_page_table_decode_fast(**kwargs)
    return transform_index_page_table_decode_ref(**kwargs)


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    max_seqlen_k: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * max_seqlen_k
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    max_seqlen_k = page_table.shape[1]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        max_seqlen_k=max_seqlen_k,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    # TODO(baizhou): can be implemented with another triton kernel
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


def _stable_unique_pages_by_first_pos(
    pages: torch.Tensor, invalid_page: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stable unique for small per-row page lists on GPU.

    Args:
        pages: int32/int64 tensor of shape (B, N). Entries equal to invalid_page are ignored.
        invalid_page: sentinel value treated as invalid.

    Returns:
        unique_pages: int32 tensor of shape (B, N), stable by first occurrence; padded with -1.
        unique_counts: int32 tensor of shape (B,), number of valid unique pages in each row.
    """
    assert pages.ndim == 2
    bsz, n = pages.shape
    if bsz == 0:
        return (
            pages.new_empty((0, n), dtype=torch.int32),
            pages.new_empty((0,), dtype=torch.int32),
        )

    pages_i64 = pages.to(torch.int64)
    pos = torch.arange(n, device=pages.device, dtype=torch.int64).unsqueeze(0).expand(
        bsz, -1
    )

    # Sort by (page, pos) so duplicates group and the first occurrence is trivially detected.
    key = pages_i64 * int(n) + pos
    _, order = torch.sort(key, dim=1)
    pages_sorted = pages_i64.gather(1, order)
    pos_sorted = pos.gather(1, order)

    first = torch.ones((bsz, n), device=pages.device, dtype=torch.bool)
    first[:, 1:] = pages_sorted[:, 1:] != pages_sorted[:, :-1]
    valid_first = first & (pages_sorted != int(invalid_page))

    large_pos = torch.full_like(pos_sorted, n + 1)
    cand_pos = torch.where(valid_first, pos_sorted, large_pos)

    cand_pos_sorted, cand_order = torch.sort(cand_pos, dim=1)
    unique_pages = pages_sorted.gather(1, cand_order).to(torch.int32)

    valid = cand_pos_sorted <= n
    unique_pages = torch.where(valid, unique_pages, torch.full_like(unique_pages, -1))
    unique_counts = valid.sum(dim=1).to(torch.int32)
    return unique_pages, unique_counts


def select_topk_pages_decode(
    topk_indices: torch.Tensor,
    *,
    page_size: int,
    pages_out: int,
    prefix_len: int = 512,
    sink_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert token-level topk indices into a page-level selection suitable for
    FA3 paged-KV decode (page_size > 1).

    This intentionally selects whole KV pages (blocks) to avoid per-token page_size=1
    decode, which is slow on Hopper.

    Output pages are **logical page ids** (token_idx // page_size), padded with -1.

    Note: This is an approximation of token-topk; it attends to all tokens in the
    selected pages.
    """
    assert topk_indices.ndim == 2
    assert page_size > 1
    assert pages_out > 0

    if topk_indices.numel() == 0:
        return (
            topk_indices.new_empty((0, pages_out), dtype=torch.int32),
            topk_indices.new_empty((0,), dtype=torch.int32),
        )

    topk = topk_indices.shape[1]
    prefix = min(int(prefix_len), int(topk))

    idx_prefix = topk_indices[:, :prefix].to(torch.int64)
    valid_idx = idx_prefix >= 0
    pages_prefix = torch.where(
        valid_idx,
        (idx_prefix // int(page_size)).to(torch.int64),
        torch.full_like(idx_prefix, _INT32_MAX),
    )

    unique_pages_prefix, _ = _stable_unique_pages_by_first_pos(
        pages_prefix, invalid_page=_INT32_MAX
    )
    unique_pages_prefix = unique_pages_prefix[:, :pages_out].to(torch.int32)

    if sink_lens is None:
        pages, counts = _stable_unique_pages_by_first_pos(
            unique_pages_prefix.to(torch.int64), invalid_page=-1
        )
        return pages[:, :pages_out], counts.clamp(max=pages_out)

    # Always include sink pages [0..sink_pages-1] (capped), then stable-unique with the
    # topk-derived pages.
    sink_lens_i64 = sink_lens.to(torch.int64).clamp(min=0)
    sink_pages = ((sink_lens_i64 + int(page_size) - 1) // int(page_size)).to(torch.int32)
    sink_pages = sink_pages.clamp(min=0, max=int(pages_out))

    sink_page_ids = (
        torch.arange(pages_out, device=topk_indices.device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(topk_indices.shape[0], -1)
    )
    sink_pages_mask = sink_page_ids < sink_pages.unsqueeze(1)
    sink_page_ids = torch.where(
        sink_pages_mask,
        sink_page_ids,
        torch.full_like(sink_page_ids, _INT32_MAX),
    )

    seed = torch.cat([sink_page_ids, unique_pages_prefix], dim=1).to(torch.int64)
    unique_pages, counts = _stable_unique_pages_by_first_pos(
        seed, invalid_page=_INT32_MAX
    )
    unique_pages = unique_pages[:, :pages_out].to(torch.int32)
    counts = counts.clamp(max=int(pages_out))
    return unique_pages, counts


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
