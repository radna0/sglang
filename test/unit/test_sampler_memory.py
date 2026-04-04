import ast
from pathlib import Path
from typing import Tuple

import torch


def _load_sampler_helpers():
    sampler_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "sampler.py"
    )
    source = sampler_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(sampler_path))
    wanted = {
        "_sanitize_sampling_probs_for_multinomial_",
        "prepare_top_k_top_p_min_p_sorted_probs_torch",
        "_apply_top_k_top_p_min_p_filters_inplace",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "torch": torch,
        "Tuple": Tuple,
        "TOP_K_ALL": 1000000,
    }
    exec(compile(module, str(sampler_path), "exec"), namespace, namespace)
    return (
        namespace["_sanitize_sampling_probs_for_multinomial_"],
        namespace["prepare_top_k_top_p_min_p_sorted_probs_torch"],
    )


(
    _sanitize_sampling_probs_for_multinomial_,
    prepare_top_k_top_p_min_p_sorted_probs_torch,
) = _load_sampler_helpers()


def _legacy_prepare_top_k_top_p_min_p_sorted_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def _densify_sorted_probs(
    probs_sort: torch.Tensor, probs_idx: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    dense = torch.zeros(
        probs_sort.shape[0],
        vocab_size,
        dtype=probs_sort.dtype,
        device=probs_sort.device,
    )
    dense.scatter_(dim=-1, index=probs_idx, src=probs_sort)
    return dense


def test_prepare_top_k_top_p_min_p_sorted_probs_torch_bounded_topk_matches_legacy():
    probs = torch.tensor(
        [
            [0.30, 0.20, 0.19, 0.12, 0.11, 0.08],
            [0.26, 0.22, 0.18, 0.14, 0.12, 0.08],
        ],
        dtype=torch.float32,
    )
    top_ks = torch.tensor([3, 5], dtype=torch.int32)
    top_ps = torch.tensor([0.75, 0.90], dtype=torch.float32)
    min_ps = torch.tensor([0.0, 0.0], dtype=torch.float32)

    probs_sort_new, probs_idx_new = prepare_top_k_top_p_min_p_sorted_probs_torch(
        probs.clone(),
        top_ks.clone(),
        top_ps.clone(),
        min_ps.clone(),
        need_min_p_sampling=False,
    )
    probs_sort_old, probs_idx_old = _legacy_prepare_top_k_top_p_min_p_sorted_probs_torch(
        probs.clone(),
        top_ks.clone(),
        top_ps.clone(),
        min_ps.clone(),
        need_min_p_sampling=False,
    )

    dense_new = _densify_sorted_probs(probs_sort_new, probs_idx_new, probs.shape[-1])
    dense_old = _densify_sorted_probs(probs_sort_old, probs_idx_old, probs.shape[-1])

    assert torch.allclose(dense_new, dense_old, atol=0.0, rtol=0.0)


def test_sanitize_sampling_probs_for_multinomial_repairs_bad_rows():
    probs_sort = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [float("nan"), 0.0, 0.0],
            [-1.0, -2.0, -3.0],
            [float("inf"), 1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    _sanitize_sampling_probs_for_multinomial_(probs_sort)

    assert torch.all(torch.isfinite(probs_sort))
    assert torch.all(probs_sort >= 0)
    assert torch.allclose(
        probs_sort.sum(dim=-1),
        torch.ones(probs_sort.shape[0], dtype=probs_sort.dtype),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(probs_sort[1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(probs_sort[2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(
        probs_sort[3],
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=probs_sort.dtype),
        atol=1e-6,
        rtol=0.0,
    )
