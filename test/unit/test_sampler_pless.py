import ast
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from sglang.srt.layers.utils.hash import murmur_hash32


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
        "sampling_from_probs_torch",
        "p_less_sampling_from_probs_torch",
        "top_k_top_p_min_p_sampling_from_probs_torch",
        "prepare_top_k_top_p_min_p_sorted_probs_torch",
        "_apply_top_k_top_p_min_p_filters_inplace",
        "multinomial_with_seed",
        "sample_from_probs_default_torch",
        "sample_from_probs_with_strategies_torch",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "List": List,
        "Tuple": Tuple,
        "TOP_K_ALL": 1 << 30,
        "murmur_hash32": murmur_hash32,
    }
    exec(compile(module, str(sampler_path), "exec"), namespace, namespace)
    return namespace


_HELPERS = _load_sampler_helpers()
p_less_sampling_from_probs_torch = _HELPERS["p_less_sampling_from_probs_torch"]
sample_from_probs_with_strategies_torch = _HELPERS[
    "sample_from_probs_with_strategies_torch"
]


def test_p_less_keeps_only_tokens_above_squared_mass_threshold():
    probs = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)

    sampled = p_less_sampling_from_probs_torch(
        probs,
        normalized_variant=False,
    )

    assert sampled.item() == 0


def test_p_less_norm_relaxes_threshold_vs_p_less():
    probs = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)

    sampled = p_less_sampling_from_probs_torch(
        probs,
        normalized_variant=True,
    )

    assert sampled.item() in {0, 1, 2}


def test_strategy_mixed_batch_keeps_default_row_behavior():
    probs = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float32,
    )
    top_ks = torch.tensor([1 << 30, 1], dtype=torch.int32)
    top_ps = torch.tensor([1.0, 1.0], dtype=torch.float32)
    min_ps = torch.tensor([0.0, 0.0], dtype=torch.float32)
    positions = torch.tensor([0, 1], dtype=torch.int64)

    sampled = sample_from_probs_with_strategies_torch(
        probs=probs,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        sampling_seed=None,
        positions=positions,
        strategy_names=["p_less", None],
    )

    assert sampled.shape == (2,)
    assert sampled[0].item() == 0
    assert sampled[1].item() == 2
