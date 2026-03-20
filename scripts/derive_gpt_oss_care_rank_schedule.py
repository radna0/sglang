#!/usr/bin/env python3

import argparse
import heapq
import json
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a CARE-E style per-layer rank schedule for GPT-OSS MLA conversion."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--covariance-dir", action="append", required=True)
    parser.add_argument("--covariance-weight", action="append", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--target-rank", type=int, default=512)
    parser.add_argument("--target-total-rank", type=int, default=None)
    parser.add_argument("--min-rank", type=int, default=128)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=None)
    parser.add_argument("--covariance-shrinkage", type=float, default=1e-4)
    parser.add_argument(
        "--round-multiple",
        type=int,
        default=1,
        help="If >1, emit a production schedule rounded to a fixed multiple (for example 8 or 16).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    return parser.parse_args()


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


class TensorIndex:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.index = _load_json(model_path / "model.safetensors.index.json")
        self.weight_map = self.index["weight_map"]

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def _repeat_gqa_rows(
    weight: torch.Tensor, num_attention_heads: int, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    repeated = weight.view(num_kv_heads, head_dim, -1)
    repeat_factor = num_attention_heads // num_kv_heads
    return repeated.repeat_interleave(repeat_factor, dim=0)


def _split_head_channels(
    tensor: torch.Tensor, qk_nope_head_dim: int, qk_rope_head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if qk_rope_head_dim == 0:
        return tensor, tensor.new_empty(tensor.shape[0], 0, *tensor.shape[2:])
    if qk_nope_head_dim == 0:
        return tensor.new_empty(tensor.shape[0], 0, *tensor.shape[2:]), tensor
    return (
        tensor[:, :qk_nope_head_dim],
        tensor[:, qk_nope_head_dim : qk_nope_head_dim + qk_rope_head_dim],
    )


def _apply_covariance(
    target: torch.Tensor,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
) -> torch.Tensor:
    if covariance is None:
        return target
    covariance = covariance.to(dtype=target.dtype, device=target.device)
    eye = torch.eye(covariance.shape[0], dtype=target.dtype, device=target.device)
    diag_mean = covariance.diagonal().mean()
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diag_mean * eye
    chol = torch.linalg.cholesky(covariance)
    return target @ chol


def _load_covariance(covariance_dir: Path, layer_id: int) -> Optional[torch.Tensor]:
    candidates = [
        covariance_dir / f"layer_{layer_id:02d}.pt",
        covariance_dir / f"layer_{layer_id}.pt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = torch.load(candidate, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("covariance", "cov", "C", "matrix"):
                if key in payload:
                    return payload[key]
        if isinstance(payload, torch.Tensor):
            return payload
    raise FileNotFoundError(
        f"Missing covariance stats for layer {layer_id} under {covariance_dir}"
    )


def _resolve_covariance_sources(
    covariance_dirs: list[str], covariance_weights: Optional[list[float]]
) -> list[tuple[Path, float]]:
    resolved_dirs = [Path(path).resolve() for path in covariance_dirs]
    if covariance_weights is None or len(covariance_weights) == 0:
        weights = [1.0] * len(resolved_dirs)
    else:
        if len(covariance_weights) != len(resolved_dirs):
            raise ValueError(
                "Expected the same number of --covariance-weight values as --covariance-dir values."
            )
        weights = [float(weight) for weight in covariance_weights]
    if any(weight < 0 for weight in weights):
        raise ValueError("covariance weights must be non-negative.")
    if sum(weights) <= 0:
        raise ValueError("At least one covariance weight must be positive.")
    return list(zip(resolved_dirs, weights))


def _load_fused_covariance(
    covariance_sources: list[tuple[Path, float]], layer_id: int
) -> torch.Tensor:
    fused = None
    total_weight = 0.0
    for covariance_dir, weight in covariance_sources:
        if weight <= 0:
            continue
        covariance = _load_covariance(covariance_dir, layer_id).to(dtype=torch.float64)
        if fused is None:
            fused = covariance * float(weight)
        else:
            fused.add_(covariance, alpha=float(weight))
        total_weight += float(weight)
    if fused is None or total_weight <= 0:
        raise ValueError("No covariance sources were available for fusion.")
    return (fused / total_weight).to(dtype=torch.float32)


def _round_down_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(value // multiple) * multiple


def _round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def _nearest_multiple_in_range(value: int, multiple: int, lo: int, hi: int) -> int:
    if multiple <= 1:
        return min(max(int(value), int(lo)), int(hi))
    candidates: list[int] = []
    floor_v = _round_down_multiple(value, multiple)
    ceil_v = _round_up_multiple(value, multiple)
    for candidate in (floor_v, ceil_v, lo, hi):
        if lo <= candidate <= hi and candidate % multiple == 0:
            candidates.append(candidate)
    if not candidates:
        raise ValueError(
            f"No feasible multiple-of-{multiple} value inside range [{lo}, {hi}]"
        )
    candidates = sorted(set(candidates), key=lambda x: (abs(x - value), x))
    return int(candidates[0])


def _round_schedule_to_multiple(
    raw_schedule: list[int],
    layer_spectra: list[list[float]],
    layer_caps: list[int],
    *,
    total_budget: int,
    min_rank: int,
    round_multiple: int,
) -> tuple[list[int], dict]:
    if round_multiple <= 1:
        return [int(x) for x in raw_schedule], {
            "enabled": False,
            "round_multiple": 1,
            "target_total_rank": int(total_budget),
            "rounded_total_rank": int(sum(raw_schedule)),
        }

    rounded_min = _round_up_multiple(int(min_rank), round_multiple)
    lower_bounds: list[int] = []
    upper_bounds: list[int] = []
    rounded_schedule: list[int] = []
    adjustment_report: list[dict] = []

    for layer_id, (raw_rank, cap) in enumerate(zip(raw_schedule, layer_caps)):
        upper = _round_down_multiple(int(cap), round_multiple)
        if upper < rounded_min:
            raise ValueError(
                f"Layer {layer_id} cap={cap} cannot satisfy min_rank={min_rank} "
                f"under round_multiple={round_multiple}."
            )
        lower = max(rounded_min, _round_down_multiple(int(raw_rank), round_multiple))
        lower = min(lower, upper)
        rounded = _nearest_multiple_in_range(
            int(raw_rank), round_multiple, lower, upper
        )
        lower_bounds.append(int(lower))
        upper_bounds.append(int(upper))
        rounded_schedule.append(int(rounded))
        adjustment_report.append(
            {
                "layer_id": int(layer_id),
                "raw_rank": int(raw_rank),
                "rounded_rank_initial": int(rounded),
                "lower_bound": int(lower),
                "upper_bound": int(upper),
            }
        )

    sum_lower = int(sum(lower_bounds))
    sum_upper = int(sum(upper_bounds))
    target_total = _round_up_multiple(total_budget, round_multiple)
    if target_total > sum_upper:
        target_total = _round_down_multiple(total_budget, round_multiple)
    target_total = min(max(target_total, sum_lower), sum_upper)
    if target_total % round_multiple != 0:
        raise RuntimeError(
            f"Rounded target total {target_total} is not divisible by {round_multiple}."
        )

    current_total = int(sum(rounded_schedule))

    if current_total < target_total:
        add_heap: list[tuple[float, int]] = []
        for layer_id, gains in enumerate(layer_spectra):
            next_rank = rounded_schedule[layer_id]
            if next_rank + round_multiple <= upper_bounds[layer_id]:
                block_gain = sum(float(gains[idx]) for idx in range(next_rank, next_rank + round_multiple))
                heapq.heappush(add_heap, (-block_gain, layer_id))
        while current_total < target_total and add_heap:
            neg_gain, layer_id = heapq.heappop(add_heap)
            rounded_schedule[layer_id] += round_multiple
            current_total += round_multiple
            next_rank = rounded_schedule[layer_id]
            if next_rank + round_multiple <= upper_bounds[layer_id]:
                gains = layer_spectra[layer_id]
                block_gain = sum(float(gains[idx]) for idx in range(next_rank, next_rank + round_multiple))
                heapq.heappush(add_heap, (-block_gain, layer_id))
        if current_total != target_total:
            raise RuntimeError(
                f"Failed to increase rounded schedule to target_total={target_total}; "
                f"reached {current_total}."
            )
    elif current_total > target_total:
        remove_heap: list[tuple[float, int]] = []
        for layer_id, gains in enumerate(layer_spectra):
            current_rank = rounded_schedule[layer_id]
            if current_rank - round_multiple >= lower_bounds[layer_id]:
                block_cost = sum(
                    float(gains[idx]) for idx in range(current_rank - round_multiple, current_rank)
                )
                heapq.heappush(remove_heap, (block_cost, layer_id))
        while current_total > target_total and remove_heap:
            block_cost, layer_id = heapq.heappop(remove_heap)
            rounded_schedule[layer_id] -= round_multiple
            current_total -= round_multiple
            current_rank = rounded_schedule[layer_id]
            if current_rank - round_multiple >= lower_bounds[layer_id]:
                gains = layer_spectra[layer_id]
                next_cost = sum(
                    float(gains[idx]) for idx in range(current_rank - round_multiple, current_rank)
                )
                heapq.heappush(remove_heap, (next_cost, layer_id))
        if current_total != target_total:
            raise RuntimeError(
                f"Failed to decrease rounded schedule to target_total={target_total}; "
                f"reached {current_total}."
            )

    for item, final_rank in zip(adjustment_report, rounded_schedule):
        item["rounded_rank_final"] = int(final_rank)

    metadata = {
        "enabled": True,
        "round_multiple": int(round_multiple),
        "rounded_min_rank": int(rounded_min),
        "target_total_rank": int(target_total),
        "rounded_total_rank": int(sum(rounded_schedule)),
        "raw_total_rank": int(sum(raw_schedule)),
        "strategy": "spectral_block_rebalance",
        "per_layer": adjustment_report,
    }
    return [int(x) for x in rounded_schedule], metadata


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    covariance_sources = _resolve_covariance_sources(
        args.covariance_dir, args.covariance_weight
    )
    output_json = Path(args.output_json).resolve()

    config = _load_json(model_path / "config.json")
    num_hidden_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    qk_rope_head_dim = int(args.qk_rope_head_dim)
    qk_nope_head_dim = (
        int(args.qk_nope_head_dim)
        if args.qk_nope_head_dim is not None
        else int(head_dim - qk_rope_head_dim)
    )
    if qk_nope_head_dim + qk_rope_head_dim != head_dim:
        raise ValueError(
            "This allocator requires qk_nope_head_dim + qk_rope_head_dim == original head_dim."
        )

    per_layer_target = int(args.target_rank)
    total_budget = (
        int(args.target_total_rank)
        if args.target_total_rank is not None
        else per_layer_target * num_hidden_layers
    )
    max_rank = int(args.max_rank) if args.max_rank is not None else max(per_layer_target * 2, per_layer_target)
    min_rank = int(args.min_rank)
    work_dtype = getattr(torch, args.dtype)
    tensor_device = _choose_device(args.device)

    index = TensorIndex(model_path)
    layer_spectra: list[list[float]] = []
    layer_caps: list[int] = []
    layer_reports: list[dict] = []

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "covariance_dirs": [str(path) for path, _ in covariance_sources],
                "covariance_weights": [float(weight) for _, weight in covariance_sources],
                "output_json": str(output_json),
                "target_rank": per_layer_target,
                "target_total_rank": total_budget,
                "min_rank": min_rank,
                "max_rank": max_rank,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
            },
            indent=2,
        ),
        flush=True,
    )

    for layer_id in range(num_hidden_layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        w_k = index.get_tensor(f"{prefix}.k_proj.weight")
        w_v = index.get_tensor(f"{prefix}.v_proj.weight")
        w_k_rep = _repeat_gqa_rows(w_k, num_heads, num_kv_heads, head_dim)
        w_v_rep = _repeat_gqa_rows(w_v, num_heads, num_kv_heads, head_dim)
        w_k_nope, _ = _split_head_channels(w_k_rep, qk_nope_head_dim, qk_rope_head_dim)

        k_rows = w_k_nope.reshape(-1, w_k_nope.shape[-1]).to(device=tensor_device, dtype=work_dtype)
        v_rows = w_v_rep.reshape(-1, w_v_rep.shape[-1]).to(device=tensor_device, dtype=work_dtype)
        target = torch.cat([k_rows, v_rows], dim=0)
        covariance = _load_fused_covariance(covariance_sources, layer_id)
        target_weighted = _apply_covariance(
            target,
            covariance.to(device=tensor_device, dtype=work_dtype),
            args.covariance_shrinkage,
        )
        singular_values = torch.linalg.svdvals(target_weighted).to(device="cpu", dtype=torch.float64)
        gains = (singular_values.square()).tolist()
        cap = min(int(max_rank), len(gains))
        gains = gains[:cap]
        layer_spectra.append(gains)
        layer_caps.append(cap)
        layer_reports.append(
            {
                "layer_id": layer_id,
                "max_gain": float(gains[0]) if gains else 0.0,
                "cap": int(cap),
            }
        )
        print(
            f"[spectrum] layer={layer_id} cap={cap} top_gain={layer_reports[-1]['max_gain']:.6f}",
            flush=True,
        )

    if min_rank * num_hidden_layers > total_budget:
        raise ValueError("min_rank exceeds the total budget.")
    if sum(layer_caps) < total_budget:
        raise ValueError("Requested total budget exceeds the available rank capacity.")

    schedule = [min(min_rank, cap) for cap in layer_caps]
    remaining = total_budget - sum(schedule)
    heap: list[tuple[float, int]] = []
    for layer_id, gains in enumerate(layer_spectra):
        next_idx = schedule[layer_id]
        if next_idx < len(gains):
            heapq.heappush(heap, (-float(gains[next_idx]), layer_id))

    while remaining > 0 and heap:
        neg_gain, layer_id = heapq.heappop(heap)
        schedule[layer_id] += 1
        remaining -= 1
        next_idx = schedule[layer_id]
        if next_idx < len(layer_spectra[layer_id]):
            heapq.heappush(heap, (-float(layer_spectra[layer_id][next_idx]), layer_id))

    if remaining != 0:
        raise RuntimeError("Failed to exhaust the target budget during allocation.")

    raw_schedule = [int(x) for x in schedule]
    final_schedule, rounding_metadata = _round_schedule_to_multiple(
        raw_schedule,
        layer_spectra,
        layer_caps,
        total_budget=total_budget,
        min_rank=min_rank,
        round_multiple=int(args.round_multiple),
    )

    payload = {
        "model_path": str(model_path),
        "covariance_dirs": [str(path) for path, _ in covariance_sources],
        "covariance_weights": [float(weight) for _, weight in covariance_sources],
        "covariance_fusion": "weighted_average",
        "target_rank": per_layer_target,
        "target_total_rank": total_budget,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "allocation_metric": "weighted_singular_value_squared",
        "raw_kv_lora_rank_per_layer": [int(x) for x in raw_schedule],
        "kv_lora_rank_per_layer": [int(x) for x in final_schedule],
        "rounding": rounding_metadata,
        "layer_reports": [
            {
                **report,
                "allocated_rank_raw": int(raw_schedule[idx]),
                "allocated_rank_final": int(final_schedule[idx]),
            }
            for idx, report in enumerate(layer_reports)
        ],
    }
    _save_json(output_json, payload)
    print(f"[done] wrote rank schedule to {output_json}", flush=True)


if __name__ == "__main__":
    main()
