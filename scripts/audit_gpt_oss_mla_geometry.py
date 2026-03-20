#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TensorIndex:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.index = _load_json(model_path / "model.safetensors.index.json")
        self.weight_map = self.index["weight_map"]

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GPT-OSS -> MLA geometry mismatch and shared-RoPE loss."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--converted-config", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=None)
    parser.add_argument("--qk-nope-head-dim", type=int, default=None)
    return parser.parse_args()


def _relative_frob(tensor: torch.Tensor, mean_dim: int) -> float:
    centered = tensor - tensor.mean(dim=mean_dim, keepdim=True)
    denom = float(torch.linalg.norm(tensor).item())
    if denom == 0.0:
        return 0.0
    return float(torch.linalg.norm(centered).item() / denom)


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    model_config = _load_json(model_path / "config.json")
    converted_config = (
        _load_json(Path(args.converted_config).resolve())
        if args.converted_config
        else None
    )
    index = TensorIndex(model_path)

    num_heads = int(model_config["num_attention_heads"])
    num_kv_heads = int(model_config["num_key_value_heads"])
    head_dim = int(model_config["head_dim"])
    num_layers = int(model_config["num_hidden_layers"])
    hidden_size = int(model_config["hidden_size"])
    repeat_factor = num_heads // num_kv_heads

    qk_rope_head_dim = (
        int(args.qk_rope_head_dim)
        if args.qk_rope_head_dim is not None
        else int(
            (converted_config or {}).get(
                "qk_rope_head_dim",
                head_dim // 2,
            )
        )
    )
    qk_nope_head_dim = (
        int(args.qk_nope_head_dim)
        if args.qk_nope_head_dim is not None
        else int(
            (converted_config or {}).get(
                "qk_nope_head_dim",
                head_dim - qk_rope_head_dim,
            )
        )
    )
    if qk_rope_head_dim + qk_nope_head_dim != head_dim:
        raise ValueError("qk_rope_head_dim + qk_nope_head_dim must equal original head_dim")

    layer_reports = []
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        w_k = index.get_tensor(f"{prefix}.k_proj.weight").view(num_kv_heads, head_dim, hidden_size)
        rope_slice = w_k[:, qk_nope_head_dim : qk_nope_head_dim + qk_rope_head_dim, :]
        nope_slice = w_k[:, :qk_nope_head_dim, :]

        rope_shared_rel_frob = _relative_frob(rope_slice, mean_dim=0)
        nope_head_variation_rel_frob = _relative_frob(nope_slice, mean_dim=0) if qk_nope_head_dim > 0 else 0.0

        unique_factorized_rows = num_kv_heads * (qk_nope_head_dim + head_dim)
        repeated_factorized_rows = num_heads * (qk_nope_head_dim + head_dim)
        exact_factorization_at_r1024 = 1024 >= unique_factorized_rows

        layer_reports.append(
            {
                "layer_id": layer_id,
                "rope_shared_rel_frob": rope_shared_rel_frob,
                "nope_head_variation_rel_frob": nope_head_variation_rel_frob,
                "unique_factorized_rows": unique_factorized_rows,
                "repeated_factorized_rows": repeated_factorized_rows,
                "exact_factorization_at_r1024": exact_factorization_at_r1024,
            }
        )

    rope_values = [x["rope_shared_rel_frob"] for x in layer_reports]
    nope_values = [x["nope_head_variation_rel_frob"] for x in layer_reports]

    payload = {
        "model_path": str(model_path),
        "converted_config": str(Path(args.converted_config).resolve()) if args.converted_config else None,
        "original_geometry": {
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "repeat_factor": repeat_factor,
            "gpt_oss_original_rope_mode": "full_head_rope",
        },
        "mla_geometry_under_audit": {
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "latent_mode": "shared_rope_path_with_per_head_nope_value_projection",
        },
        "summary": {
            "mean_rope_shared_rel_frob": sum(rope_values) / len(rope_values),
            "max_rope_shared_rel_frob": max(rope_values),
            "mean_nope_head_variation_rel_frob": sum(nope_values) / len(nope_values),
            "max_nope_head_variation_rel_frob": max(nope_values),
            "unique_factorized_rows_at_r1024": num_kv_heads * (qk_nope_head_dim + head_dim),
            "r1024_exact_for_nope_plus_value_factorization": bool(
                1024 >= num_kv_heads * (qk_nope_head_dim + head_dim)
            ),
        },
        "interpretation": {
            "current_split_mismatch": (
                "Original GPT-OSS applies RoPE to the full head_dim. Any nonzero qk_nope_head_dim "
                "forces some originally-RoPE channels into a non-RoPE path."
            ),
            "shared_rope_mismatch": (
                "The rope slice is forced into a shared path. rope_shared_rel_frob quantifies head-to-head "
                "variation in that slice that cannot be preserved by a single shared rope key."
            ),
            "r1024_note": (
                "If r=1024 is already enough to represent the unique noPE+V rows exactly, then remaining "
                "error is dominated by attention-geometry mismatch rather than latent-rank shortage."
            ),
        },
        "layers": layer_reports,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_path:
        Path(args.output_path).resolve().write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
