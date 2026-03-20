#!/usr/bin/env python3

import argparse
import copy
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gpt_oss_care_checkpoint_utils import (
    build_repaired_config,
    build_repaired_index,
    load_json,
    resolve_source_model_path,
    save_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a GPT-OSS CARE MLA checkpoint with absorbed w_kc/w_vc tensors."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

class TensorIndex:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.index = build_repaired_index(
            model_path, source_model_path=resolve_source_model_path(model_path)
        )
        self.weight_map = self.index["weight_map"]

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map


def _symlink_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    config = build_repaired_config(model_path, clear_stale_healing_metadata=False)
    if config.get("architectures", [None])[0] != "GptOssMlaForCausalLM":
        raise ValueError("This exporter expects a converted GptOssMlaForCausalLM checkpoint.")

    num_layers = int(config["num_hidden_layers"])
    qk_nope_head_dim = int(config["qk_nope_head_dim"])
    v_head_dim = int(config["v_head_dim"])
    index = TensorIndex(model_path)

    absorbed_tensors: dict[str, torch.Tensor] = {}
    layers_absorbed: list[int] = []
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        if index.has_tensor(f"{prefix}.w_kc") and index.has_tensor(f"{prefix}.w_vc"):
            absorbed_tensors[f"{prefix}.w_kc"] = index.get_tensor(f"{prefix}.w_kc").contiguous()
            absorbed_tensors[f"{prefix}.w_vc"] = index.get_tensor(f"{prefix}.w_vc").contiguous()
            layers_absorbed.append(layer_id)
            continue
        if not index.has_tensor(f"{prefix}.kv_b_proj.weight"):
            continue

        kv_b_weight = index.get_tensor(f"{prefix}.kv_b_proj.weight")
        w_kc, w_vc = kv_b_weight.unflatten(
            0, (-1, qk_nope_head_dim + v_head_dim)
        ).split([qk_nope_head_dim, v_head_dim], dim=1)
        absorbed_tensors[f"{prefix}.w_kc"] = w_kc.contiguous()
        absorbed_tensors[f"{prefix}.w_vc"] = w_vc.transpose(1, 2).contiguous()
        layers_absorbed.append(layer_id)

    if not absorbed_tensors:
        raise RuntimeError("No MLA attention tensors were found to absorb.")

    absorbed_shard = "model-care-mla-absorbed.safetensors"
    save_file(absorbed_tensors, output_dir / absorbed_shard)

    for src in model_path.iterdir():
        if src.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if src.name == absorbed_shard:
            continue
        _symlink_or_copy(src, output_dir / src.name, copy_files=args.copy_files)

    new_config = copy.deepcopy(config)
    care_meta = copy.deepcopy(new_config.get("care_mla_conversion", {}))
    care_meta["absorbed_export"] = {
        "mla_absorbed_shard": absorbed_shard,
        "layers_absorbed": layers_absorbed,
    }
    new_config["care_mla_conversion"] = care_meta
    save_json(output_dir / "config.json", new_config)

    new_index = build_repaired_index(
        model_path, source_model_path=resolve_source_model_path(model_path)
    )
    for layer_id in layers_absorbed:
        prefix = f"model.layers.{layer_id}.self_attn"
        new_index["weight_map"].pop(f"{prefix}.kv_b_proj.weight", None)
    for name in absorbed_tensors:
        new_index["weight_map"][name] = absorbed_shard
    save_json(output_dir / "model.safetensors.index.json", new_index)

    save_json(
        output_dir / "care_mla_absorbed_manifest.json",
        {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "absorbed_shard": absorbed_shard,
            "layers_absorbed": layers_absorbed,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "absorbed_shard": absorbed_shard,
                "layers_absorbed": layers_absorbed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
