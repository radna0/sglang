#!/usr/bin/env python3

import copy
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_source_model_path(model_path: Path) -> Path:
    config = load_json(model_path / "config.json")
    source_path = config.get("care_mla_conversion", {}).get("source_model_path")
    if source_path:
        return Path(source_path).resolve()
    return model_path.resolve()


def _existing_safetensors_shards(model_path: Path) -> list[Path]:
    shards: list[Path] = []
    for shard_path in sorted(model_path.glob("*.safetensors")):
        try:
            if shard_path.exists():
                shards.append(shard_path)
        except OSError:
            continue
    return shards


def list_local_mla_shards(model_path: Path) -> list[Path]:
    return [
        shard_path
        for shard_path in _existing_safetensors_shards(model_path)
        if shard_path.name.startswith("model-care-mla")
    ]


def _mla_pruned_weight_map(
    weight_map: dict[str, str],
    config: dict[str, Any],
) -> dict[str, str]:
    """Remove dense KV tensors that are invalid for converted MLA checkpoints."""
    architectures = config.get("architectures") or []
    if "GptOssMlaForCausalLM" not in architectures:
        return weight_map

    pruned = dict(weight_map)
    for name in list(pruned):
        if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
            pruned.pop(name, None)
    return pruned


def build_effective_weight_map(
    model_path: Path,
    *,
    source_model_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any] | None]:
    if source_model_path is None:
        source_model_path = resolve_source_model_path(model_path)

    base_index = load_json(source_model_path / "model.safetensors.index.json")
    weight_map = dict(base_index.get("weight_map", {}))

    local_index_path = model_path / "model.safetensors.index.json"
    local_index = load_json(local_index_path) if local_index_path.exists() else None
    if local_index is not None:
        for name, shard in local_index.get("weight_map", {}).items():
            shard_path = model_path / shard
            if shard_path.exists():
                weight_map[name] = shard

    for shard_path in list_local_mla_shards(model_path):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard_path.name

    config_path = model_path / "config.json"
    if config_path.exists():
        config = load_json(config_path)
        weight_map = _mla_pruned_weight_map(weight_map, config)

    return weight_map, base_index, local_index


def build_repaired_index(
    model_path: Path,
    *,
    source_model_path: Path | None = None,
) -> dict[str, Any]:
    weight_map, base_index, _ = build_effective_weight_map(
        model_path, source_model_path=source_model_path
    )
    repaired_index = copy.deepcopy(base_index)
    repaired_index["weight_map"] = weight_map
    total_size = int(base_index.get("metadata", {}).get("total_size", 0))
    total_size += sum(int(p.stat().st_size) for p in list_local_mla_shards(model_path))
    repaired_index["metadata"] = {"total_size": total_size}
    return repaired_index


def build_repaired_config(
    model_path: Path,
    *,
    clear_stale_healing_metadata: bool = True,
) -> dict[str, Any]:
    config = load_json(model_path / "config.json")
    care_meta = copy.deepcopy(config.get("care_mla_conversion", {}))
    absorbed_export = care_meta.get("absorbed_export")
    if isinstance(absorbed_export, dict):
        shard_name = absorbed_export.get("mla_absorbed_shard")
        if not shard_name or not (model_path / str(shard_name)).exists():
            care_meta.pop("absorbed_export", None)
    config["care_mla_conversion"] = care_meta
    if clear_stale_healing_metadata and not (model_path / "model-care-mla-healed-absorbed.safetensors").exists():
        config.pop("care_mla_healing_fsdp", None)
    return config
