#!/usr/bin/env python3
"""Materialize a local GPT-OSS CARE MLA overlay for SGLang.

This helper downloads the base GPT-OSS-120B model shards once, then builds a
rank-specific overlay directory that contains:

- the CARE MLA config/index/manifest
- the rank-specific MLA attention shard
- the GPT-OSS remote-code files
- symlinks to the 14 base GPT-OSS safetensor shards

The overlay directory can then be passed directly to SGLang or the reference
probe script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open


BASE_REPO_ID = "openai/gpt-oss-120b"
DATASET_REPO_ID = "radna0/sglang_gpt_oss_care_runs"

OVERLAY_SUBPATHS = {
    512: "gptoss120b_care_u_convert_only_sweep_20260315_0622/care_u_r512/conversion/converted_checkpoint",
    1024: "gptoss120b_care_u_convert_only_sweep_20260315_0622/care_u_r1024/conversion/converted_checkpoint",
}

BASE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
]

OVERLAY_SMALL_FILES = [
    "config.json",
    "care_mla_manifest.json",
    "model.safetensors.index.json",
    "model-care-mla-attention.safetensors",
    "modeling_gpt_oss_mla.py",
    "__init__.py",
    ".gpt_oss_mla_loader.lock",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the GPT-OSS base model and materialize a rank-specific CARE MLA overlay."
    )
    parser.add_argument("--rank", type=int, choices=sorted(OVERLAY_SUBPATHS), required=True)
    parser.add_argument("--base-dir", default="/workspace/offload_root/gpt-oss-120b")
    parser.add_argument("--overlay-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-base", action="store_true", default=True)
    parser.add_argument("--no-download-base", dest="download_base", action="store_false")
    return parser.parse_args()


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_base_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[base] downloading {BASE_REPO_ID} into {base_dir}", flush=True)
    snapshot_download(
        repo_id=BASE_REPO_ID,
        repo_type="model",
        local_dir=str(base_dir),
        local_dir_use_symlinks=False,
        allow_patterns=BASE_PATTERNS,
    )


def _download_overlay_file(rank: int, filename: str) -> Path:
    subpath = OVERLAY_SUBPATHS[rank]
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=f"{subpath}/{filename}",
        )
    )


def _base_shards(base_dir: Path) -> list[Path]:
    return sorted(p for p in base_dir.glob("model-*.safetensors") if p.is_file())


def _ensure_overlay_dir(overlay_dir: Path, *, overwrite: bool) -> None:
    if overlay_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{overlay_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)


def _validate_overlay(overlay_dir: Path, *, rank: int, base_dir: Path) -> None:
    index_path = overlay_dir / "model.safetensors.index.json"
    config_path = overlay_dir / "config.json"
    attn_path = overlay_dir / "model-care-mla-attention.safetensors"
    if not index_path.exists() or not config_path.exists() or not attn_path.exists():
        raise FileNotFoundError(f"Overlay is incomplete: {overlay_dir}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    missing = []
    for name, shard in weight_map.items():
        shard_path = overlay_dir / shard
        if not shard_path.exists():
            missing.append((name, shard))
    if missing:
        sample = ", ".join(f"{n}->{s}" for n, s in missing[:6])
        raise FileNotFoundError(f"Overlay is missing weight shards: {sample}")

    with safe_open(attn_path, framework="pt", device="cpu") as handle:
        attn_keys = list(handle.keys())
    if not attn_keys:
        raise RuntimeError(f"Attention shard is empty: {attn_path}")

    print(
        json.dumps(
            {
                "overlay_dir": str(overlay_dir),
                "rank": int(rank),
                "base_dir": str(base_dir),
                "base_shards": len(_base_shards(base_dir)),
                "attention_keys": len(attn_keys),
                "attention_shard_size": attn_path.stat().st_size,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    rank = int(args.rank)
    overlay_subpath = OVERLAY_SUBPATHS[rank]
    base_dir = Path(args.base_dir).expanduser().resolve()
    overlay_dir = Path(args.overlay_dir).expanduser().resolve() if args.overlay_dir else Path(
        f"/workspace/r{rank}_absorbed"
    ).resolve()

    if args.download_base:
        _materialize_base_dir(base_dir)
    elif not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    _ensure_overlay_dir(overlay_dir, overwrite=args.overwrite)

    for filename in OVERLAY_SMALL_FILES:
        src = _download_overlay_file(rank, filename)
        dst = overlay_dir / filename
        if filename == "model.safetensors.index.json":
            shutil.copy2(src, dst)
        elif filename == "config.json":
            with src.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            care_meta = payload.get("care_mla_conversion", {})
            care_meta["source_model_path"] = str(base_dir)
            payload["care_mla_conversion"] = care_meta
            with dst.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        elif filename == ".gpt_oss_mla_loader.lock":
            dst.touch(exist_ok=True)
        else:
            shutil.copy2(src, dst)

    for shard_path in _base_shards(base_dir):
        _copy_or_link(shard_path, overlay_dir / shard_path.name)

    for tokenizer_name in [
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src = base_dir / tokenizer_name
        if src.exists() and not (overlay_dir / tokenizer_name).exists():
            _copy_or_link(src, overlay_dir / tokenizer_name)

    _validate_overlay(overlay_dir, rank=rank, base_dir=base_dir)
    print(f"[done] {overlay_dir}", flush=True)


if __name__ == "__main__":
    main()
