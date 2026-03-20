#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

from gpt_oss_care_checkpoint_utils import (
    build_repaired_config,
    build_repaired_index,
    list_local_mla_shards,
    resolve_source_model_path,
    save_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair a GPT-OSS CARE checkpoint by rebuilding config/index from actual local MLA shards over the source GPT-OSS checkpoint."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-stale-healing-metadata", action="store_true")
    return parser.parse_args()


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
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    for src in model_path.iterdir():
        if src.name in {"config.json", "model.safetensors.index.json"}:
            continue
        _symlink_or_copy(src, output_dir / src.name, copy_files=args.copy_files)

    repaired_config = build_repaired_config(
        model_path,
        clear_stale_healing_metadata=not args.keep_stale_healing_metadata,
    )
    repaired_index = build_repaired_index(
        model_path, source_model_path=resolve_source_model_path(model_path)
    )

    save_json(output_dir / "config.json", repaired_config)
    save_json(output_dir / "model.safetensors.index.json", repaired_index)
    save_json(
        output_dir / "care_mla_repair_manifest.json",
        {
            "input_model_path": str(model_path),
            "output_dir": str(output_dir),
            "source_model_path": str(resolve_source_model_path(model_path)),
            "local_mla_shards": [p.name for p in list_local_mla_shards(model_path)],
            "copy_files": bool(args.copy_files),
            "cleared_stale_healing_metadata": not bool(args.keep_stale_healing_metadata),
        },
    )


if __name__ == "__main__":
    main()
