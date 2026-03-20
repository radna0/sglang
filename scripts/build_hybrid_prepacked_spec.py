#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from collect_gpt_oss_kv_covariance import _load_json, _save_json, _safe_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed raw/prepacked dataset spec from completed prepack manifests."
    )
    parser.add_argument("--raw-spec-json", required=True)
    parser.add_argument("--prepack-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _replace_source(raw_source: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    source = {
        "name": raw_source["name"],
        "kind": "packed_torch_filelist",
        "filelist": manifest["filelist"],
        # Preserve the original row accounting so target_total_rows stays consistent.
        "max_rows": int(manifest.get("rows_used", 0) or 0),
        # The packed path consumes already-packed full-length sequences directly.
        "max_sequences": int(manifest.get("num_sequences", 0) or 0),
        "seq_len": int(manifest.get("seq_len", 0) or 0),
        "original_kind": raw_source.get("kind"),
        "source_path": manifest.get("source_path", raw_source.get("path", "")),
    }
    return source


def main() -> None:
    args = _parse_args()
    raw_spec = _load_json(Path(args.raw_spec_json))
    prepack_root = Path(args.prepack_root)

    completed = []
    hybrid_sources = []
    for raw_source in raw_spec["sources"]:
        manifest_path = prepack_root / _safe_name(raw_source["name"]) / "prepack_manifest.json"
        if manifest_path.exists():
            hybrid_sources.append(_replace_source(raw_source, manifest_path))
            completed.append(raw_source["name"])
            continue
        hybrid_sources.append(raw_source)

    payload = {
        "sources": hybrid_sources,
        "hybrid_prepacked_sources": completed,
        "raw_spec_json": str(Path(args.raw_spec_json).resolve()),
        "prepack_root": str(prepack_root.resolve()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(output_path, payload)
    print(
        {
            "output_json": str(output_path),
            "num_sources": len(hybrid_sources),
            "packed_sources": completed,
        }
    )


if __name__ == "__main__":
    main()
