#!/usr/bin/env python3
"""Materialize the Kaggle AIMO3 integrated-reasoning CSV into JSONL shards.

The source dataset is a CSV export with ``prompt`` / ``completion`` style fields.
The output format is a JSONL shard directory compatible with the existing
``collect_gpt_oss_kv_covariance.py`` jsonl_filelist spec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def _to_harmony_text(prompt: str, completion: str, system_prompt: str) -> str:
    return (
        f"<|start|>system<|message|>{system_prompt}<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{completion}<|end|>"
    )


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _iter_rows(csv_path: Path) -> Iterable[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out-root", default="/workspace/data/jsonl_shards_kaggle")
    parser.add_argument("--dataset-slug", default="jeannkouagou/aimo3-tool-integrated-reasoning")
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--rows-per-shard", type=int, default=1024)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="CSV column containing user input / prompt text.",
    )
    parser.add_argument(
        "--completion-field",
        default="completion",
        help="CSV column containing target assistant text.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    ds_tag = args.dataset_slug.replace("/", "__")
    out_root = Path(args.out_root).resolve()
    shard_dir = (
        out_root
        / ds_tag
        / args.split_name
        / f"rows{args.rows_per_shard}"
        / "shards"
    )
    filelist_dir = (
        out_root
        / ds_tag
        / args.split_name
        / f"rows{args.rows_per_shard}"
        / "filelists"
        / "mod1_i0"
    )
    filelist_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    filelist = filelist_dir / "shard_jsonl_filelist_ws.txt"

    wrote = 0
    seen_ids: set[str] = set()
    shard_idx = 0
    row_buffer = 0
    out_fh = None
    file_paths: list[str] = []

    def open_new_shard() -> None:
        nonlocal out_fh, shard_idx, row_buffer
        if out_fh is not None:
            out_fh.close()
        out_path = shard_dir / f"shard_{shard_idx:06d}.jsonl"
        shard_idx += 1
        out_fh = out_path.open("w", encoding="utf-8")
        file_paths.append(str(out_path))
        row_buffer = 0

    open_new_shard()

    for row in _iter_rows(csv_path):
        if args.limit_rows is not None and wrote >= int(args.limit_rows):
            break

        prompt = str(row.get(args.prompt_field, "") or "").strip()
        completion = str(row.get(args.completion_field, "") or "").strip()
        if not prompt or not completion:
            continue

        text = _to_harmony_text(prompt=prompt, completion=completion, system_prompt=args.system_prompt)
        if not text:
            continue

        row_id = str(row.get("problem_id") or row.get("id") or row.get("uuid") or "").strip()
        if not row_id:
            row_id = _stable_id(text)
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)

        if out_fh is None:
            open_new_shard()
        out_fh.write(json.dumps({"id": row_id, "text": text}, ensure_ascii=False) + "\n")
        wrote += 1
        row_buffer += 1

        if row_buffer >= int(args.rows_per_shard):
            open_new_shard()

    if out_fh is not None:
        out_fh.close()

    # remove last empty shard if nothing was written after rotation
    if file_paths:
        last_path = Path(file_paths[-1])
        if last_path.exists() and last_path.stat().st_size == 0:
            file_paths.pop()
            last_path.unlink(missing_ok=True)

    filelist.write_text("\n".join(file_paths) + ("\n" if file_paths else ""), encoding="utf-8")

    print(f"dataset_slug={args.dataset_slug}")
    print(f"out_root={out_root}")
    print(f"rows_written={wrote}")
    print(f"shards={len(file_paths)}")
    print(f"filelist={filelist}")


if __name__ == "__main__":
    main()

