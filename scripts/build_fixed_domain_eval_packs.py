#!/usr/bin/env python3

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic small fixed-sample domain packs for GPT-OSS CARE evals."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples-per-source", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--aimo3-jsonl",
        default="/workspace/data/kaggle_snapshots/wenliangtlh__aimo3-high-difficulty-tool-calling-dataset/AIMO3-High-Difficulty-Tool-Calling-Dataset.jsonl",
    )
    parser.add_argument(
        "--aimo3-shards-dir",
        default="/workspace/data/jsonl_shards_modelscope/Isekai-Creation__aimo3-high-difficulty-tool-calling-jsonl-shards",
    )
    parser.add_argument(
        "--calib-dir",
        default="/workspace/data/jsonl_shards_modelscope/Isekai-Creation__harmony-qwen3-calib-packs-20260113",
    )
    return parser.parse_args()


def _stable_score(seed: int, *parts: str) -> int:
    payload = "||".join((str(seed),) + tuple(parts)).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest(), 16)


def _write_pack(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _pick_longest_jsonl(
    paths: list[Path],
    *,
    source: str,
    text_field: str,
    id_field: str,
    sample_count: int,
) -> list[dict]:
    heap: list[tuple[int, int, dict]] = []
    for path in paths:
        for row in _read_jsonl(path):
            text = str(row.get(text_field, "") or "")
            if not text:
                continue
            row_id = str(row.get(id_field, "")) or f"{path.name}:{len(heap)}"
            packed = {
                "source": source,
                "sample_id": row_id,
                "text": text,
                "char_len": len(text),
            }
            item = (len(text), row_id, packed)
            if len(heap) < sample_count:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: (-int(row["char_len"]), str(row["sample_id"])))
    return rows


def _pick_seeded_jsonl(
    paths: list[Path],
    *,
    source: str,
    text_field: str,
    id_field: str,
    sample_count: int,
    seed: int,
    require_tool: bool = False,
) -> list[dict]:
    rows = []
    for path in paths:
        for row in _read_jsonl(path):
            if require_tool and not bool(row.get("quality_has_tool", False)):
                continue
            text = str(row.get(text_field, "") or "")
            if not text:
                continue
            row_id = str(row.get(id_field, "")) or f"{path.name}:{len(rows)}"
            rows.append(
                {
                    "source": source,
                    "sample_id": row_id,
                    "text": text,
                    "char_len": len(text),
                    "_score": _stable_score(seed, source, row_id),
                }
            )
    rows.sort(key=lambda row: (row["_score"], row["sample_id"]))
    rows = rows[:sample_count]
    for row in rows:
        row.pop("_score", None)
    return rows


def _pick_seeded_hf(
    dataset_name: str,
    *,
    split: str,
    source: str,
    sample_count: int,
    seed: int,
) -> list[dict]:
    ds = load_dataset(dataset_name, split=split)
    rows = []
    if dataset_name == "livecodebench/code_generation":
        for row in ds:
            text = (
                f"Title: {row['question_title']}\n\n"
                f"Platform: {row['platform']}\nDifficulty: {row['difficulty']}\n\n"
                f"{row['question_content']}"
            )
            row_id = str(row["question_id"])
            rows.append(
                {
                    "source": source,
                    "sample_id": row_id,
                    "text": text,
                    "char_len": len(text),
                    "_score": _stable_score(seed, source, row_id),
                }
            )
    elif dataset_name == "HuggingFaceH4/MATH-500":
        for row in ds:
            text = (
                f"Subject: {row['subject']}\nLevel: {row['level']}\n\n"
                f"{row['problem']}"
            )
            row_id = str(row["unique_id"])
            rows.append(
                {
                    "source": source,
                    "sample_id": row_id,
                    "text": text,
                    "char_len": len(text),
                    "_score": _stable_score(seed, source, row_id),
                }
            )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    rows.sort(key=lambda row: (row["_score"], row["sample_id"]))
    rows = rows[:sample_count]
    for row in rows:
        row.pop("_score", None)
    return rows


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aimo3_jsonl = Path(args.aimo3_jsonl).resolve()
    aimo3_shards_dir = Path(args.aimo3_shards_dir).resolve()
    calib_dir = Path(args.calib_dir).resolve()

    aimo3_paths: list[Path] = []
    if aimo3_jsonl.exists():
        aimo3_paths.append(aimo3_jsonl)
    if aimo3_shards_dir.exists():
        aimo3_paths.extend(sorted(aimo3_shards_dir.rglob("*.jsonl")))
    calib_paths = sorted(calib_dir.rglob("*.jsonl"))

    if not aimo3_paths:
        raise FileNotFoundError("No AIMO3 JSONL sources found.")
    if not calib_paths:
        raise FileNotFoundError("No Harmony calib-pack JSONLs found.")

    packs = {
        "aimo3_long.jsonl": _pick_longest_jsonl(
            aimo3_paths,
            source="aimo3_long",
            text_field="text",
            id_field="id",
            sample_count=args.samples_per_source,
        ),
        "calib_tool_seeded.jsonl": _pick_seeded_jsonl(
            calib_paths,
            source="calib_tool",
            text_field="text",
            id_field="id",
            sample_count=args.samples_per_source,
            seed=args.seed,
            require_tool=True,
        ),
        "livecodebench_seeded.jsonl": _pick_seeded_hf(
            "livecodebench/code_generation",
            split="test",
            source="livecodebench",
            sample_count=args.samples_per_source,
            seed=args.seed,
        ),
        "math500_seeded.jsonl": _pick_seeded_hf(
            "HuggingFaceH4/MATH-500",
            split="test",
            source="math500",
            sample_count=args.samples_per_source,
            seed=args.seed,
        ),
    }

    combined = []
    for filename, rows in packs.items():
        _write_pack(out_dir / filename, rows)
        combined.extend(rows)
    _write_pack(out_dir / "combined_fixed_eval_pack.jsonl", combined)

    manifest = {
        "seed": args.seed,
        "samples_per_source": args.samples_per_source,
        "packs": {
            filename: {
                "count": len(rows),
                "source": rows[0]["source"] if rows else None,
                "max_char_len": max((row["char_len"] for row in rows), default=0),
                "min_char_len": min((row["char_len"] for row in rows), default=0),
            }
            for filename, rows in packs.items()
        },
        "combined_count": len(combined),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
