#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_gpt_oss_kv_covariance import (  # noqa: E402
    _compute_row_budgets,
    _iter_rows,
    _load_json,
    _row_to_text,
    _safe_name,
    _save_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretokenize and prepack GPT-OSS CARE corpus sources into fixed-length token shards."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--shard-sequences", type=int, default=1024)
    return parser.parse_args()


def _flush_shard(
    source_dir: Path,
    shard_idx: int,
    shard_sequences: list[list[int]],
    shard_paths: list[str],
) -> int:
    if not shard_sequences:
        return shard_idx
    tensor = torch.tensor(shard_sequences, dtype=torch.int32)
    shard_path = source_dir / f"packed_{shard_idx:05d}.pt"
    torch.save({"input_ids": tensor, "num_sequences": int(tensor.shape[0])}, shard_path)
    shard_paths.append(str(shard_path))
    shard_sequences.clear()
    return shard_idx + 1


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    dataset_spec_path = Path(args.dataset_spec_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_json(dataset_spec_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    row_budgets = _compute_row_budgets(spec, args.target_total_rows)

    packed_sources: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []

    for source, row_budget in zip(spec["sources"], row_budgets):
        source_name = source.get("name", source.get("path", source["kind"]))
        safe_name = _safe_name(source_name)
        source_dir = output_dir / safe_name
        source_dir.mkdir(parents=True, exist_ok=True)
        shard_filelist_path = source_dir / "packed_shard_filelist.txt"

        buffer: list[int] = []
        rows_used = 0
        sequences_emitted = 0
        shard_idx = 0
        shard_paths: list[str] = []
        shard_sequences: list[list[int]] = []

        for row in _iter_rows(source):
            if row_budget and rows_used >= row_budget:
                break
            text = _row_to_text(row, source)
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            if args.append_eos and eos_token_id is not None:
                token_ids = list(token_ids) + [int(eos_token_id)]
            buffer.extend(int(x) for x in token_ids)
            rows_used += 1

            while len(buffer) >= int(args.seq_len):
                shard_sequences.append(buffer[: int(args.seq_len)])
                buffer = buffer[int(args.seq_len) :]
                sequences_emitted += 1
                if len(shard_sequences) >= int(args.shard_sequences):
                    shard_idx = _flush_shard(source_dir, shard_idx, shard_sequences, shard_paths)

        shard_idx = _flush_shard(source_dir, shard_idx, shard_sequences, shard_paths)
        shard_filelist_path.write_text("\n".join(shard_paths) + ("\n" if shard_paths else ""), encoding="utf-8")

        source_manifest = {
            "name": source_name,
            "kind": source["kind"],
            "row_budget": int(row_budget),
            "rows_used": int(rows_used),
            "num_sequences": int(sequences_emitted),
            "seq_len": int(args.seq_len),
            "filelist": str(shard_filelist_path),
            "source_path": source.get("path", source.get("filelist", "<local-files>")),
        }
        _save_json(source_dir / "prepack_manifest.json", source_manifest)
        source_manifests.append(source_manifest)
        packed_sources.append(
            {
                "name": source_name,
                "kind": "packed_torch_filelist",
                "filelist": str(shard_filelist_path),
                "max_sequences": int(sequences_emitted),
                "seq_len": int(args.seq_len),
            }
        )
        print(
            json.dumps(
                {
                    "source": source_name,
                    "rows_used": rows_used,
                    "num_sequences": sequences_emitted,
                    "filelist": str(shard_filelist_path),
                }
            ),
            flush=True,
        )

    packed_spec = {
        "model_path": str(model_path),
        "raw_dataset_spec_json": str(dataset_spec_path),
        "seq_len": int(args.seq_len),
        "sources": packed_sources,
    }
    packed_spec_path = output_dir / "prepacked_dataset_spec.json"
    _save_json(packed_spec_path, packed_spec)
    _save_json(
        output_dir / "prepack_manifest.json",
        {
            "model_path": str(model_path),
            "raw_dataset_spec_json": str(dataset_spec_path),
            "seq_len": int(args.seq_len),
            "sources": source_manifests,
            "prepacked_dataset_spec_json": str(packed_spec_path),
        },
    )
    print(json.dumps({"prepacked_dataset_spec_json": str(packed_spec_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
