#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-task GPT-OSS eval JSON outputs into one artifact."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    inputs = [Path(path).resolve() for path in args.inputs]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    merged_summary: dict[str, Any] = {}
    merged_raw_results: dict[str, Any] = {}
    payloads: list[dict[str, Any]] = []

    for path in inputs:
        payload = _load_json(path)
        payloads.append({"path": str(path), "payload": payload})
        for task_name, metrics in payload.get("summary", {}).items():
            merged_summary[task_name] = metrics
        raw = payload.get("raw", {})
        for task_name, metrics in raw.get("results", {}).items():
            merged_raw_results[task_name] = metrics

    merged = {
        "summary": merged_summary,
        "raw": {"results": merged_raw_results},
        "inputs": [item["path"] for item in payloads],
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(json.dumps({"output": str(output), "inputs": merged["inputs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
