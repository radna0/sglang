#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any


TASK_ORDER = [
    "wikitext",
    "c4",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "mmlu",
    "openbookqa",
    "race",
    "winogrande",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a combined benchmark matrix for GPT-OSS CARE sweep outputs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form LABEL=DIR where DIR contains tasks/, passkey/, and fixed_pack/ subdirs.",
    )
    parser.add_argument("--markdown-out", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_metric(metrics: dict[str, Any]) -> tuple[str, Any]:
    preferred = [
        "word_perplexity",
        "byte_perplexity",
        "bits_per_byte",
        "acc_norm",
        "acc",
        "exact_match",
    ]
    for token in preferred:
        for key, value in metrics.items():
            if token in key and not key.endswith("_stderr") and not key.endswith("_stderr,none"):
                return key, value
    if metrics:
        key = sorted(metrics)[0]
        return key, metrics[key]
    return "missing", "pending"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
        return f"{value:.6f}"
    return str(value)


def _scan_task_dir(task_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not task_dir.exists():
        return out
    for task in TASK_ORDER:
        payload = _load_json(task_dir / f"{task}.json")
        if not payload:
            continue
        summary = payload.get("summary", {})
        if task not in summary:
            continue
        metric_name, metric_value = _pick_metric(summary[task])
        out[task] = {"metric": metric_name, "value": metric_value}
    return out


def _scan_passkey_dir(passkey_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    payload = _load_json(passkey_dir / "passkey.json")
    if not payload:
        return out
    summary = payload.get("summary", {})
    grouped: dict[int, list[float]] = {}
    for bucket, values in summary.items():
        try:
            context_str, _position = bucket.split(":", 1)
            context = int(context_str)
            acc = float(values["accuracy"])
        except Exception:
            continue
        grouped.setdefault(context, []).append(acc)
    for context, vals in sorted(grouped.items()):
        if not vals:
            continue
        out[f"passkey@{context}"] = {
            "metric": "accuracy_mean",
            "value": sum(vals) / len(vals),
        }
    return out


def _scan_fixed_pack_dir(fixed_pack_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    mapping = {
        "combined_fixed_eval_pack.json": "combined_fixed",
        "aimo3_long.json": "aimo3_long",
    }
    for filename, label in mapping.items():
        payload = _load_json(fixed_pack_dir / filename)
        if not payload:
            continue
        results = payload.get("results", {})
        for context, ctx_payload in sorted(results.items(), key=lambda kv: int(kv[0])):
            summary = ctx_payload.get("summary", {})
            value = summary.get("word_perplexity")
            out[f"{label}@{context}"] = {
                "metric": "word_perplexity",
                "value": value,
            }
    return out


def _scan_run(run_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    out.update(_scan_task_dir(run_root / "tasks"))
    out.update(_scan_passkey_dir(run_root / "passkey"))
    out.update(_scan_fixed_pack_dir(run_root / "fixed_pack"))
    return out


def main() -> None:
    args = _parse_args()
    runs: list[tuple[str, Path]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec: {spec!r}")
        label, raw_dir = spec.split("=", 1)
        runs.append((label.strip(), Path(raw_dir).resolve()))

    run_data = {label: _scan_run(path) for label, path in runs}

    ordered_rows = list(TASK_ORDER) + [
        "passkey@2048",
        "passkey@4096",
        "combined_fixed@2048",
        "combined_fixed@8192",
        "aimo3_long@2048",
        "aimo3_long@8192",
    ]
    for key in sorted({k for data in run_data.values() for k in data.keys()}):
        if key not in ordered_rows:
            ordered_rows.append(key)

    headers = ["Benchmark", "Metric"] + [label for label, _ in runs]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for benchmark in ordered_rows:
        metric = None
        values = []
        present = False
        for label, _ in runs:
            row = run_data[label].get(benchmark)
            if row is not None:
                metric = metric or row["metric"]
                values.append(_fmt(row["value"]))
                present = True
            else:
                values.append("pending")
        if not present:
            continue
        lines.append("| " + " | ".join([benchmark, metric or "missing", *values]) + " |")

    out_path = Path(args.markdown_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

