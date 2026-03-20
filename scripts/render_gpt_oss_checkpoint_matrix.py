#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_ORDER = [
    "word_perplexity",
    "byte_perplexity",
    "bits_per_byte",
    "acc_norm",
    "acc",
    "exact_match",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side markdown matrix across GPT-OSS checkpoint eval outputs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form LABEL=DIR, where DIR contains per-task JSON outputs.",
    )
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument(
        "--task-order",
        default="wikitext,c4,arc_easy,arc_challenge,hellaswag,piqa,mmlu,openbookqa,race,winogrande",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_metric(metrics: dict[str, Any]) -> tuple[str, Any]:
    for token in METRIC_ORDER:
        for key, value in metrics.items():
            if token in key and not key.endswith("_stderr") and not key.endswith("_stderr,none"):
                return key, value
    if metrics:
        key = sorted(metrics)[0]
        return key, metrics[key]
    return "missing", "N/A"


def _task_summary_from_file(path: Path) -> tuple[str, str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    summary = payload.get("summary", {})
    if not summary:
        return None
    task_name = next(iter(summary))
    metric_name, metric_value = _pick_metric(summary[task_name])
    return task_name, metric_name, metric_value


def _scan_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    task_map: dict[str, dict[str, Any]] = {}
    if not run_dir.exists():
        return task_map
    for path in sorted(run_dir.glob("*.json")):
        parsed = _task_summary_from_file(path)
        if parsed is None:
            continue
        task_name, metric_name, metric_value = parsed
        task_map[task_name] = {
            "metric": metric_name,
            "value": metric_value,
            "source": str(path),
        }
    return task_map


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = _parse_args()
    runs: list[tuple[str, Path]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run spec: {spec!r}")
        label, raw_dir = spec.split("=", 1)
        runs.append((label.strip(), Path(raw_dir).resolve()))

    run_data = {label: _scan_run(path) for label, path in runs}
    configured_order = [task.strip() for task in args.task_order.split(",") if task.strip()]
    task_names = list(configured_order)
    for task in sorted({task for data in run_data.values() for task in data}):
        if task not in task_names:
            task_names.append(task)

    headers = ["Task", "Metric"] + [label for label, _ in runs]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for task in task_names:
        metric = None
        values: list[str] = []
        present = False
        for label, _ in runs:
            row = run_data[label].get(task)
            if row is not None:
                metric = metric or row["metric"]
                values.append(_format_value(row["value"]))
                present = True
            else:
                values.append("pending")
        if not present:
            continue
        lines.append("| " + " | ".join([task, metric or "missing", *values]) + " |")

    out_path = Path(args.markdown_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

