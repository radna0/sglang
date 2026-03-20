#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


METRIC_PRIORITY = [
    "exact_match",
    "acc_norm",
    "acc",
    "word_perplexity",
    "byte_perplexity",
    "bits_per_byte",
]

PPL_TASK_ORDER = ["wikitext", "c4"]
BENCHMARK_TASK_ORDER = [
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
        description="Render a markdown/CSV summary table from GPT-OSS baseline eval artifacts."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_metric(metrics: dict) -> tuple[str, object]:
    for token in METRIC_PRIORITY:
        for key, value in metrics.items():
            if token in key and not key.endswith("_stderr,none") and not key.endswith("_stderr"):
                return key, value
    if metrics:
        key = sorted(metrics)[0]
        return key, metrics[key]
    return "missing", "N/A"


def _as_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _render_markdown(rows: list[dict], avg_row: dict | None) -> str:
    lines = [
        "| Group | Task | Metric | Value |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['task']} | {row['metric']} | {row['value']} |"
        )
    if avg_row is not None:
        lines.append(
            f"| {avg_row['group']} | {avg_row['task']} | {avg_row['metric']} | {avg_row['value']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _append_rows(rows: list[dict], group: str, payload: dict) -> None:
    for task_name, metrics in payload.get("summary", {}).items():
        metric, value = _pick_metric(metrics)
        rows.append({"group": group, "task": task_name, "metric": metric, "value": value})


def _load_if_exists(path: Path) -> dict | None:
    if path.exists():
        return _load_json(path)
    return None


def _sorted_rows(rows: list[dict]) -> list[dict]:
    group_priority = {"ppl": 0, "benchmark": 1, "long_context": 2}
    ppl_priority = {name: idx for idx, name in enumerate(PPL_TASK_ORDER)}
    benchmark_priority = {name: idx for idx, name in enumerate(BENCHMARK_TASK_ORDER)}

    def _key(row: dict) -> tuple[int, int, str]:
        group = row["group"]
        task = row["task"]
        if group == "ppl":
            return (group_priority[group], ppl_priority.get(task, 999), task)
        if group == "benchmark":
            return (group_priority[group], benchmark_priority.get(task, 999), task)
        return (group_priority.get(group, 999), 999, task)

    return sorted(rows, key=_key)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    markdown_out = (
        Path(args.markdown_out).resolve()
        if args.markdown_out
        else run_dir / "baseline_table.md"
    )
    csv_out = Path(args.csv_out).resolve() if args.csv_out else run_dir / "baseline_table.csv"
    json_out = (
        Path(args.json_out).resolve()
        if args.json_out
        else run_dir / "baseline_table.json"
    )

    rows: list[dict] = []

    ppl_dir = run_dir / "ppl"
    if ppl_dir.exists():
        for task_name in PPL_TASK_ORDER:
            payload = _load_if_exists(ppl_dir / f"{task_name}.json")
            if payload is not None:
                _append_rows(rows, "ppl", payload)
    else:
        for task_name in PPL_TASK_ORDER:
            payload = _load_if_exists(run_dir / f"{task_name}.json")
            if payload is not None:
                _append_rows(rows, "ppl", payload)

    tasks_dir = run_dir / "tasks"
    if tasks_dir.exists():
        for task_name in BENCHMARK_TASK_ORDER:
            payload = _load_if_exists(tasks_dir / f"{task_name}.json")
            if payload is not None:
                _append_rows(rows, "benchmark", payload)
    else:
        benchmark_payload = _load_if_exists(run_dir / "benchmarks.json")
        if benchmark_payload is not None:
            _append_rows(rows, "benchmark", benchmark_payload)

    passkey_payload = _load_if_exists(run_dir / "hf_passkey.json")
    if passkey_payload is not None:
        for bucket, metrics in passkey_payload.get("summary", {}).items():
            rows.append(
                {
                    "group": "long_context",
                    "task": f"passkey:{bucket}",
                    "metric": "accuracy",
                    "value": metrics.get("accuracy", "N/A"),
                }
            )

    longbench_payload = _load_if_exists(run_dir / "hf_longbench_v2.json")
    if longbench_payload is not None:
        for bucket, metrics in longbench_payload.get("summary", {}).items():
            metric, value = _pick_metric(metrics if isinstance(metrics, dict) else {"value": metrics})
            rows.append(
                {
                    "group": "long_context",
                    "task": f"longbench_v2:{bucket}",
                    "metric": metric,
                    "value": value,
                }
            )

    rows = _sorted_rows(rows)

    avg_candidates = [
        _as_float(row["value"])
        for row in rows
        if row["group"] == "benchmark" and _as_float(row["value"]) is not None
    ]
    avg_row = None
    if avg_candidates:
        avg_value = sum(avg_candidates) / len(avg_candidates)
        avg_row = {
            "group": "benchmark",
            "task": "AVG",
            "metric": "mean_selected_metric",
            "value": round(avg_value, 6),
        }

    payload = {
        "run_dir": str(run_dir),
        "rows": rows,
        "average": avg_row,
    }

    markdown_out.write_text(_render_markdown(rows, avg_row), encoding="utf-8")
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
        if avg_row is not None:
            writer.writerow(avg_row)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
