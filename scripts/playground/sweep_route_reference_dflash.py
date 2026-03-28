#!/usr/bin/env python3
"""Run explore32->route8 DFlash routing on the reference problems one at a time."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_question_ids(reference_csv: Path) -> list[str]:
    with reference_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [str(row.get("id") or "").strip() for row in rows if str(row.get("id") or "").strip()]


def _parse_question_ids(args: argparse.Namespace) -> list[str]:
    if str(args.question_ids or "").strip():
        return [q.strip() for q in str(args.question_ids).split(",") if q.strip()]
    return _load_question_ids(Path(args.reference_csv))


def _route_cmd(args: argparse.Namespace, question_id: str, out_json: Path) -> list[str]:
    route_script = Path(args.route_script)
    return [
        sys.executable,
        str(route_script),
        "--model-path",
        str(args.model_path),
        "--draft-model-path",
        str(args.draft_model_path),
        "--reference-csv",
        str(args.reference_csv),
        "--question-ids",
        str(question_id),
        "--out-json",
        str(out_json),
        "--final-context-length",
        str(int(args.final_context_length)),
        "--exploration-decode-len",
        str(int(args.exploration_decode_len)),
        "--exploration-round-len",
        str(int(args.exploration_round_len)),
        "--exploration-min-rounds",
        str(int(args.exploration_min_rounds)),
        "--exploration-concurrency",
        str(int(args.exploration_concurrency)),
        "--exploration-num-prompts",
        str(int(args.exploration_num_prompts)),
        "--buffer-tokens",
        str(int(args.buffer_tokens)),
        "--mem-fraction-static",
        str(float(args.mem_fraction_static)),
        "--promotion-mode",
        str(args.promotion_mode),
        "--promote-total-k",
        str(int(args.promote_total_k)),
        "--min-keep-per-qid",
        str(int(args.min_keep_per_qid)),
    ]


def _extract_row(question_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    explore = payload["exploration"]["summary"]
    continue_ = payload["continuation"]["summary"]
    selection = payload["selection"]
    cont_mv = payload["continuation"].get("majority_vote") or {}
    sel_mv = selection.get("majority_vote") or {}
    return {
        "question_id": str(question_id),
        "explore_wall_tok_s": float(explore["wall_tok_s"]),
        "explore_accept_length": float(explore["accept_length"]),
        "explore_verify_ct_sum": int(explore["verify_ct_sum"]),
        "explore_avg_output_tokens": float(explore["avg_output_tokens"]),
        "selected_count": int(selection["selected_count"]),
        "selected_green": int(selection["selected_by_label"].get("green", 0)),
        "selected_neutral": int(selection["selected_by_label"].get("neutral", 0)),
        "selected_hard_tail": int(selection["selected_by_label"].get("hard_tail", 0)),
        "selected_confident_conflict": int(
            selection["selected_by_label"].get("confident_conflict", 0)
        ),
        "selected_majority_answer": sel_mv.get("majority_answer"),
        "selected_majority_support": int(sel_mv.get("majority_support") or 0),
        "selected_is_correct_majority": bool(sel_mv.get("is_correct_majority")),
        "continue_wall_tok_s": float(continue_["wall_tok_s"]),
        "continue_accept_length": float(continue_["accept_length"]),
        "continue_verify_ct_sum": int(continue_["verify_ct_sum"]),
        "continue_avg_output_tokens": float(continue_["avg_output_tokens"]),
        "final_majority_answer": cont_mv.get("majority_answer"),
        "final_majority_support": int(cont_mv.get("majority_support") or 0),
        "final_is_correct_majority": bool(cont_mv.get("is_correct_majority")),
    }


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Route Sweep",
        "",
        "| qid | explore tok/s | explore accept | selected labels | final tok/s | final accept | final majority | support | correct |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        labels = (
            f"g={row['selected_green']},n={row['selected_neutral']},"
            f"h={row['selected_hard_tail']},c={row['selected_confident_conflict']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["question_id"]),
                    f"{row['explore_wall_tok_s']:.3f}",
                    f"{row['explore_accept_length']:.3f}",
                    labels,
                    f"{row['continue_wall_tok_s']:.3f}",
                    f"{row['continue_accept_length']:.3f}",
                    str(row["final_majority_answer"]),
                    str(row["final_majority_support"]),
                    str(bool(row["final_is_correct_majority"])),
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep explore32->route8 over the reference problems one problem at a time."
    )
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--route-script", default=str(Path(__file__).with_name("route_reference_dflash.py")))
    p.add_argument("--model-path", default="/workspace/offload_root/gpt-oss-120b")
    p.add_argument("--draft-model-path", default="/root/epoch_65_step_23760")
    p.add_argument("--reference-csv", default="/root/reference.csv")
    p.add_argument("--question-ids", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--final-context-length", type=int, default=65536)
    p.add_argument("--exploration-decode-len", type=int, default=8192)
    p.add_argument("--exploration-round-len", type=int, default=2048)
    p.add_argument("--exploration-min-rounds", type=int, default=2)
    p.add_argument("--exploration-concurrency", type=int, default=32)
    p.add_argument("--exploration-num-prompts", type=int, default=32)
    p.add_argument("--buffer-tokens", type=int, default=512)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--promotion-mode", choices=("strict", "throughput"), default="strict")
    p.add_argument("--promote-total-k", type=int, default=8)
    p.add_argument("--min-keep-per-qid", type=int, default=1)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    question_ids = _parse_question_ids(args)
    repo_python = str(Path(__file__).resolve().parents[2] / "python")
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_python if not env.get("PYTHONPATH") else f"{repo_python}:{env['PYTHONPATH']}"
    env.setdefault("SGLANG_SERVER_PYTHON_EXECUTABLE", sys.executable)

    rows: list[dict[str, Any]] = []
    for qid in question_ids:
        out_json = out_dir / f"{qid}.json"
        if not out_json.exists() or not bool(args.resume):
            cmd = _route_cmd(args, qid, out_json)
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        rows.append(_extract_row(qid, payload))
        _write_summary(out_dir, rows)

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
