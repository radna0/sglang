#!/usr/bin/env python3
"""Run the AIMO3 reference questions on multiple GPT-OSS checkpoints.

This wrapper executes the small deterministic GPT-OSS reference probe on a
sequence of model paths and then renders a side-by-side markdown table from the
probe outputs.

The intent is to compare:
  - original GPT-OSS-120B GQA
  - GPT-OSS-120B CARE MLA r512
  - GPT-OSS-120B CARE MLA r1024

The probe itself uses the showtime-style OpenAI Harmony path and expects the
final answer in \\boxed{}.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_QIDS = ["92ba6a", "9c1c5f", "a295e9"]
DEFAULT_BACKENDS = {
    "original": "fa3",
    "r512": "flashmla",
    "r1024": "flashmla",
}
DEFAULT_MEM_FRACTIONS = {
    "original": 0.95,
    "r512": 0.90,
    "r1024": 0.90,
}
DEFAULT_MOE_BACKENDS = {
    "original": None,
    "r512": "triton",
    "r1024": "triton",
}
DEFAULT_FLASHINFER_MXFP4_PRECISION = {
    "original": None,
    "r512": None,
    "r1024": None,
}
DEFAULT_CPU_OFFLOAD_GB = {
    "original": 0,
    "r512": 2,
    "r1024": 2,
}


def _validate_backend_policy(name: str, backend: str) -> None:
    if name == "original" and backend != "fa3":
        raise ValueError("The original GPT-OSS-GQA lane must use fa3.")
    if name in {"r512", "r1024"} and backend != "flashmla":
        raise ValueError("The GPT-OSS-MLA lanes must use flashmla.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS AIMO3 reference matrix probes.")
    parser.add_argument("--probe-script", default="/workspace/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_reference_probe.py")
    parser.add_argument("--csv-path", default="/root/reference.csv")
    parser.add_argument("--question-ids", default=",".join(DEFAULT_QIDS))
    parser.add_argument("--out-root", default="/workspace/gptoss120b_aimo3_reference_matrix")
    parser.add_argument("--original-model-path", required=True)
    parser.add_argument("--r512-model-path", required=True)
    parser.add_argument("--r1024-model-path", required=True)
    parser.add_argument("--original-backend", default=DEFAULT_BACKENDS["original"])
    parser.add_argument("--r512-backend", default=DEFAULT_BACKENDS["r512"])
    parser.add_argument("--r1024-backend", default=DEFAULT_BACKENDS["r1024"])
    parser.add_argument("--original-moe-runner-backend", default=DEFAULT_MOE_BACKENDS["original"])
    parser.add_argument("--r512-moe-runner-backend", default=DEFAULT_MOE_BACKENDS["r512"])
    parser.add_argument("--r1024-moe-runner-backend", default=DEFAULT_MOE_BACKENDS["r1024"])
    parser.add_argument(
        "--original-flashinfer-mxfp4-moe-precision",
        default=DEFAULT_FLASHINFER_MXFP4_PRECISION["original"],
        choices=("default", "bf16"),
    )
    parser.add_argument(
        "--r512-flashinfer-mxfp4-moe-precision",
        default=DEFAULT_FLASHINFER_MXFP4_PRECISION["r512"],
        choices=("default", "bf16"),
    )
    parser.add_argument(
        "--r1024-flashinfer-mxfp4-moe-precision",
        default=DEFAULT_FLASHINFER_MXFP4_PRECISION["r1024"],
        choices=("default", "bf16"),
    )
    parser.add_argument("--original-mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTIONS["original"])
    parser.add_argument("--r512-mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTIONS["r512"])
    parser.add_argument("--r1024-mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTIONS["r1024"])
    parser.add_argument("--original-cpu-offload-gb", type=int, default=DEFAULT_CPU_OFFLOAD_GB["original"])
    parser.add_argument("--r512-cpu-offload-gb", type=int, default=DEFAULT_CPU_OFFLOAD_GB["r512"])
    parser.add_argument("--r1024-cpu-offload-gb", type=int, default=DEFAULT_CPU_OFFLOAD_GB["r1024"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-p", type=float, default=0.01)
    parser.add_argument("--mode", choices=("harmony", "completion"), default="harmony")
    return parser.parse_args()


def _load_reference(csv_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id") or "").strip()
            if qid:
                out[qid] = {
                    "problem": str(row.get("problem") or ""),
                    "answer": str(row.get("answer") or ""),
                }
    return out


def _run_probe(
    *,
    probe_script: Path,
    out_json: Path,
    model_path: str,
    backend: str,
    moe_runner_backend: str | None,
    flashinfer_mxfp4_moe_precision: str | None,
    csv_path: str,
    question_ids: str,
    mode: str,
    port: int,
    max_tokens: int,
    max_total_tokens: int,
    mem_fraction_static: float,
    cpu_offload_gb: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_json.parent / "run.log"
    cmd = [
        sys.executable,
        str(probe_script),
        "--model-path",
        model_path,
        "--tokenizer-path",
        model_path,
        "--csv-path",
        csv_path,
        "--question-ids",
        question_ids,
        "--port",
        str(port),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--attention-backend",
        backend,
        *(
            ["--moe-runner-backend", moe_runner_backend]
            if moe_runner_backend
            else []
        ),
        *(
            ["--flashinfer-mxfp4-moe-precision", flashinfer_mxfp4_moe_precision]
            if flashinfer_mxfp4_moe_precision
            else []
        ),
        "--cpu-offload-gb",
        str(cpu_offload_gb),
        "--mode",
        mode,
        "--max-tokens",
        str(max_tokens),
        "--max-total-tokens",
        str(max_total_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--min-p",
        str(min_p),
        "--out-json",
        str(out_json),
    ]
    print("[probe]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"[probe-log] {log_path}", flush=True)
    with out_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_table(
    *,
    ref: dict[str, dict[str, str]],
    runs: dict[str, dict[str, Any]],
    qids: list[str],
) -> str:
    lines = []
    lines.append("| id | expected | original | r512 | r1024 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for qid in qids:
        expected = ref[qid]["answer"]
        cells = [qid, expected]
        for key in ("original", "r512", "r1024"):
            run = runs[key]
            row = next((r for r in run["results"] if r["id"] == qid), None)
            if row is None:
                cells.append("missing")
            else:
                ans = row.get("answer")
                correct = "✓" if row.get("correct") else "✗"
                wall_s = row.get("wall_s")
                cells.append(f"{ans} {correct} ({wall_s}s)")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Totals")
    lines.append("| run | correct / total |")
    lines.append("| --- | --- |")
    for key in ("original", "r512", "r1024"):
        run = runs[key]
        lines.append(f"| {key} | {run['correct']} / {run['total']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    probe_script = Path(args.probe_script).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    ref = _load_reference(csv_path)
    qids = [q.strip() for q in str(args.question_ids).split(",") if q.strip()]
    if not qids:
        raise ValueError("No question IDs provided")
    for qid in qids:
        if qid not in ref:
            raise KeyError(f"Question id not found in reference CSV: {qid}")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    model_specs = [
        (
            "original",
            args.original_model_path,
            args.original_backend,
            args.original_moe_runner_backend,
            args.original_flashinfer_mxfp4_moe_precision,
            args.original_mem_fraction_static,
            args.original_cpu_offload_gb,
            30100,
        ),
        (
            "r512",
            args.r512_model_path,
            args.r512_backend,
            args.r512_moe_runner_backend,
            args.r512_flashinfer_mxfp4_moe_precision,
            args.r512_mem_fraction_static,
            args.r512_cpu_offload_gb,
            30110,
        ),
        (
            "r1024",
            args.r1024_model_path,
            args.r1024_backend,
            args.r1024_moe_runner_backend,
            args.r1024_flashinfer_mxfp4_moe_precision,
            args.r1024_mem_fraction_static,
            args.r1024_cpu_offload_gb,
            30120,
        ),
    ]

    runs: dict[str, dict[str, Any]] = {}
    for name, model_path, backend, moe_runner_backend, flashinfer_mxfp4_moe_precision, mem_fraction_static, cpu_offload_gb, port in model_specs:
        _validate_backend_policy(name, backend)
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_combined = run_dir / "combined.json"
        if existing_combined.exists():
            print(f"[skip] {name} -> {existing_combined}", flush=True)
            with existing_combined.open("r", encoding="utf-8") as f:
                runs[name] = json.load(f)
            continue
        print(f"[run] {name} -> {run_dir}", flush=True)
        runs[name] = _run_probe(
            probe_script=probe_script,
            out_json=run_dir / "combined.json",
            model_path=model_path,
            backend=backend,
            moe_runner_backend=moe_runner_backend,
            flashinfer_mxfp4_moe_precision=flashinfer_mxfp4_moe_precision,
            csv_path=str(csv_path),
            question_ids=",".join(qids),
            mode=args.mode,
            port=port,
            max_tokens=args.max_tokens,
            max_total_tokens=args.max_total_tokens,
            mem_fraction_static=mem_fraction_static,
            cpu_offload_gb=cpu_offload_gb,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
        )
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model_path": model_path,
                    "attention_backend": backend,
                    "moe_runner_backend": moe_runner_backend,
                    "flashinfer_mxfp4_moe_precision": flashinfer_mxfp4_moe_precision,
                    "mem_fraction_static": mem_fraction_static,
                    "cpu_offload_gb": cpu_offload_gb,
                    "mode": args.mode,
                    "port": port,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    matrix_md = _render_table(ref=ref, runs=runs, qids=qids)
    (out_root / "matrix.md").write_text(matrix_md, encoding="utf-8")
    (out_root / "summary.json").write_text(
        json.dumps({"runs": runs, "question_ids": qids}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(matrix_md, flush=True)
    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
