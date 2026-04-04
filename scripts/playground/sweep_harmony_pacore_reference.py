#!/usr/bin/env python3
"""Run showtime-style Harmony tool-calling and PaCoRe on the local reference set.

This script uses SGLang's native /v1/responses path with builtin Harmony tool execution.
The server runs the whole Harmony/tool loop internally; the client only orchestrates
attempt-level majority vote and optional PaCoRe rounds.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from sglang.test.test_utils import kill_process_tree

from bench_reference_dflash import _launch_server

SHOWTIME_SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver with expertise at the "
    "International Mathematical Olympiad (IMO) level. Your goal is to find "
    "the correct answer through rigorous mathematical reasoning.\n\n"
    "The final answer must be a non-negative integer between 0 and 99999.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}.\n\n"
    "Think step-by-step and show your complete reasoning process. Quality of "
    "reasoning is as important as the final answer."
)

SHOWTIME_TOOL_PROMPT = (
    "Use this tool to execute Python code for:\n"
    "- Complex calculations that would be error-prone by hand\n"
    "- Numerical verification of analytical results\n"
    "- Generating examples or testing conjectures\n"
    "- Visualizing problem structure when helpful\n"
    "- Brute-force verification for small cases\n\n"
    "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
    "Always use print() to display results. Write clear, well-commented code.\n\n"
    "Remember: Code should support your mathematical reasoning, not replace it. "
    "Explain what you're computing and why before running code."
)

SHOWTIME_PREFERENCE_PROMPT = (
    "You have access to math, numpy, and sympy. Use them when they reduce algebraic "
    "or arithmetic error. Always end with a single boxed non-negative integer answer."
)

BOXED_RE = re.compile(r"\\boxed\s*\{\s*([0-9,]+)\s*\}")
FINAL_INT_RE = re.compile(r"final\s+answer\s+is\s*([0-9,]+)", re.IGNORECASE)


def _scan_for_answer(text: str) -> int | None:
    if not text:
        return None
    boxed = BOXED_RE.findall(text)
    if boxed:
        with contextlib.suppress(Exception):
            return int(boxed[-1].replace(",", ""))
    final = FINAL_INT_RE.findall(text)
    if final:
        with contextlib.suppress(Exception):
            return int(final[-1].replace(",", ""))
    return None


def _parse_widths(raw: str) -> list[int]:
    vals: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        with contextlib.suppress(Exception):
            val = int(part)
            if val > 0:
                vals.append(val)
    return vals


def _extract_reference_text(transcript: str, *, max_chars: int) -> str:
    raw = str(transcript or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    parts = raw.split("</think>")
    ref = parts[-1].strip() if len(parts) > 1 else raw
    ref = ref or raw
    if int(max_chars) > 0 and len(ref) > int(max_chars):
        ref = ref[-int(max_chars) :]
    return ref.strip()


def _build_pacore_prompt(
    *, problem: str, reference_responses: list[str], preference_prompt: str
) -> str:
    refs = [str(r or "").strip() for r in reference_responses if str(r or "").strip()]
    original_problem = f"{problem.strip()} {preference_prompt.strip()}".strip()
    if not refs:
        return original_problem
    refs_blob = "\n".join(
        f"Reference {idx}: {text}" for idx, text in enumerate(refs, start=1)
    )
    return (
        "You are given a problem and a list of reference responses. Your job is to analyze "
        "these references and provide your own response.\n\n"
        f"Original Problem:\n{original_problem}\n\n"
        f"Reference Responses:\n{refs_blob}\n\n"
        "Now, based on the original problem and reference responses above, provide your own "
        "comprehensive solution. Critically evaluate the references: they may be incomplete, "
        "conflicting, or wrong. Do not merely copy or majority-vote; synthesize a better solution."
    )


def _rank_refs(
    attempts: list[dict[str, Any]], *, max_refs: int, max_ref_chars: int, max_dup_per_answer: int
) -> list[str]:
    answers = [a.get("answer") for a in attempts if isinstance(a.get("answer"), int)]
    support = Counter(int(x) for x in answers)
    items: list[tuple[int, int, int, int, float, dict[str, Any]]] = []
    for attempt in attempts:
        answer = attempt.get("answer") if isinstance(attempt.get("answer"), int) else None
        has_answer = int(answer is not None)
        sup = int(support.get(int(answer), 0) if answer is not None else 0)
        python_calls = int(attempt.get("python_calls") or 0)
        token_count = int(attempt.get("completion_tokens") or 0)
        wall_s = float(attempt.get("wall_s") or 0.0)
        items.append((has_answer, sup, int(python_calls > 0), token_count, -wall_s, attempt))
    items.sort(reverse=True, key=lambda item: item[:5])

    refs: list[str] = []
    seen_norm: set[str] = set()
    per_answer: dict[int, int] = {}
    for _has_answer, _sup, _tool_bonus, _tok, _neg_wall, attempt in items:
        if len(refs) >= int(max_refs):
            break
        text = _extract_reference_text(
            str(attempt.get("transcript") or ""),
            max_chars=int(max_ref_chars),
        )
        if not text:
            continue
        norm = " ".join(text.split())
        if norm in seen_norm:
            continue
        answer = _scan_for_answer(text)
        if answer is not None:
            if int(per_answer.get(answer, 0)) >= int(max_dup_per_answer):
                continue
            per_answer[answer] = int(per_answer.get(answer, 0)) + 1
        refs.append(text)
        seen_norm.add(norm)
    return refs


def _majority_answer(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    hist = Counter(
        int(a["answer"]) for a in attempts if isinstance(a.get("answer"), int)
    )
    if not hist:
        return {
            "majority_answer": None,
            "majority_support": 0,
            "answer_hist": {},
        }
    answer, support = hist.most_common(1)[0]
    return {
        "majority_answer": int(answer),
        "majority_support": int(support),
        "answer_hist": {str(k): int(v) for k, v in hist.most_common()},
    }


def _parse_response_payload(payload: dict[str, Any]) -> tuple[str, int, int]:
    transcript_parts: list[str] = []
    python_calls = 0
    python_errors = 0
    for item in payload.get("output", []):
        item_type = item.get("type")
        if item_type == "reasoning":
            for content in item.get("content", []):
                text = str(content.get("text") or "")
                if text:
                    transcript_parts.append(text)
        elif item_type == "message":
            for content in item.get("content", []):
                text = str(content.get("text") or "")
                if text:
                    transcript_parts.append(text)
        elif item_type == "function_call":
            python_calls += 1
            name = str(item.get("name") or "")
            args = str(item.get("arguments") or "")
            transcript_parts.append(f"[tool_call:{name}] {args}")
        elif item_type == "code_interpreter_call":
            python_calls += 1
    transcript = "\n".join(part for part in transcript_parts if part).strip()
    if "[ERROR]" in transcript:
        python_errors += 1
    return transcript, python_calls, python_errors


def _call_responses(
    *,
    base_url: str,
    model: str,
    input_payload: Any,
    instructions: str,
    max_output_tokens: int,
    reasoning_effort: str,
    timeout_s: float,
    use_python_tool: bool,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": str(model),
        "input": input_payload,
        "instructions": str(instructions),
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": str(reasoning_effort)},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "tool_choice": "auto",
        "stream": False,
    }
    if use_python_tool:
        payload["tools"] = [{"type": "code_interpreter"}]
    if previous_response_id:
        payload["previous_response_id"] = str(previous_response_id)

    t0 = time.time()
    response = requests.post(
        f"{base_url}/v1/responses",
        json=payload,
        timeout=float(timeout_s),
    )
    if not response.ok:
        body = ""
        with contextlib.suppress(Exception):
            body = str(response.text or "")
        return {
            "response": None,
            "transcript": "",
            "answer": None,
            "python_calls": 0,
            "python_errors": 0,
            "wall_s": float(time.time() - t0),
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "reasoning_tokens": 0,
            "error": f"http_{response.status_code}: {body[:4000]}",
        }
    result = response.json()
    transcript, python_calls, python_errors = _parse_response_payload(result)
    usage = result.get("usage") or {}
    answer = _scan_for_answer(transcript)
    return {
        "response": result,
        "response_id": result.get("id"),
        "transcript": transcript,
        "answer": answer,
        "python_calls": int(python_calls),
        "python_errors": int(python_errors),
        "wall_s": float(time.time() - t0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "error": "",
    }


def _run_harmony_attempt(
    *,
    base_url: str,
    model: str,
    user_prompt: str,
    instructions: str,
    max_turn_output_tokens: int,
    turns: int,
    reasoning_effort: str,
    timeout_s: float,
    use_python_tool: bool,
) -> dict[str, Any]:
    transcript_parts: list[str] = []
    response_ids: list[str] = []
    total_python_calls = 0
    total_python_errors = 0
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_reasoning_tokens = 0
    total_wall_s = 0.0
    final_answer: int | None = None
    last_error = ""
    last_response = None
    prev_response_id: str | None = None

    for turn_idx in range(int(turns)):
        result = _call_responses(
            base_url=base_url,
            model=model,
            input_payload=(str(user_prompt) if turn_idx == 0 else []),
            instructions=instructions,
            max_output_tokens=int(max_turn_output_tokens),
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            use_python_tool=use_python_tool,
            previous_response_id=prev_response_id,
        )
        total_wall_s += float(result.get("wall_s") or 0.0)
        total_python_calls += int(result.get("python_calls") or 0)
        total_python_errors += int(result.get("python_errors") or 0)
        total_completion_tokens += int(result.get("completion_tokens") or 0)
        total_prompt_tokens += int(result.get("prompt_tokens") or 0)
        total_reasoning_tokens += int(result.get("reasoning_tokens") or 0)
        if result.get("transcript"):
            transcript_parts.append(str(result["transcript"]))
        if result.get("response_id"):
            response_ids.append(str(result["response_id"]))
            prev_response_id = str(result["response_id"])
        if result.get("error"):
            last_error = str(result["error"])
            last_response = result.get("response")
            break
        final_answer = _scan_for_answer("\n".join(transcript_parts))
        last_response = result.get("response")
        if final_answer is not None:
            break
        if not result.get("response") or not (result["response"].get("output") or []):
            break

    return {
        "response": last_response,
        "response_ids": response_ids,
        "transcript": "\n".join(part for part in transcript_parts if part).strip(),
        "answer": final_answer,
        "python_calls": int(total_python_calls),
        "python_errors": int(total_python_errors),
        "wall_s": float(total_wall_s),
        "completion_tokens": int(total_completion_tokens),
        "prompt_tokens": int(total_prompt_tokens),
        "reasoning_tokens": int(total_reasoning_tokens),
        "error": last_error,
    }


def _load_reference_rows(csv_path: Path, question_ids: list[str]) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id") or "").strip()
            if qid:
                rows[qid] = {
                    "id": qid,
                    "problem": str(row.get("problem") or "").strip(),
                    "answer": str(row.get("answer") or "").strip(),
                }
    selected: list[dict[str, str]] = []
    for qid in question_ids:
        if qid not in rows:
            raise KeyError(f"Question {qid!r} not found in {csv_path}")
        selected.append(rows[qid])
    return selected


def _solve_one_problem(
    *,
    base_url: str,
    model_path: str,
    problem_row: dict[str, str],
    attempts: int,
    early_stop: int,
    pacore_widths: list[int],
    pacore_max_refs: int,
    pacore_max_ref_chars: int,
    pacore_max_dup_per_answer: int,
    reasoning_effort: str,
    max_turn_output_tokens: int,
    turns: int,
    timeout_s: float,
    use_python_tool: bool,
) -> dict[str, Any]:
    round_widths = [*pacore_widths, 1] if pacore_widths else [int(attempts)]
    round_input_refs: list[str] = []
    rounds: list[dict[str, Any]] = []
    final_attempts: list[dict[str, Any]] = []

    for round_idx, width in enumerate(round_widths, start=1):
        is_final = bool(pacore_widths) and round_idx == len(round_widths)
        stage = "final_synthesis" if is_final else f"round_{round_idx}"
        user_prompt = _build_pacore_prompt(
            problem=problem_row["problem"],
            reference_responses=round_input_refs,
            preference_prompt=SHOWTIME_PREFERENCE_PROMPT,
        )
        instructions = SHOWTIME_SYSTEM_PROMPT + "\n\n" + SHOWTIME_TOOL_PROMPT

        attempts_out: list[dict[str, Any]] = []
        vote_hist: Counter[int] = Counter()
        with ThreadPoolExecutor(max_workers=int(width)) as executor:
            futures = [
                executor.submit(
                    _run_harmony_attempt,
                    base_url=base_url,
                    model=model_path,
                    user_prompt=user_prompt,
                    instructions=instructions,
                    max_turn_output_tokens=max_turn_output_tokens,
                    turns=turns,
                    reasoning_effort=reasoning_effort,
                    timeout_s=timeout_s,
                    use_python_tool=use_python_tool,
                )
                for _ in range(int(width))
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "response": None,
                        "transcript": "",
                        "answer": None,
                        "python_calls": 0,
                        "python_errors": 0,
                        "wall_s": 0.0,
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "reasoning_tokens": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                result["is_correct"] = (
                    str(result.get("answer")) == str(problem_row["answer"])
                )
                attempts_out.append(result)
                if isinstance(result.get("answer"), int):
                    vote_hist[int(result["answer"])] += 1
                if vote_hist and vote_hist.most_common(1)[0][1] >= int(early_stop):
                    break

        majority = _majority_answer(attempts_out)
        round_record = {
            "round": int(round_idx),
            "stage": str(stage),
            "width": int(width),
            "input_ref_count": int(len(round_input_refs)),
            "majority": majority,
            "attempts": attempts_out,
        }
        rounds.append(round_record)

        if pacore_widths and not is_final:
            round_input_refs = _rank_refs(
                attempts_out,
                max_refs=int(pacore_max_refs),
                max_ref_chars=int(pacore_max_ref_chars),
                max_dup_per_answer=int(pacore_max_dup_per_answer),
            )
        final_attempts = attempts_out

    final_majority = _majority_answer(final_attempts)
    final_answer = final_majority["majority_answer"]
    return {
        "question_id": problem_row["id"],
        "expected_answer": int(problem_row["answer"]),
        "final_answer": final_answer,
        "is_correct": int(final_answer) == int(problem_row["answer"])
        if final_answer is not None
        else False,
        "final_majority": final_majority,
        "rounds": rounds,
    }


def _parse_question_ids(raw: str, csv_path: Path) -> list[str]:
    if str(raw or "").strip():
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [
            str(row.get("id") or "").strip()
            for row in csv.DictReader(f)
            if str(row.get("id") or "").strip()
        ]


def _write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    if rows:
        fieldnames = [
            "question_id",
            "expected_answer",
            "final_answer",
            "is_correct",
            "majority_support",
        ]
        with (out_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "expected_answer": row["expected_answer"],
                        "final_answer": row["final_answer"],
                        "is_correct": row["is_correct"],
                        "majority_support": row["final_majority"]["majority_support"],
                    }
                )
        lines = [
            "# Harmony PaCoRe Sweep",
            "",
            "| qid | expected | final | support | correct |",
            "|---|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['question_id']} | {row['expected_answer']} | {row['final_answer']} | "
                f"{row['final_majority']['majority_support']} | {row['is_correct']} |"
            )
        lines.append("")
        lines.append(
            f"Accuracy: {sum(1 for row in rows if row['is_correct'])}/{len(rows)}"
        )
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run native Harmony/tool Responses + PaCoRe on the reference set."
    )
    p.add_argument("--model-path", default="/workspace/offload_root/gpt-oss-120b")
    p.add_argument("--draft-model-path", default="/root/epoch_65_step_23760")
    p.add_argument("--reference-csv", default="/root/reference.csv")
    p.add_argument("--question-ids", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--port", type=int, default=23201)
    p.add_argument("--context-length", type=int, default=65536)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--cuda-graph-max-bs", type=int, default=8)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--max-turn-output-tokens", type=int, default=96)
    p.add_argument("--turns", type=int, default=128)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--early-stop", type=int, default=4)
    p.add_argument("--pacore-widths", default="")
    p.add_argument("--pacore-max-refs", type=int, default=32)
    p.add_argument("--pacore-max-ref-chars", type=int, default=2000)
    p.add_argument("--pacore-max-dup-per-answer", type=int, default=1)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--disable-dflash", action="store_true")
    p.add_argument("--disable-python-tool", action="store_true")
    p.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("SGLANG_SERVER_PYTHON_EXECUTABLE", sys.executable)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    question_ids = _parse_question_ids(args.question_ids, Path(args.reference_csv))
    problems = _load_reference_rows(Path(args.reference_csv), question_ids)

    server_proc = _launch_server(
        model_path=str(args.model_path),
        port=int(args.port),
        attention_backend="fa3",
        moe_runner_backend="triton_kernel",
        kv_cache_dtype="fp8_e4m3",
        context_length=int(args.context_length),
        cuda_graph_max_bs=int(args.cuda_graph_max_bs),
        max_running_requests=int(args.max_running_requests),
        page_size=1,
        enable_piecewise_cuda_graph=not bool(args.disable_piecewise_cuda_graph),
        piecewise_cuda_graph_max_tokens=(
            None if args.disable_piecewise_cuda_graph else 8192
        ),
        disable_cuda_graph=False,
        speculative=not bool(args.disable_dflash),
        draft_model_path=None if args.disable_dflash else str(args.draft_model_path),
        draft_attention_backend="fa3",
        draft_kv_cache_dtype="bfloat16",
        draft_page_size=1,
        speculative_moe_runner_backend="triton_kernel",
        speculative_dflash_block_size=16,
        mem_fraction_static=float(args.mem_fraction_static),
        tool_server="demo",
    )
    base_url = f"http://127.0.0.1:{int(args.port)}"

    try:
        rows: list[dict[str, Any]] = []
        for problem in problems:
            out_json = out_dir / f"{problem['id']}.json"
            result = _solve_one_problem(
                base_url=base_url,
                model_path=str(args.model_path),
                problem_row=problem,
                attempts=int(args.attempts),
                early_stop=int(args.early_stop),
                pacore_widths=_parse_widths(args.pacore_widths),
                pacore_max_refs=int(args.pacore_max_refs),
                pacore_max_ref_chars=int(args.pacore_max_ref_chars),
                pacore_max_dup_per_answer=int(args.pacore_max_dup_per_answer),
                reasoning_effort=str(args.reasoning_effort),
                max_turn_output_tokens=int(args.max_turn_output_tokens),
                turns=int(args.turns),
                timeout_s=float(args.timeout_s),
                use_python_tool=not bool(args.disable_python_tool),
            )
            out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            rows.append(result)
            _write_summary(out_dir, rows)
            print(json.dumps({"question_id": problem["id"], "is_correct": result["is_correct"]}))
        return 0
    finally:
        with contextlib.suppress(Exception):
            kill_process_tree(server_proc.pid)


if __name__ == "__main__":
    raise SystemExit(main())
