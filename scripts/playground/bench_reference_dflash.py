#!/usr/bin/env python3
"""Benchmark GPT-OSS DFLASH on the local reference problems."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import os
import re
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    _launch_server_process,
    _wait_for_server_health,
    kill_process_tree,
)


REFERENCE_IDS = ("92ba6a", "9c1c5f", "a295e9")

SHOWTIME_SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver with expertise at the "
    "International Mathematical Olympiad (IMO) level. Your goal is to find "
    "the correct answer through rigorous mathematical reasoning.\n\n"
    "The final answer must be a non-negative integer between 0 and 99999.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}.\n\n"
    "Think step-by-step and show your complete reasoning process. Quality of "
    "reasoning is as important as the final answer."
)


@dataclass(frozen=True)
class BenchResult:
    completed: int
    wall_s: float
    req_s: float
    wall_tok_s: float
    accept_length: float
    accept_length_min_step: Optional[int]
    accept_length_max_step: Optional[int]
    accept_draft_tokens_min: Optional[int]
    accept_draft_tokens_max: Optional[int]
    verify_ct_sum: Optional[int]
    verify_ct_avg: Optional[float]
    verify_ct_min: Optional[int]
    verify_ct_max: Optional[int]
    accept_token_sum: Optional[int]
    step_time_p20_s: float | None
    output_tok_s_p20: float | None
    avg_output_tokens: float


@dataclass(frozen=True)
class BenchRun:
    summary: BenchResult
    request_metrics: list[dict[str, Any]]
    request_metric_aggregate: dict[str, Any]


def _bench_result_from_payload(payload: dict[str, Any]) -> BenchResult:
    defaults = {
        "accept_length_min_step": None,
        "accept_length_max_step": None,
        "accept_draft_tokens_min": None,
        "accept_draft_tokens_max": None,
        "verify_ct_sum": None,
        "verify_ct_avg": None,
        "verify_ct_min": None,
        "verify_ct_max": None,
        "accept_token_sum": None,
    }
    return BenchResult(**(defaults | payload))


def _build_prompt(problem: str) -> str:
    return (
        f"{SHOWTIME_SYSTEM_PROMPT}\n\n"
        "Solve the following problem. Give a concise but complete derivation, "
        "then end with the final integer answer in \\boxed{}.\n\n"
        f"Problem:\n{problem.strip()}\n"
    )


def _load_reference_prompts(
    csv_path: Path, question_ids: tuple[str, ...], num_prompts: int
) -> tuple[list[str], list[str], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id") or "").strip()
            if qid:
                rows[qid] = {
                    "problem": str(row.get("problem") or ""),
                    "answer": str(row.get("answer") or ""),
                }

    problems: list[str] = []
    answers: list[str] = []
    for qid in question_ids:
        row = rows.get(qid)
        if not row:
            raise KeyError(f"Reference problem {qid!r} not found in {csv_path}")
        problems.append(_build_prompt(row["problem"]))
        answers.append(row["answer"].strip())

    if not problems:
        raise RuntimeError("No reference prompts loaded.")

    repeated = (problems * ((int(num_prompts) + len(problems) - 1) // len(problems)))[
        : int(num_prompts)
    ]
    repeated_qids = (
        list(question_ids) * ((int(num_prompts) + len(question_ids) - 1) // len(question_ids))
    )[: int(num_prompts)]
    repeated_answers = (
        list(answers) * ((int(num_prompts) + len(answers) - 1) // len(answers))
    )[: int(num_prompts)]
    return repeated, repeated_qids, repeated_answers


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_INT_RE = re.compile(r"(?<!\d)(\d{1,5})(?!\d)")


def _extract_boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    if not matches:
        return None
    candidate = matches[-1].strip()
    int_matches = _INT_RE.findall(candidate)
    if int_matches:
        return int_matches[-1]
    return None


def _extract_fallback_answer(text: str) -> str | None:
    matches = _INT_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1]


def _build_sampling_params(
    *, temperature: float, top_p: float, top_k: int, min_p: float
) -> dict[str, Any]:
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "ignore_eos": True,
    }


def _compute_output_lens(
    *,
    prompts: list[str],
    tokenizer,
    context_length: int,
    decode_len: int,
    decode_to_context_limit: bool,
    buffer_tokens: int,
) -> list[int]:
    if not decode_to_context_limit:
        return [int(decode_len)] * len(prompts)

    output_lens: list[int] = []
    for prompt in prompts:
        prompt_len = int(len(tokenizer.encode(prompt, add_special_tokens=False)))
        remaining = int(context_length) - int(prompt_len) - int(buffer_tokens)
        if remaining <= 0:
            raise ValueError(
                f"Prompt length {prompt_len} leaves no decode budget under "
                f"context_length={context_length} buffer_tokens={buffer_tokens}"
            )
        output_lens.append(int(remaining))
    return output_lens


def _effective_draft_share_pools(page_size: int, draft_page_size: int | None) -> bool:
    draft_ps = int(page_size if draft_page_size is None else draft_page_size)
    if draft_ps != int(page_size):
        return False
    env_val = os.environ.get("SGLANG_DFLASH_DRAFT_SHARE_POOLS")
    if env_val is None:
        return True
    return str(env_val).strip().lower() not in {"0", "false", "no", "off"}


def _launch_server(
    *,
    model_path: str,
    port: int,
    attention_backend: str,
    moe_runner_backend: str,
    kv_cache_dtype: str,
    context_length: int,
    cuda_graph_max_bs: int,
    max_running_requests: int,
    page_size: int,
    enable_piecewise_cuda_graph: bool,
    piecewise_cuda_graph_max_tokens: int | None,
    disable_cuda_graph: bool,
    speculative: bool,
    speculative_algorithm: str,
    draft_model_path: str | None,
    draft_attention_backend: str | None,
    draft_kv_cache_dtype: str | None,
    draft_page_size: int | None,
    speculative_moe_runner_backend: str | None,
    speculative_dflash_block_size: int | None,
    speculative_num_steps: int | None,
    speculative_eagle_topk: int | None,
    speculative_num_draft_tokens: int | None,
    mem_fraction_static: float | None,
    speculative_draft_mem_fraction_static: float | None,
    disable_overlap_schedule: bool,
    tool_server: str | None = None,
) -> object:
    base_url = f"http://127.0.0.1:{int(port)}"
    server_python = (
        os.environ.get("SGLANG_SERVER_PYTHON_EXECUTABLE") or sys.executable or "python3"
    )
    cmd = [
        str(server_python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--tensor-parallel-size",
        "1",
        "--attention-backend",
        str(attention_backend),
        "--moe-runner-backend",
        str(moe_runner_backend),
        "--kv-cache-dtype",
        str(kv_cache_dtype),
        "--context-length",
        str(int(context_length)),
        "--cuda-graph-max-bs",
        str(int(cuda_graph_max_bs)),
        "--max-running-requests",
        str(int(max_running_requests)),
        "--page-size",
        str(int(page_size)),
        "--trust-remote-code",
        "--device",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
    ]
    if mem_fraction_static is not None:
        cmd += ["--mem-fraction-static", str(float(mem_fraction_static))]
    if speculative_draft_mem_fraction_static is not None:
        cmd += [
            "--speculative-draft-mem-fraction-static",
            str(float(speculative_draft_mem_fraction_static)),
        ]
    if tool_server:
        cmd += ["--tool-server", str(tool_server)]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if enable_piecewise_cuda_graph:
        cmd.append("--enable-piecewise-cuda-graph")
    if piecewise_cuda_graph_max_tokens is not None:
        cmd += [
            "--piecewise-cuda-graph-max-tokens",
            str(int(piecewise_cuda_graph_max_tokens)),
        ]
    if speculative:
        if not draft_model_path:
            raise ValueError("draft_model_path is required for speculative launch")
        cmd += [
            "--speculative-algorithm",
            str(speculative_algorithm),
            "--speculative-draft-model-path",
            str(draft_model_path),
        ]
        if draft_attention_backend:
            cmd += [
                "--speculative-draft-attention-backend",
                str(draft_attention_backend),
            ]
        if draft_kv_cache_dtype:
            cmd += ["--speculative-draft-kv-cache-dtype", str(draft_kv_cache_dtype)]
        if draft_page_size is not None:
            cmd += ["--speculative-draft-page-size", str(int(draft_page_size))]
        if speculative_moe_runner_backend:
            cmd += [
                "--speculative-moe-runner-backend",
                str(speculative_moe_runner_backend),
            ]
        if speculative_dflash_block_size is not None:
            cmd += [
                "--speculative-dflash-block-size",
                str(int(speculative_dflash_block_size)),
            ]
        if speculative_num_steps is not None:
            cmd += [
                "--speculative-num-steps",
                str(int(speculative_num_steps)),
            ]
        if speculative_eagle_topk is not None:
            cmd += [
                "--speculative-eagle-topk",
                str(int(speculative_eagle_topk)),
            ]
        if speculative_num_draft_tokens is not None:
            cmd += [
                "--speculative-num-draft-tokens",
                str(int(speculative_num_draft_tokens)),
            ]

    source_root = os.environ.get("SGLANG_SOURCE_ROOT")
    repo_python = str(
        Path(source_root).resolve() / "python"
        if source_root
        else (Path(__file__).resolve().parents[2] / "python")
    )
    env = {"SGLANG_RECORD_STEP_TIME": "1", **os.environ}
    env["PYTHONPATH"] = (
        repo_python
        if not env.get("PYTHONPATH")
        else f"{repo_python}:{env['PYTHONPATH']}"
    )
    server_launch_timeout = int(
        env.get(
            "SGLANG_BENCH_SERVER_LAUNCH_TIMEOUT",
            str(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
        )
    )
    print(f"command={shlex.join(cmd)}")
    proc = _launch_server_process(cmd, env, None, model_path)
    success, error_msg = _wait_for_server_health(
        proc, base_url, None, server_launch_timeout
    )
    if success:
        return proc
    try:
        kill_process_tree(proc.pid)
    except Exception:
        pass
    raise RuntimeError(error_msg or "server failed to start")


def _summarize_spec_meta(meta_infos: list[dict[str, Any]]) -> dict[str, Any]:
    verify_counts: list[int] = []
    accept_token_counts: list[int] = []
    aggregate_hist: list[int] = []

    for meta in meta_infos:
        if not isinstance(meta, dict) or not meta:
            continue

        if "spec_verify_ct" in meta:
            try:
                verify_counts.append(int(meta["spec_verify_ct"]))
            except Exception:
                pass

        if "spec_accept_token_num" in meta:
            try:
                accept_token_counts.append(int(meta["spec_accept_token_num"]))
            except Exception:
                pass

        hist = meta.get("spec_accept_histogram")
        if isinstance(hist, list) and hist:
            for idx, count in enumerate(hist):
                count_i = int(count)
                if len(aggregate_hist) <= idx:
                    aggregate_hist.extend([0] * (idx - len(aggregate_hist) + 1))
                aggregate_hist[idx] += count_i

    accept_draft_tokens_min: Optional[int] = None
    accept_draft_tokens_max: Optional[int] = None
    if aggregate_hist:
        nonzero = [idx for idx, count in enumerate(aggregate_hist) if int(count) > 0]
        if nonzero:
            accept_draft_tokens_min = int(nonzero[0])
            accept_draft_tokens_max = int(nonzero[-1])

    return {
        "accept_draft_tokens_min": accept_draft_tokens_min,
        "accept_draft_tokens_max": accept_draft_tokens_max,
        "accept_length_min_step": (
            None
            if accept_draft_tokens_min is None
            else int(accept_draft_tokens_min + 1)
        ),
        "accept_length_max_step": (
            None
            if accept_draft_tokens_max is None
            else int(accept_draft_tokens_max + 1)
        ),
        "verify_ct_sum": (sum(verify_counts) if verify_counts else None),
        "verify_ct_avg": (
            float(sum(verify_counts)) / float(len(verify_counts))
            if verify_counts
            else None
        ),
        "verify_ct_min": (min(verify_counts) if verify_counts else None),
        "verify_ct_max": (max(verify_counts) if verify_counts else None),
        "accept_token_sum": (sum(accept_token_counts) if accept_token_counts else None),
    }


def _extract_request_metrics(
    *,
    meta_infos: list[dict[str, Any]],
    request_question_ids: list[str],
    output_lens: list[int],
    generated_texts: list[str],
    expected_answers: list[str],
) -> list[dict[str, Any]]:
    metric_keys = (
        "spec_accept_rate",
        "spec_accept_length",
        "spec_accept_token_num",
        "spec_draft_token_num",
        "spec_verify_ct",
        "spec_accept_length_step_min",
        "spec_accept_length_step_max",
        "spec_dflash_verify_mode_last",
        "spec_dflash_debug_stat_ct",
        "spec_dflash_max_steps_last",
        "spec_dflash_max_steps_min",
        "spec_dflash_max_steps_max",
        "spec_dflash_max_steps_mean",
        "spec_dflash_effective_draft_token_num_last",
        "spec_dflash_effective_draft_token_num_min",
        "spec_dflash_effective_draft_token_num_max",
        "spec_dflash_effective_draft_token_num_mean",
        "spec_dflash_effective_step_count_last",
        "spec_dflash_effective_step_count_min",
        "spec_dflash_effective_step_count_max",
        "spec_dflash_effective_step_count_mean",
        "spec_dflash_total_draft_token_num",
        "spec_dflash_accept_ratio_mean",
        "spec_dflash_tv_mean",
        "spec_dflash_p_entropy_mean",
        "spec_dflash_q_entropy_mean",
        "spec_dflash_p_max_mean",
        "spec_dflash_q_max_mean",
        "spec_dflash_q_max_mean_first",
        "spec_dflash_q_max_min_first",
        "spec_dflash_q_ent_mean_first",
        "spec_dflash_adaptive_temp_mul",
        "spec_dflash_pq_disabled_rounds_left",
    )
    rows: list[dict[str, Any]] = []
    row_count = max(
        len(meta_infos),
        len(request_question_ids),
        len(output_lens),
        len(generated_texts),
        len(expected_answers),
    )
    for i in range(row_count):
        meta = meta_infos[i] if i < len(meta_infos) else {}
        row = {
            "request_index": int(i),
            "question_id": (
                request_question_ids[i] if i < len(request_question_ids) else None
            ),
            "requested_output_len": (
                int(output_lens[i]) if i < len(output_lens) else None
            ),
        }
        generated_text = generated_texts[i] if i < len(generated_texts) else ""
        expected_answer = expected_answers[i] if i < len(expected_answers) else None
        boxed_answer = _extract_boxed_answer(generated_text)
        fallback_answer = _extract_fallback_answer(generated_text)
        row["expected_answer"] = expected_answer
        row["boxed_answer"] = boxed_answer
        row["fallback_answer"] = fallback_answer
        row["is_correct_boxed"] = (
            bool(boxed_answer == expected_answer)
            if expected_answer is not None and boxed_answer is not None
            else False
        )
        row["is_correct_fallback"] = (
            bool(fallback_answer == expected_answer)
            if expected_answer is not None and fallback_answer is not None
            else False
        )
        if isinstance(meta, dict):
            for key in metric_keys:
                if key in meta:
                    row[key] = meta.get(key)
            if "completion_tokens" in meta:
                row["completion_tokens"] = meta.get("completion_tokens")
            if "cached_tokens" in meta:
                row["cached_tokens"] = meta.get("cached_tokens")
        rows.append(row)
    return rows


def _aggregate_request_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "spec_accept_rate",
        "spec_accept_length",
        "spec_accept_token_num",
        "spec_draft_token_num",
        "spec_verify_ct",
        "spec_accept_length_step_min",
        "spec_accept_length_step_max",
        "spec_dflash_debug_stat_ct",
        "spec_dflash_max_steps_last",
        "spec_dflash_max_steps_min",
        "spec_dflash_max_steps_max",
        "spec_dflash_max_steps_mean",
        "spec_dflash_effective_draft_token_num_last",
        "spec_dflash_effective_draft_token_num_min",
        "spec_dflash_effective_draft_token_num_max",
        "spec_dflash_effective_draft_token_num_mean",
        "spec_dflash_effective_step_count_last",
        "spec_dflash_effective_step_count_min",
        "spec_dflash_effective_step_count_max",
        "spec_dflash_effective_step_count_mean",
        "spec_dflash_total_draft_token_num",
        "spec_dflash_accept_ratio_mean",
        "spec_dflash_tv_mean",
        "spec_dflash_p_entropy_mean",
        "spec_dflash_q_entropy_mean",
        "spec_dflash_p_max_mean",
        "spec_dflash_q_max_mean",
        "spec_dflash_q_max_mean_first",
        "spec_dflash_q_max_min_first",
        "spec_dflash_q_ent_mean_first",
        "spec_dflash_adaptive_temp_mul",
        "spec_dflash_pq_disabled_rounds_left",
    )
    out: dict[str, Any] = {}
    for key in numeric_keys:
        vals = []
        for row in rows:
            val = row.get(key)
            if val is None:
                continue
            try:
                vals.append(float(val))
            except Exception:
                continue
        if not vals:
            continue
        out[f"{key}_mean"] = round(float(sum(vals)) / float(len(vals)), 6)
        out[f"{key}_min"] = round(float(min(vals)), 6)
        out[f"{key}_max"] = round(float(max(vals)), 6)
    if rows:
        boxed_correct = [1.0 for row in rows if row.get("is_correct_boxed")]
        fallback_correct = [1.0 for row in rows if row.get("is_correct_fallback")]
        out["correct_boxed_rate"] = round(len(boxed_correct) / len(rows), 6)
        out["correct_fallback_rate"] = round(len(fallback_correct) / len(rows), 6)
    return out


def _bench_one(
    *,
    base_url: str,
    tokenizer,
    prompts: list[str],
    prompt_question_ids: list[str],
    prompt_expected_answers: list[str],
    concurrency: int,
    output_lens: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    disable_stream: bool = False,
) -> BenchRun:
    if len(prompts) != len(output_lens):
        raise ValueError("prompts and output_lens must have the same length")
    reqs = [
        DatasetRow(prompt=p, prompt_len=0, output_len=int(out))
        for p, out in zip(prompts, output_lens)
    ]

    args = argparse.Namespace(
        disable_ignore_eos=False,
        disable_stream=bool(disable_stream),
        return_logprob=False,
        return_routed_experts=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        plot_throughput=False,
        backend="sglang",
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=True,
    )
    set_global_args(args)

    extra_request_body = {
        "sampling_params": _build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
    }

    t0 = time.perf_counter()
    results = asyncio.run(
        benchmark(
            backend="sglang",
            api_url=f"{base_url}/generate",
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=reqs,
            request_rate=float("inf"),
            max_concurrency=int(concurrency),
            disable_tqdm=False,
            lora_names=None,
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body=extra_request_body,
            profile=False,
        )
    )
    wall_s = time.perf_counter() - t0

    raw_generated_texts = list(results.get("generated_texts") or [])
    raw_meta_infos = list(results.get("meta_infos") or [])
    if len(raw_generated_texts) != len(reqs):
        raise RuntimeError(
            "Benchmark returned an unexpected number of generated texts: "
            f"{len(raw_generated_texts)} vs expected {len(reqs)}"
        )
    normalized_meta_infos: list[dict[str, Any]] = []
    for i in range(len(reqs)):
        meta = raw_meta_infos[i] if i < len(raw_meta_infos) else {}
        if isinstance(meta, dict) and isinstance(meta.get("meta_info"), dict):
            meta = meta["meta_info"]
        if not isinstance(meta, dict):
            meta = {}
        if meta.get("completion_tokens") is None:
            with contextlib.suppress(Exception):
                meta["completion_tokens"] = len(
                    tokenizer.encode(
                        raw_generated_texts[i], add_special_tokens=False
                    )
                )
        normalized_meta_infos.append(meta)

    if int(results.get("completed", 0)) != len(reqs):
        raise RuntimeError(
            f"Benchmark incomplete: completed={results.get('completed')} expected={len(reqs)}"
        )

    server_info = requests.get(base_url + "/get_server_info", timeout=60).json()
    step_time_p20_s: float | None = None
    output_tok_s_p20: float | None = None
    with contextlib.suppress(Exception):
        step_times = server_info["internal_states"][0]["step_time_dict"][
            str(int(concurrency))
        ]
        step_time_p20_s = float(np.percentile(step_times, 20))
        accept_length = float(results.get("accept_length") or 1.0)
        output_tok_s_p20 = (1.0 / step_time_p20_s) * accept_length

    completed = int(results["completed"])
    total_output_tokens = float(results["total_output_tokens"])
    accept_length = float(results.get("accept_length") or 1.0)
    spec_meta = _summarize_spec_meta(normalized_meta_infos)
    request_metrics = _extract_request_metrics(
        meta_infos=normalized_meta_infos,
        request_question_ids=prompt_question_ids,
        output_lens=output_lens,
        generated_texts=list(results.get("generated_texts") or []),
        expected_answers=prompt_expected_answers,
    )

    summary = BenchResult(
        completed=completed,
        # Use the benchmark phase duration from bench_serving so warmup/setup does
        # not contaminate the reported serving throughput.
        wall_s=round(float(results.get("duration") or wall_s), 6),
        req_s=round(completed / float(results.get("duration") or wall_s), 3),
        wall_tok_s=round(total_output_tokens / float(results.get("duration") or wall_s), 3),
        accept_length=round(accept_length, 3),
        accept_length_min_step=spec_meta["accept_length_min_step"],
        accept_length_max_step=spec_meta["accept_length_max_step"],
        accept_draft_tokens_min=spec_meta["accept_draft_tokens_min"],
        accept_draft_tokens_max=spec_meta["accept_draft_tokens_max"],
        verify_ct_sum=spec_meta["verify_ct_sum"],
        verify_ct_avg=(
            round(spec_meta["verify_ct_avg"], 3)
            if spec_meta["verify_ct_avg"] is not None
            else None
        ),
        verify_ct_min=spec_meta["verify_ct_min"],
        verify_ct_max=spec_meta["verify_ct_max"],
        accept_token_sum=spec_meta["accept_token_sum"],
        step_time_p20_s=(
            round(step_time_p20_s, 6) if step_time_p20_s is not None else None
        ),
        output_tok_s_p20=(
            round(output_tok_s_p20, 3) if output_tok_s_p20 is not None else None
        ),
        avg_output_tokens=round(total_output_tokens / completed, 3),
    )
    return BenchRun(
        summary=summary,
        request_metrics=request_metrics,
        request_metric_aggregate=_aggregate_request_metrics(request_metrics),
    )


def _run_single(
    *,
    model_path: str,
    port: int,
    attention_backend: str,
    moe_runner_backend: str,
    kv_cache_dtype: str,
    context_length: int,
    cuda_graph_max_bs: int,
    max_running_requests: int,
    page_size: int,
    enable_piecewise_cuda_graph: bool,
    piecewise_cuda_graph_max_tokens: int | None,
    disable_cuda_graph: bool,
    speculative: bool,
    speculative_algorithm: str,
    draft_model_path: str | None,
    draft_attention_backend: str | None,
    draft_kv_cache_dtype: str | None,
    draft_page_size: int | None,
    speculative_moe_runner_backend: str | None,
    speculative_dflash_block_size: int | None,
    speculative_num_steps: int | None,
    speculative_eagle_topk: int | None,
    speculative_num_draft_tokens: int | None,
    mem_fraction_static: float | None,
    speculative_draft_mem_fraction_static: float | None,
    disable_overlap_schedule: bool,
    prompts: list[str],
    prompt_question_ids: list[str],
    prompt_expected_answers: list[str],
    concurrency: int,
    output_lens: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    disable_stream: bool,
) -> BenchRun:
    proc = _launch_server(
        model_path=model_path,
        port=port,
        attention_backend=attention_backend,
        moe_runner_backend=moe_runner_backend,
        kv_cache_dtype=kv_cache_dtype,
        context_length=context_length,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_running_requests=max_running_requests,
        page_size=page_size,
        enable_piecewise_cuda_graph=enable_piecewise_cuda_graph,
        piecewise_cuda_graph_max_tokens=piecewise_cuda_graph_max_tokens,
        disable_cuda_graph=disable_cuda_graph,
        speculative=speculative,
        speculative_algorithm=speculative_algorithm,
        draft_model_path=draft_model_path,
        draft_attention_backend=draft_attention_backend,
        draft_kv_cache_dtype=draft_kv_cache_dtype,
        draft_page_size=draft_page_size,
        speculative_moe_runner_backend=speculative_moe_runner_backend,
        speculative_dflash_block_size=speculative_dflash_block_size,
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=speculative_eagle_topk,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        mem_fraction_static=mem_fraction_static,
        speculative_draft_mem_fraction_static=speculative_draft_mem_fraction_static,
        disable_overlap_schedule=disable_overlap_schedule,
    )
    base_url = f"http://127.0.0.1:{int(port)}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return _bench_one(
            base_url=base_url,
            tokenizer=tokenizer,
            prompts=prompts,
            prompt_question_ids=prompt_question_ids,
            prompt_expected_answers=prompt_expected_answers,
            concurrency=concurrency,
            output_lens=output_lens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            disable_stream=disable_stream,
        )
    finally:
        kill_process_tree(proc.pid)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark GPT-OSS DFlash on the local reference problems."
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-model-path", required=True)
    p.add_argument("--reference-csv", default="/root/reference.csv")
    p.add_argument("--question-ids", default="92ba6a,9c1c5f,a295e9")
    p.add_argument("--out-json", default=None)
    p.add_argument("--baseline-json-in", default=None)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--baseline-port", type=int, default=21000)
    p.add_argument("--dflash-port", type=int, default=21001)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--decode-len", type=int, default=2048)
    p.add_argument("--decode-to-context-limit", action="store_true")
    p.add_argument("--buffer-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--cuda-graph-max-bs", type=int, default=8)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--piecewise-cuda-graph-max-tokens", type=int, default=8192)
    p.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    p.add_argument("--draft-kv-cache-dtype", default="bfloat16")
    p.add_argument("--draft-page-size", type=int, default=None)
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument("--moe-runner-backend", default="triton_kernel")
    p.add_argument("--draft-attention-backend", default="fa3")
    p.add_argument("--speculative-moe-runner-backend", default="triton_kernel")
    p.add_argument("--speculative-dflash-block-size", type=int, default=8)
    p.add_argument("--speculative-num-steps", type=int, default=None)
    p.add_argument("--speculative-eagle-topk", type=int, default=None)
    p.add_argument("--speculative-num-draft-tokens", type=int, default=None)
    p.add_argument("--mem-fraction-static", type=float, default=None)
    p.add_argument("--speculative-draft-mem-fraction-static", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--disable-cuda-graph", action="store_true")
    p.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    p.add_argument("--disable-overlap-schedule", action="store_true")
    p.add_argument(
        "--speculative-algorithm",
        type=str,
        choices=["DFLASH", "DFLASH_TREE"],
        default="DFLASH",
    )
    p.add_argument("--disable-stream", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.reference_csv)
    question_ids = tuple(
        q.strip() for q in str(args.question_ids).split(",") if q.strip()
    )
    prompts, prompt_question_ids, prompt_expected_answers = _load_reference_prompts(
        csv_path, question_ids, args.num_prompts
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    output_lens = _compute_output_lens(
        prompts=prompts,
        tokenizer=tokenizer,
        context_length=args.context_length,
        decode_len=args.decode_len,
        decode_to_context_limit=args.decode_to_context_limit,
        buffer_tokens=args.buffer_tokens,
    )

    baseline: BenchRun | None
    if args.skip_baseline:
        baseline = None
    elif args.baseline_json_in:
        baseline_payload = json.loads(
            Path(args.baseline_json_in).read_text(encoding="utf-8")
        )
        baseline = BenchRun(
            summary=_bench_result_from_payload(baseline_payload["baseline"]),
            request_metrics=list(baseline_payload.get("baseline_request_metrics") or []),
            request_metric_aggregate=dict(
                baseline_payload.get("baseline_request_metric_aggregate") or {}
            ),
        )
    else:
        baseline = _run_single(
            model_path=args.model_path,
            port=args.baseline_port,
            attention_backend=args.attention_backend,
            moe_runner_backend=args.moe_runner_backend,
            kv_cache_dtype=args.kv_cache_dtype,
            context_length=args.context_length,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            max_running_requests=args.max_running_requests,
            page_size=args.page_size,
            enable_piecewise_cuda_graph=(
                not args.disable_cuda_graph and not args.disable_piecewise_cuda_graph
            ),
            piecewise_cuda_graph_max_tokens=args.piecewise_cuda_graph_max_tokens,
            disable_cuda_graph=bool(args.disable_cuda_graph),
            speculative=False,
            speculative_algorithm="DFLASH",
            draft_model_path=None,
            draft_attention_backend=None,
            draft_kv_cache_dtype=None,
            draft_page_size=None,
            speculative_moe_runner_backend=None,
            speculative_dflash_block_size=None,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
            mem_fraction_static=args.mem_fraction_static,
            speculative_draft_mem_fraction_static=None,
            disable_overlap_schedule=bool(args.disable_overlap_schedule),
            prompts=prompts,
            prompt_question_ids=prompt_question_ids,
            prompt_expected_answers=prompt_expected_answers,
            concurrency=args.concurrency,
            output_lens=output_lens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            disable_stream=bool(args.disable_stream),
        )

    dflash = _run_single(
        model_path=args.model_path,
        port=args.dflash_port,
        attention_backend=args.attention_backend,
        moe_runner_backend=args.moe_runner_backend,
        kv_cache_dtype=args.kv_cache_dtype,
        context_length=args.context_length,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_requests=args.max_running_requests,
        page_size=args.page_size,
        enable_piecewise_cuda_graph=(
            not args.disable_cuda_graph and not args.disable_piecewise_cuda_graph
        ),
        piecewise_cuda_graph_max_tokens=args.piecewise_cuda_graph_max_tokens,
        disable_cuda_graph=bool(args.disable_cuda_graph),
        speculative=True,
        speculative_algorithm=args.speculative_algorithm,
        draft_model_path=args.draft_model_path,
        draft_attention_backend=args.draft_attention_backend,
        draft_kv_cache_dtype=args.draft_kv_cache_dtype,
        draft_page_size=args.draft_page_size,
        speculative_moe_runner_backend=args.speculative_moe_runner_backend,
        speculative_dflash_block_size=args.speculative_dflash_block_size,
        speculative_num_steps=args.speculative_num_steps,
        speculative_eagle_topk=args.speculative_eagle_topk,
        speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        mem_fraction_static=args.mem_fraction_static,
        speculative_draft_mem_fraction_static=args.speculative_draft_mem_fraction_static,
        disable_overlap_schedule=bool(args.disable_overlap_schedule),
        prompts=prompts,
        prompt_question_ids=prompt_question_ids,
        prompt_expected_answers=prompt_expected_answers,
        concurrency=args.concurrency,
        output_lens=output_lens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        disable_stream=bool(args.disable_stream),
    )

    report = {
        "regime": {
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "reference_csv": str(csv_path),
            "question_ids": list(question_ids),
            "context_length": args.context_length,
            "decode_len": args.decode_len,
            "decode_to_context_limit": bool(args.decode_to_context_limit),
            "buffer_tokens": args.buffer_tokens,
            "requested_output_len_min": int(min(output_lens)),
            "requested_output_len_max": int(max(output_lens)),
            "requested_output_len_avg": round(
                float(sum(output_lens)) / float(len(output_lens)), 3
            ),
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "page_size": args.page_size,
            "kv_cache_dtype": args.kv_cache_dtype,
            "draft_kv_cache_dtype": args.draft_kv_cache_dtype,
            "draft_page_size": args.draft_page_size,
            "draft_share_pools_env": os.environ.get("SGLANG_DFLASH_DRAFT_SHARE_POOLS"),
            "draft_share_pools_effective": _effective_draft_share_pools(
                page_size=args.page_size,
                draft_page_size=args.draft_page_size,
            ),
            "attention_backend": args.attention_backend,
            "moe_runner_backend": args.moe_runner_backend,
            "draft_attention_backend": args.draft_attention_backend,
            "speculative_moe_runner_backend": args.speculative_moe_runner_backend,
            "speculative_algorithm": args.speculative_algorithm,
            "speculative_dflash_block_size": args.speculative_dflash_block_size,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
            "speculative_draft_mem_fraction_static": args.speculative_draft_mem_fraction_static,
            "cuda_graph": not args.disable_cuda_graph,
            "piecewise_cuda_graph": (
                not args.disable_cuda_graph and not args.disable_piecewise_cuda_graph
            ),
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "max_running_requests": args.max_running_requests,
            "piecewise_cuda_graph_max_tokens": args.piecewise_cuda_graph_max_tokens,
            "mem_fraction_static": args.mem_fraction_static,
            "disable_overlap_schedule": bool(args.disable_overlap_schedule),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "disable_stream": bool(args.disable_stream),
            "skip_baseline": bool(args.skip_baseline),
        },
        "baseline": asdict(baseline.summary) if baseline is not None else None,
        "baseline_request_metrics": baseline.request_metrics if baseline is not None else [],
        "baseline_request_metric_aggregate": (
            baseline.request_metric_aggregate if baseline is not None else {}
        ),
        "dflash": asdict(dflash.summary),
        "dflash_request_metrics": dflash.request_metrics,
        "dflash_request_metric_aggregate": dflash.request_metric_aggregate,
        "speedup_req_s": (
            round(dflash.summary.req_s / baseline.summary.req_s, 4)
            if baseline is not None and baseline.summary.req_s
            else None
        ),
        "speedup_wall_tok_s": (
            round(dflash.summary.wall_tok_s / baseline.summary.wall_tok_s, 4)
            if baseline is not None and baseline.summary.wall_tok_s
            else None
        ),
        "speedup_output_tok_s_p20": (
            round(
                (dflash.summary.output_tok_s_p20 or 0.0)
                / (baseline.summary.output_tok_s_p20 or 1.0),
                4,
            )
            if baseline is not None and baseline.summary.output_tok_s_p20
            else None
        ),
    }

    print(json.dumps(report, indent=2))
    print(
        "\n| Mode | req/s | wall tok/s | accept len | accept min/max | verify sum | verify avg | p20 step (s) | p20 tok/s | avg out tok |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows: list[tuple[str, BenchResult]] = []
    if baseline is not None:
        rows.append(("baseline", baseline.summary))
    rows.append(("dflash", dflash.summary))
    for name, bench in rows:
        accept_min_max = (
            "n/a"
            if bench.accept_length_min_step is None or bench.accept_length_max_step is None
            else f"{bench.accept_length_min_step}/{bench.accept_length_max_step}"
        )
        print(
            f"| {name} | {bench.req_s:.3f} | {bench.wall_tok_s:.3f} | {bench.accept_length:.3f} | "
            f"{accept_min_max} | {bench.verify_ct_sum if bench.verify_ct_sum is not None else 'n/a'} | "
            f"{bench.verify_ct_avg if bench.verify_ct_avg is not None else 'n/a'} | "
            f"{bench.step_time_p20_s if bench.step_time_p20_s is not None else 'n/a'} | "
            f"{bench.output_tok_s_p20 if bench.output_tok_s_p20 is not None else 'n/a'} | {bench.avg_output_tokens:.3f} |"
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
