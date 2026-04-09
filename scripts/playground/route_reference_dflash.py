#!/usr/bin/env python3
"""Two-phase exploration router for GPT-OSS DFLASH reference problems.

Phase 1:
- oversample branches for a fixed exploration decode budget
- score each branch from DFlash acceptance / entropy / confidence signals

Phase 2:
- continue only the promoted branches to the remaining context budget

This is an application-layer prototype. It does not preserve KV cache across phases.
It is meant to test routing policy quality, not final serving efficiency.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.test.test_utils import kill_process_tree

from bench_reference_dflash import (
    BenchResult,
    _aggregate_request_metrics,
    _build_sampling_params,
    _compute_output_lens,
    _extract_request_metrics,
    _launch_server,
    _load_reference_prompts,
    _summarize_spec_meta,
)


@dataclass(frozen=True)
class PhaseRun:
    summary: BenchResult
    request_metrics: list[dict[str, Any]]
    request_metric_aggregate: dict[str, Any]
    generated_texts: list[str]


@dataclass(frozen=True)
class RoutePolicy:
    green_accept_ge: float
    hard_accept_lt: float
    conflict_accept_ge: float
    conflict_accept_lt: float
    conflict_q_entropy_le: float
    conflict_q_max_ge: float
    min_keep_per_qid: int
    promote_total_k: int
    promotion_mode: str


@dataclass(frozen=True)
class ChunkedPhaseRun:
    final_phase: PhaseRun
    rounds: list[dict[str, Any]]
    total_wall_s: float
    stop_reason: str
    stop_round: int


_TREE_SPEC_CONFIG_BY_BLOCK_SIZE: dict[int, tuple[int, int, int]] = {
    4: (3, 4, 4),
    8: (4, 2, 6),
    16: (8, 1, 9),
}


def _resolve_tree_spec_config(block_size: int) -> tuple[int | None, int | None, int | None]:
    return _TREE_SPEC_CONFIG_BY_BLOCK_SIZE.get(int(block_size), (None, None, None))


def _bench_one_with_texts(
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
    sampling_strategy: str | None = None,
    disable_stream: bool = False,
) -> PhaseRun:
    reqs = []
    for p, out in zip(prompts, output_lens):
        prompt_ids = tokenizer.encode(p, add_special_tokens=False)
        reqs.append(
            DatasetRow(
                prompt=prompt_ids,
                prompt_len=len(prompt_ids),
                output_len=int(out),
            )
        )
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
            sampling_strategy=sampling_strategy,
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

    if int(results.get("completed", 0)) != len(reqs):
        raise RuntimeError(
            f"Benchmark incomplete: completed={results.get('completed')} expected={len(reqs)}"
        )

    server_info = None
    with contextlib.suppress(Exception):
        resp = requests.get(base_url + "/get_server_info", timeout=60)
        if resp.status_code == 200:
            server_info = resp.json()
    step_time_p20_s: float | None = None
    output_tok_s_p20: float | None = None
    with contextlib.suppress(Exception):
        if server_info is None:
            raise RuntimeError("server_info unavailable")
        step_times = server_info["internal_states"][0]["step_time_dict"][
            str(int(concurrency))
        ]
        step_time_p20_s = float(np.percentile(step_times, 20))
        accept_length = float(results.get("accept_length") or 1.0)
        output_tok_s_p20 = (1.0 / step_time_p20_s) * accept_length

    completed = int(results["completed"])
    total_output_tokens = float(results["total_output_tokens"])
    accept_length = float(results.get("accept_length") or 1.0)
    raw_meta_infos = results.get("meta_infos") or []
    spec_meta = _summarize_spec_meta(raw_meta_infos)
    generated_texts = list(results.get("generated_texts") or [])
    request_metrics = _extract_request_metrics(
        meta_infos=raw_meta_infos,
        request_question_ids=prompt_question_ids,
        output_lens=output_lens,
        generated_texts=generated_texts,
        expected_answers=prompt_expected_answers,
    )
    summary = BenchResult(
        completed=completed,
        wall_s=round(wall_s, 6),
        req_s=round(completed / wall_s, 3),
        wall_tok_s=round(total_output_tokens / wall_s, 3),
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
    return PhaseRun(
        summary=summary,
        request_metrics=request_metrics,
        request_metric_aggregate=_aggregate_request_metrics(request_metrics),
        generated_texts=generated_texts,
    )


@contextlib.contextmanager
def _temporary_environ(extra_env: dict[str, str] | None):
    if not extra_env:
        yield
        return
    old_env: dict[str, str | None] = {}
    try:
        for key, value in extra_env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextlib.contextmanager
def _server_session(
    *,
    model_path: str,
    draft_model_path: str,
    port: int,
    context_length: int,
    concurrency: int,
    output_lens: list[int],
    prompts: list[str],
    prompt_question_ids: list[str],
    prompt_expected_answers: list[str],
    mem_fraction_static: float,
    dflash_block_size: int,
    draft_attention_backend: str,
    draft_kv_cache_dtype: str,
    speculative_algorithm: str = "DFLASH",
    speculative: bool = True,
    server_env: dict[str, str] | None = None,
):
    speculative_num_steps: int | None = None
    speculative_eagle_topk: int | None = None
    speculative_num_draft_tokens: int | None = None
    if str(speculative_algorithm) == "DFLASH_TREE":
        (
            speculative_num_steps,
            speculative_eagle_topk,
            speculative_num_draft_tokens,
        ) = _resolve_tree_spec_config(int(dflash_block_size))
        if speculative_num_steps is None or speculative_eagle_topk is None or speculative_num_draft_tokens is None:
            raise ValueError(
                "DFLASH_TREE route benchmark requires a known tree config for "
                f"dflash_block_size={int(dflash_block_size)}."
            )
    with _temporary_environ(server_env):
        proc = _launch_server(
            model_path=model_path,
            port=port,
            attention_backend="fa3",
            moe_runner_backend="triton_kernel",
            kv_cache_dtype="fp8_e4m3",
            context_length=context_length,
            cuda_graph_max_bs=concurrency,
            max_running_requests=concurrency,
            page_size=1,
            enable_piecewise_cuda_graph=True,
            piecewise_cuda_graph_max_tokens=8192,
            disable_cuda_graph=False,
            speculative=speculative,
            speculative_algorithm=speculative_algorithm,
            draft_model_path=draft_model_path,
            draft_attention_backend=str(draft_attention_backend),
            draft_kv_cache_dtype=str(draft_kv_cache_dtype),
            draft_page_size=1,
            speculative_moe_runner_backend="triton_kernel",
            speculative_dflash_block_size=int(dflash_block_size),
            speculative_num_steps=speculative_num_steps,
            speculative_eagle_topk=speculative_eagle_topk,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            mem_fraction_static=mem_fraction_static,
            draft_mem_fraction_static=None,
            sampling_backend="pytorch",
            skip_tokenizer_init=True,
            disable_overlap_schedule=True,
        )
    base_url = f"http://127.0.0.1:{int(port)}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        yield base_url, tokenizer
    finally:
        kill_process_tree(proc.pid)


def _run_phase(
    *,
    model_path: str,
    draft_model_path: str,
    port: int,
    context_length: int,
    concurrency: int,
    output_lens: list[int],
    prompts: list[str],
    prompt_question_ids: list[str],
    prompt_expected_answers: list[str],
    mem_fraction_static: float,
    dflash_block_size: int,
    draft_attention_backend: str,
    draft_kv_cache_dtype: str,
    speculative_algorithm: str = "DFLASH",
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    sampling_strategy: str | None,
    speculative: bool = True,
    server_env: dict[str, str] | None = None,
    disable_stream: bool = False,
) -> PhaseRun:
    with _server_session(
        model_path=model_path,
        draft_model_path=draft_model_path,
        port=port,
        context_length=context_length,
        concurrency=concurrency,
        output_lens=output_lens,
        prompts=prompts,
        prompt_question_ids=prompt_question_ids,
        prompt_expected_answers=prompt_expected_answers,
        mem_fraction_static=mem_fraction_static,
        dflash_block_size=dflash_block_size,
        draft_attention_backend=draft_attention_backend,
        draft_kv_cache_dtype=draft_kv_cache_dtype,
        speculative_algorithm=speculative_algorithm,
        speculative=speculative,
        server_env=server_env,
    ) as (base_url, tokenizer):
        return _bench_one_with_texts(
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
            sampling_strategy=sampling_strategy,
            disable_stream=bool(disable_stream),
        )


def _bench_total_generated_tokens(summary: BenchResult) -> float:
    return float(summary.avg_output_tokens) * float(summary.completed)


def _server_internal_state(base_url: str) -> dict[str, Any]:
    try:
        server_info = None
        with contextlib.suppress(Exception):
            resp = requests.get(base_url + "/get_server_info", timeout=60)
            if resp.status_code == 200:
                server_info = resp.json()
        internal_states = server_info.get("internal_states") if server_info else []
        if internal_states:
            return dict(internal_states[0])
    except Exception:
        pass
    return {}


def _run_chunked_exploration(
    *,
    model_path: str,
    draft_model_path: str,
    port: int,
    context_length: int,
    concurrency: int,
    prompts: list[str],
    prompt_question_ids: list[str],
    prompt_expected_answers: list[str],
    full_output_lens: list[int],
    exploration_output_lens: list[int],
    round_output_len: int,
    min_rounds: int,
    stop_accept_le: float,
    stop_selected_mean_accept_ge: float,
    stop_selected_margin_ge: float,
    mem_fraction_static: float,
    dflash_block_size: int,
    draft_attention_backend: str,
    draft_kv_cache_dtype: str,
    speculative_algorithm: str = "DFLASH",
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    sampling_strategy: str | None,
    policy: RoutePolicy,
    disable_stream: bool = False,
) -> tuple[ChunkedPhaseRun, list[dict[str, Any]], list[dict[str, Any]], list[str], list[int]]:
    current_prompts = list(prompts)
    cumulative_texts = [""] * len(prompts)
    remaining_exploration_lens = [int(v) for v in exploration_output_lens]
    remaining_full_lens = [int(v) for v in full_output_lens]
    rounds: list[dict[str, Any]] = []
    total_wall_s = 0.0
    last_phase: PhaseRun | None = None
    last_branches: list[dict[str, Any]] = []
    last_selected: list[dict[str, Any]] = []
    stop_reason = "max_exploration_budget"
    stop_round = 0

    with _server_session(
        model_path=model_path,
        draft_model_path=draft_model_path,
        port=port,
        context_length=context_length,
        concurrency=concurrency,
        output_lens=exploration_output_lens,
        prompts=prompts,
        prompt_question_ids=prompt_question_ids,
        prompt_expected_answers=prompt_expected_answers,
        mem_fraction_static=mem_fraction_static,
        dflash_block_size=dflash_block_size,
        draft_attention_backend=draft_attention_backend,
        draft_kv_cache_dtype=draft_kv_cache_dtype,
        speculative_algorithm=speculative_algorithm,
        speculative=True,
        server_env=None,
    ) as (base_url, tokenizer):
        round_idx = 0
        while True:
            active = [i for i, rem in enumerate(remaining_exploration_lens) if int(rem) > 0]
            if not active:
                stop_reason = "completed_exploration_budget"
                break

            round_prompts = [current_prompts[i] for i in active]
            round_qids = [prompt_question_ids[i] for i in active]
            round_answers = [prompt_expected_answers[i] for i in active]
            round_output_lens = [
                min(int(round_output_len), int(remaining_exploration_lens[i])) for i in active
            ]
            phase = _bench_one_with_texts(
                base_url=base_url,
                tokenizer=tokenizer,
                prompts=round_prompts,
                prompt_question_ids=round_qids,
                prompt_expected_answers=round_answers,
                concurrency=min(int(concurrency), len(round_prompts)),
                output_lens=round_output_lens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                sampling_strategy=sampling_strategy,
                disable_stream=bool(disable_stream),
            )
            total_wall_s += float(phase.summary.wall_s)
            round_idx += 1

            merged_request_metrics: list[dict[str, Any]] = []
            merged_generated_texts: list[str] = []
            for active_pos, req_idx in enumerate(active):
                cumulative_texts[req_idx] += phase.generated_texts[active_pos]
                current_prompts[req_idx] += phase.generated_texts[active_pos]
                remaining_exploration_lens[req_idx] = max(
                    0, int(remaining_exploration_lens[req_idx]) - int(round_output_lens[active_pos])
                )
                remaining_full_lens[req_idx] = max(
                    0, int(remaining_full_lens[req_idx]) - int(round_output_lens[active_pos])
                )

            idx_to_metric = {req_idx: phase.request_metrics[pos] for pos, req_idx in enumerate(active)}
            for req_idx in range(len(prompts)):
                base_metric: dict[str, Any]
                if req_idx in idx_to_metric:
                    base_metric = dict(idx_to_metric[req_idx])
                else:
                    base_metric = {
                        "request_index": int(req_idx),
                        "question_id": prompt_question_ids[req_idx],
                        "requested_output_len": 0,
                        "completion_tokens": 0,
                    }
                base_metric["request_index"] = int(req_idx)
                merged_request_metrics.append(base_metric)
                merged_generated_texts.append(cumulative_texts[req_idx])

            merged_phase = PhaseRun(
                summary=phase.summary,
                request_metrics=merged_request_metrics,
                request_metric_aggregate=_aggregate_request_metrics(merged_request_metrics),
                generated_texts=merged_generated_texts,
            )
            branches = _annotate_branches(
                prompts=prompts,
                question_ids=prompt_question_ids,
                expected_answers=prompt_expected_answers,
                output_lens=remaining_full_lens,
                phase=merged_phase,
                policy=policy,
            )
            selected = _select_promotions(branches=branches, policy=policy)
            internal_state = _server_internal_state(base_url)
            selected_accepts = [
                float(branch.get("spec_accept_length") or 0.0) for branch in selected
            ]
            selected_mean_accept = (
                float(sum(selected_accepts)) / float(len(selected_accepts))
                if selected_accepts
                else 0.0
            )
            round_record = {
                "round_index": int(round_idx),
                "summary": asdict(phase.summary),
                "request_metric_aggregate": merged_phase.request_metric_aggregate,
                "selected_count": int(len(selected)),
                "selected_mean_accept_length": round(selected_mean_accept, 6),
                "selected_by_label": {
                    label: sum(1 for row in selected if row["route_label"] == label)
                    for label in ("green", "neutral", "hard_tail", "confident_conflict")
                },
                "server_internal_state": {
                    "avg_spec_accept_length": internal_state.get("avg_spec_accept_length"),
                    "last_gen_throughput": internal_state.get("last_gen_throughput"),
                    "memory_usage": internal_state.get("memory_usage"),
                },
            }
            rounds.append(round_record)
            last_phase = merged_phase
            last_branches = branches
            last_selected = selected

            aggregate_accept = float(phase.summary.accept_length)
            selected_margin = float(selected_mean_accept) - float(aggregate_accept)
            if round_idx >= int(min_rounds):
                if float(selected_mean_accept) >= float(stop_selected_mean_accept_ge):
                    stop_reason = "selected_accept_good_enough"
                    stop_round = int(round_idx)
                    break
                if (
                    float(aggregate_accept) <= float(stop_accept_le)
                    and float(selected_margin) >= float(stop_selected_margin_ge)
                ):
                    stop_reason = "subset_beats_pool"
                    stop_round = int(round_idx)
                    break
            stop_round = int(round_idx)

    if last_phase is None:
        raise RuntimeError("Exploration produced no rounds")

    return (
        ChunkedPhaseRun(
            final_phase=last_phase,
            rounds=rounds,
            total_wall_s=round(float(total_wall_s), 6),
            stop_reason=str(stop_reason),
            stop_round=int(stop_round),
        ),
        last_branches,
        last_selected,
        current_prompts,
        remaining_full_lens,
    )


def _classify_branch(row: dict[str, Any], policy: RoutePolicy) -> str:
    accept = float(row.get("spec_accept_length") or 0.0)
    q_entropy = row.get("spec_dflash_q_entropy_mean")
    q_max = row.get("spec_dflash_q_max_mean")
    if accept >= float(policy.green_accept_ge):
        return "green"
    if (
        float(policy.conflict_accept_ge) <= accept < float(policy.conflict_accept_lt)
        and q_entropy is not None
        and q_max is not None
        and float(q_entropy) <= float(policy.conflict_q_entropy_le)
        and float(q_max) >= float(policy.conflict_q_max_ge)
    ):
        return "confident_conflict"
    if accept < float(policy.hard_accept_lt):
        return "hard_tail"
    return "neutral"


def _score_branch(row: dict[str, Any], label: str) -> float:
    accept = float(row.get("spec_accept_length") or 0.0)
    verify = float(row.get("spec_verify_ct") or 0.0)
    q_entropy = float(row.get("spec_dflash_q_entropy_mean") or 0.0)
    q_max = float(row.get("spec_dflash_q_max_mean") or 0.0)
    score = (accept * 10.0) + (q_max * 4.0) - (q_entropy * 3.0) - (verify / 10000.0)
    if label == "green":
        score += 100.0
    elif label == "neutral":
        score += 10.0
    elif label == "hard_tail":
        score += 0.0
    elif label == "confident_conflict":
        score -= 40.0
    return float(score)


def _priority_tuple(branch: dict[str, Any]) -> tuple[float, float, float]:
    label_priority = {
        "green": 3.0,
        "neutral": 2.0,
        "hard_tail": 1.0,
        "confident_conflict": 0.0,
    }
    return (
        label_priority.get(str(branch["route_label"]), -1.0),
        float(branch["route_score"]),
        float(branch.get("spec_accept_length") or 0.0),
    )


def _annotate_branches(
    *,
    prompts: list[str],
    question_ids: list[str],
    expected_answers: list[str],
    output_lens: list[int],
    phase: PhaseRun,
    policy: RoutePolicy,
) -> list[dict[str, Any]]:
    branches: list[dict[str, Any]] = []
    for i, row in enumerate(phase.request_metrics):
        branch = dict(row)
        branch["request_index"] = int(i)
        branch["prompt"] = prompts[i]
        branch["question_id"] = question_ids[i]
        branch["expected_answer"] = expected_answers[i]
        branch["requested_output_len"] = int(output_lens[i])
        branch["generated_text"] = phase.generated_texts[i]
        branch["generated_text_preview"] = phase.generated_texts[i][:240]
        branch["route_label"] = _classify_branch(branch, policy)
        branch["route_score"] = round(_score_branch(branch, branch["route_label"]), 6)
        branches.append(branch)
    return branches


def _select_promotions(
    *,
    branches: list[dict[str, Any]],
    policy: RoutePolicy,
) -> list[dict[str, Any]]:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for branch in branches:
        by_qid.setdefault(str(branch["question_id"]), []).append(branch)
    for qid in by_qid:
        by_qid[qid].sort(key=_priority_tuple, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    for qid in sorted(by_qid):
        keep = 0
        for branch in by_qid[qid]:
            if keep >= int(policy.min_keep_per_qid):
                break
            selected.append(branch)
            selected_ids.add(int(branch["request_index"]))
            keep += 1

    remaining = [b for b in branches if int(b["request_index"]) not in selected_ids]
    remaining.sort(key=_priority_tuple, reverse=True)

    if str(policy.promotion_mode) == "strict":
        total_k = int(policy.promote_total_k)
        slots = max(0, total_k - len(selected))
        selected.extend(remaining[:slots])
        selected.sort(key=_priority_tuple, reverse=True)
        return selected

    # throughput mode: keep all green branches, then all neutral/hard branches,
    # and only use confident-conflict if we still need minimum coverage.
    for branch in remaining:
        if str(branch["route_label"]) == "confident_conflict":
            continue
        selected.append(branch)
    selected.sort(key=_priority_tuple, reverse=True)
    return selected


def _build_continuation_prompts(
    *,
    selected: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    prompts: list[str] = []
    qids: list[str] = []
    answers: list[str] = []
    for branch in selected:
        prompts.append(str(branch["prompt"]) + str(branch["generated_text"]))
        qids.append(str(branch["question_id"]))
        answers.append(str(branch["expected_answer"]))
    return prompts, qids, answers


def _resolved_answer(row: dict[str, Any]) -> str | None:
    boxed = row.get("boxed_answer")
    if boxed is not None and str(boxed).strip():
        return str(boxed).strip()
    fallback = row.get("fallback_answer")
    if fallback is not None and str(fallback).strip():
        return str(fallback).strip()
    return None


def _majority_vote_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hist: dict[str, int] = {}
    expected: str | None = None
    for row in rows:
        answer = _resolved_answer(row)
        if answer is None:
            continue
        hist[answer] = int(hist.get(answer, 0)) + 1
        if expected is None and row.get("expected_answer") is not None:
            expected = str(row.get("expected_answer")).strip()

    majority_answer: str | None = None
    majority_support = 0
    if hist:
        majority_answer, majority_support = max(
            hist.items(), key=lambda kv: (int(kv[1]), str(kv[0]))
        )

    return {
        "answer_hist": hist,
        "majority_answer": majority_answer,
        "majority_support": int(majority_support),
        "valid_answer_count": int(sum(int(v) for v in hist.values())),
        "expected_answer": expected,
        "is_correct_majority": (
            bool(majority_answer == expected)
            if majority_answer is not None and expected is not None
            else False
        ),
    }


def _build_continuation_server_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if bool(args.continuation_adaptive_cap_enable):
        env["SGLANG_DFLASH_ADAPTIVE_CAP_ENABLE"] = "1"
        env["SGLANG_DFLASH_ADAPTIVE_CAP_VERIFY_CT_GE"] = str(
            int(args.continuation_adaptive_cap_verify_ct_ge)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_LAST_VERIFY_CT_GE"] = str(
            int(args.continuation_adaptive_cap_last_verify_ct_ge)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_EMA_HARD_LE"] = str(
            float(args.continuation_adaptive_cap_accept_ema_hard_le)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_EMA_MEDIUM_LE"] = str(
            float(args.continuation_adaptive_cap_accept_ema_medium_le)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_LAST_HARD_LE"] = str(
            float(args.continuation_adaptive_cap_accept_last_hard_le)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_ACCEPT_LAST_MEDIUM_LE"] = str(
            float(args.continuation_adaptive_cap_accept_last_medium_le)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_HARD_STEPS"] = str(
            int(args.continuation_adaptive_cap_hard_steps)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_MEDIUM_STEPS"] = str(
            int(args.continuation_adaptive_cap_medium_steps)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_Q_ENTROPY_HARD_LE"] = str(
            float(args.continuation_adaptive_cap_q_entropy_hard_le)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_Q_MAX_HARD_GE"] = str(
            float(args.continuation_adaptive_cap_q_max_hard_ge)
        )
        env["SGLANG_DFLASH_ADAPTIVE_CAP_TV_HARD_GE"] = str(
            float(args.continuation_adaptive_cap_tv_hard_ge)
        )
    if bool(args.continuation_adaptive_cap_batch_enable):
        env["SGLANG_DFLASH_ADAPTIVE_BATCH_CAP_ENABLE"] = "1"
        env["SGLANG_DFLASH_ADAPTIVE_BATCH_CAP_HARD_FRACTION_GE"] = str(
            float(args.continuation_adaptive_cap_batch_hard_fraction_ge)
        )
        env["SGLANG_DFLASH_ADAPTIVE_BATCH_CAP_MEDIUM_FRACTION_GE"] = str(
            float(args.continuation_adaptive_cap_batch_medium_fraction_ge)
        )
    return env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase exploration router for GPT-OSS DFlash reference problems."
    )
    p.add_argument("--model-path", default="/workspace/offload_root/gpt-oss-120b")
    p.add_argument("--draft-model-path", default="/root/epoch_65_step_23760")
    p.add_argument("--reference-csv", default="/root/reference.csv")
    p.add_argument("--question-ids", default="92ba6a,9c1c5f,a295e9")
    p.add_argument("--out-json", required=True)
    p.add_argument("--exploration-port", type=int, default=23101)
    p.add_argument("--continuation-port", type=int, default=23102)
    p.add_argument("--final-context-length", type=int, default=65536)
    p.add_argument("--exploration-decode-len", type=int, default=8192)
    p.add_argument("--exploration-concurrency", type=int, default=32)
    p.add_argument("--exploration-num-prompts", type=int, default=32)
    p.add_argument("--buffer-tokens", type=int, default=512)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--dflash-block-size", type=int, default=16)
    p.add_argument("--draft-attention-backend", default="fa3")
    p.add_argument("--draft-kv-cache-dtype", default="bfloat16")
    p.add_argument(
        "--speculative-algorithm",
        choices=("DFLASH", "DFLASH_TREE"),
        default="DFLASH",
    )
    p.add_argument("--exploration-dflash-block-size", type=int, default=None)
    p.add_argument("--continuation-dflash-block-size", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--sampling-strategy", default=None)
    p.add_argument("--promotion-mode", choices=("strict", "throughput"), default="strict")
    p.add_argument("--promote-total-k", type=int, default=8)
    p.add_argument("--min-keep-per-qid", type=int, default=1)
    p.add_argument("--exploration-round-len", type=int, default=2048)
    p.add_argument("--exploration-min-rounds", type=int, default=2)
    p.add_argument("--exploration-stop-accept-le", type=float, default=3.25)
    p.add_argument("--exploration-stop-selected-mean-accept-ge", type=float, default=3.5)
    p.add_argument("--exploration-stop-selected-margin-ge", type=float, default=0.10)
    p.add_argument("--green-accept-ge", type=float, default=6.0)
    p.add_argument("--hard-accept-lt", type=float, default=3.0)
    p.add_argument("--conflict-accept-ge", type=float, default=3.5)
    p.add_argument("--conflict-accept-lt", type=float, default=6.0)
    p.add_argument("--conflict-q-entropy-le", type=float, default=0.7)
    p.add_argument("--conflict-q-max-ge", type=float, default=0.85)
    p.add_argument("--continuation-adaptive-cap-enable", action="store_true", default=False)
    p.add_argument("--continuation-adaptive-cap-verify-ct-ge", type=int, default=8)
    p.add_argument("--continuation-adaptive-cap-last-verify-ct-ge", type=int, default=2)
    p.add_argument("--continuation-adaptive-cap-accept-ema-hard-le", type=float, default=3.25)
    p.add_argument("--continuation-adaptive-cap-accept-ema-medium-le", type=float, default=5.0)
    p.add_argument("--continuation-adaptive-cap-accept-last-hard-le", type=float, default=-1.0)
    p.add_argument("--continuation-adaptive-cap-accept-last-medium-le", type=float, default=2.0)
    p.add_argument("--continuation-adaptive-cap-hard-steps", type=int, default=1)
    p.add_argument("--continuation-adaptive-cap-medium-steps", type=int, default=8)
    p.add_argument("--continuation-adaptive-cap-q-entropy-hard-le", type=float, default=-1.0)
    p.add_argument("--continuation-adaptive-cap-q-max-hard-ge", type=float, default=-1.0)
    p.add_argument("--continuation-adaptive-cap-tv-hard-ge", type=float, default=-1.0)
    p.add_argument("--continuation-adaptive-cap-batch-enable", action="store_true", default=False)
    p.add_argument("--continuation-adaptive-cap-batch-hard-fraction-ge", type=float, default=0.75)
    p.add_argument("--continuation-adaptive-cap-batch-medium-fraction-ge", type=float, default=0.5)
    p.add_argument("--disable-stream", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    policy = RoutePolicy(
        green_accept_ge=float(args.green_accept_ge),
        hard_accept_lt=float(args.hard_accept_lt),
        conflict_accept_ge=float(args.conflict_accept_ge),
        conflict_accept_lt=float(args.conflict_accept_lt),
        conflict_q_entropy_le=float(args.conflict_q_entropy_le),
        conflict_q_max_ge=float(args.conflict_q_max_ge),
        min_keep_per_qid=int(args.min_keep_per_qid),
        promote_total_k=int(args.promote_total_k),
        promotion_mode=str(args.promotion_mode),
    )

    question_ids = tuple(q.strip() for q in str(args.question_ids).split(",") if q.strip())
    prompts, qids, answers = _load_reference_prompts(
        Path(args.reference_csv),
        question_ids,
        int(args.exploration_num_prompts),
    )
    exploration_dflash_block_size = int(
        args.exploration_dflash_block_size
        if args.exploration_dflash_block_size is not None
        else args.dflash_block_size
    )
    continuation_dflash_block_size = int(
        args.continuation_dflash_block_size
        if args.continuation_dflash_block_size is not None
        else args.dflash_block_size
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    full_output_lens = _compute_output_lens(
        prompts=prompts,
        tokenizer=tokenizer,
        context_length=int(args.final_context_length),
        decode_len=int(args.final_context_length),
        decode_to_context_limit=True,
        buffer_tokens=int(args.buffer_tokens),
    )
    exploration_output_lens = [
        min(int(full_budget), int(args.exploration_decode_len))
        for full_budget in full_output_lens
    ]

    exploration, branches, selected, explored_prompts, remaining_after_exploration = _run_chunked_exploration(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        port=int(args.exploration_port),
        context_length=int(args.final_context_length),
        concurrency=int(args.exploration_concurrency),
        prompts=prompts,
        prompt_question_ids=qids,
        prompt_expected_answers=answers,
        full_output_lens=full_output_lens,
        exploration_output_lens=exploration_output_lens,
        round_output_len=int(args.exploration_round_len),
        min_rounds=int(args.exploration_min_rounds),
        stop_accept_le=float(args.exploration_stop_accept_le),
        stop_selected_mean_accept_ge=float(args.exploration_stop_selected_mean_accept_ge),
        stop_selected_margin_ge=float(args.exploration_stop_selected_margin_ge),
        mem_fraction_static=float(args.mem_fraction_static),
        dflash_block_size=exploration_dflash_block_size,
        draft_attention_backend=str(args.draft_attention_backend),
        draft_kv_cache_dtype=str(args.draft_kv_cache_dtype),
        speculative_algorithm=str(args.speculative_algorithm),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        min_p=float(args.min_p),
        sampling_strategy=(
            str(args.sampling_strategy) if args.sampling_strategy is not None else None
        ),
        policy=policy,
        disable_stream=bool(args.disable_stream),
    )

    continuation_prompts, continuation_qids, continuation_answers = _build_continuation_prompts(
        selected=selected
    )
    req_to_remaining = {int(i): int(rem) for i, rem in enumerate(remaining_after_exploration)}
    continuation_output_lens = [req_to_remaining[int(branch["request_index"])] for branch in selected]
    continuation_server_env = _build_continuation_server_env(args)

    continuation = _run_phase(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        port=int(args.continuation_port),
        context_length=int(args.final_context_length),
        concurrency=max(1, len(continuation_prompts)),
        output_lens=continuation_output_lens,
        prompts=continuation_prompts,
        prompt_question_ids=continuation_qids,
        prompt_expected_answers=continuation_answers,
        mem_fraction_static=float(args.mem_fraction_static),
        dflash_block_size=continuation_dflash_block_size,
        draft_attention_backend=str(args.draft_attention_backend),
        draft_kv_cache_dtype=str(args.draft_kv_cache_dtype),
        speculative_algorithm=str(args.speculative_algorithm),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        min_p=float(args.min_p),
        sampling_strategy=(
            str(args.sampling_strategy) if args.sampling_strategy is not None else None
        ),
        speculative=True,
        server_env=continuation_server_env,
        disable_stream=bool(args.disable_stream),
    )

    selected_rows = []
    for branch in selected:
        selected_rows.append(
            {
                "request_index": int(branch["request_index"]),
                "question_id": branch["question_id"],
                "route_label": branch["route_label"],
                "route_score": branch["route_score"],
                "spec_accept_length": branch.get("spec_accept_length"),
                "spec_verify_ct": branch.get("spec_verify_ct"),
                "spec_dflash_q_entropy_mean": branch.get("spec_dflash_q_entropy_mean"),
                "spec_dflash_q_max_mean": branch.get("spec_dflash_q_max_mean"),
                "completion_tokens": branch.get("completion_tokens"),
                "generated_text_preview": branch.get("generated_text_preview"),
                "boxed_answer": branch.get("boxed_answer"),
                "expected_answer": branch.get("expected_answer"),
                "is_correct_boxed": branch.get("is_correct_boxed"),
            }
        )

    continuation_majority = _majority_vote_summary(continuation.request_metrics)
    selected_majority = _majority_vote_summary(selected_rows)

    report = {
        "policy": asdict(policy),
        "regime": {
            "final_context_length": int(args.final_context_length),
            "exploration_decode_len": int(args.exploration_decode_len),
            "exploration_round_len": int(args.exploration_round_len),
            "exploration_concurrency": int(args.exploration_concurrency),
            "exploration_num_prompts": int(args.exploration_num_prompts),
            "mem_fraction_static": float(args.mem_fraction_static),
            "buffer_tokens": int(args.buffer_tokens),
            "exploration_dflash_block_size": exploration_dflash_block_size,
            "continuation_dflash_block_size": continuation_dflash_block_size,
            "draft_kv_cache_dtype": str(args.draft_kv_cache_dtype),
            "speculative_algorithm": str(args.speculative_algorithm),
            "resolved_tree_config": (
                {
                    "exploration": {
                        "num_steps": _resolve_tree_spec_config(exploration_dflash_block_size)[0],
                        "topk": _resolve_tree_spec_config(exploration_dflash_block_size)[1],
                        "num_draft_tokens": _resolve_tree_spec_config(exploration_dflash_block_size)[2],
                    },
                    "continuation": {
                        "num_steps": _resolve_tree_spec_config(continuation_dflash_block_size)[0],
                        "topk": _resolve_tree_spec_config(continuation_dflash_block_size)[1],
                        "num_draft_tokens": _resolve_tree_spec_config(continuation_dflash_block_size)[2],
                    },
                }
                if str(args.speculative_algorithm) == "DFLASH_TREE"
                else None
            ),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "min_p": float(args.min_p),
            "sampling_strategy": (
                str(args.sampling_strategy) if args.sampling_strategy is not None else None
            ),
        },
        "exploration": {
            "summary": asdict(exploration.final_phase.summary),
            "request_metric_aggregate": exploration.final_phase.request_metric_aggregate,
            "rounds": exploration.rounds,
            "total_wall_s": exploration.total_wall_s,
            "stop_reason": exploration.stop_reason,
            "stop_round": exploration.stop_round,
        },
        "selection": {
            "selected_count": len(selected_rows),
            "selected_by_label": {
                label: sum(1 for row in selected_rows if row["route_label"] == label)
                for label in ("green", "neutral", "hard_tail", "confident_conflict")
            },
            "majority_vote": selected_majority,
            "selected_rows": selected_rows,
        },
        "continuation": {
            "summary": asdict(continuation.summary),
            "request_metric_aggregate": continuation.request_metric_aggregate,
            "majority_vote": continuation_majority,
            "request_metrics": continuation.request_metrics,
            "server_env": continuation_server_env,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
