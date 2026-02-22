"""
Benchmark/sweep DFlash speculative decoding configurations (DFLASH vs DFLASH_TREE).

This is modeled after `scripts/playground/bench_speculative.py`, but adds:
  - DFlash-specific server knobs: block_size / spec_steps / tree_topk / verify-node budget
  - Production-like sampling knobs: temperature / top_p / min_p / top_k
  - Long-decode regimes (e.g. decode_len=65536) + concurrency sweeps

Typical usage (single node, local server launch):

  python3 bench_speculative_dflash.py \
    --model-path /path/to/gpt-oss-120b \
    --speculative-draft-model-path /path/to/dflash-draft-ckpt \
    --attention-backend fa3 \
    --speculative-draft-attention-backend fa3 \
    --context-length 131072 \
    --tp-size 1 \
    --algorithms DFLASH DFLASH_TREE \
    --batch-size 1 8 \
    --decode-len 8192 65536 \
    --block-size 16 \
    --spec-steps 15 \
    --tree-topk 4 \
    --num-verify-tokens 16 \
    --sampling-temperature 1.0 \
    --sampling-top-p 1.0 \
    --sampling-min-p 0.02 \
    --output dflash_sweep.jsonl
"""

import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


PROMPTS = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step.\n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
]


def _node0_print(server_args: ServerArgs, msg: str) -> None:
    if getattr(server_args, "node_rank", 0) == 0:
        print(msg, flush=True)


def _get_server_info(base_url: str) -> Dict[str, Any]:
    info = requests.get(base_url + "/get_server_info").json()
    # Some server modes nest decode states.
    if isinstance(info, dict) and "decode" in info and info["decode"]:
        return info["decode"][0]
    return info


def _build_sampling_params(args: argparse.Namespace, *, decode_len: int) -> Dict[str, Any]:
    # This object overrides `payload["sampling_params"]` in bench_serving's sglang backend.
    sampling_params: Dict[str, Any] = {
        "temperature": float(args.sampling_temperature),
        "top_p": float(args.sampling_top_p),
        "min_p": float(args.sampling_min_p),
        "top_k": int(args.sampling_top_k),
        "max_new_tokens": int(decode_len),
        "ignore_eos": bool(args.ignore_eos),
    }
    return sampling_params


def _make_input_requests(num_prompts: int, *, decode_len: int) -> List[DatasetRow]:
    padded = (PROMPTS * ((num_prompts + len(PROMPTS) - 1) // len(PROMPTS)))[:num_prompts]
    # format: (prompt, input_len, output_len). input_len is a dummy value for `benchmark()`.
    return [DatasetRow(p, 0, int(decode_len)) for p in padded]


def _send_one_batch(
    *,
    base_url: str,
    batch_size: int,
    num_prompts: int,
    tokenizer,
    decode_len: int,
    sampling_params: Dict[str, Any],
    step_time_percentile: float,
) -> Dict[str, Any]:
    backend = "sglang"
    api_url = f"{base_url}/generate"
    input_requests = _make_input_requests(num_prompts, decode_len=decode_len)

    # We need to set some dummy values in order to call `benchmark` below.
    bench_args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        return_routed_experts=False,
        plot_throughput=False,
        backend=backend,
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    )
    set_global_args(bench_args)

    extra_request_body = {"sampling_params": sampling_params}

    results = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=int(batch_size),
            disable_tqdm=False,
            lora_names=None,
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body=extra_request_body,
            profile=None,
        )
    )

    server_info = _get_server_info(base_url)
    internal = (server_info.get("internal_states") or [{}])[0] or {}
    step_time_dict = internal.get("step_time_dict") or {}
    step_times = step_time_dict.get(str(batch_size))
    step_time = (
        float(np.percentile(step_times, step_time_percentile))
        if step_times
        else None
    )

    accept_length = results.get("accept_length", None)
    output_tok_s = results.get("output_throughput", None)

    return {
        "accept_length": accept_length,
        "output_tok_s": output_tok_s,
        "step_time_s_pctl": step_time,
        "server_internal": {
            "avg_spec_accept_length": internal.get("avg_spec_accept_length"),
            "avg_spec_verify_ct": internal.get("avg_spec_verify_ct"),
        },
        "raw_results": {
            "completed": results.get("completed"),
            "total_output_tokens": results.get("total_output_tokens"),
            "output_throughput": results.get("output_throughput"),
            "request_throughput": results.get("request_throughput"),
        },
    }


def _launch_and_bench(
    *,
    args: argparse.Namespace,
    server_args: ServerArgs,
    base_url: str,
    other_args: List[str],
    batch_size: int,
    decode_len: int,
) -> Dict[str, Any]:
    env = {"SGLANG_RECORD_STEP_TIME": "1", **os.environ}
    process = popen_launch_server(
        args.model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=env,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=server_args.trust_remote_code
    )

    sampling_params = _build_sampling_params(args, decode_len=decode_len)

    try:
        # Warmup (single request) to build graphs/kernels.
        _send_one_batch(
            base_url=base_url,
            batch_size=batch_size,
            num_prompts=max(1, min(args.num_prompts, batch_size)),
            tokenizer=tokenizer,
            decode_len=min(64, int(decode_len)),
            sampling_params={**sampling_params, "max_new_tokens": min(64, int(decode_len))},
            step_time_percentile=float(args.step_time_percentile),
        )

        bench = _send_one_batch(
            base_url=base_url,
            batch_size=batch_size,
            num_prompts=max(args.num_prompts, batch_size),
            tokenizer=tokenizer,
            decode_len=decode_len,
            sampling_params=sampling_params,
            step_time_percentile=float(args.step_time_percentile),
        )
    finally:
        kill_process_tree(process.pid)

    # Wait for the server to shutdown and port to be released.
    time.sleep(float(args.shutdown_sleep_s))
    return bench


def _base_other_args(server_args: ServerArgs, *, batch_size: int) -> List[str]:
    other_args: List[str] = [
        "--cuda-graph-max-bs",
        str(batch_size),
        "--max-running-requests",
        str(batch_size),
        "--tp-size",
        str(server_args.tp_size),
        "--mem-fraction-static",
        str(server_args.mem_fraction_static),
    ]

    if server_args.trust_remote_code:
        other_args.append("--trust-remote-code")

    if server_args.quantization:
        other_args.extend(["--quantization", str(server_args.quantization)])

    if server_args.attention_backend:
        other_args.extend(["--attention-backend", str(server_args.attention_backend)])

    if server_args.speculative_draft_attention_backend:
        other_args.extend(
            [
                "--speculative-draft-attention-backend",
                str(server_args.speculative_draft_attention_backend),
            ]
        )

    if server_args.context_length:
        other_args.extend(["--context-length", str(server_args.context_length)])

    if server_args.max_total_tokens is not None:
        other_args.extend(["--max-total-tokens", str(server_args.max_total_tokens)])

    if server_args.allow_auto_truncate:
        other_args.append("--allow-auto-truncate")

    if server_args.dtype:
        other_args.extend(["--dtype", str(server_args.dtype)])

    return other_args


def main(args: argparse.Namespace, server_args: ServerArgs) -> None:
    base_url = f"http://127.0.0.1:{args.port}"

    if (
        any(a in ("DFLASH", "DFLASH_TREE") for a in args.algorithms)
        and server_args.speculative_draft_model_path is None
    ):
        raise ValueError(
            "DFlash benchmarks require --speculative-draft-model-path (a trained DFlash draft checkpoint)."
        )

    configs: List[Dict[str, Any]] = []
    for algo in args.algorithms:
        algo = str(algo).upper()
        if algo == "BASELINE":
            for batch_size in args.batch_size:
                for decode_len in args.decode_len:
                    configs.append(
                        {
                            "algorithm": "BASELINE",
                            "batch_size": int(batch_size),
                            "decode_len": int(decode_len),
                        }
                    )
            continue

        if algo == "DFLASH":
            for batch_size in args.batch_size:
                for decode_len in args.decode_len:
                    for block_size in args.block_size:
                        configs.append(
                            {
                                "algorithm": "DFLASH",
                                "batch_size": int(batch_size),
                                "decode_len": int(decode_len),
                                "block_size": int(block_size),
                            }
                        )
            continue

        if algo == "DFLASH_TREE":
            for batch_size in args.batch_size:
                for decode_len in args.decode_len:
                    for block_size in args.block_size:
                        for spec_steps in args.spec_steps:
                            if int(spec_steps) <= 0 or int(spec_steps) >= int(block_size):
                                continue
                            for tree_topk in args.tree_topk:
                                for num_verify_tokens in args.num_verify_tokens:
                                    candidate_count = int(tree_topk) + max(
                                        0, int(spec_steps) - 1
                                    ) * (int(tree_topk) ** 2)
                                    max_verify_tokens = 1 + candidate_count
                                    if int(num_verify_tokens) > max_verify_tokens:
                                        continue
                                    configs.append(
                                        {
                                            "algorithm": "DFLASH_TREE",
                                            "batch_size": int(batch_size),
                                            "decode_len": int(decode_len),
                                            "block_size": int(block_size),
                                            "spec_steps": int(spec_steps),
                                            "tree_topk": int(tree_topk),
                                            "num_verify_tokens": int(num_verify_tokens),
                                        }
                                    )
            continue

        raise ValueError(f"Unknown algorithm: {algo!r}")

    if args.end is None:
        args.end = len(configs)

    _node0_print(server_args, f"Total configs: {len(configs)} (running [{args.start}, {args.end}))")
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for idx in range(int(args.start), int(args.end)):
        cfg = configs[idx]
        batch_size = int(cfg["batch_size"])
        decode_len = int(cfg["decode_len"])

        other_args = _base_other_args(server_args, batch_size=batch_size)

        if cfg["algorithm"] != "BASELINE":
            other_args.extend(
                [
                    "--speculative-draft-model-path",
                    str(server_args.speculative_draft_model_path),
                    "--speculative-algorithm",
                    str(cfg["algorithm"]),
                ]
            )

        if cfg["algorithm"] == "DFLASH":
            other_args.extend(
                [
                    "--speculative-dflash-block-size",
                    str(cfg["block_size"]),
                ]
            )

        if cfg["algorithm"] == "DFLASH_TREE":
            other_args.extend(
                [
                    "--speculative-dflash-block-size",
                    str(cfg["block_size"]),
                    "--speculative-num-steps",
                    str(cfg["spec_steps"]),
                    "--speculative-eagle-topk",
                    str(cfg["tree_topk"]),
                    "--speculative-num-draft-tokens",
                    str(cfg["num_verify_tokens"]),
                ]
            )

        _node0_print(server_args, f"[{idx}] Launch+bench: {json.dumps(cfg)}")

        bench = _launch_and_bench(
            args=args,
            server_args=server_args,
            base_url=base_url,
            other_args=other_args,
            batch_size=batch_size,
            decode_len=decode_len,
        )

        record = {
            **cfg,
            "tp_size": int(server_args.tp_size),
            "context_length": int(server_args.context_length or 0),
            "mem_fraction_static": float(server_args.mem_fraction_static),
            "attention_backend": str(server_args.attention_backend or ""),
            "sampling": _build_sampling_params(args, decode_len=decode_len),
            "bench": bench,
        }

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        accept_len = bench.get("accept_length") or 1.0
        step_time = bench.get("step_time_s_pctl")
        output_tok_s = bench.get("output_tok_s")
        if step_time:
            score = accept_len / step_time
        else:
            score = output_tok_s

        if score is not None:
            scored.append((float(score), cfg))

        _node0_print(
            server_args,
            f"[{idx}] Done: alg={cfg['algorithm']} bs={batch_size} decode={decode_len} "
            f"accept={accept_len} step_p{args.step_time_percentile}={step_time} output_tok_s={output_tok_s} score={score}",
        )

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        _node0_print(server_args, "Top configs by score (accept_len / step_time):")
        for rank, (score, cfg) in enumerate(scored[: min(10, len(scored))]):
            _node0_print(server_args, f"  #{rank+1}: score={score:.4f} cfg={json.dumps(cfg)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    parser.add_argument("--port", type=int, default=20000)

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=("DFLASH", "DFLASH_TREE"),
        help="Which server modes to sweep: BASELINE, DFLASH, DFLASH_TREE",
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=(1, 8))
    parser.add_argument("--decode-len", type=int, nargs="+", default=(8192,))
    parser.add_argument("--num-prompts", type=int, default=16)

    parser.add_argument("--block-size", type=int, nargs="+", default=(16,))
    parser.add_argument("--spec-steps", type=int, nargs="+", default=(15,))
    parser.add_argument("--tree-topk", type=int, nargs="+", default=(4,))
    parser.add_argument("--num-verify-tokens", type=int, nargs="+", default=(16,))

    parser.add_argument("--sampling-temperature", type=float, default=0.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-min-p", type=float, default=0.0)
    parser.add_argument("--sampling-top-k", type=int, default=-1)
    parser.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")

    parser.add_argument("--step-time-percentile", type=float, default=20.0)
    parser.add_argument("--shutdown-sleep-s", type=float, default=5.0)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=str, default="dflash_sweep.jsonl")

    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    main(args, server_args)
