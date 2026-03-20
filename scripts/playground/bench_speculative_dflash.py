"""
Quick DFLASH vs baseline benchmark harness (fail-fast friendly).

Design goals (for our repo workflow):
  - Single script to benchmark *exactly one* config quickly (no long multi-matrix sweeps).
  - Works with long decode + concurrency presets (ctx131072 / decode65536) when you *choose* to run them,
    but also supports short decode smoke runs to avoid wasting 30+ minutes.

Example (short smoke):
  python3 scripts/playground/bench_speculative_dflash.py \\
    --model-path /path/to/gpt-oss-120b \\
    --draft-model-path /path/to/dflash_draft_ckpt \\
    --attention-backend fa3 \\
    --draft-attention-backend fa3 \\
    --context-length 8192 \\
    --decode-len 2048 \\
    --concurrency 1

Example (gold regime, long decode):
  python3 scripts/playground/bench_speculative_dflash.py \\
    --model-path /path/to/gpt-oss-120b \\
    --draft-model-path /path/to/dflash_draft_ckpt \\
    --attention-backend fa3 \\
    --draft-attention-backend fa3 \\
    --context-length 131072 \\
    --decode-len 65536 \\
    --concurrency 8 \\
    --temperature 1.0 --top-p 1.0 --top-k 50 --min-p 0.02 \\
    --cuda-graph
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import benchmark, set_global_args
from sglang.benchmark.datasets import DatasetRow
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


PROMPTS = [
    "Write a short Python function that computes fibonacci(n).",
    "Explain what speculative decoding is in 2 sentences.",
]


@dataclass(frozen=True)
class BenchResult:
    accept_length: float
    step_time_s: float
    output_tok_s: float
    avg_output_tokens: float


def _bench_one(
    *,
    base_url: str,
    tokenizer,
    num_prompts: int,
    concurrency: int,
    decode_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> BenchResult:
    padded = (PROMPTS * ((num_prompts + len(PROMPTS) - 1) // len(PROMPTS)))[:num_prompts]
    reqs: List[DatasetRow] = [DatasetRow(p, 0, int(decode_len)) for p in padded]

    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        return_routed_experts=False,
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
        output_details=False,
    )
    set_global_args(args)

    extra_request_body = {
        "sampling_params": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "max_new_tokens": int(decode_len),
            # Always let the server run to max_new_tokens for throughput comparisons.
            "ignore_eos": True,
        }
    }

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
            profile=None,
        )
    )

    if results["completed"] != len(reqs):
        raise RuntimeError(
            f"Benchmark incomplete: completed={results['completed']} expected={len(reqs)}"
        )

    accept_length = float(results["accept_length"] or 1.0)
    avg_output_tokens = float(results["total_output_tokens"]) / float(results["completed"])

    server_info = requests.get(base_url + "/get_server_info", timeout=60).json()
    # Use 20th percentile like the upstream script (more robust than median for graphs warmup).
    step_time_s = float(
        np.percentile(
            server_info["internal_states"][0]["step_time_dict"][str(int(concurrency))],
            20,
        )
    )
    output_tok_s = (1.0 / step_time_s) * accept_length
    return BenchResult(
        accept_length=round(accept_length, 3),
        step_time_s=round(step_time_s, 6),
        output_tok_s=round(output_tok_s, 3),
        avg_output_tokens=round(avg_output_tokens, 3),
    )


def _launch(
    *,
    model_path: str,
    tp_size: int,
    attention_backend: str,
    context_length: int,
    cuda_graph: bool,
    speculative: bool,
    draft_model_path: Optional[str],
    draft_attention_backend: Optional[str],
    dflash_block_size: Optional[int],
    port: int,
) -> tuple[object, str]:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        str(int(tp_size)),
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--context-length",
        str(int(context_length)),
        "--attention-backend",
        str(attention_backend),
    ]
    if cuda_graph:
        cmd.append("--cuda-graph")
    else:
        cmd.append("--disable-cuda-graph")

    if speculative:
        if not draft_model_path:
            raise ValueError("draft_model_path is required for DFLASH benchmark.")
        cmd += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            draft_model_path,
        ]
        if draft_attention_backend:
            cmd += ["--speculative-draft-attention-backend", str(draft_attention_backend)]
        if dflash_block_size is not None:
            cmd += ["--speculative-dflash-block-size", str(int(dflash_block_size))]

    proc = popen_launch_server(cmd)
    base_url = f"http://127.0.0.1:{int(port)}"
    # Wait until server is up.
    start = time.time()
    while True:
        try:
            r = requests.get(base_url + "/get_model_info", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        if time.time() - start > DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH:
            kill_process_tree(proc.pid)
            raise RuntimeError("Server launch timeout.")
        time.sleep(1)
    return proc, base_url


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preset",
        default=None,
        choices=["smoke_det", "smoke_sampling", "gold_det", "gold_sampling"],
        help="Convenience preset for (ctx/decode/concurrency/sampling). Explicit flags still override after preset.",
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-model-path", required=True, help="DFLASH draft checkpoint dir")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument("--draft-attention-backend", default="fa3")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--dflash-block-size", type=int, default=None)
    p.add_argument("--decode-len", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=2)
    p.add_argument("--port", type=int, default=20000)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--min-p", type=float, default=0.0)

    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-dflash", action="store_true")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    # Presets (applied first; explicit flags can still override afterwards by user choice).
    if args.preset == "smoke_det":
        args.context_length = 8192
        args.decode_len = 2048
        args.concurrency = 1
        args.temperature, args.top_k, args.min_p, args.top_p = 0.0, 1, 0.0, 1.0
    elif args.preset == "smoke_sampling":
        args.context_length = 8192
        args.decode_len = 2048
        args.concurrency = 1
        args.temperature, args.top_k, args.min_p, args.top_p = 1.0, 50, 0.02, 1.0
    elif args.preset == "gold_det":
        args.context_length = 131072
        args.decode_len = 65536
        args.concurrency = 8
        args.temperature, args.top_k, args.min_p, args.top_p = 0.0, 1, 0.0, 1.0
        args.cuda_graph = True
    elif args.preset == "gold_sampling":
        args.context_length = 131072
        args.decode_len = 65536
        args.concurrency = 8
        args.temperature, args.top_k, args.min_p, args.top_p = 1.0, 50, 0.02, 1.0
        args.cuda_graph = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    out: dict = {
        "regime": {
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "tp_size": args.tp_size,
            "attention_backend": args.attention_backend,
            "draft_attention_backend": args.draft_attention_backend,
            "context_length": args.context_length,
            "decode_len": args.decode_len,
            "concurrency": args.concurrency,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
            },
            "cuda_graph": bool(args.cuda_graph),
        }
    }

    baseline_proc = None
    dflash_proc = None
    try:
        if not args.skip_baseline:
            baseline_proc, baseline_url = _launch(
                model_path=args.model_path,
                tp_size=args.tp_size,
                attention_backend=args.attention_backend,
                context_length=args.context_length,
                cuda_graph=args.cuda_graph,
                speculative=False,
                draft_model_path=None,
                draft_attention_backend=None,
                dflash_block_size=None,
                port=args.port,
            )
            base = _bench_one(
                base_url=baseline_url,
                tokenizer=tokenizer,
                num_prompts=args.num_prompts,
                concurrency=args.concurrency,
                decode_len=args.decode_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
            out["baseline"] = base.__dict__

        if not args.skip_dflash:
            dflash_proc, dflash_url = _launch(
                model_path=args.model_path,
                tp_size=args.tp_size,
                attention_backend=args.attention_backend,
                context_length=args.context_length,
                cuda_graph=args.cuda_graph,
                speculative=True,
                draft_model_path=args.draft_model_path,
                draft_attention_backend=args.draft_attention_backend,
                dflash_block_size=args.dflash_block_size,
                port=args.port + 1,
            )
            df = _bench_one(
                base_url=dflash_url,
                tokenizer=tokenizer,
                num_prompts=args.num_prompts,
                concurrency=args.concurrency,
                decode_len=args.decode_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
            out["dflash"] = df.__dict__

        if "baseline" in out and "dflash" in out:
            out["speedup"] = round(
                float(out["dflash"]["output_tok_s"]) / float(out["baseline"]["output_tok_s"]),
                4,
            )

        print(json.dumps(out, indent=2))
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        return 0
    finally:
        if baseline_proc is not None:
            kill_process_tree(baseline_proc.pid)
        if dflash_proc is not None:
            kill_process_tree(dflash_proc.pid)


if __name__ == "__main__":
    raise SystemExit(main())
