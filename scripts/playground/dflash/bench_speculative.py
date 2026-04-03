"""
Quick DFLASH vs baseline benchmark harness (fail-fast friendly).

Design goals (for our repo workflow):
  - Single script to benchmark *exactly one* config quickly (no long multi-matrix sweeps).
  - Works with long decode + concurrency presets (ctx131072 / decode65536) when you *choose* to run them,
    but also supports short decode smoke runs to avoid wasting 30+ minutes.

Example (short smoke):
  python3 scripts/playground/dflash/bench_speculative.py \\
    --model-path /path/to/gpt-oss-120b \\
    --draft-model-path /path/to/dflash_draft_ckpt \\
    --attention-backend fa3 \\
    --draft-attention-backend fa3 \\
    --context-length 8192 \\
    --decode-len 2048 \\
    --concurrency 1


  python3 bench_speculative.py \
    --model-path /path/to/gpt-oss-120b \
    --speculative-draft-model-path /path/to/dflash-draft-ckpt \
    --prefill-attention-backend fa3 \
    --decode-attention-backend fa3 \
    --speculative-draft-attention-backend fa3 \
    --page-size 128 \
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
from sglang.bench_serving import DatasetRow
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
    reqs: List[DatasetRow] = [
        DatasetRow(prompt=p, prompt_len=0, output_len=int(decode_len))
        for p in padded
    ]

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
    for attr in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
        "speculative_draft_attention_backend",
    ):
        if getattr(server_args, attr, None) == "fa4":
            setattr(server_args, attr, "fa3")
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
            "prefill_attention_backend": str(
                getattr(server_args, "prefill_attention_backend", "") or ""
            ),
            "decode_attention_backend": str(
                getattr(server_args, "decode_attention_backend", "") or ""
            ),
            "speculative_draft_attention_backend": str(
                getattr(server_args, "speculative_draft_attention_backend", "") or ""
            ),
            "page_size": int(getattr(server_args, "page_size", 1) or 1),
            "kv_cache_dtype": str(getattr(server_args, "kv_cache_dtype", "") or ""),
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
    raise SystemExit(main())
