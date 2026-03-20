#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any

import lm_eval

from gpt_oss_hf_loader import prepare_hf_model_path


DEFAULT_TASKS = [
    "wikitext",
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
        description="Run a pure-HF lm-eval sweep for GPT-OSS checkpoints."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parallelize", dest="parallelize", action="store_true")
    parser.add_argument("--no-parallelize", dest="parallelize", action="store_false")
    parser.add_argument("--max-memory-per-gpu", default="78GiB")
    parser.add_argument("--max-cpu-memory", default="512GiB")
    parser.add_argument("--offload-folder", default="./offload")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logits-cache", action="store_true", default=False)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.set_defaults(parallelize=True)
    return parser.parse_args()


def _summarize(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        important = {}
        for key, value in metrics.items():
            if any(
                token in key
                for token in (
                    "acc",
                    "exact_match",
                    "word_perplexity",
                    "byte_perplexity",
                    "bits_per_byte",
                )
            ):
                important[key] = value
        summary[task_name] = important or metrics
    return summary


def _dist_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def main() -> None:
    args = _parse_args()
    rank = _dist_rank()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    output_path = Path(args.output_path).resolve() if args.output_path else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ref, trust_remote_code = prepare_hf_model_path(args.model_path)

    model_args = {
        "pretrained": model_ref,
        "dtype": args.dtype,
        "parallelize": bool(args.parallelize),
        "trust_remote_code": trust_remote_code,
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "max_cpu_memory": args.max_cpu_memory,
        "offload_folder": args.offload_folder,
        "max_length": args.max_length,
        "logits_cache": bool(args.logits_cache),
    }

    if rank == 0:
        print(
            json.dumps(
                {
                    "model": "hf",
                    "model_args": model_args,
                    "tasks": tasks,
                    "limit": args.limit,
                    "num_fewshot": args.num_fewshot,
                },
                indent=2,
            ),
            flush=True,
        )

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=bool(args.log_samples),
        apply_chat_template=bool(args.apply_chat_template),
    )

    if results is None:
        return

    payload = {
        "summary": _summarize(results),
        "raw": results,
    }
    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
