#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_BENCHMARK_TASKS = [
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
        description="Run a PPL-first CARE benchmark suite for GPT-OSS checkpoints."
    )
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--sglang-model-path", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--ppl-task", default="wikitext")
    parser.add_argument("--benchmark-tasks", default=",".join(DEFAULT_BENCHMARK_TASKS))
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-memory-per-gpu", default="78GiB")
    parser.add_argument("--max-cpu-memory", default="512GiB")
    parser.add_argument("--offload-folder", default="./offload")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--run-passkey", action="store_true")
    parser.add_argument("--passkey-contexts", default="4096,8192,16384,32768")
    parser.add_argument("--passkey-positions", default="start,middle,end")
    parser.add_argument("--passkey-num-samples", type=int, default=16)
    parser.add_argument("--passkey-max-new-tokens", type=int, default=16)
    parser.add_argument("--run-longbench-v2", action="store_true")
    parser.add_argument("--longbench-v2-dataset-path", default="THUDM/LongBench-v2:train")
    parser.add_argument("--longbench-v2-num-examples", type=int, default=100)
    parser.add_argument("--longbench-v2-categories", default=None)
    parser.add_argument("--longbench-v2-min-context-length", type=int, default=None)
    parser.add_argument("--longbench-v2-max-context-length", type=int, default=None)
    parser.add_argument("--longbench-v2-max-new-tokens", type=int, default=64)
    parser.add_argument("--run-sglang", action="store_true")
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--sglang-tp-size", type=int, default=8)
    parser.add_argument("--sglang-attention-backend", default="triton")
    parser.add_argument("--sglang-batch-size", default="auto")
    parser.add_argument("--sglang-num-concurrent", type=int, default=1)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("[cmd] " + " ".join(cmd) + "\n")
        log_file.flush()
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    hf_eval_script = repo_root / "scripts" / "run_gpt_oss_hf_lm_eval.py"
    passkey_eval_script = repo_root / "scripts" / "run_gpt_oss_hf_passkey_eval.py"
    longbench_v2_eval_script = repo_root / "scripts" / "run_gpt_oss_hf_longbench_v2.py"
    sglang_eval_script = repo_root / "scripts" / "run_gpt_oss_care_lm_eval.py"

    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    benchmark_tasks = [x.strip() for x in args.benchmark_tasks.split(",") if x.strip()]

    summary: dict[str, object] = {
        "hf_model_path": str(Path(args.hf_model_path).resolve()),
        "sglang_model_path": str(Path(args.sglang_model_path).resolve()) if args.sglang_model_path else None,
        "contexts": contexts,
        "ppl_task": args.ppl_task,
        "benchmark_tasks": benchmark_tasks,
    }

    ppl_results: dict[str, dict] = {}
    for context in contexts:
        output_path = out_root / "hf_long_context" / f"{args.ppl_task}_ml{context}.json"
        cmd = [
            sys.executable,
            str(hf_eval_script),
            "--model-path",
            str(Path(args.hf_model_path).resolve()),
            "--tasks",
            args.ppl_task,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.batch_size),
            "--max-batch-size",
            str(args.max_batch_size),
            "--max-memory-per-gpu",
            args.max_memory_per_gpu,
            "--max-cpu-memory",
            args.max_cpu_memory,
            "--offload-folder",
            str((out_root / "offload").resolve()),
            "--max-length",
            str(context),
            "--output-path",
            str(output_path),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.num_fewshot:
            cmd.extend(["--num-fewshot", str(args.num_fewshot)])
        if args.apply_chat_template:
            cmd.append("--apply-chat-template")
        _run(cmd, out_root / "hf_long_context" / f"{args.ppl_task}_ml{context}.log")
        ppl_results[str(context)] = _load_json(output_path)

    benchmark_output = out_root / "hf_benchmarks.json"
    benchmark_cmd = [
        sys.executable,
        str(hf_eval_script),
        "--model-path",
        str(Path(args.hf_model_path).resolve()),
        "--tasks",
        ",".join(benchmark_tasks),
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-memory-per-gpu",
        args.max_memory_per_gpu,
        "--max-cpu-memory",
        args.max_cpu_memory,
        "--offload-folder",
        str((out_root / "offload").resolve()),
        "--max-length",
        str(min(contexts) if contexts else 2048),
        "--output-path",
        str(benchmark_output),
    ]
    if args.limit is not None:
        benchmark_cmd.extend(["--limit", str(args.limit)])
    if args.num_fewshot:
        benchmark_cmd.extend(["--num-fewshot", str(args.num_fewshot)])
    if args.apply_chat_template:
        benchmark_cmd.append("--apply-chat-template")
    _run(benchmark_cmd, out_root / "hf_benchmarks.log")

    summary["hf_long_context"] = {
        context: payload.get("summary", {}) for context, payload in ppl_results.items()
    }
    summary["hf_benchmarks"] = _load_json(benchmark_output).get("summary", {})

    if args.run_passkey:
        passkey_output = out_root / "hf_passkey.json"
        passkey_cmd = [
            sys.executable,
            str(passkey_eval_script),
            "--model-path",
            str(Path(args.hf_model_path).resolve()),
            "--out-json",
            str(passkey_output),
            "--contexts",
            args.passkey_contexts,
            "--positions",
            args.passkey_positions,
            "--num-samples",
            str(args.passkey_num_samples),
            "--dtype",
            args.dtype,
            "--device",
            "cuda",
            "--max-memory-per-gpu",
            args.max_memory_per_gpu,
            "--max-cpu-memory",
            args.max_cpu_memory,
            "--offload-folder",
            str((out_root / "offload").resolve()),
            "--max-new-tokens",
            str(args.passkey_max_new_tokens),
        ]
        _run(passkey_cmd, out_root / "hf_passkey.log")
        summary["hf_passkey"] = _load_json(passkey_output).get("summary", {})

    if args.run_longbench_v2:
        longbench_output = out_root / "hf_longbench_v2.json"
        longbench_cmd = [
            sys.executable,
            str(longbench_v2_eval_script),
            "--model-path",
            str(Path(args.hf_model_path).resolve()),
            "--out-json",
            str(longbench_output),
            "--dataset-path",
            args.longbench_v2_dataset_path,
            "--num-examples",
            str(args.longbench_v2_num_examples),
            "--dtype",
            args.dtype,
            "--device",
            "cuda",
            "--max-memory-per-gpu",
            args.max_memory_per_gpu,
            "--max-cpu-memory",
            args.max_cpu_memory,
            "--offload-folder",
            str((out_root / "offload").resolve()),
            "--max-new-tokens",
            str(args.longbench_v2_max_new_tokens),
        ]
        if args.longbench_v2_categories:
            longbench_cmd.extend(["--categories", args.longbench_v2_categories])
        if args.longbench_v2_min_context_length is not None:
            longbench_cmd.extend(
                ["--min-context-length", str(args.longbench_v2_min_context_length)]
            )
        if args.longbench_v2_max_context_length is not None:
            longbench_cmd.extend(
                ["--max-context-length", str(args.longbench_v2_max_context_length)]
            )
        _run(longbench_cmd, out_root / "hf_longbench_v2.log")
        summary["hf_longbench_v2"] = _load_json(longbench_output).get("summary", {})

    if args.run_sglang:
        sglang_model_path = args.sglang_model_path or args.hf_model_path
        sglang_output = out_root / "sglang_benchmarks.json"
        sglang_cmd = [
            sys.executable,
            str(sglang_eval_script),
            "--model-path",
            str(Path(sglang_model_path).resolve()),
            "--host",
            args.sglang_host,
            "--port",
            str(args.sglang_port),
            "--tp-size",
            str(args.sglang_tp_size),
            "--attention-backend",
            args.sglang_attention_backend,
            "--batch-size",
            str(args.sglang_batch_size),
            "--num-concurrent",
            str(args.sglang_num_concurrent),
            "--tasks",
            ",".join(benchmark_tasks),
            "--out-json",
            str(sglang_output),
        ]
        if args.limit is not None:
            sglang_cmd.extend(["--limit", str(args.limit)])
        if args.num_fewshot:
            sglang_cmd.extend(["--num-fewshot", str(args.num_fewshot)])
        if args.apply_chat_template:
            sglang_cmd.append("--apply-chat-template")
        _run(sglang_cmd, out_root / "sglang_benchmarks.log")
        summary["sglang_benchmarks"] = _load_json(sglang_output).get("results", {})

    _save_json(out_root / "benchmark_suite_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
