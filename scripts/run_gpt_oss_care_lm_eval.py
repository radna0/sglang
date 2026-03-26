#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


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
        description="Launch an SGLang server and run lm-eval against a GPT-OSS CARE checkpoint."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.95)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--server-timeout-s", type=int, default=900)
    return parser.parse_args()


def _wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready within {timeout_s} seconds.")


def _launch_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    if not args.model_path:
        raise ValueError("--model-path is required unless --skip-server is set.")

    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--attention-backend",
        args.attention_backend,
        "--trust-remote-code",
    ]
    if args.tokenizer_path:
        cmd.extend(["--tokenizer-path", args.tokenizer_path])
    if args.tokenizer_mode:
        cmd.extend(["--tokenizer-mode", args.tokenizer_mode])
    print("[server]", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, env=env)


def _run_lm_eval(args: argparse.Namespace, base_url: str) -> dict[str, Any]:
    try:
        import lm_eval
    except Exception as exc:
        raise RuntimeError(
            "lm_eval is not installed in this environment. Install lm-eval-harness first."
        ) from exc

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model_args = {
        "model": Path(args.model_path).name if args.model_path else "gpt-oss-care-mla",
        "base_url": f"{base_url}/v1/completions",
        "num_concurrent": args.num_concurrent,
    }
    return lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        batch_size=args.batch_size,
    )


def main() -> None:
    args = _parse_args()
    base_url = f"http://{args.host}:{args.port}"
    process = None
    try:
        if not args.skip_server:
            process = _launch_server(args)
            _wait_for_server(base_url, args.server_timeout_s)
        results = _run_lm_eval(args, base_url)
        if args.out_json:
            out_path = Path(args.out_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=True)
                f.write("\n")
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
