#!/usr/bin/env python3

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


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


@register_model("local-generate")
class LocalGenerateAPI(TemplateAPI):
    """lm-eval adapter for SGLang's native /generate endpoint."""

    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            verify_certificate=verify_certificate,
            ca_cert_path=ca_cert_path,
            auth_token=auth_token,
            **kwargs,
        )

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs=None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        gen_kwargs = copy.deepcopy(gen_kwargs or {})
        if generate:
            if "max_tokens" in gen_kwargs:
                max_new_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_new_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0.0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            sampling_params = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": gen_kwargs.pop("top_p", 1.0),
                "top_k": gen_kwargs.pop("top_k", -1),
                "min_p": gen_kwargs.pop("min_p", 0.0),
                "frequency_penalty": gen_kwargs.pop("frequency_penalty", 0.0),
                "presence_penalty": gen_kwargs.pop("presence_penalty", 0.0),
                "ignore_eos": gen_kwargs.pop("ignore_eos", False),
                "stop": stop,
            }
            sampling_params.update(gen_kwargs)
            payload = {
                "input_ids": messages,
                "sampling_params": sampling_params,
            }
            payload.update(kwargs)
            return payload

        payload = {
            "input_ids": messages,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "return_text_in_logprobs": True,
            "top_logprobs_num": 1,
        }
        payload.update(kwargs)
        return payload

    @staticmethod
    def parse_logprobs(outputs, tokens=None, ctxlens=None, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        res = []
        for out, ctxlen in zip(outputs, ctxlens):
            meta = out["meta_info"]
            input_logprobs = meta["input_token_logprobs"]
            score = sum(float(x[0]) for x in input_logprobs[ctxlen:])
            res.append((score, True))
        return res

    @staticmethod
    def parse_generations(outputs, **kwargs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [out["text"] for out in outputs]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an SGLang server and run lm-eval against a GPT-OSS CARE checkpoint."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--tokenizer-path",
        default="/workspace/offload_root/gpt-oss-120b",
        help="Local tokenizer directory for GPT-OSS evals.",
    )
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.95)
    parser.add_argument("--attention-backend", default="flashmla")
    parser.add_argument("--moe-runner-backend", default="triton")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--disable-cuda-graph", action="store_true")
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
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo_root = Path(__file__).resolve().parent.parent
    python_path = str(repo_root / "python")
    existing_python_path = env.get("PYTHONPATH", "")
    if python_path not in existing_python_path.split(":"):
        env["PYTHONPATH"] = (
            python_path + (":" + existing_python_path if existing_python_path else "")
        )
    ld_library_paths = [
        "/workspace/cublas_compat",
        "/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib",
        "/venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        "/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ]
    existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [p for p in existing_ld_library_path.split(":") if p]
    for path in ld_library_paths:
        if path not in ld_parts:
            ld_parts.insert(0, path)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
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
        "--page-size",
        str(args.page_size),
        "--dtype",
        str(args.dtype),
        "--kv-cache-dtype",
        str(args.kv_cache_dtype),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--attention-backend",
        args.attention_backend,
        "--moe-runner-backend",
        args.moe_runner_backend,
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        str(args.max_running_requests),
        "--trust-remote-code",
    ]
    if args.tokenizer_path:
        cmd.extend(["--tokenizer-path", args.tokenizer_path])
    if args.tokenizer_mode:
        cmd.extend(["--tokenizer-mode", args.tokenizer_mode])
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
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
        "base_url": f"{base_url}/generate",
        "num_concurrent": args.num_concurrent,
        "tokenizer_backend": "huggingface",
    }
    if args.tokenizer_path:
        model_args["tokenizer"] = args.tokenizer_path
    return lm_eval.simple_evaluate(
        model="local-generate",
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
