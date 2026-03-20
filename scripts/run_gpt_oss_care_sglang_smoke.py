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

_DISALLOWED_GPTOSS_CARE_BACKENDS = {
    "flex_attention2",
    "flex_flash2",
    "flex_flash4",
    "flex_flash2_delegate_fa3",
    "triton",
    "flashinfer",
    "trtllm_mla",
    "cutlass_mla",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an SGLang server for a GPT-OSS CARE MLA checkpoint and run a single completion smoke."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--prompt", default="Write a short Python function that returns the square of an integer.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--skip-server", action="store_true")
    return parser.parse_args()


def _load_checkpoint_config(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_attention_backend(
    requested_backend: str, checkpoint_config: dict[str, Any]
) -> str:
    architectures = checkpoint_config.get("architectures") or []
    is_gpt_oss_mla = "GptOssMlaForCausalLM" in architectures
    mla_rope_num_kv_heads = int(checkpoint_config.get("mla_rope_num_kv_heads", 1) or 1)
    uses_gpt_oss_native_mha_anchor = bool(
        is_gpt_oss_mla
        and (
            checkpoint_config.get("mla_attention_mode") == "mha_explicit"
            or mla_rope_num_kv_heads > 1
        )
    )
    kv_rank_schedule = checkpoint_config.get("kv_lora_rank_per_layer")
    has_dynamic_rank = isinstance(kv_rank_schedule, list) and len(set(kv_rank_schedule)) > 1
    layer_types = checkpoint_config.get("layer_types") or []
    has_sliding_layers = any(layer_type == "sliding_attention" for layer_type in layer_types)

    if requested_backend == "auto":
        if is_gpt_oss_mla:
            return "flashmla"
        return "flashmla"

    if requested_backend in _DISALLOWED_GPTOSS_CARE_BACKENDS:
        raise ValueError(
            "GPT-OSS CARE MLA smoke is locked to FlashMLA in this branch. "
            "Use --attention-backend flashmla."
        )

    return requested_backend


def _backend_metadata(
    selected_backend: str, checkpoint_config: dict[str, Any]
) -> dict[str, Any]:
    architectures = checkpoint_config.get("architectures") or []
    is_gpt_oss_mla = "GptOssMlaForCausalLM" in architectures
    mla_rope_num_kv_heads = int(checkpoint_config.get("mla_rope_num_kv_heads", 1) or 1)
    uses_gpt_oss_native_mha_anchor = bool(
        is_gpt_oss_mla
        and (
            checkpoint_config.get("mla_attention_mode") == "mha_explicit"
            or mla_rope_num_kv_heads > 1
        )
    )
    kv_rank_schedule = checkpoint_config.get("kv_lora_rank_per_layer")
    has_dynamic_rank = isinstance(kv_rank_schedule, list) and len(set(kv_rank_schedule)) > 1
    layer_types = checkpoint_config.get("layer_types") or []
    has_sliding_layers = any(layer_type == "sliding_attention" for layer_type in layer_types)

    backend_family = selected_backend
    experimental = False
    if is_gpt_oss_mla and selected_backend == "flashmla":
        reason = (
            "Selected flashmla because GPT-OSS CARE serving is locked to the actual MLA "
            "kernel path. GPT-OSS-specific sink and sliding-window parity is implemented here."
        )
    else:
        reason = "Using the explicitly requested backend."

    return {
        "is_gpt_oss_mla": is_gpt_oss_mla,
        "uses_gpt_oss_native_mha_anchor": uses_gpt_oss_native_mha_anchor,
        "has_dynamic_rank": has_dynamic_rank,
        "has_sliding_layers": has_sliding_layers,
        "backend_family": backend_family,
        "experimental": experimental,
        "reason": reason,
    }


def _should_disable_cuda_graph(
    selected_backend: str, checkpoint_config: dict[str, Any]
) -> bool:
    architectures = checkpoint_config.get("architectures") or []
    is_gpt_oss_mla = "GptOssMlaForCausalLM" in architectures
    uses_gpt_oss_native_mha_anchor = bool(
        is_gpt_oss_mla
        and (
            checkpoint_config.get("mla_attention_mode") == "mha_explicit"
            or int(checkpoint_config.get("mla_rope_num_kv_heads", 1) or 1) > 1
        )
    )
    kv_rank_schedule = checkpoint_config.get("kv_lora_rank_per_layer")
    has_dynamic_rank = isinstance(kv_rank_schedule, list) and len(set(kv_rank_schedule)) > 1
    layer_types = checkpoint_config.get("layer_types") or []
    has_sliding_layers = any(layer_type == "sliding_attention" for layer_type in layer_types)
    return bool(
        uses_gpt_oss_native_mha_anchor
        or (is_gpt_oss_mla and has_sliding_layers and selected_backend == "flashmla")
        or (is_gpt_oss_mla and has_dynamic_rank and selected_backend == "flashmla")
    )


def _should_disable_piecewise_cuda_graph(
    selected_backend: str, checkpoint_config: dict[str, Any]
) -> bool:
    return _should_disable_cuda_graph(selected_backend, checkpoint_config)


def _wait_for_server(base_url: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.ok:
                payload = response.json()
                if payload.get("data"):
                    return payload
                return payload
        except Exception as exc:  # pragma: no cover - operational path
            last_error = exc
        time.sleep(2)
    raise TimeoutError(
        f"SGLang server did not become ready within {timeout_s} seconds. last_error={last_error!r}"
    )


def _launch_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    python_path = str(repo_root / "python")
    existing_python_path = env.get("PYTHONPATH", "")
    if python_path not in existing_python_path.split(":"):
        env["PYTHONPATH"] = (
            python_path + (":" + existing_python_path if existing_python_path else "")
        )
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
        "--attention-backend",
        args.attention_backend,
        "--trust-remote-code",
    ]
    if getattr(args, "disable_cuda_graph", False):
        cmd.append("--disable-cuda-graph")
    print("[server]", " ".join(cmd), flush=True)

    stdout = None
    stderr = None
    if args.server_log:
        log_path = Path(args.server_log).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        stdout = handle
        stderr = subprocess.STDOUT

    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, text=True)


def _pick_model_id(models_payload: dict[str, Any], fallback: str) -> str:
    data = models_payload.get("data")
    if isinstance(data, list) and data:
        model_id = data[0].get("id")
        if model_id:
            return str(model_id)
    return fallback


def _run_completion(args: argparse.Namespace, base_url: str, model_id: str) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    response = requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return {
        "request": payload,
        "response": response.json(),
    }


def main() -> None:
    args = _parse_args()
    checkpoint_config = _load_checkpoint_config(args.model_path)
    args.attention_backend = _select_attention_backend(
        args.attention_backend, checkpoint_config
    )
    args.disable_cuda_graph = _should_disable_cuda_graph(
        args.attention_backend, checkpoint_config
    )
    if not args.disable_piecewise_cuda_graph:
        args.disable_piecewise_cuda_graph = _should_disable_piecewise_cuda_graph(
            args.attention_backend, checkpoint_config
        )
    backend_metadata = _backend_metadata(args.attention_backend, checkpoint_config)
    backend_metadata["disable_cuda_graph"] = bool(args.disable_cuda_graph)
    backend_metadata["disable_piecewise_cuda_graph"] = bool(
        args.disable_piecewise_cuda_graph
    )
    if args.attention_backend == "flashmla" and args.page_size != 64:
        print(
            "[info] FlashMLA requires page_size=64. Overriding the requested page size.",
            file=sys.stderr,
            flush=True,
        )
        args.page_size = 64
    if backend_metadata["experimental"]:
        print(
            f"[warning] attention backend '{args.attention_backend}' is experimental for GPT-OSS CARE MLA. "
            f"{backend_metadata['reason']}",
            file=sys.stderr,
            flush=True,
        )
    if args.disable_cuda_graph:
        if backend_metadata["uses_gpt_oss_native_mha_anchor"]:
            reason = (
                "the GPT-OSS native MHA-anchor path is not CUDA-graph-ready in SGLang yet"
            )
        elif (
            backend_metadata["backend_family"] == "flashmla"
            and backend_metadata["has_sliding_layers"]
        ):
            reason = (
                "FlashMLA sliding-window GPT-OSS MLA is running on the new sparse path and is not CUDA-graph-ready yet"
            )
        else:
            reason = (
                "FlashMLA does not yet support dynamic per-layer kv_lora_rank in its CUDA-graph path"
            )
        print(
            f"[info] Disabling CUDA graph because {reason}.",
            file=sys.stderr,
            flush=True,
        )
    if args.disable_piecewise_cuda_graph:
        if backend_metadata["uses_gpt_oss_native_mha_anchor"]:
            reason = "the GPT-OSS native MHA-anchor path is not piecewise-CUDA-graph-ready"
        elif (
            backend_metadata["backend_family"] == "flashmla"
            and backend_metadata["has_sliding_layers"]
        ):
            reason = "FlashMLA sliding-window GPT-OSS MLA"
        else:
            reason = "dynamic-rank FlashMLA"
        print(
            f"[info] Disabling piecewise CUDA graph because {reason}.",
            file=sys.stderr,
            flush=True,
        )
    base_url = f"http://{args.host}:{args.port}"
    process = None
    try:
        if not args.skip_server:
            process = _launch_server(args)
        models_payload = _wait_for_server(base_url, args.server_timeout_s)
        model_id = _pick_model_id(models_payload, Path(args.model_path).name)
        result = {
            "base_url": base_url,
            "model_id": model_id,
            "attention_backend": args.attention_backend,
            "attention_backend_metadata": backend_metadata,
            "tp_size": args.tp_size,
            "page_size": args.page_size,
            "kv_cache_dtype": args.kv_cache_dtype,
            "models": models_payload,
        }
        result.update(_run_completion(args, base_url, model_id))
        if args.out_json:
            out_path = Path(args.out_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - operational path
                process.kill()


if __name__ == "__main__":
    main()
