"""
Modal benchmark harness for GPT-OSS-120B GQA "DSA" (sparse decode via FA3 page_table).

Design goals:
  - Prefill stays dense; sparse only activates on decode for full-attn layers.
  - Saturate the GPU with concurrent requests (default: 8) + CUDA graphs (default: bs=8).
  - Allow auto-truncate so we can request 65k/131k prompts even if the server's max
    context is lower under the chosen max_running_requests/cuda_graph_bs.
  - Run the same benchmark matrix for BF16 KV cache and FP8_E4M3 KV cache.

Usage (examples):
  python -m modal run benchmark/gpt_oss/gqa_dsa/modal_bench.py --prompt-len 65000
  python -m modal run benchmark/gpt_oss/gqa_dsa/modal_bench.py --prompt-len 131000 --allow-auto-truncate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import modal


BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"

_DEFAULT_FLASH_ATTN_WHEEL_URL = os.environ.get("DEFAULT_FLASH_ATTN_WHEEL_URL", "")
_SGL_KERNEL_WHEEL_URL = os.environ.get("SGL_KERNEL_WHEEL_URL", "")

# Note: keep this list conservative; the user pins exact deps elsewhere.
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .run_commands(
        [
            "apt-get update && apt-get install -y --no-install-recommends git git-lfs curl wget libnuma1 libnuma-dev build-essential && rm -rf /var/lib/apt/lists/*",
            "python -m pip install --upgrade pip setuptools wheel",
            "python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1",
            # Optional: allow users to inject a prebuilt wheel URL for FA3.
            *(
                [f"python -m pip install --no-deps {_DEFAULT_FLASH_ATTN_WHEEL_URL}"]
                if _DEFAULT_FLASH_ATTN_WHEEL_URL
                else []
            ),
            "python -m pip install fastapi uvicorn orjson pydantic requests packaging numpy tqdm psutil httpx",
            "python -m pip install 'transformers==4.57.1' 'huggingface_hub==0.35.0' 'tokenizers>=0.22.0,<0.23' safetensors accelerate sentencepiece hf_transfer",
            "python -m pip install pybase64 msgspec setproctitle loguru compressed-tensors torchao==0.9.0",
            # Needed for GPT-OSS MXFP4 MoE runner paths.
            "python -m pip install --no-deps git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels",
            # flashinfer is required by upstream MXFP4 MoE modules even if we use FA3 attention.
            "python -m pip install flashinfer-python==0.6.7.post2",
            *(
                [f"python -m pip install --no-deps {_SGL_KERNEL_WHEEL_URL}"]
                if _SGL_KERNEL_WHEEL_URL
                else []
            ),
            "mkdir -p /root/tiktoken_encodings",
            "wget -O /root/tiktoken_encodings/o200k_base.tiktoken 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'",
            "wget -O /root/tiktoken_encodings/cl100k_base.tiktoken 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'",
        ]
    )
    .run_commands(
        [
            # Install SGLang from the fork/branch that contains the GPT-OSS GQA DSA work.
            "rm -rf /root/sglang && git clone --depth 1 --branch gpt-oss-dsa https://github.com/radna0/sglang.git /root/sglang",
            "python -m pip install -e /root/sglang[all]",
        ]
    )
)

app = modal.App("gpt-oss-gqa-dsa-bench")


@dataclass
class BenchResult:
    enable_dsa: bool
    kv_cache_dtype: str
    topk_source: str
    topk: int
    prompt_len_req: int
    prompt_len_effective: Optional[int]
    gen_len: int
    concurrent_requests: int
    cuda_graph_bs: int
    allow_auto_truncate: bool
    wall_s: float
    completion_tokens: int
    toks_per_s: float


_TRUNC_RE = re.compile(
    r"Truncated\.\s+len\(req\.origin_input_ids\)=(\d+),\s+max_req_input_len=(\d+)\."
)


def _load_tokenizer(model_id_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)


def _make_prompt(tokenizer, target_tokens: int) -> str:
    # Repeat one common token so the prompt length is deterministic in tokens.
    ids = tokenizer.encode(" hello", add_special_tokens=False)
    token_id = ids[0] if ids else 0
    return tokenizer.decode([token_id] * target_tokens)


async def _one_request(
    client,
    base_url: str,
    model: str,
    prompt: str,
    gen_len: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_len,
        "temperature": temperature,
        "stream": False,
    }
    r = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


async def _run_concurrent(
    base_url: str,
    model: str,
    prompt: str,
    gen_len: int,
    concurrent_requests: int,
) -> tuple[list[dict[str, Any]], float]:
    import httpx

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        out = await asyncio.gather(
            *[
                _one_request(
                    client=client,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    gen_len=gen_len,
                )
                for _ in range(concurrent_requests)
            ]
        )
    t1 = time.time()
    return out, (t1 - t0)


def _extract_usage_tokens(resp: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    usage = resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return prompt_tokens, completion_tokens


def _start_server(
    *,
    model_path: str,
    port: int,
    enable_dsa: bool,
    topk_source: Literal["recent", "indexer"],
    topk: int,
    kv_cache_dtype: Literal["bf16", "fp8_e4m3"],
    max_running_requests: int,
    cuda_graph_bs: int,
    allow_auto_truncate: bool,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "1")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--attention-backend",
        "fa3",
        "--moe-backend",
        "triton_kernel",
        "--max-running-requests",
        str(max_running_requests),
        "--cuda-graph-bs",
        str(cuda_graph_bs),
    ]
    if allow_auto_truncate:
        cmd.append("--allow-auto-truncate")
    if enable_dsa:
        cmd += [
            "--enable-gpt-oss-gqa-dsa",
            "--gpt-oss-dsa-topk-source",
            topk_source,
            "--gpt-oss-dsa-index-topk",
            str(topk),
        ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    # Use a process group so we can SIGTERM the whole tree.
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def _ensure_model_local(model_id_or_path: str) -> str:
    # If the caller passed an HF repo id, download it locally with excludes.
    # This avoids pulling `metal/*` and `original/*`.
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    if "/" not in model_id_or_path:
        return model_id_or_path

    local_dir = f"/root/hf_models/{model_id_or_path.replace('/', '__')}"
    if os.path.isdir(local_dir) and any(Path(local_dir).glob("**/*.safetensors")):
        return local_dir

    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # The user requested excluding these large folders.
    subprocess.run(
        [
            "hf",
            "download",
            model_id_or_path,
            "--exclude",
            "metal/*",
            "--exclude",
            "original/*",
            "--local-dir",
            local_dir,
        ],
        env=env,
        check=True,
    )
    return local_dir


def _stop_server(p: subprocess.Popen):
    if p.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        else:
            p.terminate()
        p.wait(timeout=30)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except Exception:
            pass


def _parse_effective_prompt_len_from_log(log_text: str) -> Optional[int]:
    # If truncation happened, infer the effective prompt length.
    # If multiple requests truncated, choose the max_req_input_len (the actual cap).
    matches = list(_TRUNC_RE.finditer(log_text))
    if not matches:
        return None
    return max(int(m.group(2)) for m in matches)


def _wait_server_ready(base_url: str, timeout_s: int = 600) -> None:
    import requests

    t0 = time.time()
    last_err: Optional[Exception] = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Server not ready after {timeout_s}s: {last_err}")


def _bench_once(
    *,
    model_id_or_path: str,
    prompt_len: int,
    gen_len: int,
    concurrent_requests: int,
    cuda_graph_bs: int,
    allow_auto_truncate: bool,
    enable_dsa: bool,
    topk_source: Literal["recent", "indexer"],
    topk: int,
    kv_cache_dtype: Literal["bf16", "fp8_e4m3"],
    workdir: Path,
) -> BenchResult:
    port = 30000
    base_url = f"http://127.0.0.1:{port}"
    log_path = workdir / f"server_enable_dsa={int(enable_dsa)}_kv={kv_cache_dtype}_topk_source={topk_source}_prompt={prompt_len}.log"

    model_local = _ensure_model_local(model_id_or_path)
    model_name_for_openai = model_local
    server = _start_server(
        model_path=model_local,
        port=port,
        enable_dsa=enable_dsa,
        topk_source=topk_source,
        topk=topk,
        kv_cache_dtype=kv_cache_dtype,
        max_running_requests=max(concurrent_requests, cuda_graph_bs),
        cuda_graph_bs=cuda_graph_bs,
        allow_auto_truncate=allow_auto_truncate,
        log_path=log_path,
    )
    try:
        _wait_server_ready(base_url)

        tokenizer = _load_tokenizer(model_local)
        prompt = _make_prompt(tokenizer, prompt_len)

        # Warmup: one short request to trigger compilation/cudagraph capture.
        asyncio.run(
            _run_concurrent(
                base_url=base_url,
                model=model_name_for_openai,
                prompt=prompt,
                gen_len=min(16, gen_len),
                concurrent_requests=1,
            )
        )

        resps, wall_s = asyncio.run(
            _run_concurrent(
                base_url=base_url,
                model=model_name_for_openai,
                prompt=prompt,
                gen_len=gen_len,
                concurrent_requests=concurrent_requests,
            )
        )

        completion_tokens_total = 0
        prompt_tokens_eff: Optional[int] = None
        for r in resps:
            p_toks, c_toks = _extract_usage_tokens(r)
            if c_toks is not None:
                completion_tokens_total += int(c_toks)
            if p_toks is not None:
                prompt_tokens_eff = int(p_toks)

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        prompt_len_from_log = _parse_effective_prompt_len_from_log(log_text)
        if prompt_len_from_log is not None:
            prompt_tokens_eff = prompt_len_from_log

        toks_per_s = (completion_tokens_total / wall_s) if wall_s > 0 else 0.0
        return BenchResult(
            enable_dsa=enable_dsa,
            kv_cache_dtype=kv_cache_dtype,
            topk_source=topk_source,
            topk=topk,
            prompt_len_req=prompt_len,
            prompt_len_effective=prompt_tokens_eff,
            gen_len=gen_len,
            concurrent_requests=concurrent_requests,
            cuda_graph_bs=cuda_graph_bs,
            allow_auto_truncate=allow_auto_truncate,
            wall_s=wall_s,
            completion_tokens=completion_tokens_total,
            toks_per_s=toks_per_s,
        )
    finally:
        _stop_server(server)


@app.function(
    gpu=modal.gpu.H100(count=1),
    image=image,
    timeout=60 * 60 * 6,
    cpu=16,
    memory=64 * 1024,
)
def run(
    prompt_len: int = 65000,
    gen_len: int = 2048,
    topk: int = 2048,
    topk_source: str = "recent",
    concurrent_requests: int = 8,
    cuda_graph_bs: int = 8,
    allow_auto_truncate: bool = True,
    model_path: str = "openai/gpt-oss-120b",
    workdir: str = "/root/out_gpt_oss_gqa_dsa_bench",
) -> None:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "meta": {
            "model_path": model_path,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "topk": topk,
            "topk_source": topk_source,
            "concurrent_requests": concurrent_requests,
            "cuda_graph_bs": cuda_graph_bs,
            "allow_auto_truncate": allow_auto_truncate,
        },
        "runs": [],
    }

    for kv in ("bf16", "fp8_e4m3"):
        dense = _bench_once(
            model_id_or_path=model_path,
            prompt_len=prompt_len,
            gen_len=gen_len,
            concurrent_requests=concurrent_requests,
            cuda_graph_bs=cuda_graph_bs,
            allow_auto_truncate=allow_auto_truncate,
            enable_dsa=False,
            topk_source=topk_source,  # ignored
            topk=topk,
            kv_cache_dtype=kv,  # type: ignore[arg-type]
            workdir=work,
        )
        dsa = _bench_once(
            model_id_or_path=model_path,
            prompt_len=prompt_len,
            gen_len=gen_len,
            concurrent_requests=concurrent_requests,
            cuda_graph_bs=cuda_graph_bs,
            allow_auto_truncate=allow_auto_truncate,
            enable_dsa=True,
            topk_source=topk_source,  # type: ignore[arg-type]
            topk=topk,
            kv_cache_dtype=kv,  # type: ignore[arg-type]
            workdir=work,
        )
        speedup = (dsa.toks_per_s / dense.toks_per_s) if dense.toks_per_s > 0 else None
        out["runs"].append(
            {
                "dense": asdict(dense),
                "dsa": asdict(dsa),
                "speedup_ratio": speedup,
            }
        )

    out_path = work / f"bench_prompt={prompt_len}_gen={gen_len}_topk={topk}_concur={concurrent_requests}_cudagraph={cuda_graph_bs}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=65000)
    p.add_argument("--gen-len", type=int, default=2048)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--topk-source", type=str, default="recent")
    p.add_argument("--concurrent-requests", type=int, default=8)
    p.add_argument("--cuda-graph-bs", type=int, default=8)
    p.add_argument("--allow-auto-truncate", action="store_true")
    p.add_argument("--model-path", type=str, default="openai/gpt-oss-120b")
    p.add_argument("--workdir", type=str, default="/root/out_gpt_oss_gqa_dsa_bench")
    args = p.parse_args()

    run.remote(
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        topk=args.topk,
        topk_source=args.topk_source,
        concurrent_requests=args.concurrent_requests,
        cuda_graph_bs=args.cuda_graph_bs,
        allow_auto_truncate=args.allow_auto_truncate,
        model_path=args.model_path,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    main()
