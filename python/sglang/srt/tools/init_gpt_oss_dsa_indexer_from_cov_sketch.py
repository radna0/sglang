"""
Initialize GPT-OSS GQA DSA indexer weights from precomputed covariance sketches.

Goal: replace DeepSeek-style DSA "Phase-1 warmup training" with a cheap, deterministic
conversion step using the same calibration samples we already run for CARE covariance.

This tool expects a lightweight sketch per layer, not a full (hidden, hidden) covariance:
  Y_l = C_l @ Omega, where:
    - C_l is the hidden-state covariance at that layer (E[x^T x])
    - Omega is a fixed random projection matrix (hidden, k)
  Y_l has shape (hidden, k) and is enough to form an approximate principal subspace.

The collector that produces Y_l should be run during covariance collection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch


def _load_config(model_dir: Path) -> Dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _is_full_attention_layer(cfg: Dict[str, Any], layer_id: int) -> bool:
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_id < len(layer_types):
        return layer_types[layer_id] == "full_attention"
    # Conservative default if the config doesn't expose per-layer types.
    return True


def _per_block_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (weight_fp8_e4m3fn, weight_scale_inv)
    The scale tensor is ue8m0 (power-of-2) style, as used by DeepGEMM block FP8.
    """
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    w2d = weight_bf16.to(torch.bfloat16)
    w2d = w2d.reshape(w2d.shape[0], w2d.shape[1])
    w_fp8, sf = per_block_cast_to_fp8(w2d)
    # Keep scale in fp32 for checkpoint compatibility.
    return w_fp8, sf.to(torch.float32)


def _maybe_cuda(x: torch.Tensor, enable_cuda: bool) -> torch.Tensor:
    if enable_cuda and torch.cuda.is_available():
        return x.cuda()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True, help="HF model dir (for config.json)")
    ap.add_argument(
        "--sketch-dir",
        type=str,
        required=True,
        help="Directory containing per-layer sketches, e.g. layer_00000.pt",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path (.pt or .safetensors) containing indexer-only params.",
    )
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument(
        "--q-lora-rank",
        type=int,
        default=1536,
        help="If > hidden, will be clamped to hidden_size.",
    )
    ap.add_argument(
        "--index-n-heads",
        type=int,
        default=0,
        help="0 means use config.num_attention_heads.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for any random init (weights_proj and any missing bases).",
    )
    ap.add_argument(
        "--enable-fp8",
        action="store_true",
        help="If set, emits fp8 weights + scale_inv for wk/wq_a/wq_b. Requires CUDA for float8 ops.",
    )
    ap.add_argument(
        "--layer-template",
        type=str,
        default="layer_{layer_id:05d}.pt",
        help="Sketch filename template inside --sketch-dir.",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))

    model_dir = Path(args.model_dir)
    sketch_dir = Path(args.sketch_dir)
    out_path = Path(args.out)

    cfg = _load_config(model_dir)
    num_layers = int(cfg.get("num_hidden_layers"))
    hidden = int(cfg.get("hidden_size"))
    num_heads = int(cfg.get("num_attention_heads"))
    head_dim = int(cfg.get("head_dim", 128))

    index_head_dim = int(args.index_head_dim)
    index_n_heads = int(args.index_n_heads) if int(args.index_n_heads) > 0 else int(num_heads)
    q_lora_rank = min(int(args.q_lora_rank), hidden)

    # Output state dict.
    state: Dict[str, torch.Tensor] = {}

    enable_cuda = bool(args.enable_fp8)
    if enable_cuda and not torch.cuda.is_available():
        raise RuntimeError("--enable-fp8 was set but CUDA is not available.")

    for layer_id in range(num_layers):
        if not _is_full_attention_layer(cfg, layer_id):
            continue

        sketch_path = sketch_dir / args.layer_template.format(layer_id=layer_id)
        if not sketch_path.exists():
            raise FileNotFoundError(f"Missing sketch for layer {layer_id}: {sketch_path}")

        obj = torch.load(sketch_path, map_location="cpu")
        if isinstance(obj, dict) and "Y" in obj:
            Y = obj["Y"]
        elif isinstance(obj, torch.Tensor):
            Y = obj
        else:
            raise TypeError(f"Unsupported sketch format in {sketch_path}: {type(obj)}")

        if Y.dim() != 2 or Y.shape[0] != hidden:
            raise ValueError(f"Bad sketch shape for layer {layer_id}: {tuple(Y.shape)} expected ({hidden}, k)")

        # Orthonormalize the sketch columns to get a usable basis.
        # Q: (hidden, k)
        Q, _ = torch.linalg.qr(Y.to(torch.float32), mode="reduced")
        Q = Q.to(torch.bfloat16)

        # wk: (index_head_dim, hidden)
        wk = Q[:, :index_head_dim].T.contiguous()

        # wq_a: (q_lora_rank, hidden)  and wq_b: (index_n_heads*index_head_dim, q_lora_rank)
        # If q_lora_rank > sketch rank, fall back to random for the missing rows.
        k = int(Q.shape[1])
        if q_lora_rank <= k:
            wq_a = Q[:, :q_lora_rank].T.contiguous()
            # Map back to hidden, then tile/trim to (index_n_heads*index_head_dim).
            wq_b_full = Q[:, :q_lora_rank].contiguous()
        else:
            # Hybrid: PCA for first k dims, then random for the remaining.
            pad = q_lora_rank - k
            wq_a = torch.empty((q_lora_rank, hidden), dtype=torch.bfloat16)
            wq_a[:k] = Q.T
            wq_a[k:] = torch.randn((pad, hidden), dtype=torch.bfloat16) * 0.02
            wq_b_full = torch.empty((hidden, q_lora_rank), dtype=torch.bfloat16)
            wq_b_full[:, :k] = Q
            wq_b_full[:, k:] = torch.randn((hidden, pad), dtype=torch.bfloat16) * 0.02

        out_q = index_n_heads * index_head_dim
        if out_q != hidden:
            # GPT-OSS: out_q often equals hidden (num_heads*head_dim == hidden). If not, tile.
            reps = (out_q + hidden - 1) // hidden
            wq_b_full = wq_b_full.repeat((reps, 1))[:out_q, :]
        wq_b = wq_b_full.contiguous()

        # weights_proj: BF16 (n_heads, hidden). Nonzero init is important.
        weights_proj = torch.randn((index_n_heads, hidden), dtype=torch.bfloat16) * (hidden**-0.5)

        prefix = f"model.layers.{layer_id}.self_attn.indexer"
        if args.enable_fp8:
            wk_fp8, wk_s = _per_block_fp8(_maybe_cuda(wk, enable_cuda))
            wq_a_fp8, wq_a_s = _per_block_fp8(_maybe_cuda(wq_a, enable_cuda))
            wq_b_fp8, wq_b_s = _per_block_fp8(_maybe_cuda(wq_b, enable_cuda))
            state[f"{prefix}.wk.weight"] = wk_fp8.cpu()
            state[f"{prefix}.wk.weight_scale_inv"] = wk_s.cpu()
            state[f"{prefix}.wq_a.weight"] = wq_a_fp8.cpu()
            state[f"{prefix}.wq_a.weight_scale_inv"] = wq_a_s.cpu()
            state[f"{prefix}.wq_b.weight"] = wq_b_fp8.cpu()
            state[f"{prefix}.wq_b.weight_scale_inv"] = wq_b_s.cpu()
        else:
            state[f"{prefix}.wk.weight"] = wk.cpu()
            state[f"{prefix}.wq_a.weight"] = wq_a.cpu()
            state[f"{prefix}.wq_b.weight"] = wq_b.cpu()

        state[f"{prefix}.weights_proj.weight"] = weights_proj.cpu()

        # LayerNorm defaults: weight=1, bias=0.
        state[f"{prefix}.k_norm.weight"] = torch.ones((index_head_dim,), dtype=torch.float32)
        state[f"{prefix}.k_norm.bias"] = torch.zeros((index_head_dim,), dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(out_path))
    else:
        torch.save(state, out_path)

    print(f"[saved] {out_path} keys={len(state)}")


if __name__ == "__main__":
    main()

