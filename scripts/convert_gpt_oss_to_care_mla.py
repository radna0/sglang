#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gpt_oss_hf_loader import ensure_gpt_oss_mla_remote_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GPT-OSS checkpoint into a CARE-style MLA checkpoint."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--kv-lora-rank", type=int, default=None)
    parser.add_argument("--rank-schedule-json", default=None)
    parser.add_argument("--qk-rope-head-dim", type=int, default=None)
    parser.add_argument("--qk-nope-head-dim", type=int, default=None)
    parser.add_argument(
        "--mla-rope-num-kv-heads",
        type=int,
        default=None,
        help=(
            "Number of KV heads used by the MLA rope path. Default is 1 (shared rope path). "
            "For GPT-OSS-native GQA-aware rope experiments, set this to the original "
            "num_key_value_heads."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-default-gpt-oss-split",
        action="store_true",
        help=(
            "Allow the legacy GPT-OSS default of qk_rope_head_dim=head_dim//2 when no "
            "explicit qk split is provided. This is unsafe for quality and exists only "
            "for reproducing the old path."
        ),
    )
    parser.add_argument("--decoupled-rope-dim", type=int, default=0)
    parser.add_argument(
        "--decoupled-rope-init",
        default="mean",
        choices=["zero", "mean", "copy"],
    )
    parser.add_argument("--rope-slice-mode", default="tail", choices=["tail"])
    parser.add_argument("--covariance-dir", default=None)
    parser.add_argument("--covariance-shrinkage", type=float, default=1e-4)
    parser.add_argument(
        "--gpt-oss-query-basis-method",
        default="orthogonal_barycenter_v4",
        choices=[
            "mean_residual_v1",
            "orthogonal_shared_rope_v3",
            "orthogonal_barycenter_v4",
            "common_rope_basis_v2",
        ],
        help=(
            "Bridge used for GPT-OSS rewritten decoupled exports. "
            "mean_residual_v1 preserves the original query RoPE branch and uses a "
            "shared mean rope key plus a NoPE residual. orthogonal_shared_rope_v3 "
            "adds an orthogonal per-group query transform on top of a shared mean "
            "rope key. orthogonal_barycenter_v4 solves a covariance-weighted "
            "orthogonal shared-rope barycenter before forming the residual. "
            "common_rope_basis_v2 fits a shared rope basis with full per-group "
            "query transforms."
        ),
    )
    parser.add_argument(
        "--gpt-oss-nope-residual-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the rewritten GPT-OSS NoPE residual bridge. "
            "1.0 preserves the full residual contribution; values below 1.0 shrink "
            "the unrotated residual path."
        ),
    )
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


class TensorIndex:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.index = _load_json(model_path / "model.safetensors.index.json")
        self.weight_map = self.index["weight_map"]

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(
            self.model_path / shard, framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name)


def _repeat_gqa_rows(
    weight: torch.Tensor, num_attention_heads: int, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    repeated = weight.view(num_kv_heads, head_dim, -1)
    repeat_factor = num_attention_heads // num_kv_heads
    return repeated.repeat_interleave(repeat_factor, dim=0)


def _repeat_gqa_bias(
    bias: torch.Tensor, num_attention_heads: int, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    repeated = bias.view(num_kv_heads, head_dim)
    repeat_factor = num_attention_heads // num_kv_heads
    return repeated.repeat_interleave(repeat_factor, dim=0)


def _split_head_channels(
    tensor: torch.Tensor, qk_nope_head_dim: int, qk_rope_head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if qk_rope_head_dim == 0:
        return tensor, tensor.new_empty(tensor.shape[0], 0, *tensor.shape[2:])
    if qk_nope_head_dim == 0:
        return tensor.new_empty(tensor.shape[0], 0, *tensor.shape[2:]), tensor
    return (
        tensor[:, :qk_nope_head_dim],
        tensor[:, qk_nope_head_dim : qk_nope_head_dim + qk_rope_head_dim],
    )


def _build_orthonormal_expansion(
    target_dim: int,
    source_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if target_dim < source_dim:
        raise ValueError(
            f"target_dim ({target_dim}) must be >= source_dim ({source_dim})"
        )
    if target_dim % source_dim != 0:
        raise ValueError(
            "GPT-OSS rewritten decoupled export currently requires "
            "qk_nope_head_dim to be an integer multiple of the original head_dim."
        )
    repeats = target_dim // source_dim
    eye = torch.eye(source_dim, device=device, dtype=dtype)
    return eye.repeat(repeats, 1) / math.sqrt(float(repeats))


def _init_extra_rows_shared(
    base: torch.Tensor,
    extra_rows: int,
    mode: str,
) -> torch.Tensor:
    if extra_rows <= 0:
        return base.new_empty((0, *base.shape[1:]))
    if base.shape[0] == 0 or mode == "zero":
        return base.new_zeros((extra_rows, *base.shape[1:]))
    if mode == "mean":
        return base.mean(dim=0, keepdim=True).expand(extra_rows, *base.shape[1:]).clone()
    if mode == "copy":
        repeats = math.ceil(extra_rows / base.shape[0])
        return base.repeat((repeats,) + (1,) * (base.ndim - 1))[:extra_rows].clone()
    raise ValueError(f"Unsupported decoupled rope init mode: {mode}")


def _init_extra_rows_per_head(
    base: torch.Tensor,
    extra_rows: int,
    mode: str,
) -> torch.Tensor:
    if extra_rows <= 0:
        return base.new_empty((base.shape[0], 0, *base.shape[2:]))
    if base.shape[1] == 0 or mode == "zero":
        return base.new_zeros((base.shape[0], extra_rows, *base.shape[2:]))
    if mode == "mean":
        return (
            base.mean(dim=1, keepdim=True)
            .expand(base.shape[0], extra_rows, *base.shape[2:])
            .clone()
        )
    if mode == "copy":
        repeats = math.ceil(extra_rows / base.shape[1])
        return base.repeat(1, repeats, *([1] * (base.ndim - 2)))[:, :extra_rows].clone()
    raise ValueError(f"Unsupported decoupled rope init mode: {mode}")


def _apply_covariance(
    target: torch.Tensor,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if covariance is None:
        return target, None
    covariance = covariance.to(dtype=target.dtype, device=target.device)
    eye = torch.eye(covariance.shape[0], dtype=target.dtype, device=target.device)
    diag_mean = covariance.diagonal().mean()
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diag_mean * eye
    chol = torch.linalg.cholesky(covariance)
    return target @ chol, chol


def _undo_covariance(
    factor: torch.Tensor, chol: Optional[torch.Tensor]
) -> torch.Tensor:
    if chol is None:
        return factor
    return torch.linalg.solve_triangular(
        chol.transpose(-2, -1),
        factor.transpose(-2, -1),
        upper=True,
        left=True,
    ).transpose(-2, -1)


def _factorize_kv(
    k_nope: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_rows = k_nope.reshape(-1, k_nope.shape[-1])
    v_rows = v.reshape(-1, v.shape[-1])
    target = torch.cat([k_rows, v_rows], dim=0)
    target_weighted, chol = _apply_covariance(target, covariance, shrinkage)
    u, s, vh = torch.linalg.svd(target_weighted, full_matrices=False)
    rank = min(rank, s.shape[0])
    s_root = s[:rank].clamp_min(0).sqrt()
    basis_out = u[:, :rank] * s_root.unsqueeze(0)
    basis_in = s_root.unsqueeze(1) * vh[:rank, :]
    basis_in = _undo_covariance(basis_in, chol)
    k_rows = k_rows.shape[0]
    b_k = basis_out[:k_rows].reshape(k_nope.shape[0], k_nope.shape[1], rank)
    b_v = basis_out[k_rows:].reshape(v.shape[0], v.shape[1], rank)
    return basis_in, b_k, b_v


def _factorize_rows(
    rows: torch.Tensor,
    rank: int,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_weighted, chol = _apply_covariance(rows, covariance, shrinkage)
    u, s, vh = torch.linalg.svd(target_weighted, full_matrices=False)
    rank = min(rank, s.shape[0])
    s_root = s[:rank].clamp_min(0).sqrt()
    basis_out = u[:, :rank] * s_root.unsqueeze(0)
    basis_in = s_root.unsqueeze(1) * vh[:rank, :]
    basis_in = _undo_covariance(basis_in, chol)
    return basis_in, basis_out


def _solve_orthogonal_row_map(
    base_rows: torch.Tensor,
    target_rows: torch.Tensor,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
) -> torch.Tensor:
    base_weighted, _ = _apply_covariance(base_rows, covariance, shrinkage)
    target_weighted, _ = _apply_covariance(target_rows, covariance, shrinkage)
    cross = target_weighted @ base_weighted.transpose(0, 1)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def _solve_orthogonal_barycenter(
    unique_rows: torch.Tensor,
    covariance: Optional[torch.Tensor],
    shrinkage: float,
    num_iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_rows, chol = _apply_covariance(unique_rows, covariance, shrinkage)
    barycenter_weighted = weighted_rows.mean(dim=0)
    maps = []
    for _ in range(num_iters):
        maps = []
        for idx in range(weighted_rows.shape[0]):
            cross = weighted_rows[idx] @ barycenter_weighted.transpose(0, 1)
            u, _, vh = torch.linalg.svd(cross, full_matrices=False)
            maps.append(u @ vh)
        maps = torch.stack(maps, dim=0)
        barycenter_weighted = torch.matmul(
            maps.transpose(1, 2), weighted_rows
        ).mean(dim=0)
    barycenter = _undo_covariance(barycenter_weighted, chol)
    return barycenter, maps


def _solve_down_bias(up_proj: torch.Tensor, target_bias: torch.Tensor) -> torch.Tensor:
    if target_bias.numel() == 0:
        return target_bias.new_empty(0)
    solution = torch.linalg.lstsq(
        up_proj.to(dtype=torch.float32), target_bias.to(dtype=torch.float32).unsqueeze(1)
    ).solution.squeeze(1)
    return solution.to(dtype=target_bias.dtype)


def _load_covariance(
    covariance_dir: Optional[Path], layer_id: int
) -> Optional[torch.Tensor]:
    if covariance_dir is None:
        return None
    candidates = [
        covariance_dir / f"layer_{layer_id:02d}.pt",
        covariance_dir / f"layer_{layer_id}.pt",
        covariance_dir / f"layer_{layer_id:02d}.pth",
        covariance_dir / f"layer_{layer_id}.pth",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = torch.load(candidate, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("covariance", "cov", "C", "matrix"):
                if key in payload:
                    return payload[key]
        if isinstance(payload, torch.Tensor):
            return payload
    raise FileNotFoundError(
        f"Could not find covariance stats for layer {layer_id} in {covariance_dir}"
    )


def _parse_rank_schedule(
    rank_schedule_json: Optional[Path],
    num_hidden_layers: int,
    default_rank: int,
) -> list[int]:
    if rank_schedule_json is None:
        return [int(default_rank)] * num_hidden_layers
    payload = _load_json(rank_schedule_json)
    if isinstance(payload, list):
        schedule = [int(x) for x in payload]
    elif isinstance(payload, dict):
        if "kv_lora_rank_per_layer" in payload:
            schedule = [int(x) for x in payload["kv_lora_rank_per_layer"]]
        elif all(str(i) in payload for i in range(num_hidden_layers)):
            schedule = [int(payload[str(i)]) for i in range(num_hidden_layers)]
        else:
            raise ValueError(
                "Unsupported rank schedule format. Expected list or dict with kv_lora_rank_per_layer."
            )
    else:
        raise ValueError("Unsupported rank schedule JSON payload.")
    if len(schedule) != num_hidden_layers:
        raise ValueError(
            f"Rank schedule length mismatch: expected {num_hidden_layers}, got {len(schedule)}"
        )
    return schedule


def _symlink_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if src.is_dir():
        return
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _write_filtered_shard(
    src_shard: Path,
    dst_shard: Path,
    keep_names: list[str],
) -> None:
    tensors = {}
    with safe_open(src_shard, framework="pt", device="cpu") as handle:
        for name in keep_names:
            tensors[name] = handle.get_tensor(name)
    save_file(tensors, dst_shard)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to reuse it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    covariance_dir = (
        Path(args.covariance_dir).resolve() if args.covariance_dir else None
    )
    rank_schedule_path = (
        Path(args.rank_schedule_json).resolve() if args.rank_schedule_json else None
    )
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    config = _load_json(model_path / "config.json")
    index = TensorIndex(model_path)

    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    num_hidden_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    model_type = str(config.get("model_type", ""))
    attention_bias = bool(config.get("attention_bias", False))
    layer_start = int(args.layer_start)
    layer_end = (
        int(args.layer_end) if args.layer_end is not None else int(num_hidden_layers)
    )
    if not (0 <= layer_start < layer_end <= num_hidden_layers):
        raise ValueError(
            f"Invalid layer slice [{layer_start}, {layer_end}) for {num_hidden_layers} layers."
        )

    kv_lora_rank = (
        int(args.kv_lora_rank)
        if args.kv_lora_rank is not None
        else int(num_kv_heads * head_dim)
    )
    mla_rope_num_kv_heads = (
        int(args.mla_rope_num_kv_heads)
        if args.mla_rope_num_kv_heads is not None
        else 1
    )
    if mla_rope_num_kv_heads not in (1, num_kv_heads):
        raise ValueError(
            f"--mla-rope-num-kv-heads must be 1 or the original num_key_value_heads ({num_kv_heads})"
        )
    if (
        model_type == "gpt_oss"
        and args.qk_rope_head_dim is None
        and args.qk_nope_head_dim is None
        and not args.allow_unsafe_default_gpt_oss_split
    ):
        raise ValueError(
            "GPT-OSS MLA conversion requires an explicit qk split. The legacy default "
            "of qk_rope_head_dim=head_dim//2 is unsafe for GPT-OSS correctness. Pass "
            "--qk-rope-head-dim/--qk-nope-head-dim explicitly, or use "
            "--allow-unsafe-default-gpt-oss-split only to reproduce the old path."
        )

    base_qk_rope_head_dim = (
        int(args.qk_rope_head_dim)
        if args.qk_rope_head_dim is not None
        else int(head_dim // 2)
    )
    qk_nope_head_dim = (
        int(args.qk_nope_head_dim)
        if args.qk_nope_head_dim is not None
        else int(head_dim - base_qk_rope_head_dim)
    )
    use_gpt_oss_rewritten_decoupled = (
        model_type == "gpt_oss"
        and mla_rope_num_kv_heads == 1
        and base_qk_rope_head_dim == head_dim
        and qk_nope_head_dim >= head_dim
        and qk_nope_head_dim % head_dim == 0
    )
    if args.gpt_oss_nope_residual_scale < 0:
        raise ValueError("--gpt-oss-nope-residual-scale must be >= 0")
    if qk_nope_head_dim + base_qk_rope_head_dim != head_dim:
        if model_type == "gpt_oss" and use_gpt_oss_rewritten_decoupled:
            pass
        elif model_type == "gpt_oss":
            raise ValueError(
                "GPT-OSS target geometries that expand beyond the original head_dim "
                "(for example the quality-first target qk_rope_head_dim=64, "
                "qk_nope_head_dim=128) require an explicit GPT-OSS query/K basis "
                "rewrite. The only supported rewritten path in this converter is the "
                "shared-k_R contract with qk_rope_head_dim equal to the original "
                "head_dim and qk_nope_head_dim as an integer multiple of it. See "
                "docs/GPTOSS120B_CARE_MLA_TARGET_CONTRACT.md for the intended "
                "contract."
            )
        else:
            raise ValueError(
                "This converter currently requires qk_nope_head_dim + base_qk_rope_head_dim == original head_dim."
            )
    decoupled_rope_dim = int(args.decoupled_rope_dim)
    if decoupled_rope_dim < 0:
        raise ValueError("--decoupled-rope-dim must be >= 0")
    qk_rope_head_dim = base_qk_rope_head_dim + decoupled_rope_dim

    gpt_oss_contract_status = None
    if model_type == "gpt_oss":
        if mla_rope_num_kv_heads != 1:
            gpt_oss_contract_status = "bridge_per_kv_head_rope"
            print(
                "[gpt-oss] bridge mode: mla_rope_num_kv_heads != 1 preserves more "
                "source-model positional structure for debugging, but it is not the "
                "final CARE/DeepSeek-style shared-k_R contract."
            )
        elif use_gpt_oss_rewritten_decoupled:
            gpt_oss_contract_status = "rewritten_decoupled_target"
            print(
                "[gpt-oss] rewritten decoupled target enabled: explicit query/K "
                "basis rewrite for shared-k_R GPT-OSS MLA export."
            )
        else:
            gpt_oss_contract_status = "experimental_geometry"
            print(
                "[gpt-oss] experimental GPT-OSS MLA geometry. The quality-first "
                "target is qk_rope_head_dim=64, qk_nope_head_dim=128, "
                "kv_lora_rank=512, mla_rope_num_kv_heads=1."
            )

    rank_schedule = _parse_rank_schedule(
        rank_schedule_path, num_hidden_layers, kv_lora_rank
    )
    tensor_device = _choose_device(args.device)
    work_dtype = getattr(torch, args.dtype)

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "device": str(tensor_device),
                "dtype": str(work_dtype),
                "kv_lora_rank": kv_lora_rank,
                "mla_rope_num_kv_heads": mla_rope_num_kv_heads,
                "qk_nope_head_dim": qk_nope_head_dim,
                "qk_rope_head_dim": qk_rope_head_dim,
                "base_qk_rope_head_dim": base_qk_rope_head_dim,
                "decoupled_rope_dim": decoupled_rope_dim,
                "layer_slice": [layer_start, layer_end],
                "gpt_oss_contract_status": gpt_oss_contract_status,
            },
            indent=2,
        )
    )

    new_tensors: Dict[str, torch.Tensor] = {}
    original_dtype = None
    for layer_id in range(layer_start, layer_end):
        prefix = f"model.layers.{layer_id}.self_attn"
        print(f"[convert] layer={layer_id} rank={rank_schedule[layer_id]}")

        w_q = index.get_tensor(f"{prefix}.q_proj.weight")
        w_k = index.get_tensor(f"{prefix}.k_proj.weight")
        w_v = index.get_tensor(f"{prefix}.v_proj.weight")
        b_q = index.get_tensor(f"{prefix}.q_proj.bias") if attention_bias else None
        b_k = index.get_tensor(f"{prefix}.k_proj.bias") if attention_bias else None
        b_v = index.get_tensor(f"{prefix}.v_proj.bias") if attention_bias else None
        original_dtype = original_dtype or w_k.dtype

        w_q_heads = w_q.view(num_heads, head_dim, hidden_size)
        w_k_unique = w_k.view(num_kv_heads, head_dim, hidden_size)
        w_k_rep = _repeat_gqa_rows(w_k, num_heads, num_kv_heads, head_dim)
        w_v_rep = _repeat_gqa_rows(w_v, num_heads, num_kv_heads, head_dim)

        if attention_bias:
            b_q_heads = b_q.view(num_heads, head_dim)
            b_k_unique = b_k.view(num_kv_heads, head_dim)
            b_k_rep = _repeat_gqa_bias(b_k, num_heads, num_kv_heads, head_dim)
            b_v_rep = _repeat_gqa_bias(b_v, num_heads, num_kv_heads, head_dim)
        else:
            b_v_rep = w_v_rep.new_zeros(num_heads, head_dim)

        covariance = _load_covariance(covariance_dir, layer_id)
        if covariance is not None:
            covariance = covariance.to(device=tensor_device, dtype=work_dtype)

        if use_gpt_oss_rewritten_decoupled:
            expansion = _build_orthonormal_expansion(
                qk_nope_head_dim,
                head_dim,
                device=tensor_device,
                dtype=work_dtype,
            )
            residual_scale = math.sqrt(float(args.gpt_oss_nope_residual_scale))
            # The rewritten GPT-OSS bridge is designed so that, before low-rank
            # approximation error, the decomposed score preserves the original
            # bilinear form:
            #
            #   <q_orig, k_orig>
            #     =
            #   <q_rope, k_shared_recon> + <q_nope, k_residual>
            #
            # because:
            #   - q_rope = R^T q_orig, k_shared_recon = R k_shared for orthogonal R
            #   - q_nope = E q_orig, k_residual_nope = E k_residual with E^T E = I
            #
            # Using the old dimensionality-based q scaling here would therefore
            # multiply an already-preserved score by sqrt((D_nope + D_rope)/D_orig),
            # which is sqrt(3) for the 64+128 target. That sharply inflates logits
            # and is inconsistent with score-preserving rewrite semantics.
            q_scale = 1.0
            w_q_heads_dev = w_q_heads.to(device=tensor_device, dtype=work_dtype)
            w_k_unique_dev = w_k_unique.to(device=tensor_device, dtype=work_dtype)
            w_k_rep_dev = w_k_rep.to(device=tensor_device, dtype=work_dtype)
            if args.gpt_oss_query_basis_method == "mean_residual_v1":
                w_k_shared_rope = w_k_unique_dev.mean(dim=0)
                w_k_shared_recon = w_k_shared_rope.unsqueeze(0).expand(
                    num_heads, -1, -1
                )
                w_k_residual = w_k_rep_dev - w_k_shared_recon
                w_q_rope = w_q_heads_dev
                w_q_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_q_heads_dev
                )
                w_k_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_k_residual
                )
                rope_coeff_rep = None
                rope_coeff_unique = None
            elif args.gpt_oss_query_basis_method == "orthogonal_shared_rope_v3":
                w_k_shared_rope = w_k_unique_dev.mean(dim=0)
                orth_maps = []
                for kv_head in range(num_kv_heads):
                    orth_maps.append(
                        _solve_orthogonal_row_map(
                            w_k_shared_rope,
                            w_k_unique_dev[kv_head],
                            covariance=covariance,
                            shrinkage=args.covariance_shrinkage,
                        )
                    )
                rope_coeff_unique = torch.stack(orth_maps, dim=0)
                repeat_factor = num_heads // num_kv_heads
                rope_coeff_rep = rope_coeff_unique.repeat_interleave(
                    repeat_factor, dim=0
                )
                w_k_shared_recon = torch.matmul(
                    rope_coeff_rep,
                    w_k_shared_rope.unsqueeze(0).expand(num_heads, -1, -1),
                )
                w_k_residual = w_k_rep_dev - w_k_shared_recon
                w_q_rope = torch.matmul(
                    rope_coeff_rep.transpose(1, 2), w_q_heads_dev
                )
                w_q_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_q_heads_dev
                )
                w_k_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_k_residual
                )
            elif args.gpt_oss_query_basis_method == "orthogonal_barycenter_v4":
                w_k_shared_rope, rope_coeff_unique = _solve_orthogonal_barycenter(
                    w_k_unique_dev,
                    covariance=covariance,
                    shrinkage=args.covariance_shrinkage,
                )
                repeat_factor = num_heads // num_kv_heads
                rope_coeff_rep = rope_coeff_unique.repeat_interleave(
                    repeat_factor, dim=0
                )
                w_k_shared_recon = torch.matmul(
                    rope_coeff_rep,
                    w_k_shared_rope.unsqueeze(0).expand(num_heads, -1, -1),
                )
                w_k_residual = w_k_rep_dev - w_k_shared_recon
                w_q_rope = torch.matmul(
                    rope_coeff_rep.transpose(1, 2), w_q_heads_dev
                )
                w_q_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_q_heads_dev
                )
                w_k_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_k_residual
                )
            else:
                rope_basis_in, rope_coeff_unique_rows = _factorize_rows(
                    w_k_unique_dev.reshape(-1, hidden_size),
                    head_dim,
                    covariance=covariance,
                    shrinkage=args.covariance_shrinkage,
                )
                rope_coeff_unique = rope_coeff_unique_rows.reshape(
                    num_kv_heads, head_dim, head_dim
                )
                repeat_factor = num_heads // num_kv_heads
                rope_coeff_rep = rope_coeff_unique.repeat_interleave(
                    repeat_factor, dim=0
                )
                w_k_shared_rope = rope_basis_in
                w_k_shared_recon = torch.matmul(
                    rope_coeff_rep,
                    w_k_shared_rope.unsqueeze(0).expand(num_heads, -1, -1),
                )
                w_k_residual = w_k_rep_dev - w_k_shared_recon
                w_q_rope = torch.matmul(
                    rope_coeff_rep.transpose(1, 2), w_q_heads_dev
                )
                w_q_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_q_heads_dev
                )
                w_k_nope = residual_scale * torch.einsum(
                    "nd,hdm->hnm", expansion, w_k_residual
                )

            if attention_bias:
                b_q_heads_dev = b_q_heads.to(device=tensor_device, dtype=work_dtype)
                b_k_unique_dev = b_k_unique.to(device=tensor_device, dtype=work_dtype)
                b_k_rep_dev = b_k_rep.to(device=tensor_device, dtype=work_dtype)
                if args.gpt_oss_query_basis_method == "mean_residual_v1":
                    b_k_shared_rope = b_k_unique_dev.mean(dim=0)
                    b_k_shared_recon = b_k_shared_rope.unsqueeze(0).expand(
                        num_heads, -1
                    )
                    b_k_residual = b_k_rep_dev - b_k_shared_recon
                    b_q_rope = b_q_heads_dev
                    b_q_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_q_heads_dev
                    )
                    b_k_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_k_residual
                    )
                elif (
                    args.gpt_oss_query_basis_method
                    == "orthogonal_shared_rope_v3"
                ):
                    b_k_shared_rope = b_k_unique_dev.mean(dim=0)
                    b_k_shared_recon = torch.matmul(
                        rope_coeff_rep,
                        b_k_shared_rope.view(1, head_dim, 1).expand(
                            num_heads, -1, -1
                        ),
                    ).squeeze(-1)
                    b_k_residual = b_k_rep_dev - b_k_shared_recon
                    b_q_rope = torch.matmul(
                        rope_coeff_rep.transpose(1, 2),
                        b_q_heads_dev.unsqueeze(-1),
                    ).squeeze(-1)
                    b_q_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_q_heads_dev
                    )
                    b_k_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_k_residual
                    )
                elif (
                    args.gpt_oss_query_basis_method
                    == "orthogonal_barycenter_v4"
                ):
                    b_k_shared_rope = torch.matmul(
                        rope_coeff_unique.transpose(1, 2),
                        b_k_unique_dev.unsqueeze(-1),
                    ).squeeze(-1).mean(dim=0)
                    b_k_shared_recon = torch.matmul(
                        rope_coeff_rep,
                        b_k_shared_rope.view(1, head_dim, 1).expand(
                            num_heads, -1, -1
                        ),
                    ).squeeze(-1)
                    b_k_residual = b_k_rep_dev - b_k_shared_recon
                    b_q_rope = torch.matmul(
                        rope_coeff_rep.transpose(1, 2),
                        b_q_heads_dev.unsqueeze(-1),
                    ).squeeze(-1)
                    b_q_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_q_heads_dev
                    )
                    b_k_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_k_residual
                    )
                else:
                    rope_coeff_unique_flat = rope_coeff_unique.reshape(
                        num_kv_heads * head_dim, head_dim
                    )
                    b_k_shared_rope = _solve_down_bias(
                        rope_coeff_unique_flat, b_k_unique_dev.reshape(-1)
                    )
                    b_k_shared_recon = torch.matmul(
                        rope_coeff_rep,
                        b_k_shared_rope.view(1, head_dim, 1).expand(
                            num_heads, -1, -1
                        ),
                    ).squeeze(-1)
                    b_k_residual = b_k_rep_dev - b_k_shared_recon
                    b_q_rope = torch.matmul(
                        rope_coeff_rep.transpose(1, 2),
                        b_q_heads_dev.unsqueeze(-1),
                    ).squeeze(-1)
                    b_q_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_q_heads_dev
                    )
                    b_k_nope = residual_scale * torch.einsum(
                        "nd,hd->hn", expansion, b_k_residual
                    )
            else:
                b_q_nope = w_q_nope.new_zeros(num_heads, qk_nope_head_dim)
                b_q_rope = w_q_rope.new_zeros(num_heads, qk_rope_head_dim)
                b_k_shared_rope = w_k_shared_rope.new_zeros(qk_rope_head_dim)
                b_k_nope = w_k_nope.new_zeros(num_heads, qk_nope_head_dim)

            q_proj_weight = (
                torch.cat([w_q_nope, w_q_rope], dim=1)
                .reshape(num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size)
                .mul(q_scale)
            )
            new_tensors[f"{prefix}.q_proj.weight"] = q_proj_weight.to(
                dtype=original_dtype, device="cpu"
            ).contiguous()
            if attention_bias:
                q_proj_bias = (
                    torch.cat([b_q_nope, b_q_rope], dim=1).reshape(-1).mul(q_scale)
                )
                new_tensors[f"{prefix}.q_proj.bias"] = q_proj_bias.to(
                    dtype=original_dtype, device="cpu"
                ).contiguous()
        else:
            w_q_nope, w_q_rope = _split_head_channels(
                w_q_heads, qk_nope_head_dim, base_qk_rope_head_dim
            )
            _, w_k_unique_rope = _split_head_channels(
                w_k_unique, qk_nope_head_dim, base_qk_rope_head_dim
            )
            w_k_nope, w_k_rope = _split_head_channels(
                w_k_rep, qk_nope_head_dim, base_qk_rope_head_dim
            )

            if attention_bias:
                b_q_nope, b_q_rope = _split_head_channels(
                    b_q_heads, qk_nope_head_dim, base_qk_rope_head_dim
                )
                _, b_k_unique_rope = _split_head_channels(
                    b_k_unique, qk_nope_head_dim, base_qk_rope_head_dim
                )
                b_k_nope, b_k_rope = _split_head_channels(
                    b_k_rep, qk_nope_head_dim, base_qk_rope_head_dim
                )
            else:
                b_q_nope = w_q_nope.new_zeros(num_heads, qk_nope_head_dim)
                b_q_rope = w_q_rope.new_zeros(num_heads, base_qk_rope_head_dim)
                b_k_unique_rope = w_k_unique_rope.new_zeros(
                    num_kv_heads, base_qk_rope_head_dim
                )
                b_k_nope = w_k_nope.new_zeros(num_heads, qk_nope_head_dim)
                b_k_rope = w_k_rope.new_zeros(num_heads, base_qk_rope_head_dim)

            q_extra_rope = _init_extra_rows_per_head(
                w_q_rope.to(device=tensor_device, dtype=work_dtype),
                decoupled_rope_dim,
                args.decoupled_rope_init,
            )
            q_proj_weight = torch.cat(
                [
                    w_q_nope.to(device=tensor_device, dtype=work_dtype),
                    w_q_rope.to(device=tensor_device, dtype=work_dtype),
                    q_extra_rope,
                ],
                dim=1,
            ).reshape(num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden_size)
            new_tensors[f"{prefix}.q_proj.weight"] = q_proj_weight.to(
                dtype=original_dtype, device="cpu"
            ).contiguous()

            if attention_bias:
                q_extra_bias = _init_extra_rows_per_head(
                    b_q_rope.to(device=tensor_device, dtype=work_dtype).unsqueeze(-1),
                    decoupled_rope_dim,
                    args.decoupled_rope_init,
                ).squeeze(-1)
                q_proj_bias = torch.cat(
                    [
                        b_q_nope.to(device=tensor_device, dtype=work_dtype),
                        b_q_rope.to(device=tensor_device, dtype=work_dtype),
                        q_extra_bias,
                    ],
                    dim=1,
                ).reshape(-1)
                new_tensors[f"{prefix}.q_proj.bias"] = q_proj_bias.to(
                    dtype=original_dtype, device="cpu"
                ).contiguous()

        basis_in, b_k_latent, b_v_latent = _factorize_kv(
            w_k_nope.to(device=tensor_device, dtype=work_dtype),
            w_v_rep.to(device=tensor_device, dtype=work_dtype),
            rank_schedule[layer_id],
            covariance=covariance,
            shrinkage=args.covariance_shrinkage,
        )

        if use_gpt_oss_rewritten_decoupled:
            rope_weight_base = w_k_shared_rope
            rope_bias_base = b_k_shared_rope if attention_bias else None
        elif base_qk_rope_head_dim > 0:
            if mla_rope_num_kv_heads == 1:
                rope_weight_base = (
                    w_k_rope.to(device=tensor_device, dtype=work_dtype).mean(dim=0)
                )
                rope_bias_base = (
                    b_k_rope.to(device=tensor_device, dtype=work_dtype).mean(dim=0)
                    if attention_bias
                    else None
                )
            else:
                rope_weight_base = (
                    w_k_unique_rope.to(device=tensor_device, dtype=work_dtype)
                    .reshape(mla_rope_num_kv_heads, base_qk_rope_head_dim, hidden_size)
                )
                rope_bias_base = (
                    b_k_unique_rope.to(device=tensor_device, dtype=work_dtype)
                    .reshape(mla_rope_num_kv_heads, base_qk_rope_head_dim)
                    if attention_bias
                    else None
                )
        else:
            if mla_rope_num_kv_heads == 1:
                rope_weight_base = basis_in.new_empty(0, hidden_size)
                rope_bias_base = basis_in.new_empty(0) if attention_bias else None
            else:
                rope_weight_base = basis_in.new_empty(mla_rope_num_kv_heads, 0, hidden_size)
                rope_bias_base = (
                    basis_in.new_empty(mla_rope_num_kv_heads, 0) if attention_bias else None
                )

        if mla_rope_num_kv_heads == 1:
            rope_extra = _init_extra_rows_shared(
                rope_weight_base,
                decoupled_rope_dim,
                args.decoupled_rope_init,
            )
            rope_weight = torch.cat([rope_weight_base, rope_extra], dim=0)
            if attention_bias:
                rope_bias = torch.cat(
                    [
                        rope_bias_base,
                        _init_extra_rows_shared(
                            rope_bias_base.unsqueeze(-1),
                            decoupled_rope_dim,
                            args.decoupled_rope_init,
                        ).squeeze(-1),
                    ],
                    dim=0,
                )
            else:
                rope_bias = None
        else:
            rope_extra = _init_extra_rows_per_head(
                rope_weight_base,
                decoupled_rope_dim,
                args.decoupled_rope_init,
            )
            rope_weight = torch.cat([rope_weight_base, rope_extra], dim=1).reshape(
                -1, hidden_size
            )
            if attention_bias:
                rope_bias = torch.cat(
                    [
                        rope_bias_base,
                        _init_extra_rows_per_head(
                            rope_bias_base.unsqueeze(-1),
                            decoupled_rope_dim,
                            args.decoupled_rope_init,
                        ).squeeze(-1),
                    ],
                    dim=1,
                ).reshape(-1)
            else:
                rope_bias = None

        kv_a_weight = torch.cat([basis_in, rope_weight], dim=0)
        k_latent_2d = b_k_latent.reshape(
            num_heads * qk_nope_head_dim, rank_schedule[layer_id]
        )
        v_latent_2d = b_v_latent.reshape(num_heads * head_dim, rank_schedule[layer_id])
        kv_b_weight = torch.cat(
            [
                k_latent_2d,
                v_latent_2d,
            ],
            dim=0,
        )

        new_tensors[f"{prefix}.kv_a_proj_with_mqa.weight"] = kv_a_weight.to(
            dtype=original_dtype, device="cpu"
        ).contiguous()
        new_tensors[f"{prefix}.kv_b_proj.weight"] = kv_b_weight.to(
            dtype=original_dtype, device="cpu"
        ).contiguous()

        if attention_bias:
            target_bias = torch.cat(
                [
                    b_k_nope.to(device=tensor_device, dtype=work_dtype).reshape(-1),
                    b_v_rep.to(device=tensor_device, dtype=work_dtype).reshape(-1),
                ],
                dim=0,
            )
            up_proj = torch.cat(
                [
                    k_latent_2d,
                    v_latent_2d,
                ],
                dim=0,
            )
            down_bias = _solve_down_bias(up_proj, target_bias)
            kv_a_bias = (
                torch.cat([down_bias, rope_bias], dim=0)
                if rope_bias is not None
                else down_bias
            )
            new_tensors[f"{prefix}.kv_a_proj_with_mqa.bias"] = kv_a_bias.to(
                dtype=original_dtype, device="cpu"
            ).contiguous()

    mla_shard_name = "model-care-mla-attention.safetensors"
    save_file(new_tensors, output_dir / mla_shard_name)

    converted_prefixes = {
        f"model.layers.{layer_id}.self_attn." for layer_id in range(layer_start, layer_end)
    }

    def _is_replaced_or_removed_attention_tensor(name: str) -> bool:
        if not any(name.startswith(prefix) for prefix in converted_prefixes):
            return False
        return (
            ".self_attn.q_proj." in name
            or ".self_attn.k_proj." in name
            or ".self_attn.v_proj." in name
        )

    original_index = _load_json(model_path / "model.safetensors.index.json")
    shard_to_names: dict[str, list[str]] = {}
    for name, shard_name in original_index["weight_map"].items():
        shard_to_names.setdefault(shard_name, []).append(name)

    shard_names = set(shard_to_names)
    for src in model_path.iterdir():
        if src.name in {"config.json", "model.safetensors.index.json"}:
            continue
        dst = output_dir / src.name
        if src.name not in shard_names:
            _symlink_or_copy(src, dst, copy_files=args.copy_files)
            continue
        keep_names = [
            name
            for name in shard_to_names[src.name]
            if not _is_replaced_or_removed_attention_tensor(name)
        ]
        if len(keep_names) == len(shard_to_names[src.name]):
            _symlink_or_copy(src, dst, copy_files=args.copy_files)
        else:
            _write_filtered_shard(src, dst, keep_names)

    new_config = copy.deepcopy(config)
    new_config["architectures"] = ["GptOssMlaForCausalLM"]
    new_config["kv_lora_rank"] = int(max(rank_schedule))
    new_config["head_dim"] = int(qk_nope_head_dim + qk_rope_head_dim)
    new_config["qk_nope_head_dim"] = int(qk_nope_head_dim)
    new_config["qk_rope_head_dim"] = int(qk_rope_head_dim)
    new_config["v_head_dim"] = int(head_dim)
    new_config["mla_rope_num_kv_heads"] = int(mla_rope_num_kv_heads)
    if config.get("model_type") == "gpt_oss":
        new_config["mla_attention_mode"] = (
            "mha_explicit" if mla_rope_num_kv_heads > 1 else "shared_latent"
        )
        new_config["gpt_oss_mla_contract"] = {
            "status": gpt_oss_contract_status,
            "recommended_qk_rope_head_dim": 64,
            "recommended_qk_nope_head_dim": 128,
            "recommended_kv_lora_rank": 512,
            "recommended_mla_rope_num_kv_heads": 1,
            "bridge_mode": mla_rope_num_kv_heads != 1,
            "query_basis_status": (
                "rewritten_decoupled"
                if use_gpt_oss_rewritten_decoupled
                else "legacy_channel_split"
            ),
            "query_basis_method": (
                args.gpt_oss_query_basis_method
                if use_gpt_oss_rewritten_decoupled
                else None
            ),
            "nope_residual_scale": (
                float(args.gpt_oss_nope_residual_scale)
                if use_gpt_oss_rewritten_decoupled
                else None
            ),
            "notes": (
                "The final GPT-OSS target contract is 64 rope + 128 nope + shared "
                "k_R. Rewritten exports use an orthonormal NoPE expansion plus a "
                "shared-k_R / per-head residual decomposition. Legacy exports do not "
                "implement that rewrite."
            ),
        }
    if len(set(rank_schedule)) > 1:
        new_config["kv_lora_rank_per_layer"] = [int(x) for x in rank_schedule]
    new_config["care_mla_conversion"] = {
        "source_model_path": str(model_path),
        "converter": "convert_gpt_oss_to_care_mla.py",
        "kv_lora_rank_per_layer": [int(x) for x in rank_schedule],
        "qk_nope_head_dim": int(qk_nope_head_dim),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "mla_rope_num_kv_heads": int(mla_rope_num_kv_heads),
        "mla_attention_mode": new_config.get(
            "mla_attention_mode", "shared_latent"
        ),
        "base_qk_rope_head_dim": int(base_qk_rope_head_dim),
        "decoupled_rope_dim": int(decoupled_rope_dim),
        "decoupled_rope_init": args.decoupled_rope_init,
        "rope_slice_mode": args.rope_slice_mode,
        "covariance_dir": str(covariance_dir) if covariance_dir else None,
        "covariance_shrinkage": float(args.covariance_shrinkage),
        "layer_slice": [int(layer_start), int(layer_end)],
        "gpt_oss_contract_status": gpt_oss_contract_status,
        "query_basis_status": (
            "rewritten_decoupled"
            if use_gpt_oss_rewritten_decoupled
            else "legacy_channel_split"
        ),
        "query_basis_method": (
            args.gpt_oss_query_basis_method
            if use_gpt_oss_rewritten_decoupled
            else None
        ),
        "query_scale_mode": (
            "score_preserving_unity"
            if use_gpt_oss_rewritten_decoupled
            else "legacy_dimensional"
        ),
        "nope_residual_scale": (
            float(args.gpt_oss_nope_residual_scale)
            if use_gpt_oss_rewritten_decoupled
            else None
        ),
    }
    _save_json(output_dir / "config.json", new_config)
    ensure_gpt_oss_mla_remote_code(output_dir)

    new_index = copy.deepcopy(original_index)
    for name in list(new_index["weight_map"]):
        if _is_replaced_or_removed_attention_tensor(name) and (
            ".self_attn.k_proj." in name or ".self_attn.v_proj." in name
        ):
            new_index["weight_map"].pop(name, None)
    for name in new_tensors:
        new_index["weight_map"][name] = mla_shard_name
    total_size = int(original_index.get("metadata", {}).get("total_size", 0))
    total_size += sum(int(t.numel() * t.element_size()) for t in new_tensors.values())
    new_index["metadata"] = {"total_size": total_size}
    _save_json(output_dir / "model.safetensors.index.json", new_index)
    _save_json(
        output_dir / "care_mla_manifest.json",
        {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "mla_shard": mla_shard_name,
            "layers_converted": list(range(layer_start, layer_end)),
            "kv_lora_rank_per_layer": [int(x) for x in rank_schedule],
            "qk_nope_head_dim": int(qk_nope_head_dim),
            "qk_rope_head_dim": int(qk_rope_head_dim),
            "mla_rope_num_kv_heads": int(mla_rope_num_kv_heads),
            "mla_attention_mode": new_config.get(
                "mla_attention_mode", "shared_latent"
            ),
            "base_qk_rope_head_dim": int(base_qk_rope_head_dim),
            "decoupled_rope_dim": int(decoupled_rope_dim),
            "v_head_dim": int(head_dim),
            "gpt_oss_contract_status": gpt_oss_contract_status,
            "query_scale_mode": (
                "score_preserving_unity"
                if use_gpt_oss_rewritten_decoupled
                else "legacy_dimensional"
            ),
        },
    )
    print(
        f"[done] wrote {output_dir} with {len(new_tensors)} converted MLA tensors in {mla_shard_name}"
    )


if __name__ == "__main__":
    main()
