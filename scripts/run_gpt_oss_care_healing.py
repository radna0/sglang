#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CARE healing for a converted GPT-OSS MLA checkpoint."
    )
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--distill-topk", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--trainable-subset",
        default="all_mla",
        choices=["all_mla", "rope_only", "decoupled_rope_only", "all_mla_plus_o"],
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_script_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
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
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def maybe_get_tensor(self, name: str) -> Optional[torch.Tensor]:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def _copy_linear_from_checkpoint(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    layer = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        layer.weight.copy_(weight.to(dtype=dtype, device=device))
        if bias is not None:
            layer.bias.copy_(bias.to(dtype=dtype, device=device))
    return layer


def _infer_input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        return embeddings.weight.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


class CareGptOssAttentionHF(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        base_qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_proj_weight: torch.Tensor,
        q_proj_bias: Optional[torch.Tensor],
        o_proj_weight: torch.Tensor,
        o_proj_bias: Optional[torch.Tensor],
        kv_a_weight: torch.Tensor,
        kv_a_bias: Optional[torch.Tensor],
        kv_b_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_groups = 1
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.base_qk_rope_head_dim = int(base_qk_rope_head_dim)
        self.decoupled_rope_dim = int(self.qk_rope_head_dim - self.base_qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(v_head_dim)
        self.kv_lora_rank = int(kv_lora_rank)
        self.scaling = self.qk_head_dim**-0.5
        self.attention_dropout = float(original_attn.attention_dropout)
        self.is_causal = True
        self.sliding_window = original_attn.sliding_window

        q_dtype = original_attn.q_proj.weight.dtype
        q_device = original_attn.q_proj.weight.device
        self.q_proj = _copy_linear_from_checkpoint(
            q_proj_weight,
            q_proj_bias,
            dtype=q_dtype,
            device=q_device,
        )
        self.o_proj = _copy_linear_from_checkpoint(
            o_proj_weight,
            o_proj_bias,
            dtype=original_attn.o_proj.weight.dtype,
            device=original_attn.o_proj.weight.device,
        )
        self.sinks = original_attn.sinks
        self.sinks.requires_grad_(False)

        self.kv_a_proj_with_mqa = nn.Linear(
            int(q_proj_weight.shape[1]),
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=kv_a_bias is not None,
            dtype=kv_a_weight.dtype,
            device=q_device,
        )
        with torch.no_grad():
            self.kv_a_proj_with_mqa.weight.copy_(
                kv_a_weight.to(dtype=kv_a_weight.dtype, device=q_device)
            )
            if kv_a_bias is not None:
                self.kv_a_proj_with_mqa.bias.copy_(
                    kv_a_bias.to(dtype=kv_a_weight.dtype, device=q_device)
                )

            kv_b_unflat = kv_b_weight.unflatten(
                0, (self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            w_kc, w_vc = kv_b_unflat.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
            self.w_kc = nn.Parameter(
                w_kc.to(device=q_device, dtype=kv_a_weight.dtype).contiguous()
            )
            self.w_vc = nn.Parameter(
                w_vc.transpose(1, 2)
                .to(device=q_device, dtype=kv_a_weight.dtype)
                .contiguous()
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape

        q_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        latent_states = latent_cache[..., : self.kv_lora_rank]

        if self.qk_nope_head_dim > 0:
            q_nope = q_states[..., : self.qk_nope_head_dim]
            k_nope = torch.einsum("bsr,hdr->bshd", latent_states, self.w_kc)
        else:
            q_nope = q_states.new_empty(batch_size, seq_len, self.num_heads, 0)
            k_nope = q_states.new_empty(batch_size, seq_len, self.num_heads, 0)

        if self.qk_rope_head_dim > 0:
            q_rope = q_states[..., self.qk_nope_head_dim :]
            k_rope = latent_cache[..., self.kv_lora_rank :].unsqueeze(2).expand(
                batch_size, seq_len, self.num_heads, self.qk_rope_head_dim
            )
            q_rope, k_rope = apply_rotary_pos_emb(
                q_rope.transpose(1, 2),
                k_rope.transpose(1, 2),
                *position_embeddings,
            )
            q_rope = q_rope.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
        else:
            q_rope = q_states.new_empty(batch_size, seq_len, self.num_heads, 0)
            k_rope = q_states.new_empty(batch_size, seq_len, self.num_heads, 0)

        v_states = torch.einsum("bsr,hrd->bshd", latent_states, self.w_vc)

        query_states = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        key_states = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        value_states = v_states.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _resolve_source_model_path(student_model_path: Path) -> Path:
    config = _load_json(student_model_path / "config.json")
    source_path = config.get("care_mla_conversion", {}).get("source_model_path")
    if source_path:
        return Path(source_path).resolve()
    return student_model_path.resolve()


def _patch_student_model(
    student_model: nn.Module,
    student_checkpoint_path: Path,
) -> dict[str, Any]:
    student_config = _load_json(student_checkpoint_path / "config.json")
    qk_nope_head_dim = int(student_config["qk_nope_head_dim"])
    qk_rope_head_dim = int(student_config["qk_rope_head_dim"])
    care_meta = student_config.get("care_mla_conversion", {})
    base_qk_rope_head_dim = int(
        care_meta.get(
            "base_qk_rope_head_dim",
            qk_rope_head_dim - int(care_meta.get("decoupled_rope_dim", 0)),
        )
    )
    v_head_dim = int(student_config["v_head_dim"])
    rank_schedule = student_config.get("kv_lora_rank_per_layer")
    if rank_schedule is None:
        rank_schedule = [int(student_config["kv_lora_rank"])] * int(student_config["num_hidden_layers"])
    rank_schedule = [int(x) for x in rank_schedule]

    index = TensorIndex(student_checkpoint_path)
    attention_modules = []
    for layer_id, layer in enumerate(student_model.model.layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        q_proj_weight = index.get_tensor(f"{prefix}.q_proj.weight")
        q_proj_bias = index.maybe_get_tensor(f"{prefix}.q_proj.bias")
        o_proj_weight = index.get_tensor(f"{prefix}.o_proj.weight")
        o_proj_bias = index.maybe_get_tensor(f"{prefix}.o_proj.bias")
        kv_a_weight = index.get_tensor(f"{prefix}.kv_a_proj_with_mqa.weight")
        kv_a_bias = index.maybe_get_tensor(f"{prefix}.kv_a_proj_with_mqa.bias")
        kv_b_weight = index.get_tensor(f"{prefix}.kv_b_proj.weight")
        patched = CareGptOssAttentionHF(
            original_attn=layer.self_attn,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            base_qk_rope_head_dim=base_qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=rank_schedule[layer_id],
            q_proj_weight=q_proj_weight.to(dtype=layer.self_attn.q_proj.weight.dtype),
            q_proj_bias=q_proj_bias.to(dtype=layer.self_attn.q_proj.weight.dtype) if q_proj_bias is not None else None,
            o_proj_weight=o_proj_weight.to(dtype=layer.self_attn.o_proj.weight.dtype),
            o_proj_bias=o_proj_bias.to(dtype=layer.self_attn.o_proj.weight.dtype) if o_proj_bias is not None else None,
            kv_a_weight=kv_a_weight.to(dtype=layer.self_attn.q_proj.weight.dtype),
            kv_a_bias=kv_a_bias.to(dtype=layer.self_attn.q_proj.weight.dtype) if kv_a_bias is not None else None,
            kv_b_weight=kv_b_weight.to(dtype=layer.self_attn.q_proj.weight.dtype),
        )
        layer.self_attn = patched
        attention_modules.append(patched)

    return {
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "base_qk_rope_head_dim": base_qk_rope_head_dim,
        "decoupled_rope_dim": qk_rope_head_dim - base_qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "kv_lora_rank_per_layer": rank_schedule,
        "attention_modules": attention_modules,
        "student_config": student_config,
    }


def _register_row_mask(param: nn.Parameter, start_row: int) -> None:
    mask = torch.zeros_like(param)
    mask[start_row:] = 1
    param.register_hook(lambda grad, mask=mask: grad * mask)


def _register_q_proj_rope_mask(
    param: nn.Parameter,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_head_dim: int,
    start_in_rope: int = 0,
) -> None:
    mask = torch.zeros_like(param)
    for head_idx in range(num_heads):
        row_start = head_idx * qk_head_dim + qk_nope_head_dim + start_in_rope
        row_end = (head_idx + 1) * qk_head_dim
        mask[row_start:row_end] = 1
    param.register_hook(lambda grad, mask=mask: grad * mask)


def _configure_trainable_subset(attention_modules: list[CareGptOssAttentionHF], mode: str) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for module in attention_modules:
        for param in module.parameters():
            param.requires_grad_(False)

        if mode == "all_mla":
            targets = [module.q_proj.weight, module.kv_a_proj_with_mqa.weight, module.w_kc, module.w_vc]
            if module.q_proj.bias is not None:
                targets.append(module.q_proj.bias)
            if module.kv_a_proj_with_mqa.bias is not None:
                targets.append(module.kv_a_proj_with_mqa.bias)
        elif mode == "rope_only":
            if module.qk_rope_head_dim <= 0:
                continue
            module.q_proj.weight.requires_grad_(True)
            _register_q_proj_rope_mask(
                module.q_proj.weight,
                num_heads=module.num_heads,
                qk_nope_head_dim=module.qk_nope_head_dim,
                qk_head_dim=module.qk_head_dim,
            )
            targets = [module.q_proj.weight]
            if module.q_proj.bias is not None:
                module.q_proj.bias.requires_grad_(True)
                _register_q_proj_rope_mask(
                    module.q_proj.bias,
                    num_heads=module.num_heads,
                    qk_nope_head_dim=module.qk_nope_head_dim,
                    qk_head_dim=module.qk_head_dim,
                )
                targets.append(module.q_proj.bias)
            module.kv_a_proj_with_mqa.weight.requires_grad_(True)
            _register_row_mask(module.kv_a_proj_with_mqa.weight, module.kv_lora_rank)
            targets.append(module.kv_a_proj_with_mqa.weight)
            if module.kv_a_proj_with_mqa.bias is not None:
                module.kv_a_proj_with_mqa.bias.requires_grad_(True)
                _register_row_mask(module.kv_a_proj_with_mqa.bias, module.kv_lora_rank)
                targets.append(module.kv_a_proj_with_mqa.bias)
        elif mode == "decoupled_rope_only":
            if module.decoupled_rope_dim <= 0:
                continue
            module.q_proj.weight.requires_grad_(True)
            _register_q_proj_rope_mask(
                module.q_proj.weight,
                num_heads=module.num_heads,
                qk_nope_head_dim=module.qk_nope_head_dim + module.base_qk_rope_head_dim,
                qk_head_dim=module.qk_head_dim,
            )
            targets = [module.q_proj.weight]
            if module.q_proj.bias is not None:
                module.q_proj.bias.requires_grad_(True)
                _register_q_proj_rope_mask(
                    module.q_proj.bias,
                    num_heads=module.num_heads,
                    qk_nope_head_dim=module.qk_nope_head_dim + module.base_qk_rope_head_dim,
                    qk_head_dim=module.qk_head_dim,
                )
                targets.append(module.q_proj.bias)
            module.kv_a_proj_with_mqa.weight.requires_grad_(True)
            _register_row_mask(
                module.kv_a_proj_with_mqa.weight,
                module.kv_lora_rank + module.base_qk_rope_head_dim,
            )
            targets.append(module.kv_a_proj_with_mqa.weight)
            if module.kv_a_proj_with_mqa.bias is not None:
                module.kv_a_proj_with_mqa.bias.requires_grad_(True)
                _register_row_mask(
                    module.kv_a_proj_with_mqa.bias,
                    module.kv_lora_rank + module.base_qk_rope_head_dim,
                )
                targets.append(module.kv_a_proj_with_mqa.bias)
        elif mode == "all_mla_plus_o":
            targets = [
                module.q_proj.weight,
                module.kv_a_proj_with_mqa.weight,
                module.w_kc,
                module.w_vc,
                module.o_proj.weight,
            ]
            if module.q_proj.bias is not None:
                targets.append(module.q_proj.bias)
            if module.kv_a_proj_with_mqa.bias is not None:
                targets.append(module.kv_a_proj_with_mqa.bias)
            if module.o_proj.bias is not None:
                targets.append(module.o_proj.bias)
        else:
            raise ValueError(f"Unsupported trainable subset: {mode}")

        for target in targets:
            target.requires_grad_(True)
            trainable.append(target)
    return trainable


def _iter_token_windows(
    dataset_spec_json: Path,
    tokenizer,
    seq_len: int,
    target_total_rows: Optional[int],
    append_eos: bool,
) -> Iterator[list[int]]:
    collect_helpers = _load_script_module(
        Path(__file__).resolve().with_name("collect_gpt_oss_kv_covariance.py")
    )
    spec = _load_json(dataset_spec_json)
    row_budgets = collect_helpers._compute_row_budgets(spec, target_total_rows)
    eos_token_id = tokenizer.eos_token_id
    for source, row_budget in zip(spec["sources"], row_budgets):
        buffer: list[int] = []
        rows_used = 0
        for row in collect_helpers._iter_rows(source):
            if row_budget and rows_used >= row_budget:
                break
            text = collect_helpers._row_to_text(row, source)
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            if append_eos and eos_token_id is not None:
                token_ids = list(token_ids) + [int(eos_token_id)]
            buffer.extend(int(x) for x in token_ids)
            rows_used += 1
            while len(buffer) >= seq_len:
                yield buffer[:seq_len]
                buffer = buffer[seq_len:]


def _build_batches(
    dataset_spec_json: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    target_total_rows: Optional[int],
    append_eos: bool,
) -> Iterator[torch.Tensor]:
    packed: list[list[int]] = []
    for token_window in _iter_token_windows(
        dataset_spec_json, tokenizer, seq_len, target_total_rows, append_eos
    ):
        packed.append(token_window)
        if len(packed) >= batch_size:
            yield torch.tensor(packed, dtype=torch.long)
            packed = []
    if packed:
        yield torch.tensor(packed, dtype=torch.long)


def _distill_topk_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    student_logits = student_logits[:, :-1].float()
    teacher_logits = teacher_logits[:, :-1].float()
    k = min(int(topk), teacher_logits.shape[-1])
    teacher_topk_vals, teacher_topk_idx = teacher_logits.topk(k=k, dim=-1)
    student_topk_vals = torch.gather(student_logits, dim=-1, index=teacher_topk_idx)
    teacher_probs = F.softmax(teacher_topk_vals / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk_vals / temperature, dim=-1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * (temperature**2)
    )


def _copy_tree(converted_model_path: Path, output_dir: Path, copy_files: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in converted_model_path.iterdir():
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        if copy_files:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


def _export_healed_checkpoint(
    student_model: nn.Module,
    student_checkpoint_path: Path,
    output_dir: Path,
    copy_files: bool,
    overwrite: bool,
    metadata: dict[str, Any],
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    _copy_tree(student_checkpoint_path, output_dir, copy_files=copy_files)

    new_tensors: dict[str, torch.Tensor] = {}
    for layer_id, layer in enumerate(student_model.model.layers):
        attn = layer.self_attn
        if not isinstance(attn, CareGptOssAttentionHF):
            continue
        prefix = f"model.layers.{layer_id}.self_attn"
        new_tensors[f"{prefix}.q_proj.weight"] = attn.q_proj.weight.detach().cpu().contiguous()
        if attn.q_proj.bias is not None:
            new_tensors[f"{prefix}.q_proj.bias"] = attn.q_proj.bias.detach().cpu().contiguous()
        kv_b_weight = torch.cat(
            [attn.w_kc.detach().cpu(), attn.w_vc.detach().cpu().transpose(1, 2)],
            dim=1,
        ).reshape(-1, attn.kv_lora_rank)
        new_tensors[f"{prefix}.kv_a_proj_with_mqa.weight"] = (
            attn.kv_a_proj_with_mqa.weight.detach().cpu().contiguous()
        )
        if attn.kv_a_proj_with_mqa.bias is not None:
            new_tensors[f"{prefix}.kv_a_proj_with_mqa.bias"] = (
                attn.kv_a_proj_with_mqa.bias.detach().cpu().contiguous()
            )
        new_tensors[f"{prefix}.kv_b_proj.weight"] = kv_b_weight.contiguous()
        new_tensors[f"{prefix}.o_proj.weight"] = attn.o_proj.weight.detach().cpu().contiguous()
        if attn.o_proj.bias is not None:
            new_tensors[f"{prefix}.o_proj.bias"] = attn.o_proj.bias.detach().cpu().contiguous()

    mla_shard_name = "model-care-mla-attention.safetensors"
    save_file(new_tensors, output_dir / mla_shard_name)

    config = _load_json(student_checkpoint_path / "config.json")
    config["care_mla_healing"] = metadata
    _save_json(output_dir / "config.json", config)

    index = _load_json(student_checkpoint_path / "model.safetensors.index.json")
    for name in new_tensors:
        index["weight_map"][name] = mla_shard_name
    _save_json(output_dir / "model.safetensors.index.json", index)


def main() -> None:
    args = _parse_args()
    student_model_path = Path(args.student_model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_spec_json = Path(args.dataset_spec_json).resolve()
    teacher_model_path = (
        Path(args.teacher_model_path).resolve()
        if args.teacher_model_path
        else _resolve_source_model_path(student_model_path)
    )
    base_model_path = _resolve_source_model_path(student_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "healing_log.jsonl"
    manifest_path = output_dir / "healing_manifest.json"

    manifest = {
        "student_model_path": str(student_model_path),
        "teacher_model_path": str(teacher_model_path),
        "base_model_path": str(base_model_path),
        "dataset_spec_json": str(dataset_spec_json),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "target_total_rows": int(args.target_total_rows) if args.target_total_rows is not None else None,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "ce_weight": float(args.ce_weight),
        "kl_weight": float(args.kl_weight),
        "distill_topk": int(args.distill_topk),
        "temperature": float(args.temperature),
        "trainable_subset": args.trainable_subset,
    }
    _save_json(manifest_path, manifest)

    device = _resolve_device(args.device)
    model_kwargs = {
        "torch_dtype": _resolve_dtype(args.dtype),
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False, use_fast=True)

    print(json.dumps({"student_base_model": str(base_model_path), **manifest}, indent=2), flush=True)

    student_model = AutoModelForCausalLM.from_pretrained(str(base_model_path), **model_kwargs)
    student_model.config._attn_implementation = args.attn_implementation
    if args.device_map is None:
        student_model.to(device)
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
    patch_info = _patch_student_model(student_model, student_model_path)
    if patch_info["qk_rope_head_dim"] > 0:
        student_model.config.head_dim = int(patch_info["qk_rope_head_dim"])
        student_model.model.rotary_emb = type(student_model.model.rotary_emb)(
            student_model.config,
            device=_infer_input_device(student_model),
        )
    trainable_params = _configure_trainable_subset(
        patch_info["attention_modules"], args.trainable_subset
    )
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters were selected for mode={args.trainable_subset}.")

    teacher_model = None
    if args.kl_weight > 0:
        teacher_model = AutoModelForCausalLM.from_pretrained(str(teacher_model_path), **model_kwargs)
        teacher_model.config._attn_implementation = args.attn_implementation
        if args.device_map is None:
            teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    student_model.train()
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    start_time = time.time()
    for batch in _build_batches(
        dataset_spec_json=dataset_spec_json,
        tokenizer=tokenizer,
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        target_total_rows=args.target_total_rows,
        append_eos=bool(args.append_eos),
    ):
        if global_step >= int(args.max_steps):
            break

        global_step += 1
        student_input_device = _infer_input_device(student_model)
        batch = batch.to(student_input_device)
        lr_scale = min(1.0, global_step / max(int(args.warmup_steps), 1))
        for group in optimizer.param_groups:
            group["lr"] = float(args.learning_rate) * lr_scale

        labels = batch.clone()
        outputs = student_model(input_ids=batch, attention_mask=torch.ones_like(batch), labels=labels, use_cache=False)
        ce_loss = outputs.loss
        total_loss = ce_loss * float(args.ce_weight)

        kl_loss = None
        if teacher_model is not None and float(args.kl_weight) > 0:
            teacher_input_device = _infer_input_device(teacher_model)
            teacher_batch = batch.to(teacher_input_device)
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids=teacher_batch,
                    attention_mask=torch.ones_like(teacher_batch),
                    use_cache=False,
                ).logits
            kl_loss = _distill_topk_loss(
                outputs.logits,
                teacher_logits.to(outputs.logits.device),
                temperature=float(args.temperature),
                topk=int(args.distill_topk),
            )
            total_loss = total_loss + kl_loss * float(args.kl_weight)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        record = {
            "step": global_step,
            "ce_loss": float(ce_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()) if kl_loss is not None else None,
            "total_loss": float(total_loss.detach().cpu()),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - start_time,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if global_step % max(int(args.log_every), 1) == 0:
            print(json.dumps(record), flush=True)

        if global_step % max(int(args.save_every), 1) == 0:
            checkpoint_dir = output_dir / f"checkpoint_step_{global_step:06d}"
            _export_healed_checkpoint(
                student_model=student_model,
                student_checkpoint_path=student_model_path,
                output_dir=checkpoint_dir,
                copy_files=bool(args.copy_files),
                overwrite=True,
                metadata={**manifest, "completed_steps": global_step},
            )
            print(f"[checkpoint] wrote {checkpoint_dir}", flush=True)

    final_dir = output_dir / "healed_checkpoint"
    _export_healed_checkpoint(
        student_model=student_model,
        student_checkpoint_path=student_model_path,
        output_dir=final_dir,
        copy_files=bool(args.copy_files),
        overwrite=bool(args.overwrite),
        metadata={
            **manifest,
            "completed_steps": global_step,
            "attention_modules": len(patch_info["attention_modules"]),
            "base_qk_rope_head_dim": int(patch_info["base_qk_rope_head_dim"]),
            "decoupled_rope_dim": int(patch_info["decoupled_rope_dim"]),
        },
    )
    print(f"[done] wrote healed checkpoint to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
