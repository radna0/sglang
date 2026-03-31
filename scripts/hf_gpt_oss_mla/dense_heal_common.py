#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_gpt_oss_care_healing import (  # noqa: E402
    _build_batches,
    _configure_trainable_subset,
    _export_healed_checkpoint,
    _infer_input_device,
    _load_student_model_for_healing,
    _resolve_device,
    _resolve_dtype,
    _resolve_source_model_path,
)

DEFAULT_SANITY_PROMPTS = Path(__file__).resolve().with_name("dense_heal_sanity_prompts.json")
DEFAULT_HELDOUT_PROMPTS = Path(__file__).resolve().with_name("dense_heal_heldout_prompts.json")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_prompt_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, str):
                records.append({"id": f"prompt_{idx:03d}", "prompt": item})
            elif isinstance(item, dict):
                prompt = item.get("prompt") or item.get("text")
                if not prompt:
                    raise ValueError(f"Prompt record {idx} in {path} is missing 'prompt'.")
                records.append({
                    "id": item.get("id", f"prompt_{idx:03d}"),
                    "category": item.get("category"),
                    "prompt": prompt,
                    "metadata": item.get("metadata"),
                })
            else:
                raise ValueError(f"Unsupported prompt record type in {path}: {type(item)}")
        return records
    raise ValueError(f"Unsupported prompt payload in {path}; expected a list.")


def _build_model_kwargs(
    *,
    dtype_name: str,
    device_map: Optional[str],
    max_memory: Optional[dict[Any, str]],
    offload_folder: Optional[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": _resolve_dtype(dtype_name),
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
    return kwargs


def build_default_max_memory(
    *,
    gpu_max_memory_gib: int,
    cpu_max_memory_gib: int,
) -> Optional[dict[Any, str]]:
    if not torch.cuda.is_available():
        return None
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return None
    max_memory: dict[Any, str] = {
        gpu_idx: f"{int(gpu_max_memory_gib)}GiB" for gpu_idx in range(gpu_count)
    }
    max_memory["cpu"] = f"{int(cpu_max_memory_gib)}GiB"
    return max_memory


def parse_gpu_indices(spec: Optional[str]) -> Optional[list[int]]:
    if spec is None:
        return None
    values = [chunk.strip() for chunk in str(spec).split(",")]
    indices = [int(value) for value in values if value]
    if not indices:
        return None
    deduped: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        deduped.append(index)
        seen.add(index)
    return deduped


def build_max_memory_for_gpu_indices(
    *,
    gpu_indices: Optional[list[int]],
    gpu_max_memory_gib: int,
    cpu_max_memory_gib: int,
) -> Optional[dict[Any, str]]:
    if not torch.cuda.is_available():
        return None
    if gpu_indices is None:
        return build_default_max_memory(
            gpu_max_memory_gib=gpu_max_memory_gib,
            cpu_max_memory_gib=cpu_max_memory_gib,
        )
    max_memory: dict[Any, str] = {
        int(gpu_idx): f"{int(gpu_max_memory_gib)}GiB" for gpu_idx in gpu_indices
    }
    max_memory["cpu"] = f"{int(cpu_max_memory_gib)}GiB"
    return max_memory


def resolve_dual_model_memory_layout(
    *,
    student_gpu_indices_spec: Optional[str],
    teacher_gpu_indices_spec: Optional[str],
    gpu_max_memory_gib: int,
    cpu_max_memory_gib: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "student_gpu_indices": None,
            "teacher_gpu_indices": None,
            "student_max_memory": None,
            "teacher_max_memory": None,
        }

    visible_gpu_indices = list(range(torch.cuda.device_count()))
    student_gpu_indices = parse_gpu_indices(student_gpu_indices_spec)
    teacher_gpu_indices = parse_gpu_indices(teacher_gpu_indices_spec)

    if student_gpu_indices is None and teacher_gpu_indices is None:
        if len(visible_gpu_indices) >= 4:
            midpoint = len(visible_gpu_indices) // 2
            student_gpu_indices = visible_gpu_indices[:midpoint]
            teacher_gpu_indices = visible_gpu_indices[midpoint:]
        elif len(visible_gpu_indices) >= 2:
            student_gpu_indices = [visible_gpu_indices[0]]
            teacher_gpu_indices = visible_gpu_indices[1:]
        else:
            student_gpu_indices = visible_gpu_indices
            teacher_gpu_indices = visible_gpu_indices
    elif student_gpu_indices is None:
        teacher_set = set(teacher_gpu_indices or [])
        student_gpu_indices = [
            gpu_idx for gpu_idx in visible_gpu_indices if gpu_idx not in teacher_set
        ] or list(teacher_gpu_indices or [])
    elif teacher_gpu_indices is None:
        student_set = set(student_gpu_indices or [])
        teacher_gpu_indices = [
            gpu_idx for gpu_idx in visible_gpu_indices if gpu_idx not in student_set
        ] or list(student_gpu_indices or [])

    for gpu_idx in list(student_gpu_indices or []) + list(teacher_gpu_indices or []):
        if gpu_idx not in visible_gpu_indices:
            raise ValueError(
                f"Requested GPU index {gpu_idx}, but visible devices are {visible_gpu_indices}."
            )

    return {
        "student_gpu_indices": student_gpu_indices,
        "teacher_gpu_indices": teacher_gpu_indices,
        "student_max_memory": build_max_memory_for_gpu_indices(
            gpu_indices=student_gpu_indices,
            gpu_max_memory_gib=gpu_max_memory_gib,
            cpu_max_memory_gib=cpu_max_memory_gib,
        ),
        "teacher_max_memory": build_max_memory_for_gpu_indices(
            gpu_indices=teacher_gpu_indices,
            gpu_max_memory_gib=gpu_max_memory_gib,
            cpu_max_memory_gib=cpu_max_memory_gib,
        ),
    }


def load_teacher_student_pair(
    *,
    student_checkpoint_path: Path,
    teacher_model_path: Optional[Path],
    dtype_name: str,
    device_name: str,
    device_map: Optional[str],
    attn_implementation: str,
    gradient_checkpointing: bool,
    student_max_memory: Optional[dict[Any, str]] = None,
    teacher_max_memory: Optional[dict[Any, str]] = None,
    offload_root: Optional[Path] = None,
):
    base_model_path = _resolve_source_model_path(student_checkpoint_path)
    teacher_model_path = teacher_model_path or base_model_path
    device = _resolve_device(device_name)
    student_offload = None
    teacher_offload = None
    if offload_root is not None:
        offload_root.mkdir(parents=True, exist_ok=True)
        student_offload = str((offload_root / "student").resolve())
        teacher_offload = str((offload_root / "teacher").resolve())
    student_kwargs = _build_model_kwargs(
        dtype_name=dtype_name,
        device_map=device_map,
        max_memory=student_max_memory,
        offload_folder=student_offload,
    )
    teacher_kwargs = _build_model_kwargs(
        dtype_name=dtype_name,
        device_map=device_map,
        max_memory=teacher_max_memory,
        offload_folder=teacher_offload,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    student_model, patch_info = _load_student_model_for_healing(
        student_model_path=student_checkpoint_path,
        model_kwargs=student_kwargs,
        device=device,
        device_map=device_map,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(str(teacher_model_path), **teacher_kwargs)
    teacher_model.config._attn_implementation = attn_implementation
    if device_map is None:
        teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    return {
        "tokenizer": tokenizer,
        "student_model": student_model,
        "teacher_model": teacher_model,
        "patch_info": patch_info,
        "base_model_path": base_model_path,
        "teacher_model_path": teacher_model_path,
        "student_max_memory": student_max_memory,
        "teacher_max_memory": teacher_max_memory,
    }


def configure_h1_trainable_params(attention_modules):
    return _configure_trainable_subset(attention_modules, "all_mla")


def compute_forward_kl(
    student_next_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    *,
    temperature: float,
    token_kl_clip: Optional[float],
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_next_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_next_logits.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if token_kl_clip is not None and token_kl_clip > 0:
        per_token_kl = per_token_kl.clamp(max=float(token_kl_clip))
    return per_token_kl.mean() * (temperature**2)


def compute_jsd(
    student_next_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    *,
    temperature: float,
    token_kl_clip: Optional[float],
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_next_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_next_logits.float() / temperature, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    mix_probs = 0.5 * (student_probs + teacher_probs)
    mix_log_probs = torch.log(mix_probs.clamp_min(1e-12))
    student_kl = (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1)
    teacher_kl = (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1)
    per_token_jsd = 0.5 * (student_kl + teacher_kl)
    if token_kl_clip is not None and token_kl_clip > 0:
        per_token_jsd = per_token_jsd.clamp(max=float(token_kl_clip))
    return per_token_jsd.mean() * (temperature**2)


def compute_topk_forward_kl(
    student_next_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    *,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    k = min(int(topk), teacher_next_logits.shape[-1])
    teacher_topk_vals, teacher_topk_idx = teacher_next_logits.float().topk(k=k, dim=-1)
    student_topk_vals = torch.gather(student_next_logits.float(), dim=-1, index=teacher_topk_idx)
    teacher_probs = F.softmax(teacher_topk_vals / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_topk_vals / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def compute_distill_loss(
    student_next_logits: torch.Tensor,
    teacher_next_logits: torch.Tensor,
    *,
    objective: str,
    temperature: float,
    token_kl_clip: Optional[float],
    topk: int,
) -> torch.Tensor:
    if objective == "forward_kl":
        return compute_forward_kl(
            student_next_logits,
            teacher_next_logits,
            temperature=temperature,
            token_kl_clip=token_kl_clip,
        )
    if objective == "jsd":
        return compute_jsd(
            student_next_logits,
            teacher_next_logits,
            temperature=temperature,
            token_kl_clip=token_kl_clip,
        )
    if objective == "topk_forward_kl":
        return compute_topk_forward_kl(
            student_next_logits,
            teacher_next_logits,
            temperature=temperature,
            topk=topk,
        )
    raise ValueError(f"Unsupported distill objective: {objective}")


def gather_unique_trainable_params(params: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(param)
    return unique


def maybe_clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> Optional[float]:
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params or max_norm <= 0:
        return None
    total = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total.detach().cpu())


def _prepare_generation_inputs(model, tokenizer, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = _infer_input_device(model)
    inputs = {k: v.to(input_device) for k, v in encoded.items()}
    return inputs, int(inputs["input_ids"].shape[1])


def evaluate_prompt_records(
    *,
    student_model,
    teacher_model,
    tokenizer,
    prompt_records: list[dict[str, Any]],
    max_new_tokens: int,
    distill_objective: str,
    temperature: float,
    token_kl_clip: Optional[float],
    topk: int,
) -> dict[str, Any]:
    student_model.eval()
    teacher_model.eval()
    rows: list[dict[str, Any]] = []
    forward_kls: list[float] = []

    for record in prompt_records:
        prompt = str(record["prompt"])
        student_inputs, prompt_len = _prepare_generation_inputs(student_model, tokenizer, prompt)
        with torch.no_grad():
            generated = student_model.generate(
                **student_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_ids = generated[:, :]
        student_eval_ids = full_ids.to(_infer_input_device(student_model))
        teacher_eval_ids = full_ids.to(_infer_input_device(teacher_model))
        student_mask = torch.ones_like(student_eval_ids)
        teacher_mask = torch.ones_like(teacher_eval_ids)
        with torch.no_grad():
            student_logits = student_model(
                input_ids=student_eval_ids,
                attention_mask=student_mask,
                use_cache=False,
            ).logits
            teacher_logits = teacher_model(
                input_ids=teacher_eval_ids,
                attention_mask=teacher_mask,
                use_cache=False,
            ).logits.to(student_logits.device)

        seq_len = int(full_ids.shape[1])
        start = max(prompt_len - 1, 0)
        stop = max(seq_len - 1, start)
        generated_tokens = max(seq_len - prompt_len, 0)
        if generated_tokens > 0 and stop > start:
            heldout_kl = compute_distill_loss(
                student_logits[:, start:stop],
                teacher_logits[:, start:stop],
                objective=distill_objective,
                temperature=temperature,
                token_kl_clip=token_kl_clip,
                topk=topk,
            )
            heldout_kl_value = float(heldout_kl.detach().cpu())
            forward_kls.append(heldout_kl_value)
        else:
            heldout_kl_value = None

        rows.append(
            {
                "id": record.get("id"),
                "category": record.get("category"),
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "generated_tokens": generated_tokens,
                "mean_teacher_student_kl": heldout_kl_value,
                "output": tokenizer.decode(full_ids[0], skip_special_tokens=True),
            }
        )

    summary = {
        "num_prompts": len(rows),
        "mean_teacher_student_kl": (
            float(sum(forward_kls) / len(forward_kls)) if forward_kls else None
        ),
        "rows": rows,
    }
    student_model.train()
    return summary


def save_eval_bundle(
    *,
    output_path: Path,
    student_model,
    teacher_model,
    tokenizer,
    sanity_prompts_path: Path,
    heldout_prompts_path: Path,
    max_new_tokens: int,
    distill_objective: str,
    temperature: float,
    token_kl_clip: Optional[float],
    topk: int,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "metadata": metadata or {},
        "sanity": evaluate_prompt_records(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            prompt_records=load_prompt_records(sanity_prompts_path),
            max_new_tokens=max_new_tokens,
            distill_objective=distill_objective,
            temperature=temperature,
            token_kl_clip=token_kl_clip,
            topk=topk,
        ),
        "heldout": evaluate_prompt_records(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            prompt_records=load_prompt_records(heldout_prompts_path),
            max_new_tokens=max_new_tokens,
            distill_objective=distill_objective,
            temperature=temperature,
            token_kl_clip=token_kl_clip,
            topk=topk,
        ),
    }
    _save_json(output_path, payload)
    return payload


def export_checkpoint_with_eval(
    *,
    student_model,
    student_checkpoint_path: Path,
    checkpoint_dir: Path,
    copy_files: bool,
    metadata: dict[str, Any],
    teacher_model,
    tokenizer,
    sanity_prompts_path: Path,
    heldout_prompts_path: Path,
    eval_max_new_tokens: int,
    distill_objective: str,
    temperature: float,
    token_kl_clip: Optional[float],
    topk: int,
) -> dict[str, Any]:
    _export_healed_checkpoint(
        student_model=student_model,
        student_checkpoint_path=student_checkpoint_path,
        output_dir=checkpoint_dir,
        copy_files=copy_files,
        overwrite=True,
        metadata=metadata,
    )
    return save_eval_bundle(
        output_path=checkpoint_dir / "eval_dense_heal_mla_only.json",
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        sanity_prompts_path=sanity_prompts_path,
        heldout_prompts_path=heldout_prompts_path,
        max_new_tokens=eval_max_new_tokens,
        distill_objective=distill_objective,
        temperature=temperature,
        token_kl_clip=token_kl_clip,
        topk=topk,
        metadata=metadata,
    )
