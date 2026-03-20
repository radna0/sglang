#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt_oss_hf_loader import prepare_hf_model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original vs converted GPT-OSS forward traces on one prompt."
    )
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trace-layers", default="0,1,2,3,4,5,10,15,20,25,30,35")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-memory-per-gpu", default="78GiB")
    parser.add_argument("--max-cpu-memory", default="512GiB")
    parser.add_argument("--offload-root", default="/workspace/offload_root/trace_compare")
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _load_model_and_tokenizer(model_path: str, *, dtype: torch.dtype, args: argparse.Namespace):
    model_ref, trust_remote_code = prepare_hf_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        max_memory=(
            {
                **{i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())},
                "cpu": args.max_cpu_memory,
            }
            if args.device == "cuda" and torch.cuda.device_count() > 0
            else None
        ),
        offload_folder=str(Path(args.offload_root).resolve()),
    )
    model.eval()
    return model_ref, model, tokenizer


def _infer_device(model) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device) and model.device.type != "meta":
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def _capture_forward(model, input_ids: torch.Tensor, trace_layers: list[int]) -> dict:
    traces = {
        "layer_attn_out": {},
        "layer_out": {},
    }
    handles = []
    for layer_id in trace_layers:
        layer = model.model.layers[layer_id]

        def _attn_hook(_mod, _inp, out, lid=layer_id):
            tensor = out[0] if isinstance(out, tuple) else out
            traces["layer_attn_out"][str(lid)] = tensor.detach().float().cpu()

        def _layer_hook(_mod, _inp, out, lid=layer_id):
            tensor = out[0] if isinstance(out, tuple) else out
            traces["layer_out"][str(lid)] = tensor.detach().float().cpu()

        handles.append(layer.self_attn.register_forward_hook(_attn_hook))
        handles.append(layer.register_forward_hook(_layer_hook))

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
    for handle in handles:
        handle.remove()
    traces["final_logits"] = outputs.logits.detach().float().cpu()
    return traces


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(a).item())
    if denom == 0.0:
        return 0.0
    return float(torch.linalg.norm(a - b).item() / denom)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return float(F.cosine_similarity(a_flat, b_flat, dim=0).item())


def _compare(teacher: dict, student: dict, topk: int) -> dict:
    out = {
        "layer_attn_out": {},
        "layer_out": {},
    }
    for key in ("layer_attn_out", "layer_out"):
        for layer_id, teacher_tensor in teacher[key].items():
            student_tensor = student[key][layer_id]
            out[key][layer_id] = {
                "rel_l2": _rel_l2(teacher_tensor, student_tensor),
                "cosine": _cosine(teacher_tensor, student_tensor),
            }

    teacher_logits = teacher["final_logits"][:, -1, :]
    student_logits = student["final_logits"][:, -1, :]
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)
    teacher_topk = teacher_logits.topk(topk, dim=-1).indices
    student_topk = student_logits.topk(topk, dim=-1).indices
    topk_overlap = (
        (teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2)).any(dim=-1).float().mean().item()
    )
    out["final_logits"] = {
        "rel_l2": _rel_l2(teacher_logits, student_logits),
        "cosine": _cosine(teacher_logits, student_logits),
        "kl_teacher_to_student": float(kl.item()),
        "topk_overlap": float(topk_overlap),
        "teacher_top1": int(teacher_logits.argmax(dim=-1).item()),
        "student_top1": int(student_logits.argmax(dim=-1).item()),
    }
    return out


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    trace_layers = [int(x.strip()) for x in args.trace_layers.split(",") if x.strip()]
    output_path = Path(args.output_path).resolve()

    teacher_ref, teacher_model, tokenizer = _load_model_and_tokenizer(
        args.teacher_model_path, dtype=dtype, args=args
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_length),
    )
    teacher_device = _infer_device(teacher_model)
    input_ids = encoded["input_ids"].to(teacher_device)
    teacher_traces = _capture_forward(teacher_model, input_ids, trace_layers)
    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    student_ref, student_model, student_tokenizer = _load_model_and_tokenizer(
        args.student_model_path, dtype=dtype, args=args
    )
    if tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError("Teacher and student tokenizers do not match.")
    student_device = _infer_device(student_model)
    input_ids = encoded["input_ids"].to(student_device)
    student_traces = _capture_forward(student_model, input_ids, trace_layers)
    del student_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = {
        "teacher_model": teacher_ref,
        "student_model": student_ref,
        "prompt": args.prompt,
        "trace_layers": trace_layers,
        "comparison": _compare(teacher_traces, student_traces, int(args.topk)),
    }
    _save_json(output_path, payload)
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
