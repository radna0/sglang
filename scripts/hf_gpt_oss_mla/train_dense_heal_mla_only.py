#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from dense_heal_common import (
    DEFAULT_HELDOUT_PROMPTS,
    DEFAULT_SANITY_PROMPTS,
    _build_batches,
    _infer_input_device,
    _save_json,
    compute_distill_loss,
    configure_h1_trainable_params,
    export_checkpoint_with_eval,
    gather_unique_trainable_params,
    load_teacher_student_pair,
    maybe_clip_grad_norm,
    resolve_dual_model_memory_layout,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense MLA-only healing for GPT-OSS CARE MLA checkpoints.")
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu-max-memory-gib", type=int, default=60)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=160)
    parser.add_argument("--student-gpu-indices", default=None)
    parser.add_argument("--teacher-gpu-indices", default=None)
    parser.add_argument("--offload-root", default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--distill-objective", default="forward_kl", choices=["forward_kl", "jsd", "topk_forward_kl"])
    parser.add_argument("--distill-topk", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--token-kl-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--sanity-prompts-json", default=str(DEFAULT_SANITY_PROMPTS))
    parser.add_argument("--heldout-prompts-json", default=str(DEFAULT_HELDOUT_PROMPTS))
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dense_heal_manifest.json"
    log_path = output_dir / "dense_heal_log.jsonl"
    dual_layout = resolve_dual_model_memory_layout(
        student_gpu_indices_spec=args.student_gpu_indices,
        teacher_gpu_indices_spec=args.teacher_gpu_indices,
        gpu_max_memory_gib=int(args.gpu_max_memory_gib),
        cpu_max_memory_gib=int(args.cpu_max_memory_gib),
    ) if args.device_map else {
        "student_gpu_indices": None,
        "teacher_gpu_indices": None,
        "student_max_memory": None,
        "teacher_max_memory": None,
    }

    bundle = load_teacher_student_pair(
        student_checkpoint_path=Path(args.student_model_path).resolve(),
        teacher_model_path=Path(args.teacher_model_path).resolve() if args.teacher_model_path else None,
        dtype_name=args.dtype,
        device_name=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        student_max_memory=dual_layout["student_max_memory"],
        teacher_max_memory=dual_layout["teacher_max_memory"],
        offload_root=Path(args.offload_root).resolve() if args.offload_root else None,
    )
    student_model = bundle["student_model"]
    teacher_model = bundle["teacher_model"]
    tokenizer = bundle["tokenizer"]
    patch_info = bundle["patch_info"]

    mla_params = configure_h1_trainable_params(patch_info["attention_modules"])
    trainable_params = gather_unique_trainable_params(mla_params)
    if not trainable_params:
        raise RuntimeError("No trainable H1 MLA parameters were selected.")

    optimizer = AdamW(trainable_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    student_model.train()
    teacher_model.eval()

    manifest = {
        "phase": "dense_heal_mla_only",
        "student_model_path": str(Path(args.student_model_path).resolve()),
        "teacher_model_path": str(bundle["teacher_model_path"]),
        "base_model_path": str(bundle["base_model_path"]),
        "dataset_spec_json": str(Path(args.dataset_spec_json).resolve()),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "target_total_rows": int(args.target_total_rows) if args.target_total_rows is not None else None,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "gpu_max_memory_gib": int(args.gpu_max_memory_gib),
        "cpu_max_memory_gib": int(args.cpu_max_memory_gib),
        "student_gpu_indices": dual_layout["student_gpu_indices"],
        "teacher_gpu_indices": dual_layout["teacher_gpu_indices"],
        "student_max_memory": bundle["student_max_memory"],
        "teacher_max_memory": bundle["teacher_max_memory"],
        "offload_root": str(Path(args.offload_root).resolve()) if args.offload_root else None,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "grad_clip": float(args.grad_clip),
        "ce_weight": float(args.ce_weight),
        "kl_weight": float(args.kl_weight),
        "distill_objective": args.distill_objective,
        "distill_topk": int(args.distill_topk),
        "temperature": float(args.temperature),
        "token_kl_clip": float(args.token_kl_clip) if args.token_kl_clip is not None else None,
        "sanity_prompts_json": str(Path(args.sanity_prompts_json).resolve()),
        "heldout_prompts_json": str(Path(args.heldout_prompts_json).resolve()),
        "eval_max_new_tokens": int(args.eval_max_new_tokens),
        "trainable_subset": ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj"],
        "trainable_param_count": int(sum(p.numel() for p in trainable_params)),
        "healing_stage": "dense_mla_only_h1",
    }
    _save_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    global_step = 0
    start_time = time.time()
    for batch in _build_batches(
        dataset_spec_json=Path(args.dataset_spec_json).resolve(),
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
        labels = batch.clone()

        lr_scale = min(1.0, global_step / max(int(args.warmup_steps), 1))
        for group in optimizer.param_groups:
            group["lr"] = float(args.learning_rate) * lr_scale

        outputs = student_model(
            input_ids=batch,
            attention_mask=torch.ones_like(batch),
            labels=labels,
            use_cache=False,
        )
        ce_loss = outputs.loss
        total_loss = ce_loss * float(args.ce_weight)

        teacher_input_device = _infer_input_device(teacher_model)
        teacher_batch = batch.to(teacher_input_device)
        with torch.no_grad():
            teacher_logits = teacher_model(
                input_ids=teacher_batch,
                attention_mask=torch.ones_like(teacher_batch),
                use_cache=False,
            ).logits.to(outputs.logits.device)

        distill_loss = compute_distill_loss(
            outputs.logits[:, :-1],
            teacher_logits[:, :-1],
            objective=args.distill_objective,
            temperature=float(args.temperature),
            token_kl_clip=float(args.token_kl_clip) if args.token_kl_clip is not None else None,
            topk=int(args.distill_topk),
        )
        total_loss = total_loss + distill_loss * float(args.kl_weight)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = maybe_clip_grad_norm(trainable_params, float(args.grad_clip))
        optimizer.step()

        record = {
            "step": global_step,
            "ce_loss": float(ce_loss.detach().cpu()),
            "distill_loss": float(distill_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
            "elapsed_s": time.time() - start_time,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if global_step % max(int(args.log_every), 1) == 0:
            print(json.dumps(record), flush=True)

        should_eval = global_step % max(int(args.eval_every), 1) == 0
        should_save = global_step % max(int(args.save_every), 1) == 0
        if should_eval or should_save:
            checkpoint_dir = output_dir / f"checkpoint_step_{global_step:06d}"
            eval_payload = export_checkpoint_with_eval(
                student_model=student_model,
                student_checkpoint_path=Path(args.student_model_path).resolve(),
                checkpoint_dir=checkpoint_dir,
                copy_files=bool(args.copy_files),
                metadata={**manifest, "completed_steps": global_step},
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                sanity_prompts_path=Path(args.sanity_prompts_json).resolve(),
                heldout_prompts_path=Path(args.heldout_prompts_json).resolve(),
                eval_max_new_tokens=int(args.eval_max_new_tokens),
                distill_objective=args.distill_objective,
                temperature=float(args.temperature),
                token_kl_clip=float(args.token_kl_clip) if args.token_kl_clip is not None else None,
                topk=int(args.distill_topk),
            )
            summary = {
                "step": global_step,
                "sanity_mean_kl": eval_payload["sanity"]["mean_teacher_student_kl"],
                "heldout_mean_kl": eval_payload["heldout"]["mean_teacher_student_kl"],
            }
            print(json.dumps({"checkpoint_eval": summary}), flush=True)

    final_dir = output_dir / "healed_checkpoint"
    export_checkpoint_with_eval(
        student_model=student_model,
        student_checkpoint_path=Path(args.student_model_path).resolve(),
        checkpoint_dir=final_dir,
        copy_files=bool(args.copy_files),
        metadata={**manifest, "completed_steps": global_step},
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        sanity_prompts_path=Path(args.sanity_prompts_json).resolve(),
        heldout_prompts_path=Path(args.heldout_prompts_json).resolve(),
        eval_max_new_tokens=int(args.eval_max_new_tokens),
        distill_objective=args.distill_objective,
        temperature=float(args.temperature),
        token_kl_clip=float(args.token_kl_clip) if args.token_kl_clip is not None else None,
        topk=int(args.distill_topk),
    )
    print(f"[done] wrote dense-healed checkpoint to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
