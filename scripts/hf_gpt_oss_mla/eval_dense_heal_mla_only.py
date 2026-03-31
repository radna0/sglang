#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from dense_heal_common import (
    DEFAULT_HELDOUT_PROMPTS,
    DEFAULT_SANITY_PROMPTS,
    load_teacher_student_pair,
    resolve_dual_model_memory_layout,
    save_eval_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense-healed GPT-OSS MLA checkpoints in HF eager mode.")
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu-max-memory-gib", type=int, default=60)
    parser.add_argument("--cpu-max-memory-gib", type=int, default=160)
    parser.add_argument("--student-gpu-indices", default=None)
    parser.add_argument("--teacher-gpu-indices", default=None)
    parser.add_argument("--offload-root", default=None)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--sanity-prompts-json", default=str(DEFAULT_SANITY_PROMPTS))
    parser.add_argument("--heldout-prompts-json", default=str(DEFAULT_HELDOUT_PROMPTS))
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--distill-objective", default="forward_kl", choices=["forward_kl", "jsd", "topk_forward_kl"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--token-kl-clip", type=float, default=5.0)
    parser.add_argument("--distill-topk", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
        gradient_checkpointing=False,
        student_max_memory=dual_layout["student_max_memory"],
        teacher_max_memory=dual_layout["teacher_max_memory"],
        offload_root=Path(args.offload_root).resolve() if args.offload_root else None,
    )
    payload = save_eval_bundle(
        output_path=Path(args.output_json).resolve(),
        student_model=bundle["student_model"],
        teacher_model=bundle["teacher_model"],
        tokenizer=bundle["tokenizer"],
        sanity_prompts_path=Path(args.sanity_prompts_json).resolve(),
        heldout_prompts_path=Path(args.heldout_prompts_json).resolve(),
        max_new_tokens=int(args.eval_max_new_tokens),
        distill_objective=args.distill_objective,
        temperature=float(args.temperature),
        token_kl_clip=float(args.token_kl_clip) if args.token_kl_clip is not None else None,
        topk=int(args.distill_topk),
        metadata={
            "student_model_path": str(Path(args.student_model_path).resolve()),
            "teacher_model_path": str(Path(args.teacher_model_path).resolve()) if args.teacher_model_path else str(bundle["teacher_model_path"]),
            "phase": "dense_eval_hf_only",
            "student_gpu_indices": dual_layout["student_gpu_indices"],
            "teacher_gpu_indices": dual_layout["teacher_gpu_indices"],
            "student_max_memory": bundle["student_max_memory"],
            "teacher_max_memory": bundle["teacher_max_memory"],
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
