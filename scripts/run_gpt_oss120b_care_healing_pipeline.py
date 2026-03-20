#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


PHASE_ORDER = ("general", "calib", "aimo")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-phase CARE healing for GPT-OSS-120B MLA checkpoints."
    )
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--general-dataset-spec-json", default=None)
    parser.add_argument("--calib-dataset-spec-json", default=None)
    parser.add_argument("--aimo-dataset-spec-json", default=None)
    parser.add_argument("--phase-order", default="general,calib,aimo")
    parser.add_argument("--runtime", default="hf", choices=["hf", "fsdp"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--general-seq-len", type=int, default=None)
    parser.add_argument("--calib-seq-len", type=int, default=None)
    parser.add_argument("--aimo-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--teacher-topk-cache-batch-size", type=int, default=None)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--fsdp-use-teacher-topk-cache", action="store_true")
    parser.add_argument("--teacher-topk-cache-device-map", default="auto")
    parser.add_argument("--torchrun-nproc-per-node", type=int, default=8)
    parser.add_argument("--torchrun-master-port", type=int, default=29600)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--mxfp4-preswizzle-dir", default=None)
    parser.add_argument(
        "--quantized-expert-layout",
        default="replicated",
        choices=["replicated", "tp_sharded"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--distill-topk", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--general-steps", type=int, default=2000)
    parser.add_argument("--calib-steps", type=int, default=1000)
    parser.add_argument("--aimo-steps", type=int, default=500)
    parser.add_argument("--general-subset", default="all_mla")
    parser.add_argument("--calib-subset", default="rope_only")
    parser.add_argument("--aimo-subset", default="all_mla_plus_o")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _resolve_source_model_path(model_path: str) -> Path:
    config_path = Path(model_path).resolve() / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    source_path = config.get("care_mla_conversion", {}).get("source_model_path")
    if source_path:
        return Path(source_path).resolve()
    return Path(model_path).resolve()


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("[cmd] " + " ".join(cmd) + "\n")
        log_file.flush()
        subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def _phase_dataset_path(args: argparse.Namespace, phase: str) -> Path | None:
    mapping = {
        "general": args.general_dataset_spec_json,
        "calib": args.calib_dataset_spec_json,
        "aimo": args.aimo_dataset_spec_json,
    }
    value = mapping[phase]
    return Path(value).resolve() if value else None


def _phase_steps(args: argparse.Namespace, phase: str) -> int:
    return int(getattr(args, f"{phase}_steps"))


def _phase_subset(args: argparse.Namespace, phase: str) -> str:
    return str(getattr(args, f"{phase}_subset"))


def _phase_seq_len(args: argparse.Namespace, phase: str) -> int:
    phase_value = getattr(args, f"{phase}_seq_len")
    if phase_value is not None:
        return int(phase_value)
    return int(args.seq_len)


def _phase_target_total_rows(args: argparse.Namespace, phase: str) -> int | None:
    if args.target_total_rows is not None:
        return int(args.target_total_rows)
    steps = max(1, _phase_steps(args, phase))
    per_step_rows = max(1, int(args.batch_size))
    if args.runtime == "fsdp":
        per_step_rows *= max(1, int(args.torchrun_nproc_per_node))
    return steps * per_step_rows


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    hf_healing_script = repo_root / "scripts" / "run_gpt_oss_care_healing.py"
    fsdp_healing_script = repo_root / "scripts" / "run_gpt_oss_care_healing_fsdp.py"
    teacher_cache_script = repo_root / "scripts" / "build_gpt_oss_teacher_topk_cache.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    phase_order = [phase.strip() for phase in args.phase_order.split(",") if phase.strip()]
    for phase in phase_order:
        if phase not in PHASE_ORDER:
            raise ValueError(f"Unsupported phase {phase}. Expected one of {PHASE_ORDER}.")

    manifest = {
        "student_model_path": str(Path(args.student_model_path).resolve()),
        "teacher_model_path": str(Path(args.teacher_model_path).resolve()) if args.teacher_model_path else None,
        "output_root": str(output_root),
        "phase_order": phase_order,
        "runtime": args.runtime,
        "device_map": args.device_map,
        "fsdp_use_teacher_topk_cache": bool(args.fsdp_use_teacher_topk_cache),
        "teacher_topk_cache_device_map": args.teacher_topk_cache_device_map,
        "torchrun_nproc_per_node": int(args.torchrun_nproc_per_node),
        "torchrun_master_port": int(args.torchrun_master_port),
        "attn_implementation": args.attn_implementation,
        "mxfp4_preswizzle_dir": args.mxfp4_preswizzle_dir,
        "quantized_expert_layout": args.quantized_expert_layout,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "teacher_topk_cache_batch_size": (
            int(args.teacher_topk_cache_batch_size)
            if args.teacher_topk_cache_batch_size is not None
            else None
        ),
    }
    _save_json(output_root / "healing_pipeline_manifest.json", manifest)

    current_model_path = str(Path(args.student_model_path).resolve())
    completed: list[dict] = []

    for phase in phase_order:
        dataset_path = _phase_dataset_path(args, phase)
        steps = _phase_steps(args, phase)
        subset = _phase_subset(args, phase)
        seq_len = _phase_seq_len(args, phase)
        target_total_rows = _phase_target_total_rows(args, phase)
        if dataset_path is None or steps <= 0:
            continue

        phase_dir = output_root / phase
        healed_checkpoint = (
            phase_dir / "healed_absorbed_checkpoint"
            if args.runtime == "fsdp"
            else phase_dir / "healed_checkpoint"
        )
        if args.resume and healed_checkpoint.exists():
            current_model_path = str(healed_checkpoint)
            completed.append(
                {
                    "phase": phase,
                    "dataset_spec_json": str(dataset_path),
                    "seq_len": seq_len,
                    "steps": steps,
                    "target_total_rows": target_total_rows,
                    "trainable_subset": subset,
                    "output_dir": str(phase_dir),
                    "resumed": True,
                }
            )
            continue

        teacher_topk_cache_dir = phase_dir / "teacher_topk_cache"
        if (
            args.runtime == "fsdp"
            and args.fsdp_use_teacher_topk_cache
            and float(args.kl_weight) > 0
        ):
            teacher_cache_batch_size = (
                int(args.teacher_topk_cache_batch_size)
                if args.teacher_topk_cache_batch_size is not None
                else int(args.batch_size)
            )
            teacher_for_cache = (
                Path(args.teacher_model_path).resolve()
                if args.teacher_model_path
                else _resolve_source_model_path(current_model_path)
            )
            cache_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(args.torchrun_nproc_per_node),
                "--master_port",
                str(args.torchrun_master_port + 100 + len(completed)),
                str(teacher_cache_script),
                "--teacher-model-path",
                str(teacher_for_cache),
                "--dataset-spec-json",
                str(dataset_path),
                "--output-dir",
                str(teacher_topk_cache_dir),
                "--seq-len",
                str(seq_len),
                "--batch-size",
                str(teacher_cache_batch_size),
                "--world-size",
                str(args.torchrun_nproc_per_node),
                "--dtype",
                args.dtype,
                "--topk",
                str(args.distill_topk),
                "--max-steps",
                str(steps),
                "--target-total-rows",
                str(target_total_rows),
                "--attn-implementation",
                args.attn_implementation,
            ]
            if args.append_eos:
                cache_cmd.append("--append-eos")
            if args.overwrite:
                cache_cmd.append("--overwrite")
            _run_logged(cache_cmd, phase_dir / "teacher_topk_cache.log")

        if args.runtime == "fsdp":
            cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(args.torchrun_nproc_per_node),
                "--master_port",
                str(args.torchrun_master_port + len(completed)),
                str(fsdp_healing_script),
                "--student-model-path",
                current_model_path,
                "--dataset-spec-json",
                str(dataset_path),
                "--output-dir",
                str(phase_dir),
                "--seq-len",
                str(seq_len),
                "--batch-size",
                str(args.batch_size),
                "--dtype",
                args.dtype,
                "--attn-implementation",
                args.attn_implementation,
                "--quantized-expert-layout",
                args.quantized_expert_layout,
                "--max-steps",
                str(steps),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--warmup-steps",
                str(args.warmup_steps),
                "--grad-clip",
                str(args.grad_clip),
                "--ce-weight",
                str(args.ce_weight),
                "--kl-weight",
                str(args.kl_weight),
                "--distill-topk",
                str(args.distill_topk),
                "--temperature",
                str(args.temperature),
                "--trainable-subset",
                subset,
                "--log-every",
                str(args.log_every),
                "--save-every",
                str(args.save_every),
                "--target-total-rows",
                str(target_total_rows),
            ]
            if args.mxfp4_preswizzle_dir:
                cmd.extend(["--mxfp4-preswizzle-dir", str(args.mxfp4_preswizzle_dir)])
            if args.fsdp_use_teacher_topk_cache and float(args.kl_weight) > 0:
                cmd.extend(["--teacher-topk-cache-dir", str(teacher_topk_cache_dir)])
        else:
            cmd = [
                sys.executable,
                str(hf_healing_script),
                "--student-model-path",
                current_model_path,
                "--dataset-spec-json",
                str(dataset_path),
                "--output-dir",
                str(phase_dir),
                "--seq-len",
                str(seq_len),
                "--batch-size",
                str(args.batch_size),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--attn-implementation",
                args.attn_implementation,
                "--max-steps",
                str(steps),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--warmup-steps",
                str(args.warmup_steps),
                "--grad-clip",
                str(args.grad_clip),
                "--ce-weight",
                str(args.ce_weight),
                "--kl-weight",
                str(args.kl_weight),
                "--distill-topk",
                str(args.distill_topk),
                "--temperature",
                str(args.temperature),
                "--trainable-subset",
                subset,
                "--log-every",
                str(args.log_every),
                "--save-every",
                str(args.save_every),
            ]
        if args.teacher_model_path:
            cmd.extend(["--teacher-model-path", str(Path(args.teacher_model_path).resolve())])
        if args.runtime != "fsdp":
            cmd.extend(["--target-total-rows", str(target_total_rows)])
        if args.append_eos:
            cmd.append("--append-eos")
        if args.runtime == "hf" and args.device_map:
            cmd.extend(["--device-map", args.device_map])
        if args.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        if args.copy_files:
            cmd.append("--copy-files")
        if args.overwrite:
            cmd.append("--overwrite")

        _run_logged(cmd, phase_dir / "launcher.log")
        current_model_path = str(healed_checkpoint)
        completed.append(
            {
                "phase": phase,
                "dataset_spec_json": str(dataset_path),
                "seq_len": seq_len,
                "steps": steps,
                "target_total_rows": target_total_rows,
                "trainable_subset": subset,
                "output_dir": str(phase_dir),
                "resumed": False,
            }
        )

    summary = {
        **manifest,
        "completed_phases": completed,
        "final_model_path": current_model_path,
    }
    _save_json(output_root / "healing_pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
