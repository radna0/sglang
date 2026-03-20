#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dataset-conditioned GPT-OSS-120B CARE conversion pipeline."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--covariance-world-size", type=int, default=1)
    parser.add_argument("--covariance-master-port", type=int, default=29700)
    parser.add_argument("--covariance-replica-device-map", default="auto")
    parser.add_argument("--mxfp4-preswizzle-dir", default="")
    parser.add_argument("--target-rank", type=int, default=512)
    parser.add_argument("--target-total-rank", type=int, default=None)
    parser.add_argument("--min-rank", type=int, default=128)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument(
        "--round-multiple",
        type=int,
        default=1,
        help="If >1, round the final CARE-E schedule to this multiple while preserving budget as closely as possible.",
    )
    parser.add_argument("--qk-rope-head-dim", type=int, default=32)
    parser.add_argument("--qk-nope-head-dim", type=int, default=None)
    parser.add_argument("--decoupled-rope-dim", type=int, default=0)
    parser.add_argument(
        "--decoupled-rope-init",
        default="mean",
        choices=["zero", "mean", "copy"],
    )
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--covariance-shrinkage", type=float, default=1e-4)
    parser.add_argument("--skip-covariance", action="store_true")
    parser.add_argument("--skip-rank-schedule", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--uniform-rank", action="store_true")
    parser.add_argument("--rank-source-fusion", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _covariance_complete(covariance_dir: Path) -> bool:
    return (covariance_dir / "covariance_manifest.json").exists()


def _covariance_has_partials(covariance_dir: Path) -> bool:
    return any((covariance_dir / "partials").glob("rank_*/covariance_manifest.json"))


def _rank_schedule_complete(rank_schedule_json: Path) -> bool:
    return rank_schedule_json.exists()


def _converted_checkpoint_complete(converted_dir: Path) -> bool:
    return (converted_dir / "config.json").exists()


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("._") or "source"


def _compute_row_budgets(spec: dict, target_total_rows: int | None) -> list[int]:
    sources = spec["sources"]
    if target_total_rows is None:
        return [int(src.get("max_rows", 0) or 0) for src in sources]

    explicit = [int(src.get("max_rows", 0) or 0) for src in sources]
    remaining = int(target_total_rows) - sum(explicit)
    if remaining < 0:
        raise ValueError("Explicit max_rows exceed target_total_rows.")

    weights = [float(src.get("weight", 0.0) or 0.0) for src in sources]
    total_weight = sum(weights)
    if remaining > 0 and total_weight <= 0:
        raise ValueError("target_total_rows requires source weights or explicit max_rows.")

    budgets = explicit[:]
    assigned = 0
    for idx, weight in enumerate(weights):
        if explicit[idx] > 0:
            continue
        quota = int(round(remaining * weight / total_weight))
        budgets[idx] = quota
        assigned += quota
    if assigned != remaining:
        for idx in range(len(sources)):
            if explicit[idx] == 0:
                budgets[idx] += remaining - assigned
                break
    return budgets


def _rank_source_fusion_args(
    dataset_spec_json: Path, covariance_dir: Path, target_total_rows: int | None
) -> list[str]:
    spec = json.loads(dataset_spec_json.read_text(encoding="utf-8"))
    row_budgets = _compute_row_budgets(spec, target_total_rows)
    total_budget = float(sum(row_budgets) or 1)
    schedule_args: list[str] = []
    for source, row_budget in zip(spec["sources"], row_budgets):
        source_name = source.get("name", source.get("path", source.get("kind", "source")))
        source_covariance_dir = covariance_dir / "by_source" / _safe_name(source_name)
        schedule_args.extend(["--covariance-dir", str(source_covariance_dir)])
        weight = float(row_budget) / total_budget
        schedule_args.extend(["--covariance-weight", str(weight)])
    return schedule_args


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model_path).resolve()
    dataset_spec_json = Path(args.dataset_spec_json).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    covariance_dir = out_root / "covariance"
    rank_schedule_json = out_root / "kv_lora_rank_schedule.json"
    converted_dir = out_root / "converted_checkpoint"
    pipeline_manifest = out_root / "pipeline_manifest.json"

    manifest = {
        "model_path": str(model_path),
        "dataset_spec_json": str(dataset_spec_json),
        "out_root": str(out_root),
        "covariance_dir": str(covariance_dir),
        "rank_schedule_json": str(rank_schedule_json),
        "converted_dir": str(converted_dir),
        "covariance_world_size": int(args.covariance_world_size),
        "covariance_master_port": int(args.covariance_master_port),
        "covariance_replica_device_map": args.covariance_replica_device_map,
        "mxfp4_preswizzle_dir": args.mxfp4_preswizzle_dir or None,
        "target_rank": int(args.target_rank),
        "target_total_rank": int(args.target_total_rank)
        if args.target_total_rank is not None
        else None,
        "min_rank": int(args.min_rank),
        "max_rank": int(args.max_rank) if args.max_rank is not None else None,
        "round_multiple": int(args.round_multiple),
        "qk_rope_head_dim": int(args.qk_rope_head_dim),
        "qk_nope_head_dim": int(args.qk_nope_head_dim)
        if args.qk_nope_head_dim is not None
        else None,
        "decoupled_rope_dim": int(args.decoupled_rope_dim),
        "decoupled_rope_init": args.decoupled_rope_init,
        "rank_source_fusion": bool(args.rank_source_fusion),
    }
    _save_json(pipeline_manifest, manifest)

    python_exe = sys.executable

    covariance_complete = _covariance_complete(covariance_dir)
    covariance_has_partials = _covariance_has_partials(covariance_dir)
    if not args.skip_covariance and not covariance_complete:
        collect_cmd = [
            str(repo_root / "scripts" / "collect_gpt_oss_kv_covariance.py"),
            "--model-path",
            str(model_path),
            "--dataset-spec-json",
            str(dataset_spec_json),
            "--output-dir",
            str(covariance_dir),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
            "--replica-device-map",
            args.covariance_replica_device_map,
        ]
        if args.mxfp4_preswizzle_dir:
            collect_cmd.extend(["--mxfp4-preswizzle-dir", args.mxfp4_preswizzle_dir])
        if int(args.covariance_world_size) > 1:
            collect_cmd = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(args.covariance_world_size),
                "--master_port",
                str(args.covariance_master_port),
                *collect_cmd,
                "--dp-world-size",
                str(args.covariance_world_size),
            ]
        else:
            collect_cmd = [python_exe, *collect_cmd]
        if args.rank_source_fusion:
            collect_cmd.append("--save-per-source-covariance")
        if args.attn_implementation:
            collect_cmd.extend(["--attn-implementation", args.attn_implementation])
        if args.target_total_rows is not None:
            collect_cmd.extend(["--target-total-rows", str(args.target_total_rows)])
        if covariance_has_partials:
            print(f"[resume] resuming covariance from {covariance_dir}", flush=True)
        _run(collect_cmd)
        covariance_complete = _covariance_complete(covariance_dir)
    elif covariance_complete:
        print(f"[skip] covariance already complete at {covariance_dir}", flush=True)

    use_rank_schedule = not args.uniform_rank and not args.skip_rank_schedule
    rank_schedule_complete = _rank_schedule_complete(rank_schedule_json)
    if use_rank_schedule and not rank_schedule_complete:
        schedule_cmd = [
            python_exe,
            str(repo_root / "scripts" / "derive_gpt_oss_care_rank_schedule.py"),
            "--model-path",
            str(model_path),
            "--output-json",
            str(rank_schedule_json),
            "--target-rank",
            str(args.target_rank),
            "--min-rank",
            str(args.min_rank),
            "--qk-rope-head-dim",
            str(args.qk_rope_head_dim),
            "--covariance-shrinkage",
            str(args.covariance_shrinkage),
        ]
        if args.rank_source_fusion:
            schedule_cmd.extend(
                _rank_source_fusion_args(
                    dataset_spec_json, covariance_dir, args.target_total_rows
                )
            )
        else:
            schedule_cmd.extend(["--covariance-dir", str(covariance_dir)])
        if args.target_total_rank is not None:
            schedule_cmd.extend(["--target-total-rank", str(args.target_total_rank)])
        if args.max_rank is not None:
            schedule_cmd.extend(["--max-rank", str(args.max_rank)])
        if int(args.round_multiple) > 1:
            schedule_cmd.extend(["--round-multiple", str(args.round_multiple)])
        if args.qk_nope_head_dim is not None:
            schedule_cmd.extend(["--qk-nope-head-dim", str(args.qk_nope_head_dim)])
        _run(schedule_cmd)
        rank_schedule_complete = _rank_schedule_complete(rank_schedule_json)
    elif use_rank_schedule and rank_schedule_complete:
        print(f"[skip] rank schedule already exists at {rank_schedule_json}", flush=True)

    converted_complete = _converted_checkpoint_complete(converted_dir)
    if not args.skip_convert and not converted_complete:
        convert_cmd = [
            python_exe,
            str(repo_root / "scripts" / "convert_gpt_oss_to_care_mla.py"),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(converted_dir),
            "--device",
            "auto",
            "--kv-lora-rank",
            str(args.target_rank),
            "--qk-rope-head-dim",
            str(args.qk_rope_head_dim),
            "--decoupled-rope-dim",
            str(args.decoupled_rope_dim),
            "--decoupled-rope-init",
            args.decoupled_rope_init,
            "--covariance-dir",
            str(covariance_dir),
            "--covariance-shrinkage",
            str(args.covariance_shrinkage),
            "--overwrite",
        ]
        if args.qk_nope_head_dim is not None:
            convert_cmd.extend(["--qk-nope-head-dim", str(args.qk_nope_head_dim)])
        if use_rank_schedule:
            convert_cmd.extend(["--rank-schedule-json", str(rank_schedule_json)])
        _run(convert_cmd)
    elif converted_complete:
        print(f"[skip] converted checkpoint already exists at {converted_dir}", flush=True)

    print(
        json.dumps(
            {
                "pipeline_manifest": str(pipeline_manifest),
                "covariance_dir": str(covariance_dir),
                "rank_schedule_json": str(rank_schedule_json) if use_rank_schedule else None,
                "converted_dir": str(converted_dir),
                "rank_source_fusion": bool(args.rank_source_fusion),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
