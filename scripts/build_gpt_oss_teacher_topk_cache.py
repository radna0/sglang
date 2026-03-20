#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunked teacher top-k distillation caches for GPT-OSS CARE healing."
    )
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_fsdp_helpers():
    helper_path = Path(__file__).resolve().with_name("run_gpt_oss_care_healing_fsdp.py")
    spec = importlib.util.spec_from_file_location("gpt_oss_care_healing_fsdp_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _distributed_context(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    enabled = world_size > 1 and "RANK" in os.environ
    if enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return enabled, rank, world_size, local_rank


def _cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _iter_sharded_batches(base_iter, rank: int, world_size: int):
    batch_index = 0
    for batch in base_iter:
        if batch_index % world_size == rank:
            yield batch
        batch_index += 1


def _full_batch_group_available(
    distributed: bool,
    local_batch: torch.Tensor | None,
    rank: int,
    world_size: int,
    batch_size: int,
) -> tuple[bool, list[int]]:
    local_rows = int(local_batch.shape[0]) if local_batch is not None else 0
    if distributed:
        rows_by_rank = [0 for _ in range(world_size)]
        dist.all_gather_object(rows_by_rank, local_rows)
    else:
        rows_by_rank = [local_rows]
    ready = all(rows == int(batch_size) for rows in rows_by_rank)
    if not ready and rank == 0:
        print(
            f"[teacher-topk-cache] stopping before incomplete global batch rows_by_rank={rows_by_rank}",
            flush=True,
        )
    return ready, rows_by_rank


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    teacher_model_path = Path(args.teacher_model_path).resolve()
    dataset_spec_json = Path(args.dataset_spec_json).resolve()
    distributed, rank, world_size, local_rank = _distributed_context(args)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not distributed or rank == 0:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
            for child in output_dir.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    helpers = _load_fsdp_helpers()
    dtype = getattr(torch, args.dtype)
    local_files_only = teacher_model_path.exists()
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_path,
        trust_remote_code=False,
        use_fast=True,
        local_files_only=local_files_only,
    )
    load_kwargs = {
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if distributed:
        load_kwargs["device_map"] = {"": f"cuda:{local_rank}"}
    else:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(teacher_model_path, **load_kwargs)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    base_iter = helpers._build_batches(
        dataset_spec_json=dataset_spec_json,
        tokenizer=tokenizer,
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        target_total_rows=args.target_total_rows,
        append_eos=bool(args.append_eos),
    )
    if distributed:
        base_iter = _iter_sharded_batches(base_iter, rank=rank, world_size=world_size)

    local_topk_values: list[torch.Tensor] = []
    local_topk_indices: list[torch.Tensor] = []
    completed_steps = 0

    while True:
        if int(args.max_steps) > 0 and completed_steps >= int(args.max_steps):
            break
        try:
            batch = next(base_iter)
        except StopIteration:
            batch = None
        ready, _ = _full_batch_group_available(
            distributed=distributed,
            local_batch=batch,
            rank=rank,
            world_size=world_size,
            batch_size=int(args.batch_size),
        )
        if not ready:
            break
        assert batch is not None
        batch = batch.to(input_device)
        with torch.no_grad():
            logits = model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                use_cache=False,
            ).logits[:, :-1].float().cpu()
        k = min(int(args.topk), logits.shape[-1])
        topk_vals, topk_idx = logits.topk(k=k, dim=-1)
        local_topk_values.append(topk_vals.to(dtype))
        local_topk_indices.append(topk_idx.to(torch.int32))
        completed_steps += 1
        if rank == 0 and completed_steps % max(1, int(args.log_every)) == 0:
            print(
                f"[teacher-topk-cache] step={completed_steps} batch_size={int(args.batch_size)} "
                f"world_size={world_size} rows={completed_steps * int(args.batch_size) * world_size}",
                flush=True,
            )

    local_steps = int(completed_steps)
    steps_by_rank = [0 for _ in range(world_size)]
    if distributed:
        dist.all_gather_object(steps_by_rank, local_steps)
    else:
        steps_by_rank[0] = local_steps

    if local_topk_values:
        topk_values = torch.stack(local_topk_values, dim=0).contiguous()
        topk_indices = torch.stack(local_topk_indices, dim=0).contiguous()
    else:
        topk_values = torch.empty((0, int(args.batch_size), int(args.seq_len) - 1, int(args.topk)), dtype=dtype)
        topk_indices = torch.empty((0, int(args.batch_size), int(args.seq_len) - 1, int(args.topk)), dtype=torch.int32)
    torch.save(
        {
            "rank": rank,
            "num_steps": int(topk_values.shape[0]),
            "topk_values": topk_values,
            "topk_indices": topk_indices,
        },
        output_dir / f"rank{rank:02d}.pt",
    )

    if distributed:
        dist.barrier()

    if rank == 0:
        manifest = {
            "teacher_model_path": str(teacher_model_path),
            "dataset_spec_json": str(dataset_spec_json),
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "world_size": int(world_size),
            "target_total_rows": int(args.target_total_rows) if args.target_total_rows is not None else None,
            "append_eos": bool(args.append_eos),
            "dtype": args.dtype,
            "topk": int(args.topk),
            "completed_steps": int(max(steps_by_rank)),
            "completed_steps_by_rank": [int(step) for step in steps_by_rank],
            "attn_implementation": args.attn_implementation,
            "device_map": "single_gpu_per_rank" if distributed else args.device_map,
        }
        _save_json(output_dir / "manifest.json", manifest)
        print(f"[done] wrote {output_dir}", flush=True)

    _cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
