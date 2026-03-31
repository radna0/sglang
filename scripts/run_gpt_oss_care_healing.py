#!/usr/bin/env python3

import argparse
import copy
import importlib.util
import json
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

_EMPTY_CACHE_MODE = os.environ.get(
    "MLA_EMPTY_CUDA_CACHE_MODE",
    os.environ.get("MLA_EMPTY_CUDA_CACHE", "step"),
).strip().lower()
if _EMPTY_CACHE_MODE in {"1", "true", "yes", "y", "on"}:
    _EMPTY_CACHE_MODE = "inner"
elif _EMPTY_CACHE_MODE in {"0", "false", "no", "off", "disabled", "step-only", "step_only"}:
    _EMPTY_CACHE_MODE = "step"
try:
    _EMPTY_CACHE_EVERY = max(1, int(os.environ.get("MLA_EMPTY_CUDA_CACHE_EVERY", "1")))
except ValueError:
    _EMPTY_CACHE_EVERY = 1

try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None  # type: ignore[assignment]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
)

from gpt_oss_hf_loader import prepare_hf_model_path
from hf_gpt_oss_mla.modeling_gpt_oss_mla import GptOssMlaAttention


def _phase_start() -> float:
    return time.perf_counter()


def _log_phase_timing(
    *,
    phase_log_path: Path,
    dist_ctx: dict[str, Any],
    phase: str,
    started_at: float,
    step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "phase_timing",
        "phase": phase,
        "rank": int(dist_ctx["rank"]),
        "world_size": int(dist_ctx["world_size"]),
        "local_rank": int(dist_ctx["local_rank"]),
        "duration_s": time.perf_counter() - started_at,
    }
    if step is not None:
        record["step"] = int(step)
    if extra:
        record.update(extra)
    if dist_ctx["is_main_process"]:
        with phase_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    return record


def _effective_attention_backend(args: argparse.Namespace) -> str:
    if bool(args.use_unsloth_flex_attention):
        return "unsloth_flex_attention"
    return str(args.attn_implementation)


class _AllToAllRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, output_split_sizes: list[int], input_split_sizes: list[int]):
        if not dist.is_available() or not dist.is_initialized():
            return input_tensor
        ctx.output_split_sizes = tuple(int(x) for x in output_split_sizes)
        ctx.input_split_sizes = tuple(int(x) for x in input_split_sizes)
        output_rows = int(sum(ctx.output_split_sizes))
        output = input_tensor.new_empty((output_rows, *input_tensor.shape[1:]))
        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=list(ctx.output_split_sizes),
            input_split_sizes=list(ctx.input_split_sizes),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input_rows = int(sum(ctx.input_split_sizes))
        grad_input = grad_output.new_empty((grad_input_rows, *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.input_split_sizes),
            input_split_sizes=list(ctx.output_split_sizes),
        )
        return grad_input, None, None


_ASYNC_ALL_TO_ALL_PENDING: dict[int, tuple[torch.cuda.Event, Any]] = {}
_ASYNC_ALL_TO_ALL_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_async_all_to_all_stream(device: torch.device) -> torch.cuda.Stream:
    key = f"{device.type}:{device.index}"
    stream = _ASYNC_ALL_TO_ALL_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _ASYNC_ALL_TO_ALL_STREAMS[key] = stream
    return stream


def _register_async_all_to_all_tensor(tensor: torch.Tensor, event: torch.cuda.Event, work: Any) -> None:
    _ASYNC_ALL_TO_ALL_PENDING[tensor.untyped_storage().data_ptr()] = (event, work)


def _wait_async_all_to_all_tensor(tensor: torch.Tensor) -> torch.Tensor:
    state = _ASYNC_ALL_TO_ALL_PENDING.pop(tensor.untyped_storage().data_ptr(), None)
    if state is None:
        return tensor
    event, work = state
    current_stream = torch.cuda.current_stream(device=tensor.device)
    current_stream.wait_event(event)
    tensor.record_stream(current_stream)
    _ = work
    return tensor


class _AllToAllRowsAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, output_split_sizes: list[int], input_split_sizes: list[int]):
        if not dist.is_available() or not dist.is_initialized():
            return input_tensor
        ctx.output_split_sizes = tuple(int(x) for x in output_split_sizes)
        ctx.input_split_sizes = tuple(int(x) for x in input_split_sizes)
        output_rows = int(sum(ctx.output_split_sizes))
        output = input_tensor.new_empty((output_rows, *input_tensor.shape[1:]))
        if input_tensor.is_cuda:
            comm_stream = _get_async_all_to_all_stream(input_tensor.device)
            with torch.cuda.stream(comm_stream):
                work = dist.all_to_all_single(
                    output,
                    input_tensor.contiguous(),
                    output_split_sizes=list(ctx.output_split_sizes),
                    input_split_sizes=list(ctx.input_split_sizes),
                    async_op=True,
                )
                event = torch.cuda.Event()
                event.record(comm_stream)
            _register_async_all_to_all_tensor(output, event, work)
        else:
            dist.all_to_all_single(
                output,
                input_tensor.contiguous(),
                output_split_sizes=list(ctx.output_split_sizes),
                input_split_sizes=list(ctx.input_split_sizes),
            )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input_rows = int(sum(ctx.input_split_sizes))
        grad_input = grad_output.new_empty((grad_input_rows, *grad_output.shape[1:]))
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.input_split_sizes),
            input_split_sizes=list(ctx.output_split_sizes),
        )
        return grad_input, None, None


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
    parser.add_argument("--use-unsloth-flex-attention", action="store_true")
    parser.add_argument("--disable-hybrid-shared-backbone", action="store_true")
    parser.add_argument(
        "--hybrid-dual-checkpointing-mode",
        default="packed",
        choices=["packed", "split_student_teacher", "packed_forward_student_backward"],
    )
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument(
        "--expert-parallel-overlap-mode",
        default="none",
        choices=["none", "return_streamed", "pipeline2"],
    )
    parser.add_argument(
        "--local-mxfp4-routing-mode",
        default="dense_logits",
        choices=["dense_logits", "top1_local"],
    )
    parser.add_argument(
        "--expert-partition-mode",
        default="contiguous",
        choices=["contiguous", "strided"],
    )
    parser.add_argument("--expert-parallel-profile-occupancy", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing-mode",
        default="native",
        choices=["native", "unsloth", "unsloth_plain"],
    )
    parser.add_argument("--layers-per-checkpoint", default="sqrt")
    parser.add_argument("--distributed-wrapper", default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--fsdp-forward-prefetch", action="store_true")
    parser.add_argument(
        "--fsdp-backward-prefetch",
        default="none",
        choices=["none", "pre", "post"],
    )
    parser.add_argument("--fsdp-disable-limit-all-gathers", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--dataloader-pin-memory", action="store_true")
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--distill-chunk-tokens", type=int, default=128)
    parser.add_argument("--distill-vocab-chunk", type=int, default=2048)
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


def _parse_layers_per_checkpoint(value: str) -> Optional[Any]:
    if value is None:
        return "sqrt"
    text = str(value).strip().lower()
    if text in {"", "none"}:
        return None
    if text == "sqrt":
        return "sqrt"
    return int(text)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _distributed_context() -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
        "is_main_process": rank == 0,
    }


def _init_distributed_if_needed(device: torch.device, ctx: dict[str, Any]) -> torch.device:
    if not ctx["is_distributed"]:
        return device
    if device.type != "cuda":
        raise ValueError("Distributed healing requires CUDA devices.")
    local_rank = int(ctx["local_rank"])
    visible_device_count = torch.cuda.device_count()
    if visible_device_count <= 0:
        raise ValueError("No visible CUDA devices available for distributed healing.")
    if local_rank >= visible_device_count:
        local_rank = local_rank % visible_device_count
    torch.cuda.set_device(local_rank)
    bound_device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl", device_id=bound_device)
        except TypeError:
            dist.init_process_group(backend="nccl")
    return bound_device


def _barrier_if_needed(ctx: dict[str, Any]) -> None:
    if ctx["is_distributed"] and dist.is_initialized():
        try:
            dist.barrier(device_ids=[ctx["local_rank"]])
        except TypeError:
            dist.barrier()


def _reduce_int_across_ranks(value: int) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return int(value)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_trainable_grads(trainable_params: list[torch.nn.Parameter], *, world_size: int) -> None:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    scale = 1.0 / float(world_size)
    for param in trainable_params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.mul_(scale)


def _collect_cuda_visibility_runtime(device: torch.device) -> dict[str, Any]:
    visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    current_device = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            current_device = int(torch.cuda.current_device())
        except Exception:
            current_device = None
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_cuda_visible_devices": os.environ.get("MLA_EXPECTED_CUDA_VISIBLE_DEVICES"),
        "expect_single_visible_device": os.environ.get("MLA_EXPECT_SINGLE_VISIBLE_DEVICE"),
        "visible_cuda_device_count": visible_count,
        "current_cuda_device": current_device,
        "launcher_local_rank": os.environ.get("MLA_ORIGINAL_LOCAL_RANK"),
    }


def _validate_rank_cuda_visibility(device: torch.device) -> dict[str, Any]:
    runtime = _collect_cuda_visibility_runtime(device)
    expected_single = str(runtime["expect_single_visible_device"]).lower() in {"1", "true", "yes", "on"}
    if not expected_single:
        return runtime
    if runtime["visible_cuda_device_count"] != 1:
        raise RuntimeError(
            "Per-rank launcher expected exactly one visible CUDA device, "
            f"but saw {runtime['visible_cuda_device_count']} with "
            f"CUDA_VISIBLE_DEVICES={runtime['cuda_visible_devices']!r}."
        )
    expected_cuda_visible_devices = runtime["expected_cuda_visible_devices"]
    actual_cuda_visible_devices = runtime["cuda_visible_devices"]
    if expected_cuda_visible_devices and actual_cuda_visible_devices != expected_cuda_visible_devices:
        raise RuntimeError(
            "Per-rank launcher expected CUDA_VISIBLE_DEVICES="
            f"{expected_cuda_visible_devices!r}, got {actual_cuda_visible_devices!r}."
        )
    return runtime


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (DDP, FSDP)) else model


def _wrap_trainable_mla_submodules_fsdp(student_model: nn.Module, *, local_rank: int) -> None:
    device_id = torch.device(f"cuda:{local_rank}")
    base_fsdp_kwargs = _build_fsdp_kwargs(
        device_id=device_id,
        limit_all_gathers=True,
        forward_prefetch=False,
        backward_prefetch="none",
    )
    trainable_child_names = (
        "q_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    )
    for module in student_model.modules():
        if not isinstance(module, GptOssMlaAttention):
            continue
        for child_name in trainable_child_names:
            child = getattr(module, child_name, None)
            if child is None or isinstance(child, FSDP):
                continue
            has_trainable = any(param.requires_grad for param in child.parameters())
            if not has_trainable:
                continue
            wrapped = FSDP(
                child,
                **base_fsdp_kwargs,
            )
            setattr(module, child_name, wrapped)


def _wrap_shared_moe_experts_fsdp(layer: nn.Module, *, fsdp_kwargs: dict[str, Any]) -> None:
    layer = _unwrap_model(layer)
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    if experts is None or isinstance(experts, FSDP) or isinstance(experts, ExpertParallelGptOssExperts):
        return
    mlp.experts = FSDP(experts, **fsdp_kwargs)


def _build_fsdp_kwargs(
    *,
    device_id: torch.device,
    limit_all_gathers: bool,
    forward_prefetch: bool,
    backward_prefetch: str,
) -> dict[str, Any]:
    backward_prefetch_mode = None
    if backward_prefetch == "pre":
        backward_prefetch_mode = BackwardPrefetch.BACKWARD_PRE
    elif backward_prefetch == "post":
        backward_prefetch_mode = BackwardPrefetch.BACKWARD_POST
    return {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": device_id,
        "use_orig_params": True,
        "limit_all_gathers": limit_all_gathers,
        "forward_prefetch": forward_prefetch,
        "backward_prefetch": backward_prefetch_mode,
        "sync_module_states": False,
    }


def _wrap_decoder_layers_fsdp(
    student_model: nn.Module,
    *,
    local_rank: int,
    fsdp_kwargs: dict[str, Any],
) -> None:
    device_id = torch.device(f"cuda:{local_rank}")
    base_model = _unwrap_model(student_model)
    layers = getattr(getattr(base_model, "model", None), "layers", None)
    if layers is None:
        return
    for layer_idx, layer in enumerate(layers):
        if isinstance(layer, FSDP):
            continue
        _wrap_shared_moe_experts_fsdp(layer, fsdp_kwargs=fsdp_kwargs)
        layers[layer_idx] = FSDP(layer, **fsdp_kwargs)


def _wrap_student_model_fsdp(
    student_model: nn.Module,
    *,
    local_rank: int,
    wrap_decoder_layers: bool = False,
    forward_prefetch: bool = False,
    backward_prefetch: str = "none",
    limit_all_gathers: bool = True,
) -> nn.Module:
    device_id = torch.device(f"cuda:{local_rank}")
    fsdp_kwargs = _build_fsdp_kwargs(
        device_id=device_id,
        limit_all_gathers=limit_all_gathers,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
    )
    if wrap_decoder_layers:
        _wrap_decoder_layers_fsdp(
            student_model,
            local_rank=local_rank,
            fsdp_kwargs=fsdp_kwargs,
        )
    else:
        _wrap_trainable_mla_submodules_fsdp(student_model, local_rank=local_rank)
    return FSDP(
        student_model,
        **fsdp_kwargs,
        ignored_modules=[student_model.lm_head],
    )


def _run_rank_local_load(loader_fn, *, ctx: dict[str, Any]):
    if not ctx["is_distributed"]:
        return loader_fn()
    if os.environ.get("MLA_SERIAL_RANK_LOAD", "").lower() not in {"1", "true", "yes", "on"}:
        return loader_fn()
    result = None
    for load_rank in range(ctx["world_size"]):
        if ctx["rank"] == load_rank:
            result = loader_fn()
        _barrier_if_needed(ctx)
    return result


def _normalize_module_device(spec: Any) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if isinstance(spec, int):
        return torch.device(f"cuda:{spec}")
    if isinstance(spec, str):
        if spec == "disk":
            return torch.device("cpu")
        if spec.startswith("cuda:"):
            return torch.device(spec)
        if spec == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def _summarize_module_devices(module: nn.Module, *, limit: int = 32) -> dict[str, Any]:
    param_counts: dict[str, int] = {}
    cpu_params: list[str] = []
    for name, param in module.named_parameters():
        key = str(param.device)
        param_counts[key] = param_counts.get(key, 0) + int(param.numel())
        if param.device.type == "cpu" and len(cpu_params) < limit:
            cpu_params.append(name)
    buffer_counts: dict[str, int] = {}
    cpu_buffers: list[str] = []
    for name, buffer in module.named_buffers():
        key = str(buffer.device)
        buffer_counts[key] = buffer_counts.get(key, 0) + int(buffer.numel())
        if buffer.device.type == "cpu" and len(cpu_buffers) < limit:
            cpu_buffers.append(name)
    return {
        "param_counts": param_counts,
        "buffer_counts": buffer_counts,
        "cpu_params": cpu_params,
        "cpu_buffers": cpu_buffers,
    }


def _normalize_module_floating_dtypes(module: nn.Module, *, target_dtype: torch.dtype) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point() and param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)
        for buffer in module.buffers():
            if buffer.is_floating_point() and buffer.dtype != target_dtype:
                buffer.data = buffer.data.to(dtype=target_dtype)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _canonicalize_dataset_spec_paths(spec: dict[str, Any], dataset_spec_json: Path) -> dict[str, Any]:
    spec_dir = dataset_spec_json.resolve().parent
    for source in spec.get("sources", []):
        filelist = source.get("filelist")
        if not filelist:
            continue
        filelist_path = Path(filelist)
        if filelist_path.exists():
            source["filelist"] = str(filelist_path.resolve())
            continue
        candidates: list[Path] = []
        if source.get("name"):
            candidates.append(spec_dir / str(source["name"]) / filelist_path.name)
        if dataset_spec_json.parent.name in filelist_path.parts:
            try:
                dataset_dir_index = filelist_path.parts.index(dataset_spec_json.parent.name)
                relative_tail = Path(*filelist_path.parts[dataset_dir_index + 1 :])
                candidates.append(spec_dir / relative_tail)
            except ValueError:
                pass
        candidates.append(spec_dir / filelist_path.name)
        for candidate in candidates:
            if candidate.exists():
                source["filelist"] = str(candidate.resolve())
                break
    return spec


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
    model = _unwrap_model(model)
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
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
        sinks_tensor: torch.Tensor,
        target_device: Optional[torch.device] = None,
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
        q_device = target_device or _infer_input_device(original_attn)
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
            device=q_device,
        )
        self.sinks = sinks_tensor.to(device=q_device, dtype=q_dtype)
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


def _resolve_workspace_mirror_path(candidate: Path) -> Path:
    candidate = Path(candidate)
    if candidate.exists():
        return candidate.resolve()

    anchor = Path("/workspace/MLA_TRANSFER/workspace")
    try:
        relative = candidate.resolve(strict=False).relative_to(anchor)
    except Exception:
        return candidate.resolve(strict=False)

    local_workspace_root = Path(__file__).resolve().parents[2]
    mirrored = local_workspace_root / relative
    if mirrored.exists():
        return mirrored.resolve()
    return candidate.resolve(strict=False)



def _resolve_source_model_path(student_model_path: Path) -> Path:
    config = _load_json(student_model_path / "config.json")
    source_path = config.get("care_mla_conversion", {}).get("source_model_path")
    if source_path:
        return _resolve_workspace_mirror_path(Path(source_path))
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
    hf_device_map = getattr(student_model, "hf_device_map", None) or {}

    index = TensorIndex(student_checkpoint_path)
    attention_modules = []
    for layer_id, layer in enumerate(student_model.model.layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        layer_prefix = f"model.layers.{layer_id}"
        layer_device_spec = hf_device_map.get(prefix, hf_device_map.get(layer_prefix, None))
        target_device = _normalize_module_device(layer_device_spec)
        q_proj_weight = index.get_tensor(f"{prefix}.q_proj.weight")
        q_proj_bias = index.maybe_get_tensor(f"{prefix}.q_proj.bias")
        o_proj_weight = index.get_tensor(f"{prefix}.o_proj.weight")
        o_proj_bias = index.maybe_get_tensor(f"{prefix}.o_proj.bias")
        kv_a_weight = index.get_tensor(f"{prefix}.kv_a_proj_with_mqa.weight")
        kv_a_bias = index.maybe_get_tensor(f"{prefix}.kv_a_proj_with_mqa.bias")
        kv_b_weight = index.get_tensor(f"{prefix}.kv_b_proj.weight")
        sinks_tensor = index.get_tensor(f"{prefix}.sinks")
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
            sinks_tensor=sinks_tensor,
            target_device=target_device,
        )
        layer.self_attn = patched
        attention_modules.append(patched)

    return {
        "student_load_mode": "live_patch",
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "base_qk_rope_head_dim": base_qk_rope_head_dim,
        "decoupled_rope_dim": qk_rope_head_dim - base_qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "kv_lora_rank_per_layer": rank_schedule,
        "attention_modules": attention_modules,
        "student_config": student_config,
    }


def _build_direct_mla_patch_info(student_model: nn.Module, student_checkpoint_path: Path) -> dict[str, Any]:
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
    attention_modules = [layer.self_attn for layer in student_model.model.layers]
    return {
        "student_load_mode": "direct_mla",
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "base_qk_rope_head_dim": base_qk_rope_head_dim,
        "decoupled_rope_dim": qk_rope_head_dim - base_qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "kv_lora_rank_per_layer": rank_schedule,
        "attention_modules": attention_modules,
        "student_config": student_config,
    }


class HybridGptOssDecoderLayerForHealing(nn.Module):
    def __init__(self, shared_layer: nn.Module, student_attn: nn.Module) -> None:
        super().__init__()
        self.hidden_size = shared_layer.hidden_size
        self.teacher_attn = shared_layer.self_attn
        self.self_attn = student_attn
        self.mlp = shared_layer.mlp
        self.input_layernorm = shared_layer.input_layernorm
        self.post_attention_layernorm = shared_layer.post_attention_layernorm
        self.attention_type = shared_layer.attention_type
        self._dual_microphase_stats: dict[str, float] = {
            "student_attn": 0.0,
            "teacher_attn": 0.0,
            "post_attn_layernorm": 0.0,
            "mlp_router": 0.0,
            "mlp_experts": 0.0,
            "mlp_total": 0.0,
        }

    def reset_dual_microphase_stats(self) -> None:
        for key in self._dual_microphase_stats:
            self._dual_microphase_stats[key] = 0.0

    def consume_dual_microphase_stats(self) -> dict[str, float]:
        stats = {key: float(value) for key, value in self._dual_microphase_stats.items()}
        self.reset_dual_microphase_stats()
        return stats

    def _record_dual_microphase(self, phase: str, duration_s: float) -> None:
        if phase in self._dual_microphase_stats:
            self._dual_microphase_stats[phase] += float(duration_s)

    def _forward_post_attention(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_dtype = next(attn_module.parameters()).dtype
        if hidden_states.dtype != attn_dtype:
            hidden_states = hidden_states.to(attn_dtype)
        if position_embeddings is not None:
            position_embeddings = tuple(
                emb.to(attn_dtype) if torch.is_floating_point(emb) and emb.dtype != attn_dtype else emb
                for emb in position_embeddings
        )
        hidden_states, _ = attn_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if hidden_states.dtype != residual.dtype:
            hidden_states = hidden_states.to(residual.dtype)
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward_with_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp.experts, ExpertParallelGptOssExperts):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
            _, router_scores, router_indices = self.mlp.router(hidden_states_flat)
            hidden_states_flat = self.mlp.experts(hidden_states_flat, router_indices, router_scores)
            hidden_states = hidden_states_flat.reshape(batch_size, sequence_length, hidden_dim)
        else:
            hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward_with_attn(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self._forward_post_attention(
            attn_module,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self._forward_with_mlp(hidden_states)
        return hidden_states

    def forward_student(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_with_attn(self.self_attn, *args, **kwargs)

    def forward_teacher(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_with_attn(self.teacher_attn, *args, **kwargs)

    def forward_dual(
        self,
        hidden_states: torch.Tensor,
        *,
        teacher_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_attn_started_at = _phase_start()
        student_hidden_states = self._forward_post_attention(
            self.self_attn,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        self._record_dual_microphase("student_attn", time.perf_counter() - student_attn_started_at)
        with torch.no_grad():
            teacher_attn_started_at = _phase_start()
            teacher_hidden_states = self._forward_post_attention(
                self.teacher_attn,
                teacher_hidden_states,
                attention_mask,
                position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        self._record_dual_microphase("teacher_attn", time.perf_counter() - teacher_attn_started_at)

        student_residual = student_hidden_states
        teacher_residual = teacher_hidden_states
        mlp_total_started_at = _phase_start()
        post_attn_layernorm_started_at = _phase_start()
        student_mlp_inputs = self.post_attention_layernorm(student_hidden_states)
        teacher_mlp_inputs = self.post_attention_layernorm(teacher_hidden_states)
        packed_mlp_inputs = torch.cat([student_mlp_inputs, teacher_mlp_inputs], dim=0)
        self._record_dual_microphase("post_attn_layernorm", time.perf_counter() - post_attn_layernorm_started_at)
        if isinstance(self.mlp.experts, ExpertParallelGptOssExperts):
            packed_batch, packed_sequence, hidden_dim = packed_mlp_inputs.shape
            packed_hidden_flat = packed_mlp_inputs.reshape(-1, hidden_dim)
            mlp_router_started_at = _phase_start()
            _, router_scores, router_indices = self.mlp.router(packed_hidden_flat)
            self._record_dual_microphase("mlp_router", time.perf_counter() - mlp_router_started_at)
            mlp_experts_started_at = _phase_start()
            packed_hidden_flat = self.mlp.experts(packed_hidden_flat, router_indices, router_scores)
            self._record_dual_microphase("mlp_experts", time.perf_counter() - mlp_experts_started_at)
            packed_hidden_states = packed_hidden_flat.reshape(packed_batch, packed_sequence, hidden_dim)
        else:
            mlp_experts_started_at = _phase_start()
            packed_hidden_states, _ = self.mlp(packed_mlp_inputs)
            self._record_dual_microphase("mlp_experts", time.perf_counter() - mlp_experts_started_at)
        student_hidden_states, teacher_hidden_states = packed_hidden_states.split(
            [student_hidden_states.shape[0], teacher_hidden_states.shape[0]],
            dim=0,
        )
        student_hidden_states = student_residual + student_hidden_states
        teacher_hidden_states = (teacher_residual + teacher_hidden_states).detach()
        self._record_dual_microphase("mlp_total", time.perf_counter() - mlp_total_started_at)
        return student_hidden_states, teacher_hidden_states

    def forward(self, *args, route: str = "student", **kwargs) -> torch.Tensor:
        if route == "student":
            return self.forward_student(*args, **kwargs)
        if route == "teacher":
            return self.forward_teacher(*args, **kwargs)
        if route == "dual":
            return self.forward_dual(*args, **kwargs)
        raise ValueError(f"Unsupported route: {route}")


class ExpertParallelGptOssExperts(nn.Module):
    def __init__(
        self,
        full_experts: nn.Module,
        *,
        rank: int,
        world_size: int,
        overlap_mode: str = "none",
        local_mxfp4_routing_mode: str = "dense_logits",
        partition_mode: str = "contiguous",
        profile_occupancy: bool = False,
    ) -> None:
        super().__init__()
        self.intermediate_size = int(full_experts.intermediate_size)
        self.num_experts = int(full_experts.num_experts)
        self.hidden_size = int(full_experts.hidden_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.overlap_mode = str(overlap_mode)
        self.local_mxfp4_routing_mode = str(local_mxfp4_routing_mode)
        self.partition_mode = str(partition_mode)
        self.profile_occupancy = bool(profile_occupancy)
        if self.num_experts % self.world_size != 0:
            raise ValueError(
                f"Expert-parallel requires num_experts divisible by world_size; "
                f"got num_experts={self.num_experts} world_size={self.world_size}"
            )
        self.local_num_experts = self.num_experts // self.world_size
        if self.partition_mode == "contiguous":
            local_global_expert_indices = torch.arange(
                self.rank * self.local_num_experts,
                (self.rank + 1) * self.local_num_experts,
                dtype=torch.long,
            )
        elif self.partition_mode == "strided":
            local_global_expert_indices = torch.arange(
                self.rank,
                self.num_experts,
                self.world_size,
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported expert partition mode: {self.partition_mode!r}")
        self.local_global_expert_indices_cpu = local_global_expert_indices.clone()
        self.expert_start = int(local_global_expert_indices[0].item())
        self.expert_stop = int(local_global_expert_indices[-1].item()) + 1
        self.alpha = float(getattr(full_experts, "alpha", 1.702))
        self.limit = float(getattr(full_experts, "limit", 7.0))
        self.local_mxfp4_experts: Optional[nn.Module] = None
        self._microphase_stats: dict[str, float] = {
            "dispatch_counts": 0.0,
            "hidden_exchange": 0.0,
            "metadata_exchange": 0.0,
            "packed_local_compute": 0.0,
            "return_exchange": 0.0,
            "merge": 0.0,
        }
        self._occupancy_stats: dict[str, float] = {
            "sample_count": 0.0,
            "routed_tokens": 0.0,
            "local_tokens": 0.0,
            "rank_load_imbalance": 0.0,
            "rank_load_std": 0.0,
            "local_active_experts": 0.0,
            "local_max_tokens_per_expert": 0.0,
            "global_active_experts": 0.0,
            "global_max_tokens_per_expert": 0.0,
            "global_mean_tokens_per_active_expert": 0.0,
            "global_top4_expert_share": 0.0,
            "global_expert_imbalance": 0.0,
        }
        self._pending_rank_loads: Optional[torch.Tensor] = None
        self._pending_routed_tokens: int = 0
        self._pending_local_tokens: int = 0

        gate_up_precision = getattr(full_experts, "gate_up_proj_precision_config", None)
        down_precision = getattr(full_experts, "down_proj_precision_config", None)
        if gate_up_precision is not None and down_precision is not None:
            self.local_mxfp4_experts = self._build_local_mxfp4_experts(
                full_experts,
                expert_indices=self.local_global_expert_indices_cpu,
            )
            return

        gate_up_proj = self._select_expert_tensor_like(
            full_experts,
            expert_indices=self.local_global_expert_indices_cpu,
            attr_name="gate_up_proj",
        )
        gate_up_proj_bias = self._select_expert_tensor_like(
            full_experts.gate_up_proj_bias,
            expert_indices=self.local_global_expert_indices_cpu,
        )
        down_proj = self._select_expert_tensor_like(
            full_experts,
            expert_indices=self.local_global_expert_indices_cpu,
            attr_name="down_proj",
        )
        down_proj_bias = self._select_expert_tensor_like(
            full_experts.down_proj_bias,
            expert_indices=self.local_global_expert_indices_cpu,
        )

        self.gate_up_proj = nn.Parameter(gate_up_proj, requires_grad=False)
        self.gate_up_proj_bias = nn.Parameter(gate_up_proj_bias, requires_grad=False)
        self.down_proj = nn.Parameter(down_proj, requires_grad=False)
        self.down_proj_bias = nn.Parameter(down_proj_bias, requires_grad=False)

    @staticmethod
    def _slice_layout_for_experts(layout: Any, *, length: int) -> Any:
        import copy

        new_layout = copy.copy(layout)
        old_shape = tuple(getattr(layout, "shape", ()))
        if old_shape:
            new_shape = (int(length), *old_shape[1:])
            new_layout.shape = new_shape
            if hasattr(new_layout, "leading_shape"):
                new_layout.leading_shape = list(new_shape[:-2])
            if hasattr(new_layout, "K"):
                new_layout.K = new_shape[-2]
            if hasattr(new_layout, "N"):
                new_layout.N = new_shape[-1]
        return new_layout

    @staticmethod
    def _select_custom_tensor_wrapper(value: Any, *, expert_indices: torch.Tensor) -> Any:
        storage = getattr(value, "storage", None)
        if storage is None or not isinstance(getattr(storage, "data", None), torch.Tensor):
            raise TypeError(f"Unsupported custom expert tensor wrapper type for slicing: {type(value)!r}")
        tensor_cls = type(value)
        storage_cls = type(storage)
        source_data = storage.data
        expert_indices = expert_indices.to(device=source_data.device, dtype=torch.long)
        sliced_view = source_data.index_select(0, expert_indices)
        source_shape = tuple(source_data.shape)
        source_stride = tuple(source_data.stride())
        sliced_shape = (int(expert_indices.numel()), *source_shape[1:])
        sliced_stride = (source_stride[0], *source_stride[1:])
        sliced_data = torch.empty_strided(
            sliced_shape,
            sliced_stride,
            dtype=sliced_view.dtype,
            device=sliced_view.device,
        )
        sliced_data.copy_(sliced_view)
        sliced_layout = ExpertParallelGptOssExperts._slice_layout_for_experts(
            storage.layout,
            length=int(expert_indices.numel()),
        )
        sliced_storage = storage_cls(sliced_data, layout=sliced_layout)
        shape = getattr(value, "shape", None)
        if shape is not None:
            shape = torch.Size((int(expert_indices.numel()), *tuple(shape)[1:]))
        shape_max = getattr(value, "shape_max", None)
        if shape_max is not None:
            shape_max = torch.Size((int(expert_indices.numel()), *tuple(shape_max)[1:]))
        return tensor_cls(sliced_storage, value.dtype, shape=shape, shape_max=shape_max)

    @staticmethod
    def _select_expert_tensor_like(
        value: Any,
        *,
        expert_indices: torch.Tensor,
        attr_name: Optional[str] = None,
    ) -> torch.Tensor:
        if attr_name is not None:
            value = getattr(value, attr_name)
        expert_indices = expert_indices.to(dtype=torch.long)
        if isinstance(value, nn.Parameter):
            return value.detach().index_select(0, expert_indices.to(device=value.device)).clone().contiguous()
        if isinstance(value, torch.Tensor):
            return value.detach().index_select(0, expert_indices.to(device=value.device)).clone().contiguous()
        data = getattr(value, "data", None)
        if isinstance(data, torch.Tensor):
            return data.index_select(0, expert_indices.to(device=data.device)).clone().contiguous()
        raise TypeError(f"Unsupported expert tensor wrapper type for expert-parallel slicing: {type(value)!r}")

    @staticmethod
    def _select_precision_config_weight_scale(
        precision_config: Any,
        expert_indices: torch.Tensor,
    ) -> Any:
        import copy

        new_precision_config = copy.copy(precision_config)
        new_precision_config.weight_scale = ExpertParallelGptOssExperts._select_custom_tensor_wrapper(
            precision_config.weight_scale,
            expert_indices=expert_indices,
        )
        return new_precision_config

    @staticmethod
    def _build_local_mxfp4_experts(full_experts: nn.Module, *, expert_indices: torch.Tensor) -> nn.Module:
        from transformers.integrations.mxfp4 import Mxfp4GptOssExperts

        local_num_experts = int(expert_indices.numel())
        local_config = type(
            "LocalMxfp4ExpertsConfig",
            (),
            {
                "num_local_experts": local_num_experts,
                "intermediate_size": int(full_experts.intermediate_size),
                "hidden_size": int(full_experts.hidden_size),
                "swiglu_limit": float(getattr(full_experts, "limit", 7.0)),
            },
        )()
        local_experts = Mxfp4GptOssExperts(local_config)
        for proj_name in ("gate_up_proj", "down_proj"):
            if proj_name in local_experts._parameters:
                del local_experts._parameters[proj_name]
        local_experts.gate_up_proj = ExpertParallelGptOssExperts._select_custom_tensor_wrapper(
            full_experts.gate_up_proj,
            expert_indices=expert_indices,
        )
        local_experts.down_proj = ExpertParallelGptOssExperts._select_custom_tensor_wrapper(
            full_experts.down_proj,
            expert_indices=expert_indices,
        )
        local_experts.gate_up_proj_bias = nn.Parameter(
            ExpertParallelGptOssExperts._select_expert_tensor_like(
                full_experts.gate_up_proj_bias,
                expert_indices=expert_indices,
            ),
            requires_grad=False,
        )
        local_experts.down_proj_bias = nn.Parameter(
            ExpertParallelGptOssExperts._select_expert_tensor_like(
                full_experts.down_proj_bias,
                expert_indices=expert_indices,
            ),
            requires_grad=False,
        )
        local_experts.gate_up_proj_precision_config = ExpertParallelGptOssExperts._select_precision_config_weight_scale(
            full_experts.gate_up_proj_precision_config,
            expert_indices,
        )
        local_experts.down_proj_precision_config = ExpertParallelGptOssExperts._select_precision_config_weight_scale(
            full_experts.down_proj_precision_config,
            expert_indices,
        )
        local_experts.alpha = float(getattr(full_experts, "alpha", 1.702))
        local_experts.limit = float(getattr(full_experts, "limit", 7.0))
        local_experts.eval()
        return local_experts

    def _owner_and_local_expert_indices(
        self,
        router_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.partition_mode == "contiguous":
            owner_ranks = torch.div(router_indices, self.local_num_experts, rounding_mode="floor")
            local_expert_indices = router_indices % self.local_num_experts
            return owner_ranks.to(torch.long), local_expert_indices.to(torch.long)
        if self.partition_mode == "strided":
            owner_ranks = router_indices % self.world_size
            local_expert_indices = torch.div(router_indices, self.world_size, rounding_mode="floor")
            return owner_ranks.to(torch.long), local_expert_indices.to(torch.long)
        raise ValueError(f"Unsupported expert partition mode: {self.partition_mode!r}")

    def reset_microphase_stats(self) -> None:
        for key in self._microphase_stats:
            self._microphase_stats[key] = 0.0
        for key in self._occupancy_stats:
            self._occupancy_stats[key] = 0.0
        self._pending_rank_loads = None
        self._pending_routed_tokens = 0
        self._pending_local_tokens = 0

    def consume_microphase_stats(self) -> dict[str, Any]:
        stats = {key: float(value) for key, value in self._microphase_stats.items()}
        occupancy = {key: float(value) for key, value in self._occupancy_stats.items()}
        self.reset_microphase_stats()
        return {
            "durations": stats,
            "occupancy": occupancy,
        }

    def _record_microphase(self, phase: str, duration_s: float) -> None:
        if phase in self._microphase_stats:
            self._microphase_stats[phase] += float(duration_s)

    def _record_occupancy_snapshot(
        self,
        *,
        rank_loads: torch.Tensor,
        local_hist: torch.Tensor,
    ) -> None:
        if not self.profile_occupancy:
            return
        local_hist = local_hist.to(dtype=torch.int64)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered = [torch.empty_like(local_hist) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_hist)
            global_hist = torch.cat(gathered, dim=0)
        else:
            global_hist = local_hist

        rank_loads_f = rank_loads.to(dtype=torch.float32)
        rank_load_mean = float(rank_loads_f.mean().item()) if rank_loads_f.numel() > 0 else 0.0
        rank_load_max = float(rank_loads_f.max().item()) if rank_loads_f.numel() > 0 else 0.0
        rank_load_std = (
            float(rank_loads_f.std(unbiased=False).item()) if rank_loads_f.numel() > 1 else 0.0
        )
        rank_load_imbalance = rank_load_max / max(rank_load_mean, 1e-6) if rank_load_mean > 0 else 0.0

        local_active_experts = int((local_hist > 0).sum().item()) if local_hist.numel() > 0 else 0
        local_max_tokens = int(local_hist.max().item()) if local_hist.numel() > 0 else 0

        active_global_hist = global_hist[global_hist > 0]
        global_active_experts = int(active_global_hist.numel())
        global_max_tokens = int(global_hist.max().item()) if global_hist.numel() > 0 else 0
        global_mean_active = (
            float(active_global_hist.to(torch.float32).mean().item()) if global_active_experts > 0 else 0.0
        )
        topk = min(4, int(global_hist.numel()))
        global_top4_expert_share = 0.0
        if topk > 0:
            global_total = float(global_hist.sum().item())
            if global_total > 0:
                global_top4_expert_share = float(
                    torch.topk(global_hist.to(torch.float32), k=topk).values.sum().item() / global_total
                )
        global_expert_imbalance = (
            global_max_tokens / max(global_mean_active, 1e-6) if global_mean_active > 0 else 0.0
        )

        self._occupancy_stats["sample_count"] += 1.0
        self._occupancy_stats["routed_tokens"] += float(self._pending_routed_tokens)
        self._occupancy_stats["local_tokens"] += float(self._pending_local_tokens)
        self._occupancy_stats["rank_load_imbalance"] += float(rank_load_imbalance)
        self._occupancy_stats["rank_load_std"] += float(rank_load_std)
        self._occupancy_stats["local_active_experts"] += float(local_active_experts)
        self._occupancy_stats["local_max_tokens_per_expert"] += float(local_max_tokens)
        self._occupancy_stats["global_active_experts"] += float(global_active_experts)
        self._occupancy_stats["global_max_tokens_per_expert"] += float(global_max_tokens)
        self._occupancy_stats["global_mean_tokens_per_active_expert"] += float(global_mean_active)
        self._occupancy_stats["global_top4_expert_share"] += float(global_top4_expert_share)
        self._occupancy_stats["global_expert_imbalance"] += float(global_expert_imbalance)
        self._pending_rank_loads = None
        self._pending_routed_tokens = 0
        self._pending_local_tokens = 0

    @staticmethod
    def _routing_torch_dist_for_rank(
        logits: torch.Tensor,
        n_expts_act: int,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[Any, Any, Any]:
        from transformers.integrations import mxfp4 as mxfp4_mod

        GatherIndx = mxfp4_mod.triton_kernels_hub.routing.GatherIndx
        RoutingData = mxfp4_mod.triton_kernels_hub.routing.RoutingData
        ScatterIndx = mxfp4_mod.triton_kernels_hub.routing.ScatterIndx
        compute_expt_data_torch = mxfp4_mod.triton_kernels_hub.routing.compute_expt_data_torch

        replace_value = -1
        n_tokens = logits.shape[0]
        n_expts_tot = logits.shape[1]
        n_local_experts = n_expts_tot // world_size
        local_expert_start = rank * n_local_experts
        local_expert_end = (rank + 1) * n_local_experts
        n_gates_pad = n_tokens * n_expts_act

        tk_indx = torch.argsort(-logits, dim=1, stable=True)[:, :n_expts_act].long()
        tk_val = torch.take_along_dim(logits, tk_indx, dim=1)
        expt_scal = torch.softmax(tk_val, dim=-1)
        expt_indx, sort_indices = torch.sort(tk_indx.int(), dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        expt_scal = expt_scal.reshape(-1)

        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)[local_expert_start:local_expert_end]
        expt_indx = expt_indx.reshape(-1).to(torch.int32)

        var = 1000
        expt_indx = torch.where(expt_indx < local_expert_start, var, expt_indx)
        topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        expt_indx = torch.where(expt_indx < local_expert_end, expt_indx, replace_value)
        expt_indx = torch.where(local_expert_start <= expt_indx, expt_indx, replace_value)

        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_scal = expt_scal[topk_indx]
        topk_indx = torch.where(gate_indx[topk_indx] == replace_value, replace_value, topk_indx)

        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        expt_data = compute_expt_data_torch(hist, n_local_experts, n_gates_pad)
        hit_experts = n_expts_act
        return RoutingData(gate_scal, hist, n_local_experts, hit_experts, expt_data), gather_indx, scatter_indx

    @staticmethod
    def _mxfp4_routing_primitives() -> tuple[Any, Any, Any, Any]:
        import importlib

        mxfp4_mod = importlib.import_module("transformers.integrations.mxfp4")
        routing_hub = mxfp4_mod.routing_torch_dist.__globals__["triton_kernels_hub"].routing
        return (
            routing_hub.GatherIndx,
            routing_hub.RoutingData,
            routing_hub.ScatterIndx,
            routing_hub.compute_expt_data_torch,
        )

    def _routing_top1_local_experts(
        self,
        local_expert_indices: torch.Tensor,
    ) -> tuple[Any, Any, Any]:
        GatherIndx, RoutingData, ScatterIndx, compute_expt_data_torch = self._mxfp4_routing_primitives()
        row_count = int(local_expert_indices.shape[0])
        if row_count <= 0:
            raise ValueError("Top-1 local routing requires at least one token row.")
        local_expert_indices = local_expert_indices.to(torch.long)
        hist = torch.bincount(local_expert_indices, minlength=self.local_num_experts).to(
            device=local_expert_indices.device,
            dtype=torch.int32,
        )
        global_expert_indices = self.local_global_expert_indices_cpu.to(local_expert_indices.device).index_select(
            0,
            local_expert_indices,
        ).to(torch.int32)
        topk_indx = torch.argsort(global_expert_indices, stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        gate_scal = torch.ones((row_count,), device=local_expert_indices.device, dtype=torch.float32)
        gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
        scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
        expt_data = compute_expt_data_torch(hist, self.local_num_experts, row_count)
        return RoutingData(gate_scal, hist, self.local_num_experts, 1, expt_data), gather_indx, scatter_indx

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def _local_dense_forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        if hidden_states.numel() == 0:
            return next_states
        with torch.no_grad():
            expert_hit = torch.unique(local_expert_indices, sorted=True)
        for local_expert_idx_tensor in expert_hit:
            local_expert_idx = int(local_expert_idx_tensor.item())
            token_mask = local_expert_indices == local_expert_idx
            if not torch.any(token_mask):
                continue
            current_state = hidden_states[token_mask]
            gate_up = current_state @ self.gate_up_proj[local_expert_idx] + self.gate_up_proj_bias[local_expert_idx]
            gated_output = self._apply_gate(gate_up)
            out = gated_output @ self.down_proj[local_expert_idx] + self.down_proj_bias[local_expert_idx]
            next_states[token_mask] = out * routing_weights[token_mask, None].to(out.dtype)
        return next_states

    def _local_mxfp4_forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_mxfp4_experts is None:
            raise RuntimeError("Expected local_mxfp4_experts to be initialized.")
        row_count = int(hidden_states.shape[0])
        if row_count == 0:
            return torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        if self.local_mxfp4_routing_mode == "top1_local" or self.partition_mode != "contiguous":
            routing_data, gather_idx, scatter_idx = self._routing_top1_local_experts(local_expert_indices)
        else:
            global_expert_indices = self.local_global_expert_indices_cpu.to(local_expert_indices.device).index_select(
                0,
                local_expert_indices.to(torch.long),
            )
            logits = hidden_states.new_full((row_count, self.num_experts), -1e9)
            logits.scatter_(1, global_expert_indices.unsqueeze(1), 0.0)
            routing_data, gather_idx, scatter_idx = self._routing_torch_dist_for_rank(
                logits,
                1,
                rank=self.rank,
                world_size=self.world_size,
            )
        local_output = self.local_mxfp4_experts(
            hidden_states,
            routing_data,
            gather_idx,
            scatter_idx=scatter_idx,
        )
        return local_output * routing_weights[:, None].to(local_output.dtype)

    def _local_forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_mxfp4_experts is not None:
            return self._local_mxfp4_forward(hidden_states, local_expert_indices, routing_weights)
        return self._local_dense_forward(hidden_states, local_expert_indices, routing_weights)

    def _local_forward_split(
        self,
        recv_hidden: list[torch.Tensor],
        recv_local_expert_indices: list[torch.Tensor],
        recv_routing_weights: list[torch.Tensor],
        recv_counts: list[int],
        *,
        hidden_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        nonempty_hidden: list[torch.Tensor] = []
        nonempty_local_expert_indices: list[torch.Tensor] = []
        nonempty_routing_weights: list[torch.Tensor] = []
        nonempty_counts: list[int] = []
        for src_rank in range(self.world_size):
            count = int(recv_counts[src_rank])
            if count <= 0:
                continue
            nonempty_hidden.append(recv_hidden[src_rank])
            nonempty_local_expert_indices.append(recv_local_expert_indices[src_rank])
            nonempty_routing_weights.append(recv_routing_weights[src_rank])
            nonempty_counts.append(count)

        if not nonempty_hidden:
            if self.profile_occupancy and self._pending_rank_loads is not None:
                self._record_occupancy_snapshot(
                    rank_loads=self._pending_rank_loads,
                    local_hist=torch.zeros(
                        (self.local_num_experts,),
                        dtype=torch.int64,
                        device=hidden_states.device,
                    ),
                )
            return [self._empty_like_rows(hidden_states, 0) for _ in range(self.world_size)]

        packed_hidden = torch.cat(nonempty_hidden, dim=0)
        packed_local_expert_indices = torch.cat(nonempty_local_expert_indices, dim=0)
        packed_routing_weights = torch.cat(nonempty_routing_weights, dim=0)
        if self.profile_occupancy and self._pending_rank_loads is not None:
            local_hist = torch.bincount(
                packed_local_expert_indices,
                minlength=self.local_num_experts,
            )
            self._record_occupancy_snapshot(
                rank_loads=self._pending_rank_loads,
                local_hist=local_hist,
            )
        packed_output = self._local_forward(
            packed_hidden,
            packed_local_expert_indices,
            packed_routing_weights,
        ).contiguous()

        outputs: list[torch.Tensor] = []
        offset = 0
        for src_rank in range(self.world_size):
            count = int(recv_counts[src_rank])
            if count <= 0:
                outputs.append(self._empty_like_rows(hidden_states, 0))
                continue
            next_offset = offset + count
            outputs.append(packed_output[offset:next_offset].contiguous())
            offset = next_offset
        return outputs

    def _dispatch_counts(self, send_counts: torch.Tensor) -> list[int]:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        return [int(count) for count in recv_counts.tolist()]

    def _gather_recv_count_matrix(self, recv_counts: list[int], device: torch.device) -> list[list[int]]:
        local_counts = torch.tensor(recv_counts, dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local_counts) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_counts)
        return [[int(x) for x in tensor.tolist()] for tensor in gathered]

    @staticmethod
    def _empty_like_rows(hidden_states: torch.Tensor, rows: int) -> torch.Tensor:
        return hidden_states.new_empty((int(rows), hidden_states.shape[-1]))

    @staticmethod
    def _split_tensor_by_counts(tensor: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            next_offset = offset + int(count)
            outputs.append(tensor[offset:next_offset].contiguous())
            offset = next_offset
        return outputs

    @staticmethod
    def _concat_or_empty(send_tensors: list[torch.Tensor]) -> torch.Tensor:
        if not send_tensors:
            raise ValueError("send_tensors must not be empty")
        total_rows = int(sum(int(t.shape[0]) for t in send_tensors))
        prototype = send_tensors[0]
        if total_rows <= 0:
            return prototype.new_empty((0, *prototype.shape[1:]))
        return torch.cat(send_tensors, dim=0).contiguous()

    def _exchange_packed_rows_with_grad(
        self,
        send_tensors: list[torch.Tensor],
        *,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> list[torch.Tensor]:
        packed_send = self._concat_or_empty(send_tensors)
        packed_recv = _AllToAllRows.apply(packed_send, recv_counts, send_counts)
        return self._split_tensor_by_counts(packed_recv, recv_counts)

    def _exchange_packed_rows_no_grad(
        self,
        send_tensors: list[torch.Tensor],
        *,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> list[torch.Tensor]:
        packed_send = self._concat_or_empty(send_tensors)
        prototype = send_tensors[0]
        packed_recv = prototype.new_empty((int(sum(recv_counts)), *prototype.shape[1:]))
        dist.all_to_all_single(
            packed_recv,
            packed_send,
            output_split_sizes=[int(x) for x in recv_counts],
            input_split_sizes=[int(x) for x in send_counts],
        )
        return self._split_tensor_by_counts(packed_recv, recv_counts)

    def _exchange_packed_rows_async_with_grad(
        self,
        send_tensors: list[torch.Tensor],
        *,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> torch.Tensor:
        packed_send = self._concat_or_empty(send_tensors)
        return _AllToAllRowsAsync.apply(packed_send, recv_counts, send_counts)

    def _wait_and_split_async_packed_rows(
        self,
        packed_recv: torch.Tensor,
        recv_counts: list[int],
    ) -> list[torch.Tensor]:
        packed_recv = _wait_async_all_to_all_tensor(packed_recv)
        return self._split_tensor_by_counts(packed_recv, recv_counts)

    @staticmethod
    def _split_tensor_even(tensor: torch.Tensor, num_chunks: int) -> list[torch.Tensor]:
        count = int(tensor.shape[0])
        base = count // int(num_chunks)
        rem = count % int(num_chunks)
        outputs: list[torch.Tensor] = []
        offset = 0
        for chunk_idx in range(int(num_chunks)):
            chunk_count = base + (1 if chunk_idx < rem else 0)
            next_offset = offset + chunk_count
            outputs.append(tensor[offset:next_offset].contiguous())
            offset = next_offset
        return outputs

    @staticmethod
    def _split_count_even(count: int, num_chunks: int) -> list[int]:
        count = int(count)
        base = count // int(num_chunks)
        rem = count % int(num_chunks)
        return [base + (1 if chunk_idx < rem else 0) for chunk_idx in range(int(num_chunks))]

    def _split_send_buckets_into_chunks(
        self,
        *,
        send_token_indices: list[torch.Tensor],
        send_hidden: list[torch.Tensor],
        send_local_expert_indices: list[torch.Tensor],
        send_routing_weights: list[torch.Tensor],
        recv_counts_total: list[int],
        num_chunks: int,
    ) -> list[dict[str, Any]]:
        chunk_data: list[dict[str, Any]] = []
        for _ in range(int(num_chunks)):
            chunk_data.append(
                {
                    "token_indices": [],
                    "hidden": [],
                    "local_expert_indices": [],
                    "routing_weights": [],
                    "send_counts": [],
                    "recv_counts": [],
                }
            )
        for owner_rank in range(self.world_size):
            token_chunks = self._split_tensor_even(send_token_indices[owner_rank], num_chunks)
            hidden_chunks = self._split_tensor_even(send_hidden[owner_rank], num_chunks)
            local_index_chunks = self._split_tensor_even(send_local_expert_indices[owner_rank], num_chunks)
            routing_weight_chunks = self._split_tensor_even(send_routing_weights[owner_rank], num_chunks)
            for chunk_idx in range(int(num_chunks)):
                chunk = chunk_data[chunk_idx]
                chunk["token_indices"].append(token_chunks[chunk_idx])
                chunk["hidden"].append(hidden_chunks[chunk_idx])
                chunk["local_expert_indices"].append(local_index_chunks[chunk_idx])
                chunk["routing_weights"].append(routing_weight_chunks[chunk_idx])
                chunk["send_counts"].append(int(hidden_chunks[chunk_idx].shape[0]))
        recv_counts_chunks = [
            self._split_count_even(recv_count, int(num_chunks))
            for recv_count in recv_counts_total
        ]
        for chunk_idx, chunk in enumerate(chunk_data):
            chunk["recv_counts"] = [
                int(recv_counts_chunks[src_rank][chunk_idx])
                for src_rank in range(self.world_size)
            ]
        return chunk_data

    def _forward_overlap_pipeline2(
        self,
        *,
        hidden_states: torch.Tensor,
        send_token_indices: list[torch.Tensor],
        send_hidden: list[torch.Tensor],
        send_local_expert_indices: list[torch.Tensor],
        send_routing_weights: list[torch.Tensor],
        recv_counts_total: list[int],
    ) -> torch.Tensor:
        device = hidden_states.device
        chunk_data = self._split_send_buckets_into_chunks(
            send_token_indices=send_token_indices,
            send_hidden=send_hidden,
            send_local_expert_indices=send_local_expert_indices,
            send_routing_weights=send_routing_weights,
            recv_counts_total=recv_counts_total,
            num_chunks=2,
        )

        hidden_exchange_started_at = _phase_start()
        for chunk in chunk_data:
            chunk["recv_hidden_async"] = self._exchange_packed_rows_async_with_grad(
                chunk["hidden"],
                send_counts=chunk["send_counts"],
                recv_counts=chunk["recv_counts"],
            )
        self._record_microphase("hidden_exchange", time.perf_counter() - hidden_exchange_started_at)

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=device)
        pending_return_chunks: list[tuple[torch.Tensor, list[int], list[torch.Tensor]]] = []

        for chunk in chunk_data:
            recv_hidden = self._wait_and_split_async_packed_rows(
                chunk["recv_hidden_async"],
                chunk["recv_counts"],
            )

            metadata_exchange_started_at = _phase_start()
            recv_local_expert_indices = self._exchange_packed_rows_no_grad(
                chunk["local_expert_indices"],
                send_counts=chunk["send_counts"],
                recv_counts=chunk["recv_counts"],
            )
            recv_routing_weights = self._exchange_packed_rows_with_grad(
                chunk["routing_weights"],
                send_counts=chunk["send_counts"],
                recv_counts=chunk["recv_counts"],
            )
            self._record_microphase("metadata_exchange", time.perf_counter() - metadata_exchange_started_at)

            packed_local_compute_started_at = _phase_start()
            return_hidden = self._local_forward_split(
                recv_hidden,
                recv_local_expert_indices,
                recv_routing_weights,
                chunk["recv_counts"],
                hidden_states=hidden_states,
            )
            self._record_microphase("packed_local_compute", time.perf_counter() - packed_local_compute_started_at)

            return_exchange_started_at = _phase_start()
            return_async = self._exchange_packed_rows_async_with_grad(
                return_hidden,
                send_counts=chunk["recv_counts"],
                recv_counts=chunk["send_counts"],
            )
            self._record_microphase("return_exchange", time.perf_counter() - return_exchange_started_at)
            pending_return_chunks.append((return_async, chunk["send_counts"], chunk["token_indices"]))

        merge_started_at = _phase_start()
        for return_async, send_counts, token_index_chunks in pending_return_chunks:
            gathered_outputs = self._wait_and_split_async_packed_rows(return_async, send_counts)
            for owner_rank, token_idx in enumerate(token_index_chunks):
                if token_idx.numel() == 0:
                    continue
                next_states.index_add_(0, token_idx, gathered_outputs[owner_rank].to(hidden_states.dtype))
        self._record_microphase("merge", time.perf_counter() - merge_started_at)
        return next_states

    def _build_send_buckets(
        self,
        *,
        hidden_states: torch.Tensor,
        token_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        owner_ranks: torch.Tensor,
        local_expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
        device = hidden_states.device
        flat_owner = owner_ranks[valid_mask].reshape(-1).to(torch.long)
        if flat_owner.numel() == 0:
            send_counts_tensor = torch.zeros((self.world_size,), dtype=torch.long, device=device)
            dispatch_started_at = _phase_start()
            count_matrix = torch.empty((self.world_size, self.world_size), dtype=torch.long, device=device)
            dist.all_gather_into_tensor(count_matrix, send_counts_tensor)
            recv_counts = [int(x) for x in count_matrix[:, self.rank].tolist()]
            self._record_microphase("dispatch_counts", time.perf_counter() - dispatch_started_at)
            empty_rows = self._empty_like_rows(hidden_states, 0)
            empty_indices = torch.empty((0,), dtype=torch.long, device=device)
            empty_weights = hidden_states.new_empty((0,))
            return (
                [empty_indices for _ in range(self.world_size)],
                [empty_rows for _ in range(self.world_size)],
                [empty_indices for _ in range(self.world_size)],
                [empty_weights for _ in range(self.world_size)],
                [0 for _ in range(self.world_size)],
                recv_counts,
            )

        flat_token_idx = token_positions[valid_mask].reshape(-1).to(torch.long)
        flat_local_expert_indices = local_expert_indices[valid_mask].reshape(-1).to(torch.long)
        flat_routing_weights = routing_weights[valid_mask].reshape(-1).to(hidden_states.dtype)

        send_counts_tensor = torch.bincount(flat_owner, minlength=self.world_size).to(dtype=torch.long, device=device)
        send_counts = [int(x) for x in send_counts_tensor.tolist()]

        dispatch_started_at = _phase_start()
        count_matrix = torch.empty((self.world_size, self.world_size), dtype=torch.long, device=device)
        count_work = dist.all_gather_into_tensor(count_matrix, send_counts_tensor, async_op=True)

        order = torch.argsort(flat_owner, stable=True)
        flat_token_idx = flat_token_idx.index_select(0, order).contiguous()
        flat_local_expert_indices = flat_local_expert_indices.index_select(0, order).contiguous()
        flat_routing_weights = flat_routing_weights.index_select(0, order).contiguous()
        flat_hidden = hidden_states.index_select(0, flat_token_idx).contiguous()

        count_work.wait()
        recv_counts = [int(x) for x in count_matrix[:, self.rank].tolist()]
        if self.profile_occupancy:
            self._pending_rank_loads = count_matrix.sum(dim=0).to(dtype=torch.float32)
            self._pending_routed_tokens = int(send_counts_tensor.sum().item())
            self._pending_local_tokens = int(sum(recv_counts))
        self._record_microphase("dispatch_counts", time.perf_counter() - dispatch_started_at)

        send_token_indices = self._split_tensor_by_counts(flat_token_idx, send_counts)
        send_hidden = self._split_tensor_by_counts(flat_hidden, send_counts)
        send_local_expert_indices = self._split_tensor_by_counts(flat_local_expert_indices, send_counts)
        send_routing_weights = self._split_tensor_by_counts(flat_routing_weights, send_counts)
        return (
            send_token_indices,
            send_hidden,
            send_local_expert_indices,
            send_routing_weights,
            send_counts,
            recv_counts,
        )

    def _forward_overlap_return_streamed(
        self,
        *,
        hidden_states: torch.Tensor,
        send_token_indices: list[torch.Tensor],
        recv_hidden: list[torch.Tensor],
        recv_local_expert_indices: list[torch.Tensor],
        recv_routing_weights: list[torch.Tensor],
        recv_counts: list[int],
    ) -> torch.Tensor:
        device = hidden_states.device
        current_stream = torch.cuda.current_stream(device=device)
        comm_stream = torch.cuda.Stream(device=device)
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=device)
        metadata_started_at = _phase_start()
        recv_count_matrix = self._gather_recv_count_matrix(recv_counts, device=device)
        self._record_microphase("metadata_exchange", time.perf_counter() - metadata_started_at)

        origin_returned: Optional[torch.Tensor] = None
        origin_return_splits: Optional[list[int]] = None
        origin_event: Optional[torch.cuda.Event] = None
        pending_send_buffers: list[torch.Tensor] = []
        packed_local_compute_started_at = _phase_start()
        local_outputs = self._local_forward_split(
            recv_hidden,
            recv_local_expert_indices,
            recv_routing_weights,
            recv_counts,
            hidden_states=hidden_states,
        )
        self._record_microphase("packed_local_compute", time.perf_counter() - packed_local_compute_started_at)

        return_exchange_started_at = _phase_start()
        for src_rank in range(self.world_size):
            send_tensor = local_outputs[src_rank]

            local_ready = torch.cuda.Event()
            local_ready.record(current_stream)

            input_split_sizes = [0] * self.world_size
            input_split_sizes[src_rank] = int(send_tensor.shape[0])
            if self.rank == src_rank:
                output_split_sizes = [int(recv_count_matrix[owner_rank][src_rank]) for owner_rank in range(self.world_size)]
            else:
                output_split_sizes = [0] * self.world_size

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(local_ready)
                returned = _AllToAllRows.apply(send_tensor, output_split_sizes, input_split_sizes)
                comm_done = torch.cuda.Event()
                comm_done.record(comm_stream)

            pending_send_buffers.append(send_tensor)
            if self.rank == src_rank:
                origin_returned = returned
                origin_return_splits = output_split_sizes
                origin_event = comm_done

        final_comm_event = torch.cuda.Event()
        with torch.cuda.stream(comm_stream):
            final_comm_event.record(comm_stream)
        current_stream.wait_event(final_comm_event)
        self._record_microphase("return_exchange", time.perf_counter() - return_exchange_started_at)

        merge_started_at = _phase_start()
        if origin_returned is not None and origin_event is not None and origin_return_splits is not None:
            current_stream.wait_event(origin_event)
            returned_chunks = list(origin_returned.split(origin_return_splits, dim=0))
            for owner_rank, token_idx in enumerate(send_token_indices):
                if token_idx.numel() == 0:
                    continue
                next_states.index_add_(0, token_idx, returned_chunks[owner_rank].to(hidden_states.dtype))
        self._record_microphase("merge", time.perf_counter() - merge_started_at)
        return next_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if router_indices is None or routing_weights is None:
            raise ValueError("Expert-parallel experts require router_indices and routing_weights.")
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expert-parallel experts expect flattened hidden states [tokens, hidden], got {tuple(hidden_states.shape)}"
            )
        if self.world_size == 1:
            valid_mask = (router_indices >= 0) & (router_indices < self.num_experts)
            local_expert_indices = router_indices.clamp(min=0).to(torch.long)
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            token_positions = torch.arange(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(1).expand_as(router_indices)
            if valid_mask.any():
                token_idx = token_positions[valid_mask]
                weighted = self._local_forward(
                    hidden_states.index_select(0, token_idx),
                    local_expert_indices[valid_mask],
                    routing_weights[valid_mask].to(hidden_states.dtype),
                )
                next_states.index_add_(0, token_idx, weighted.to(hidden_states.dtype))
            return next_states

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Expert-parallel mode requires torch.distributed to be initialized.")

        device = hidden_states.device
        valid_mask = (router_indices >= 0) & (router_indices < self.num_experts)
        owner_ranks, local_expert_indices = self._owner_and_local_expert_indices(router_indices.clamp(min=0))
        token_positions = torch.arange(hidden_states.shape[0], device=device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(1).expand_as(router_indices)

        (
            send_token_indices,
            send_hidden,
            send_local_expert_indices,
            send_routing_weights,
            send_counts,
            recv_counts,
        ) = self._build_send_buckets(
            hidden_states=hidden_states,
            token_positions=token_positions,
            valid_mask=valid_mask,
            owner_ranks=owner_ranks,
            local_expert_indices=local_expert_indices,
            routing_weights=routing_weights,
        )

        if self.overlap_mode == "pipeline2" and hidden_states.is_cuda:
            return self._forward_overlap_pipeline2(
                hidden_states=hidden_states,
                send_token_indices=send_token_indices,
                send_hidden=send_hidden,
                send_local_expert_indices=send_local_expert_indices,
                send_routing_weights=send_routing_weights,
                recv_counts_total=recv_counts,
            )

        hidden_exchange_started_at = _phase_start()
        recv_hidden = self._exchange_packed_rows_with_grad(
            send_hidden,
            send_counts=send_counts,
            recv_counts=recv_counts,
        )
        self._record_microphase("hidden_exchange", time.perf_counter() - hidden_exchange_started_at)
        metadata_exchange_started_at = _phase_start()
        recv_local_expert_indices = self._exchange_packed_rows_no_grad(
            send_local_expert_indices,
            send_counts=send_counts,
            recv_counts=recv_counts,
        )
        recv_routing_weights = self._exchange_packed_rows_with_grad(
            send_routing_weights,
            send_counts=send_counts,
            recv_counts=recv_counts,
        )
        self._record_microphase("metadata_exchange", time.perf_counter() - metadata_exchange_started_at)

        if self.overlap_mode == "return_streamed" and hidden_states.is_cuda:
            return self._forward_overlap_return_streamed(
                hidden_states=hidden_states,
                send_token_indices=send_token_indices,
                recv_hidden=recv_hidden,
                recv_local_expert_indices=recv_local_expert_indices,
                recv_routing_weights=recv_routing_weights,
                recv_counts=recv_counts,
            )

        packed_local_compute_started_at = _phase_start()
        return_hidden = self._local_forward_split(
            recv_hidden,
            recv_local_expert_indices,
            recv_routing_weights,
            recv_counts,
            hidden_states=hidden_states,
        )
        self._record_microphase("packed_local_compute", time.perf_counter() - packed_local_compute_started_at)

        return_exchange_started_at = _phase_start()
        gathered_outputs = self._exchange_packed_rows_with_grad(
            return_hidden,
            send_counts=recv_counts,
            recv_counts=send_counts,
        )
        self._record_microphase("return_exchange", time.perf_counter() - return_exchange_started_at)

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=device)
        merge_started_at = _phase_start()
        for owner_rank, token_idx in enumerate(send_token_indices):
            if token_idx.numel() == 0:
                continue
            next_states.index_add_(0, token_idx, gathered_outputs[owner_rank].to(hidden_states.dtype))
        self._record_microphase("merge", time.perf_counter() - merge_started_at)
        return next_states


def _enable_expert_parallel_for_hybrid_model(
    student_model: nn.Module,
    *,
    rank: int,
    world_size: int,
    overlap_mode: str = "none",
    local_mxfp4_routing_mode: str = "dense_logits",
    partition_mode: str = "contiguous",
    profile_occupancy: bool = False,
) -> dict[str, Any]:
    base_model = _unwrap_model(student_model)
    layers = getattr(getattr(base_model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Expert-parallel patch expected student_model.model.layers to exist.")
    layer_count = 0
    experts_per_rank = None
    num_experts = None
    for layer in layers:
        decoder_layer = _unwrap_model(layer)
        mlp = getattr(decoder_layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue
        if isinstance(experts, ExpertParallelGptOssExperts):
            experts_per_rank = experts.local_num_experts
            num_experts = experts.num_experts
            layer_count += 1
            continue
        patched = ExpertParallelGptOssExperts(
            experts,
            rank=rank,
            world_size=world_size,
            overlap_mode=overlap_mode,
            local_mxfp4_routing_mode=local_mxfp4_routing_mode,
            partition_mode=partition_mode,
            profile_occupancy=profile_occupancy,
        )
        mlp.experts = patched
        experts_per_rank = patched.local_num_experts
        num_experts = patched.num_experts
        layer_count += 1
    if layer_count <= 0:
        raise RuntimeError("Expert-parallel patch did not find any MoE experts to replace.")
    return {
        "layer_count": int(layer_count),
        "num_experts": int(num_experts) if num_experts is not None else None,
        "experts_per_rank": int(experts_per_rank) if experts_per_rank is not None else None,
        "local_mxfp4_routing_mode": str(local_mxfp4_routing_mode),
        "partition_mode": str(partition_mode),
        "profile_occupancy": bool(profile_occupancy),
    }


def _reset_expert_parallel_microphase_stats(student_model: nn.Module) -> None:
    base_model = _unwrap_model(student_model)
    for module in base_model.modules():
        if isinstance(module, ExpertParallelGptOssExperts):
            module.reset_microphase_stats()


def _consume_expert_parallel_microphase_stats(student_model: nn.Module) -> dict[str, Any]:
    base_model = _unwrap_model(student_model)
    aggregate = {
        "dispatch_counts": 0.0,
        "hidden_exchange": 0.0,
        "metadata_exchange": 0.0,
        "packed_local_compute": 0.0,
        "return_exchange": 0.0,
        "merge": 0.0,
    }
    occupancy_sum_keys = (
        "routed_tokens",
        "local_tokens",
        "rank_load_imbalance",
        "rank_load_std",
        "local_active_experts",
        "local_max_tokens_per_expert",
        "global_active_experts",
        "global_max_tokens_per_expert",
        "global_mean_tokens_per_active_expert",
        "global_top4_expert_share",
        "global_expert_imbalance",
    )
    occupancy_sums = {key: 0.0 for key in occupancy_sum_keys}
    occupancy_max = {key: 0.0 for key in occupancy_sum_keys}
    occupancy_sample_count = 0
    module_count = 0
    for module in base_model.modules():
        if not isinstance(module, ExpertParallelGptOssExperts):
            continue
        module_count += 1
        stats = module.consume_microphase_stats()
        for key in aggregate:
            aggregate[key] += float(stats["durations"].get(key, 0.0))
        occupancy = stats.get("occupancy", {})
        occupancy_sample_count += int(occupancy.get("sample_count", 0.0))
        for key in occupancy_sum_keys:
            value = float(occupancy.get(key, 0.0))
            occupancy_sums[key] += value
            occupancy_max[key] = max(occupancy_max[key], value)
    occupancy_mean = {
        key: (value / float(occupancy_sample_count) if occupancy_sample_count > 0 else 0.0)
        for key, value in occupancy_sums.items()
    }
    return {
        "module_count": module_count,
        "durations": aggregate,
        "occupancy": {
            "sample_count": int(occupancy_sample_count),
            "mean": occupancy_mean,
            "max": occupancy_max,
        },
    }


def _reset_hybrid_dual_microphase_stats(student_model: nn.Module) -> None:
    base_model = _unwrap_model(student_model)
    for module in base_model.modules():
        if isinstance(module, HybridGptOssDecoderLayerForHealing):
            module.reset_dual_microphase_stats()


def _consume_hybrid_dual_microphase_stats(student_model: nn.Module) -> dict[str, Any]:
    base_model = _unwrap_model(student_model)
    aggregate = {
        "student_attn": 0.0,
        "teacher_attn": 0.0,
        "post_attn_layernorm": 0.0,
        "mlp_router": 0.0,
        "mlp_experts": 0.0,
        "mlp_total": 0.0,
    }
    module_count = 0
    for module in base_model.modules():
        if not isinstance(module, HybridGptOssDecoderLayerForHealing):
            continue
        module_count += 1
        stats = module.consume_dual_microphase_stats()
        for key in aggregate:
            aggregate[key] += float(stats.get(key, 0.0))
    return {
        "module_count": module_count,
        "durations": aggregate,
    }


class HybridGptOssSharedBackbone(nn.Module):
    def __init__(self, shared_model: nn.Module, student_attn_modules: list[nn.Module]) -> None:
        super().__init__()
        self.embed_tokens = shared_model.embed_tokens
        self.layers = nn.ModuleList(
            [
                HybridGptOssDecoderLayerForHealing(layer, student_attn_modules[layer_idx])
                for layer_idx, layer in enumerate(shared_model.layers)
            ]
        )
        self.norm = shared_model.norm
        self.rotary_emb = shared_model.rotary_emb
        self.gradient_checkpointing = getattr(shared_model, "gradient_checkpointing", False)
        self._gradient_checkpointing_boundaries = getattr(
            shared_model, "_gradient_checkpointing_boundaries", None
        )
        self._gradient_checkpointing_use_reentrant = getattr(
            shared_model, "_gradient_checkpointing_use_reentrant", True
        )


class HybridGptOssTeacherStudentForHealing(nn.Module):
    def __init__(self, base_model: nn.Module, student_attn_modules: list[nn.Module]) -> None:
        super().__init__()
        self.config = base_model.config
        self.vocab_size = base_model.config.vocab_size
        self.model = HybridGptOssSharedBackbone(base_model.model, student_attn_modules)
        self.lm_head = base_model.lm_head
        self._loss_function = base_model.loss_function
        self.router_aux_loss_coef = getattr(base_model, "router_aux_loss_coef", None)
        self.num_experts = getattr(base_model, "num_experts", None)
        self.num_experts_per_tok = getattr(base_model, "num_experts_per_tok", None)
        self._hybrid_dual_checkpointing_mode = "packed"

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing = True

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.model.layers:
            layer.teacher_attn.eval()
            for param in layer.teacher_attn.parameters():
                param.requires_grad_(False)
        return self

    @staticmethod
    def _move_to_device(value, device: torch.device):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(HybridGptOssTeacherStudentForHealing._move_to_device(x, device) for x in value)
        if isinstance(value, list):
            return [HybridGptOssTeacherStudentForHealing._move_to_device(x, device) for x in value]
        if isinstance(value, dict):
            return {k: HybridGptOssTeacherStudentForHealing._move_to_device(v, device) for k, v in value.items()}
        return value

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        module = _unwrap_model(module)
        for tensor in module.parameters():
            return tensor.device
        for tensor in module.buffers():
            return tensor.device
        return torch.device("cpu")

    def _forward_hidden_states(
        self,
        *,
        route: str,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SimpleNamespace:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        use_student_checkpointing = (
            route == "student"
            and self.model.gradient_checkpointing
            and self.training
            and not use_cache
        )

        def run_layer_range(
            layer_start: int,
            layer_stop: int,
            current_hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            segment_hidden_states = current_hidden_states
            for layer_idx in range(layer_start, layer_stop):
                decoder_layer = self.model.layers[layer_idx]
                decoder_layer_base = _unwrap_model(decoder_layer)
                layer_device = decoder_layer_base.input_layernorm.weight.device
                segment_hidden_states = self._move_to_device(segment_hidden_states, layer_device)
                layer_kwargs = dict(
                    attention_mask=self._move_to_device(
                        causal_mask_mapping[decoder_layer_base.attention_type], layer_device
                    ),
                    position_ids=self._move_to_device(position_ids, layer_device),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=self._move_to_device(cache_position, layer_device),
                    position_embeddings=self._move_to_device(position_embeddings, layer_device),
                    **kwargs,
                )
                if isinstance(decoder_layer, FSDP):
                    segment_hidden_states = decoder_layer(
                        segment_hidden_states,
                        route=route,
                        **layer_kwargs,
                    )
                else:
                    layer_forward = (
                        decoder_layer.forward_teacher if route == "teacher" else decoder_layer.forward_student
                    )
                    segment_hidden_states = layer_forward(
                        segment_hidden_states,
                        **layer_kwargs,
                    )
            return segment_hidden_states

        if use_student_checkpointing:
            boundaries = getattr(self.model, "_gradient_checkpointing_boundaries", None)
            if not boundaries:
                boundaries = list(range(0, len(self.model.layers) + 1))
            use_reentrant = bool(getattr(self.model, "_gradient_checkpointing_use_reentrant", True))
            if not hidden_states.requires_grad:
                hidden_states = hidden_states.detach().requires_grad_(True)
            for layer_start, layer_stop in zip(boundaries[:-1], boundaries[1:]):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda x, ls=layer_start, le=layer_stop: (run_layer_range(ls, le, x),),
                    hidden_states,
                    use_reentrant=use_reentrant,
                )[0]
        else:
            hidden_states = run_layer_range(0, len(self.model.layers), hidden_states)

        hidden_states = self._move_to_device(hidden_states, self.model.norm.weight.device)
        hidden_states = self.model.norm(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _forward_hidden_states_dual(
        self,
        *,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SimpleNamespace:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        student_hidden_states = inputs_embeds
        teacher_hidden_states = inputs_embeds.detach()
        position_embeddings = self.model.rotary_emb(student_hidden_states, position_ids)
        use_student_checkpointing = self.model.gradient_checkpointing and self.training and not use_cache

        def run_layer_range_dual(
            layer_start: int,
            layer_stop: int,
            current_student_hidden_states: torch.Tensor,
            current_teacher_hidden_states: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            segment_student_hidden_states = current_student_hidden_states
            segment_teacher_hidden_states = current_teacher_hidden_states
            for layer_idx in range(layer_start, layer_stop):
                decoder_layer = self.model.layers[layer_idx]
                decoder_layer_base = _unwrap_model(decoder_layer)
                layer_device = decoder_layer_base.input_layernorm.weight.device
                segment_student_hidden_states = self._move_to_device(segment_student_hidden_states, layer_device)
                segment_teacher_hidden_states = self._move_to_device(segment_teacher_hidden_states, layer_device)
                layer_kwargs = dict(
                    attention_mask=self._move_to_device(
                        causal_mask_mapping[decoder_layer_base.attention_type], layer_device
                    ),
                    position_ids=self._move_to_device(position_ids, layer_device),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=self._move_to_device(cache_position, layer_device),
                    position_embeddings=self._move_to_device(position_embeddings, layer_device),
                    **kwargs,
                )
                if isinstance(decoder_layer, FSDP):
                    segment_student_hidden_states, segment_teacher_hidden_states = decoder_layer(
                        segment_student_hidden_states,
                        route="dual",
                        teacher_hidden_states=segment_teacher_hidden_states,
                        **layer_kwargs,
                    )
                else:
                    segment_student_hidden_states, segment_teacher_hidden_states = decoder_layer.forward_dual(
                        segment_student_hidden_states,
                        teacher_hidden_states=segment_teacher_hidden_states,
                        **layer_kwargs,
                    )
            return segment_student_hidden_states, segment_teacher_hidden_states

        def run_layer_range_dual_packed(
            layer_start: int,
            layer_stop: int,
            packed_hidden_states: torch.Tensor,
            student_batch_size: int,
        ) -> torch.Tensor:
            current_student_hidden_states, current_teacher_hidden_states = packed_hidden_states.split(
                [student_batch_size, packed_hidden_states.shape[0] - student_batch_size],
                dim=0,
            )
            next_student_hidden_states, next_teacher_hidden_states = run_layer_range_dual(
                layer_start,
                layer_stop,
                current_student_hidden_states,
                current_teacher_hidden_states,
            )
            return torch.cat([next_student_hidden_states, next_teacher_hidden_states], dim=0)

        def run_layer_range_single(
            layer_start: int,
            layer_stop: int,
            current_hidden_states: torch.Tensor,
            *,
            route: str,
        ) -> torch.Tensor:
            segment_hidden_states = current_hidden_states
            for layer_idx in range(layer_start, layer_stop):
                decoder_layer = self.model.layers[layer_idx]
                decoder_layer_base = _unwrap_model(decoder_layer)
                layer_device = decoder_layer_base.input_layernorm.weight.device
                segment_hidden_states = self._move_to_device(segment_hidden_states, layer_device)
                layer_kwargs = dict(
                    attention_mask=self._move_to_device(
                        causal_mask_mapping[decoder_layer_base.attention_type], layer_device
                    ),
                    position_ids=self._move_to_device(position_ids, layer_device),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=self._move_to_device(cache_position, layer_device),
                    position_embeddings=self._move_to_device(position_embeddings, layer_device),
                    **kwargs,
                )
                if isinstance(decoder_layer, FSDP):
                    segment_hidden_states = decoder_layer(
                        segment_hidden_states,
                        route=route,
                        **layer_kwargs,
                    )
                else:
                    layer_forward = (
                        decoder_layer.forward_teacher if route == "teacher" else decoder_layer.forward_student
                    )
                    segment_hidden_states = layer_forward(
                        segment_hidden_states,
                        **layer_kwargs,
                    )
            return segment_hidden_states

        if use_student_checkpointing:
            boundaries = getattr(self.model, "_gradient_checkpointing_boundaries", None)
            if not boundaries:
                boundaries = list(range(0, len(self.model.layers) + 1))
            use_reentrant = bool(getattr(self.model, "_gradient_checkpointing_use_reentrant", True))
            checkpoint_mode = getattr(self, "_hybrid_dual_checkpointing_mode", "packed")
            if checkpoint_mode == "split_student_teacher":
                with torch.no_grad():
                    teacher_hidden_states = run_layer_range_single(
                        0,
                        len(self.model.layers),
                        teacher_hidden_states,
                        route="teacher",
                    )
                if not student_hidden_states.requires_grad:
                    student_hidden_states = student_hidden_states.detach().requires_grad_(True)
                for layer_start, layer_stop in zip(boundaries[:-1], boundaries[1:]):
                    student_hidden_states = torch.utils.checkpoint.checkpoint(
                        lambda x, ls=layer_start, le=layer_stop: (
                            run_layer_range_single(ls, le, x, route="student"),
                        ),
                        student_hidden_states,
                        use_reentrant=use_reentrant,
                    )[0]
            elif checkpoint_mode == "packed_forward_student_backward":
                if not student_hidden_states.requires_grad:
                    student_hidden_states = student_hidden_states.detach().requires_grad_(True)
                for layer_start, layer_stop in zip(boundaries[:-1], boundaries[1:]):
                    class _PackedForwardStudentBackward(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx, student_x: torch.Tensor, teacher_x: torch.Tensor):
                            ctx.layer_start = layer_start
                            ctx.layer_stop = layer_stop
                            ctx.run_student = run_layer_range_single
                            ctx.save_for_backward(student_x)
                            with torch.no_grad():
                                next_student_x, next_teacher_x = run_layer_range_dual(
                                    layer_start,
                                    layer_stop,
                                    student_x,
                                    teacher_x,
                                )
                            return next_student_x, next_teacher_x.detach()

                        @staticmethod
                        def backward(ctx, grad_student_out: torch.Tensor, grad_teacher_out: Optional[torch.Tensor]):
                            (saved_student_x,) = ctx.saved_tensors
                            with torch.enable_grad():
                                student_x = saved_student_x.detach().requires_grad_(True)
                                student_out = ctx.run_student(
                                    ctx.layer_start,
                                    ctx.layer_stop,
                                    student_x,
                                    route="student",
                                )
                            torch.autograd.backward(student_out, grad_student_out)
                            grad_student_x = student_x.grad
                            return grad_student_x, None

                    student_hidden_states, teacher_hidden_states = _PackedForwardStudentBackward.apply(
                        student_hidden_states,
                        teacher_hidden_states,
                    )
            else:
                student_batch_size = int(student_hidden_states.shape[0])
                packed_hidden_states = torch.cat([student_hidden_states, teacher_hidden_states], dim=0)
                if not packed_hidden_states.requires_grad:
                    packed_hidden_states = packed_hidden_states.detach().requires_grad_(True)
                for layer_start, layer_stop in zip(boundaries[:-1], boundaries[1:]):
                    packed_hidden_states = torch.utils.checkpoint.checkpoint(
                        lambda packed_x, ls=layer_start, le=layer_stop, batch_size=student_batch_size: (
                            run_layer_range_dual_packed(ls, le, packed_x, batch_size),
                        ),
                        packed_hidden_states,
                        use_reentrant=use_reentrant,
                    )[0]
                student_hidden_states, teacher_hidden_states = packed_hidden_states.split(
                    [student_batch_size, packed_hidden_states.shape[0] - student_batch_size],
                    dim=0,
                )
        else:
            student_hidden_states, teacher_hidden_states = run_layer_range_dual(
                0,
                len(self.model.layers),
                student_hidden_states,
                teacher_hidden_states,
            )

        norm_device = self.model.norm.weight.device
        student_hidden_states = self._move_to_device(student_hidden_states, norm_device)
        teacher_hidden_states = self._move_to_device(teacher_hidden_states, norm_device)
        if use_student_checkpointing and getattr(self, "_hybrid_dual_checkpointing_mode", "packed") == "split_student_teacher":
            student_hidden_states = self.model.norm(student_hidden_states)
            with torch.no_grad():
                teacher_hidden_states = self.model.norm(teacher_hidden_states)
        else:
            packed_hidden_states = torch.cat([student_hidden_states, teacher_hidden_states], dim=0)
            packed_hidden_states = self.model.norm(packed_hidden_states)
            student_hidden_states, teacher_hidden_states = packed_hidden_states.split(
                [student_hidden_states.shape[0], teacher_hidden_states.shape[0]],
                dim=0,
            )
        return SimpleNamespace(
            student_last_hidden_state=student_hidden_states,
            teacher_last_hidden_state=teacher_hidden_states.detach(),
            past_key_values=past_key_values,
        )

    def _finalize_outputs(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor],
        logits_to_keep: int | torch.Tensor = 0,
        return_hidden_states_only: bool = False,
    ) -> SimpleNamespace:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep else logits_to_keep
        hidden_states = self._move_to_device(hidden_states, self._module_device(self.lm_head))
        if return_hidden_states_only:
            return SimpleNamespace(loss=None, logits=None, hidden_states=hidden_states)
        logits = self.lm_head(hidden_states[:, slice_indices, :]) if logits_to_keep else self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self._loss_function(logits, labels, self.vocab_size)
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)

    def student_forward(
        self,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_hidden_states_only: bool = False,
        **kwargs,
    ):
        outputs = self._forward_hidden_states(route="student", **kwargs)
        return self._finalize_outputs(
            hidden_states=outputs.last_hidden_state,
            labels=labels,
            logits_to_keep=logits_to_keep,
            return_hidden_states_only=return_hidden_states_only,
        )

    def teacher_forward(
        self,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_hidden_states_only: bool = False,
        **kwargs,
    ):
        outputs = self._forward_hidden_states(route="teacher", **kwargs)
        return self._finalize_outputs(
            hidden_states=outputs.last_hidden_state,
            labels=labels,
            logits_to_keep=logits_to_keep,
            return_hidden_states_only=return_hidden_states_only,
        )

    def dual_forward(
        self,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_hidden_states_only: bool = False,
        **kwargs,
    ):
        outputs = self._forward_hidden_states_dual(**kwargs)
        student_outputs = self._finalize_outputs(
            hidden_states=outputs.student_last_hidden_state,
            labels=labels,
            logits_to_keep=logits_to_keep,
            return_hidden_states_only=return_hidden_states_only,
        )
        teacher_outputs = self._finalize_outputs(
            hidden_states=outputs.teacher_last_hidden_state,
            labels=None,
            logits_to_keep=logits_to_keep,
            return_hidden_states_only=return_hidden_states_only,
        )
        return SimpleNamespace(
            loss=student_outputs.loss,
            logits=student_outputs.logits,
            hidden_states=student_outputs.hidden_states,
            student_hidden_states=student_outputs.hidden_states,
            teacher_hidden_states=teacher_outputs.hidden_states,
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
        )

    def forward(self, *args, route: str = "student", **kwargs):
        if route == "student":
            return self.student_forward(*args, **kwargs)
        if route == "teacher":
            return self.teacher_forward(*args, **kwargs)
        if route == "dual":
            return self.dual_forward(*args, **kwargs)
        raise ValueError(f"Unsupported route: {route}")


def _load_mla_attention_modules_from_checkpoint(
    *,
    base_model: nn.Module,
    student_checkpoint_path: Path,
    attn_implementation: str,
    default_device: torch.device,
) -> tuple[list[nn.Module], dict[str, Any]]:
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

    mla_config = copy.deepcopy(base_model.config)
    mla_config._attn_implementation = attn_implementation
    mla_config.head_dim = int(student_config.get("head_dim", qk_nope_head_dim + qk_rope_head_dim))
    mla_config.qk_nope_head_dim = qk_nope_head_dim
    mla_config.qk_rope_head_dim = qk_rope_head_dim
    mla_config.v_head_dim = v_head_dim
    mla_config.kv_lora_rank = int(student_config["kv_lora_rank"])
    mla_config.kv_lora_rank_per_layer = rank_schedule
    mla_config.mla_rope_num_kv_heads = int(student_config.get("mla_rope_num_kv_heads", 1))
    mla_config.gpt_oss_mla_contract = student_config.get("gpt_oss_mla_contract")
    mla_config.mla_train_attention_backend = student_config.get("mla_train_attention_backend", "eager")

    hf_device_map = getattr(base_model, "hf_device_map", None) or {}
    index = TensorIndex(student_checkpoint_path)
    attention_modules: list[nn.Module] = []
    for layer_id, layer in enumerate(base_model.model.layers):
        prefix = f"model.layers.{layer_id}.self_attn"
        layer_prefix = f"model.layers.{layer_id}"
        layer_device_spec = hf_device_map.get(prefix, hf_device_map.get(layer_prefix, None))
        target_device = (
            _normalize_module_device(layer_device_spec)
            if layer_device_spec is not None
            else layer.self_attn.q_proj.weight.device
        )
        if target_device.type == "cpu" and default_device.type == "cuda":
            target_device = default_device
        target_dtype = layer.input_layernorm.weight.dtype

        patched = GptOssMlaAttention(config=mla_config, layer_idx=layer_id)
        patched.to(device=target_device, dtype=target_dtype)
        with torch.no_grad():
            patched.q_proj.weight.copy_(index.get_tensor(f"{prefix}.q_proj.weight").to(device=target_device, dtype=target_dtype))
            q_proj_bias = index.maybe_get_tensor(f"{prefix}.q_proj.bias")
            if q_proj_bias is not None and patched.q_proj.bias is not None:
                patched.q_proj.bias.copy_(q_proj_bias.to(device=target_device, dtype=target_dtype))

            patched.kv_a_proj_with_mqa.weight.copy_(index.get_tensor(f"{prefix}.kv_a_proj_with_mqa.weight").to(device=target_device, dtype=target_dtype))
            kv_a_bias = index.maybe_get_tensor(f"{prefix}.kv_a_proj_with_mqa.bias")
            if kv_a_bias is not None and patched.kv_a_proj_with_mqa.bias is not None:
                patched.kv_a_proj_with_mqa.bias.copy_(kv_a_bias.to(device=target_device, dtype=target_dtype))

            patched.kv_b_proj.weight.copy_(index.get_tensor(f"{prefix}.kv_b_proj.weight").to(device=target_device, dtype=target_dtype))
            patched.o_proj.weight.copy_(index.get_tensor(f"{prefix}.o_proj.weight").to(device=target_device, dtype=target_dtype))
            o_proj_bias = index.maybe_get_tensor(f"{prefix}.o_proj.bias")
            if o_proj_bias is not None and patched.o_proj.bias is not None:
                patched.o_proj.bias.copy_(o_proj_bias.to(device=target_device, dtype=target_dtype))
            patched.sinks.copy_(index.get_tensor(f"{prefix}.sinks").to(device=target_device, dtype=target_dtype))
        attention_modules.append(patched)

    patch_info = {
        "student_load_mode": "hybrid_shared_backbone",
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "base_qk_rope_head_dim": base_qk_rope_head_dim,
        "decoupled_rope_dim": qk_rope_head_dim - base_qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "kv_lora_rank_per_layer": rank_schedule,
        "attention_modules": attention_modules,
        "student_config": student_config,
    }
    return attention_modules, patch_info


def _load_hybrid_model_for_healing(
    *,
    student_model_path: Path,
    teacher_model_path: Path,
    model_kwargs: dict[str, Any],
    device: torch.device,
    device_map: Optional[str],
    attn_implementation: str,
    phase_log_path: Optional[Path] = None,
    dist_ctx: Optional[dict[str, Any]] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    base_model_load_started_at = _phase_start()
    base_model = AutoModelForCausalLM.from_pretrained(str(teacher_model_path), **model_kwargs)
    if phase_log_path is not None and dist_ctx is not None:
        _log_phase_timing(
            phase_log_path=phase_log_path,
            dist_ctx=dist_ctx,
            phase="hybrid_base_model_from_pretrained",
            started_at=base_model_load_started_at,
        )
    base_model.config._attn_implementation = attn_implementation
    if device_map is None:
        base_model_to_device_started_at = _phase_start()
        base_model.to(device)
        if phase_log_path is not None and dist_ctx is not None:
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="hybrid_base_model_to_device",
                started_at=base_model_to_device_started_at,
            )

    mla_module_load_started_at = _phase_start()
    student_attn_modules, patch_info = _load_mla_attention_modules_from_checkpoint(
        base_model=base_model,
        student_checkpoint_path=student_model_path,
        attn_implementation=attn_implementation,
        default_device=device,
    )
    if phase_log_path is not None and dist_ctx is not None:
        _log_phase_timing(
            phase_log_path=phase_log_path,
            dist_ctx=dist_ctx,
            phase="hybrid_mla_module_load",
            started_at=mla_module_load_started_at,
        )
    hybrid_assembly_started_at = _phase_start()
    hybrid_model = HybridGptOssTeacherStudentForHealing(base_model, student_attn_modules)
    for param in hybrid_model.parameters():
        param.requires_grad_(False)
    if phase_log_path is not None and dist_ctx is not None:
        _log_phase_timing(
            phase_log_path=phase_log_path,
            dist_ctx=dist_ctx,
            phase="hybrid_model_assembly",
            started_at=hybrid_assembly_started_at,
        )
    return hybrid_model, patch_info


def _load_student_model_for_healing(
    *,
    student_model_path: Path,
    model_kwargs: dict[str, Any],
    device: torch.device,
    device_map: Optional[str],
    attn_implementation: str,
) -> tuple[nn.Module, dict[str, Any]]:
    student_model_ref, student_trust_remote_code = prepare_hf_model_path(student_model_path)
    student_model_kwargs = dict(model_kwargs)
    student_model_kwargs["trust_remote_code"] = bool(student_trust_remote_code)
    student_model = AutoModelForCausalLM.from_pretrained(str(student_model_ref), **student_model_kwargs)
    student_model.config._attn_implementation = attn_implementation
    if device_map is None:
        student_model.to(device)
    if student_trust_remote_code:
        patch_info = _build_direct_mla_patch_info(student_model, student_model_path)
    else:
        patch_info = _patch_student_model(student_model, student_model_path)
        if patch_info["qk_rope_head_dim"] > 0:
            student_model.config.head_dim = int(patch_info["qk_rope_head_dim"])
            student_model.model.rotary_emb = type(student_model.model.rotary_emb)(
                student_model.config,
                device=_infer_input_device(student_model),
            )
    return student_model, patch_info


def _configure_gradient_checkpointing(
    *,
    model: nn.Module,
    enabled: bool,
    mode: str,
    max_seq_length: int,
    dtype: torch.dtype,
    layers_per_checkpoint: Optional[Any],
) -> None:
    if not enabled:
        try:
            from unsloth_zoo.gradient_checkpointing import (
                unpatch_gradient_checkpointing,
                unpatch_unsloth_smart_gradient_checkpointing,
            )

            unpatch_unsloth_smart_gradient_checkpointing()
            unpatch_gradient_checkpointing()
        except Exception:
            pass
        return

    if mode == "unsloth":
        from unsloth_zoo.gradient_checkpointing import (
            patch_unsloth_smart_gradient_checkpointing,
            prepare_n_gradient_checkpoints,
        )

        patch_unsloth_smart_gradient_checkpointing(dtype=dtype)
        prepare_n_gradient_checkpoints(
            model,
            layers_per_checkpoint=layers_per_checkpoint,
            use_reentrant=True,
        )
    elif mode == "unsloth_plain":
        from unsloth_zoo.gradient_checkpointing import (
            patch_gradient_checkpointing,
            prepare_n_gradient_checkpoints,
            unpatch_unsloth_smart_gradient_checkpointing,
        )

        unpatch_unsloth_smart_gradient_checkpointing()
        patch_gradient_checkpointing()
        prepare_n_gradient_checkpoints(
            model,
            layers_per_checkpoint=layers_per_checkpoint,
            use_reentrant=True,
        )
    else:
        try:
            from unsloth_zoo.gradient_checkpointing import (
                prepare_n_gradient_checkpoints,
                unpatch_gradient_checkpointing,
                unpatch_unsloth_smart_gradient_checkpointing,
            )

            unpatch_unsloth_smart_gradient_checkpointing()
            unpatch_gradient_checkpointing()
            prepare_n_gradient_checkpoints(
                model,
                layers_per_checkpoint=layers_per_checkpoint,
                use_reentrant=True,
            )
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


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


def _configure_trainable_subset(attention_modules: list[nn.Module], mode: str) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for module in attention_modules:
        for param in module.parameters():
            param.requires_grad_(False)

        kv_content_targets: list[nn.Parameter]
        if hasattr(module, "kv_b_proj"):
            kv_content_targets = [module.kv_b_proj.weight]
        else:
            kv_content_targets = [module.w_kc, module.w_vc]

        if mode == "all_mla":
            targets = [module.q_proj.weight, module.kv_a_proj_with_mqa.weight, *kv_content_targets]
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
                *kv_content_targets,
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


def _build_tracked_candidate_params(
    model: nn.Module,
    trainable_params: list[nn.Parameter],
) -> list[tuple[str, nn.Parameter]]:
    trainable_by_id = {id(param): param for param in trainable_params}
    tracked_candidates: list[tuple[str, nn.Parameter]] = []
    seen_ids: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in trainable_by_id and id(param) not in seen_ids:
            tracked_candidates.append((name, param))
            seen_ids.add(id(param))
    if tracked_candidates:
        return tracked_candidates
    return [(f"trainable_param_{idx}", param) for idx, param in enumerate(trainable_params)]


def _find_parameter_name(module: nn.Module, target: nn.Parameter) -> str | None:
    for name, param in module.named_parameters():
        if param is target:
            return name
    return None


def _iter_token_windows(
    dataset_spec_json: Path,
    tokenizer,
    seq_len: int,
    target_total_rows: Optional[int],
    append_eos: bool,
    slot_index: int = 0,
    slot_count: int = 1,
) -> Iterator[list[int]]:
    collect_helpers = _load_script_module(
        Path(__file__).resolve().with_name("collect_gpt_oss_kv_covariance.py")
    )
    spec = _canonicalize_dataset_spec_paths(_load_json(dataset_spec_json), dataset_spec_json)
    row_budgets = collect_helpers._compute_row_budgets(spec, target_total_rows)
    eos_token_id = tokenizer.eos_token_id
    for source, row_budget in zip(spec["sources"], row_budgets):
        if source["kind"] == "packed_torch_filelist":
            rows_used = 0
            global_valid_sequences = 0
            source_sequence_budget = int(source.get("max_sequences", 0) or 0)
            for _file_index, _row_index, token_ids_tensor in collect_helpers._iter_packed_torch_filelist(source):
                if row_budget and rows_used >= row_budget:
                    break
                if isinstance(token_ids_tensor, torch.Tensor):
                    token_ids = token_ids_tensor.reshape(-1).tolist()
                else:
                    token_ids = [int(token_id) for token_id in token_ids_tensor]
                if len(token_ids) < seq_len:
                    rows_used += 1
                    continue
                total_chunks = len(token_ids) // seq_len
                if total_chunks <= 0:
                    rows_used += 1
                    continue
                for chunk_index in range(total_chunks):
                    if source_sequence_budget and global_valid_sequences >= source_sequence_budget:
                        break
                    take_this_rank = (global_valid_sequences % slot_count) == slot_index
                    global_valid_sequences += 1
                    if not take_this_rank:
                        continue
                    chunk_start = chunk_index * seq_len
                    yield [int(x) for x in token_ids[chunk_start : chunk_start + seq_len]]
                rows_used += 1
            continue

        buffer: list[int] = []
        rows_used = 0
        global_valid_rows = 0
        for row in collect_helpers._iter_rows(source):
            if row_budget and rows_used >= row_budget:
                break
            text = collect_helpers._row_to_text(row, source)
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            take_this_rank = (global_valid_rows % slot_count) == slot_index
            global_valid_rows += 1
            if not take_this_rank:
                continue
            if append_eos and eos_token_id is not None:
                token_ids = list(token_ids) + [int(eos_token_id)]
            buffer.extend(int(x) for x in token_ids)
            rows_used += 1
            while len(buffer) >= seq_len:
                yield buffer[:seq_len]
                buffer = buffer[seq_len:]


class _HealingBatchDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_spec_json: Path,
        tokenizer,
        seq_len: int,
        batch_size: int,
        target_total_rows: Optional[int],
        append_eos: bool,
    ) -> None:
        self.dataset_spec_json = dataset_spec_json
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.target_total_rows = target_total_rows
        self.append_eos = append_eos

    def __iter__(self) -> Iterator[torch.Tensor]:
        dist_ctx = _distributed_context()
        worker = get_worker_info()
        worker_count = int(worker.num_workers) if worker is not None else 1
        worker_index = int(worker.id) if worker is not None else 0
        slot_count = max(1, int(dist_ctx["world_size"]) * worker_count)
        slot_index = int(dist_ctx["rank"]) * worker_count + worker_index
        packed: list[list[int]] = []
        for token_window in _iter_token_windows(
            dataset_spec_json=self.dataset_spec_json,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            target_total_rows=self.target_total_rows,
            append_eos=self.append_eos,
            slot_index=slot_index,
            slot_count=slot_count,
        ):
            packed.append(token_window)
            if len(packed) >= self.batch_size:
                yield torch.tensor(packed, dtype=torch.long)
                packed = []
        if packed:
            yield torch.tensor(packed, dtype=torch.long)


def _build_batches(
    dataset_spec_json: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    target_total_rows: Optional[int],
    append_eos: bool,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_pin_memory: bool,
    dataloader_persistent_workers: bool,
) -> Iterator[torch.Tensor]:
    dataset = _HealingBatchDataset(
        dataset_spec_json=dataset_spec_json,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        target_total_rows=target_total_rows,
        append_eos=append_eos,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": max(0, int(dataloader_num_workers)),
        "pin_memory": bool(dataloader_pin_memory),
        "persistent_workers": bool(dataloader_persistent_workers)
        and int(dataloader_num_workers) > 0,
    }
    if int(dataloader_num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(dataloader_prefetch_factor))
    return DataLoader(dataset, **loader_kwargs)


def _iter_token_chunks(total_tokens: int, chunk_tokens: int) -> Iterator[tuple[int, int]]:
    chunk_tokens = max(1, int(chunk_tokens))
    for start in range(0, int(total_tokens), chunk_tokens):
        stop = min(int(total_tokens), start + chunk_tokens)
        yield start, stop


def _maybe_empty_cuda_cache(*tensors: Optional[torch.Tensor], force: bool = False) -> None:
    mode = _EMPTY_CACHE_MODE if not force else "inner"
    if mode == "step":
        if not force:
            return
    elif mode == "interval":
        if not force:
            _maybe_empty_cuda_cache._counter = getattr(_maybe_empty_cuda_cache, "_counter", 0) + 1
            if (_maybe_empty_cuda_cache._counter % _EMPTY_CACHE_EVERY) != 0:
                return
    elif mode == "disabled":
        return
    elif mode != "inner":
        mode = "inner"

    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            torch.cuda.empty_cache()
            return


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


def _materialize_output_path(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def _training_state_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "healing_training_state.pt"


def _gather_rng_state_for_checkpoint(
    *,
    dist_ctx: dict[str, Any],
    device: torch.device,
) -> tuple[list[torch.Tensor | None] | None, list[torch.Tensor | None] | None]:
    cpu_state = torch.get_rng_state().cpu()
    cuda_state = None
    if torch.cuda.is_available():
        if device.type == "cuda" and device.index is not None:
            cuda_state = torch.cuda.get_rng_state(device.index).cpu()
        else:
            cuda_state = torch.cuda.get_rng_state().cpu()
    if dist_ctx["is_distributed"]:
        gathered_cpu = [None for _ in range(int(dist_ctx["world_size"]))] if dist_ctx["is_main_process"] else None
        gathered_cuda = [None for _ in range(int(dist_ctx["world_size"]))] if dist_ctx["is_main_process"] else None
        dist.gather_object(cpu_state, gathered_cpu, dst=0)
        dist.gather_object(cuda_state, gathered_cuda, dst=0)
        return gathered_cpu, gathered_cuda
    return [cpu_state], [cuda_state]


def _save_training_state_checkpoint(
    *,
    student_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    global_step: int,
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    dist_ctx = _distributed_context()
    is_writer = not dist_ctx["is_distributed"] or dist_ctx["rank"] == 0
    export_root = student_model.module if isinstance(student_model, DDP) else student_model
    if torch.cuda.is_available():
        if dist_ctx["is_distributed"]:
            torch.cuda.synchronize(dist_ctx["local_rank"])
        else:
            torch.cuda.synchronize()
    _barrier_if_needed(dist_ctx)

    if isinstance(export_root, FSDP):
        optimizer_state = FSDP.full_optim_state_dict(export_root, optimizer, rank0_only=True)
    else:
        optimizer_state = optimizer.state_dict()

    cpu_rng_state_by_rank, cuda_rng_state_by_rank = _gather_rng_state_for_checkpoint(
        dist_ctx=dist_ctx,
        device=device,
    )

    if is_writer:
        training_state_path = _training_state_checkpoint_path(output_dir)
        _materialize_output_path(training_state_path)
        torch.save(
            {
                "global_step": int(global_step),
                "resume_skip_steps": int(global_step),
                "optimizer_state_dict": optimizer_state,
                "cpu_rng_state_by_rank": cpu_rng_state_by_rank,
                "cuda_rng_state_by_rank": cuda_rng_state_by_rank,
                "manifest": manifest,
            },
            training_state_path,
        )

    _barrier_if_needed(dist_ctx)


def _load_training_state_checkpoint(
    *,
    student_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    student_checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    training_state_path = _training_state_checkpoint_path(student_checkpoint_path)
    if not training_state_path.exists():
        return None

    dist_ctx = _distributed_context()
    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    export_root = student_model.module if isinstance(student_model, DDP) else student_model
    optimizer_state = training_state.get("optimizer_state_dict")
    if optimizer_state is not None:
        if isinstance(export_root, FSDP):
            optimizer_state = FSDP.scatter_full_optim_state_dict(
                optimizer_state,
                export_root,
                optim=optimizer,
            )
        optimizer.load_state_dict(optimizer_state)

    rank = int(dist_ctx["rank"]) if dist_ctx["is_distributed"] else 0
    cpu_rng_state_by_rank = training_state.get("cpu_rng_state_by_rank")
    if isinstance(cpu_rng_state_by_rank, list) and rank < len(cpu_rng_state_by_rank):
        cpu_state = cpu_rng_state_by_rank[rank]
        if isinstance(cpu_state, torch.Tensor):
            torch.set_rng_state(cpu_state.cpu())

    cuda_rng_state_by_rank = training_state.get("cuda_rng_state_by_rank")
    if torch.cuda.is_available() and isinstance(cuda_rng_state_by_rank, list) and rank < len(cuda_rng_state_by_rank):
        cuda_state = cuda_rng_state_by_rank[rank]
        if isinstance(cuda_state, torch.Tensor):
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_rng_state(cuda_state.cpu(), device=device.index)
            else:
                torch.cuda.set_rng_state(cuda_state.cpu())

    _barrier_if_needed(dist_ctx)
    training_state["training_state_path"] = str(training_state_path)
    return training_state


def _export_healed_checkpoint(
    student_model: nn.Module,
    student_checkpoint_path: Path,
    output_dir: Path,
    copy_files: bool,
    overwrite: bool,
    metadata: dict[str, Any],
) -> None:
    dist_ctx = _distributed_context()
    is_writer = not dist_ctx["is_distributed"] or dist_ctx["rank"] == 0
    export_root = student_model.module if isinstance(student_model, DDP) else student_model
    if torch.cuda.is_available():
        if dist_ctx["is_distributed"]:
            torch.cuda.synchronize(dist_ctx["local_rank"])
        else:
            torch.cuda.synchronize()
    _barrier_if_needed(dist_ctx)
    full_state_dict_ctx = nullcontext()
    if isinstance(export_root, FSDP):
        needs_post_backward_drain = bool(getattr(export_root, "_post_backward_callback_queued", False))
        if not needs_post_backward_drain:
            root_training_state = getattr(getattr(export_root, "training_state", None), "name", None)
            needs_post_backward_drain = root_training_state not in (None, "IDLE")
        if not needs_post_backward_drain:
            for fsdp_state in getattr(export_root, "_all_fsdp_states", ()):
                handle = getattr(fsdp_state, "_handle", None)
                handle_state = getattr(getattr(handle, "_training_state", None), "name", None)
                if handle_state not in (None, "IDLE"):
                    needs_post_backward_drain = True
                    break
        if needs_post_backward_drain:
            _post_backward_final_callback(export_root, export_root)
            if torch.cuda.is_available():
                if dist_ctx["is_distributed"]:
                    torch.cuda.synchronize(dist_ctx["local_rank"])
                else:
                    torch.cuda.synchronize()
            _barrier_if_needed(dist_ctx)
        full_state_dict_ctx = FSDP.state_dict_type(
            export_root,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )

    with full_state_dict_ctx:
        export_state = export_root.state_dict()

    if is_writer:
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        _copy_tree(student_checkpoint_path, output_dir, copy_files=copy_files)

        new_tensors: dict[str, torch.Tensor] = {}
        allowed_suffixes = (
            ".q_proj.weight",
            ".q_proj.bias",
            ".kv_a_proj_with_mqa.weight",
            ".kv_a_proj_with_mqa.bias",
            ".kv_b_proj.weight",
            ".o_proj.weight",
            ".o_proj.bias",
        )
        for name, tensor in export_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if ".self_attn." not in name:
                continue
            if not name.startswith("model.layers."):
                continue
            if not name.endswith(allowed_suffixes):
                continue
            new_tensors[name] = tensor.detach().cpu().contiguous()

        mla_shard_name = "model-care-mla-attention.safetensors"
        mla_shard_path = output_dir / mla_shard_name
        _materialize_output_path(mla_shard_path)
        save_file(new_tensors, mla_shard_path)

        config = _load_json(student_checkpoint_path / "config.json")
        config["care_mla_healing"] = metadata
        config_path = output_dir / "config.json"
        _materialize_output_path(config_path)
        _save_json(config_path, config)

        index = _load_json(student_checkpoint_path / "model.safetensors.index.json")
        for name in new_tensors:
            index["weight_map"][name] = mla_shard_name
        index_path = output_dir / "model.safetensors.index.json"
        _materialize_output_path(index_path)
        _save_json(index_path, index)

    _barrier_if_needed(dist_ctx)


def main() -> None:
    args = _parse_args()
    if args.use_unsloth_flex_attention and args.attn_implementation != "eager":
        raise ValueError(
            "--use-unsloth-flex-attention requires --attn-implementation eager. "
            "Training uses the MLA-local Flex Attention branch; eval/inference stays on the direct HF eager path."
        )
    student_model_path = Path(args.student_model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_spec_json = Path(args.dataset_spec_json).resolve()
    teacher_model_path = (
        _resolve_workspace_mirror_path(Path(args.teacher_model_path))
        if args.teacher_model_path
        else _resolve_source_model_path(student_model_path)
    )
    base_model_path = _resolve_source_model_path(student_model_path)
    hybrid_shared_backbone = not bool(args.disable_hybrid_shared_backbone)
    if hybrid_shared_backbone and teacher_model_path != base_model_path:
        raise ValueError(
            "Hybrid shared-backbone healing requires the teacher path to be the original "
            "source GPT-OSS checkpoint recorded in the MLA artifact."
        )
    if args.expert_parallel:
        if not hybrid_shared_backbone:
            raise ValueError("Expert-parallel healing currently requires the hybrid shared-backbone path.")

    dist_ctx = _distributed_context()
    if dist_ctx["is_distributed"] and args.device_map:
        raise ValueError("device_map is not allowed when launching healing under torchrun/DDP.")
    if args.expert_parallel and not dist_ctx["is_distributed"]:
        raise ValueError("Expert-parallel healing requires torchrun/distributed launch.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "healing_log.jsonl"
    manifest_path = output_dir / "healing_manifest.json"
    phase_log_path = output_dir / "phase_timings.jsonl"

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
        "effective_attention_backend": _effective_attention_backend(args),
        "use_unsloth_flex_attention": bool(args.use_unsloth_flex_attention),
        "hybrid_shared_backbone": bool(hybrid_shared_backbone),
        "expert_parallel": bool(args.expert_parallel),
        "expert_parallel_overlap_mode": args.expert_parallel_overlap_mode,
        "local_mxfp4_routing_mode": args.local_mxfp4_routing_mode,
        "expert_partition_mode": args.expert_partition_mode,
        "expert_parallel_profile_occupancy": bool(args.expert_parallel_profile_occupancy),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
        "layers_per_checkpoint": args.layers_per_checkpoint,
        "distributed_wrapper": args.distributed_wrapper,
        "fsdp_forward_prefetch": bool(args.fsdp_forward_prefetch),
        "fsdp_backward_prefetch": args.fsdp_backward_prefetch,
        "fsdp_limit_all_gathers": not bool(args.fsdp_disable_limit_all_gathers),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "dataloader_prefetch_factor": int(args.dataloader_prefetch_factor),
        "dataloader_pin_memory": bool(args.dataloader_pin_memory),
        "dataloader_persistent_workers": bool(args.dataloader_persistent_workers),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "ce_weight": float(args.ce_weight),
        "kl_weight": float(args.kl_weight),
        "distill_chunk_tokens": int(args.distill_chunk_tokens),
        "distill_vocab_chunk": int(args.distill_vocab_chunk),
        "distill_mode": "exact_full_vocab_chunked",
        "temperature": float(args.temperature),
        "trainable_subset": args.trainable_subset,
        "hybrid_dual_checkpointing_mode": args.hybrid_dual_checkpointing_mode,
    }

    if args.use_unsloth_flex_attention:
        os.environ["GPT_OSS_MLA_USE_UNSLOTH_FLEX_ATTENTION"] = "1"
        os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"
    else:
        os.environ["GPT_OSS_MLA_USE_UNSLOTH_FLEX_ATTENTION"] = "0"

    device = _resolve_device(args.device)
    device = _init_distributed_if_needed(device, dist_ctx)
    if dist_ctx["is_distributed"] and device.type == "cuda" and device.index is not None:
        dist_ctx["local_rank"] = int(device.index)
    cuda_runtime = _validate_rank_cuda_visibility(device)
    manifest["distributed"] = {
        "rank": dist_ctx["rank"],
        "world_size": dist_ctx["world_size"],
        "local_rank": dist_ctx["local_rank"],
    }
    manifest["cuda_runtime"] = cuda_runtime
    if dist_ctx["is_main_process"]:
        _save_json(manifest_path, manifest)
    _barrier_if_needed(dist_ctx)
    model_kwargs = {
        "torch_dtype": _resolve_dtype(args.dtype),
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False, use_fast=True)

    if dist_ctx["is_main_process"]:
        print(json.dumps({"student_base_model": str(base_model_path), **manifest}, indent=2), flush=True)

    load_started_at = _phase_start()
    if hybrid_shared_backbone:
        student_model, patch_info = _run_rank_local_load(
            lambda: _load_hybrid_model_for_healing(
                student_model_path=student_model_path,
                teacher_model_path=teacher_model_path,
                model_kwargs=model_kwargs,
                device=device,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
            ),
            ctx=dist_ctx,
        )
    else:
        student_model, patch_info = _run_rank_local_load(
            lambda: _load_student_model_for_healing(
                student_model_path=student_model_path,
                model_kwargs=model_kwargs,
                device=device,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
            ),
            ctx=dist_ctx,
        )
    teacher_model = None
    _log_phase_timing(
        phase_log_path=phase_log_path,
        dist_ctx=dist_ctx,
        phase="post_load_model",
        started_at=load_started_at,
        extra={
            "hybrid_shared_backbone": bool(hybrid_shared_backbone),
            "student_load_mode": patch_info.get("student_load_mode"),
        },
    )
    if args.expert_parallel:
        expert_parallel_started_at = _phase_start()
        expert_parallel_info = _enable_expert_parallel_for_hybrid_model(
            student_model,
            rank=dist_ctx["rank"],
            world_size=dist_ctx["world_size"],
            overlap_mode=args.expert_parallel_overlap_mode,
            local_mxfp4_routing_mode=args.local_mxfp4_routing_mode,
            partition_mode=args.expert_partition_mode,
            profile_occupancy=bool(args.expert_parallel_profile_occupancy),
        )
        manifest["expert_parallel_info"] = expert_parallel_info
        if dist_ctx["is_main_process"]:
            _save_json(manifest_path, manifest)
        _log_phase_timing(
            phase_log_path=phase_log_path,
            dist_ctx=dist_ctx,
            phase="expert_parallel_patch",
            started_at=expert_parallel_started_at,
            extra=expert_parallel_info,
        )
    if hybrid_shared_backbone:
        setattr(student_model, "_hybrid_dual_checkpointing_mode", str(args.hybrid_dual_checkpointing_mode))
        teacher_model = None
    elif args.kl_weight > 0:
        teacher_model = _run_rank_local_load(
            lambda: AutoModelForCausalLM.from_pretrained(str(teacher_model_path), **model_kwargs),
            ctx=dist_ctx,
        )
        teacher_model.config._attn_implementation = args.attn_implementation
        if args.device_map is None:
            teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    checkpoint_config_started_at = _phase_start()
    _configure_gradient_checkpointing(
        model=student_model,
        enabled=bool(args.gradient_checkpointing),
        mode=args.gradient_checkpointing_mode,
        max_seq_length=int(args.seq_len),
        dtype=model_kwargs["torch_dtype"],
        layers_per_checkpoint=_parse_layers_per_checkpoint(args.layers_per_checkpoint),
    )
    _log_phase_timing(
        phase_log_path=phase_log_path,
        dist_ctx=dist_ctx,
        phase="gradient_checkpointing_config",
        started_at=checkpoint_config_started_at,
        extra={
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
            "layers_per_checkpoint": args.layers_per_checkpoint,
        },
    )

    student_model.train()
    if args.distributed_wrapper == "fsdp":
        _normalize_module_floating_dtypes(
            student_model,
            target_dtype=model_kwargs["torch_dtype"],
        )
    residency = _summarize_module_devices(student_model)
    if residency["cpu_params"] or residency["cpu_buffers"]:
        raise RuntimeError(
            "Hybrid healing model still has CPU-resident tensors before DDP wrap: "
            f"{json.dumps(residency, sort_keys=True)}"
        )
    trainable_params = _configure_trainable_subset(
        patch_info["attention_modules"], args.trainable_subset
    )
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters were selected for mode={args.trainable_subset}.")
    selected_trainable_param_refs = int(len(trainable_params))
    selected_trainable_param_count = int(sum(param.numel() for param in trainable_params))

    if dist_ctx["is_distributed"]:
        if args.expert_parallel:
            manifest["distributed_wrapper_effective"] = "manual_grad_all_reduce"
            if dist_ctx["is_main_process"]:
                _save_json(manifest_path, manifest)
        elif args.distributed_wrapper == "ddp":
            student_model = DDP(
                student_model,
                device_ids=[dist_ctx["local_rank"]],
                output_device=dist_ctx["local_rank"],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            fsdp_wrap_started_at = _phase_start()
            student_model = _wrap_student_model_fsdp(
                student_model,
                local_rank=dist_ctx["local_rank"],
                wrap_decoder_layers=bool(
                    args.gradient_checkpointing
                    and args.gradient_checkpointing_mode in {"unsloth", "unsloth_plain"}
                ),
                forward_prefetch=bool(args.fsdp_forward_prefetch),
                backward_prefetch=args.fsdp_backward_prefetch,
                limit_all_gathers=not bool(args.fsdp_disable_limit_all_gathers),
            )
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="fsdp_wrap",
                started_at=fsdp_wrap_started_at,
                extra={
                    "wrap_decoder_layers": bool(
                        args.gradient_checkpointing
                        and args.gradient_checkpointing_mode in {"unsloth", "unsloth_plain"}
                    ),
                    "fsdp_forward_prefetch": bool(args.fsdp_forward_prefetch),
                    "fsdp_backward_prefetch": args.fsdp_backward_prefetch,
                    "fsdp_limit_all_gathers": not bool(args.fsdp_disable_limit_all_gathers),
                },
            )
            trainable_params = [param for param in student_model.parameters() if param.requires_grad]
    optimizer_trainable_param_refs = int(len(trainable_params))
    optimizer_local_trainable_param_count = int(sum(param.numel() for param in trainable_params))
    optimizer_nonempty_local_param_refs = int(sum(1 for param in trainable_params if param.numel() > 0))
    optimizer_global_local_trainable_param_count = _reduce_int_across_ranks(optimizer_local_trainable_param_count)
    optimizer_global_nonempty_local_param_refs = _reduce_int_across_ranks(optimizer_nonempty_local_param_refs)
    if optimizer_global_local_trainable_param_count <= 0:
        raise RuntimeError(
            "Distributed wrapper exposed zero local trainable elements across all ranks. "
            f"selected_trainable_param_count={selected_trainable_param_count} "
            f"selected_trainable_param_refs={selected_trainable_param_refs} "
            f"optimizer_trainable_param_refs={optimizer_trainable_param_refs}"
        )
    optimizer_started_at = _phase_start()
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    _log_phase_timing(
        phase_log_path=phase_log_path,
        dist_ctx=dist_ctx,
        phase="optimizer_creation",
        started_at=optimizer_started_at,
        extra={
            "selected_trainable_param_count": selected_trainable_param_count,
            "selected_trainable_param_refs": selected_trainable_param_refs,
            "optimizer_trainable_param_refs": optimizer_trainable_param_refs,
            "optimizer_local_trainable_param_count": optimizer_local_trainable_param_count,
            "optimizer_nonempty_local_param_refs": optimizer_nonempty_local_param_refs,
            "optimizer_global_local_trainable_param_count": optimizer_global_local_trainable_param_count,
            "optimizer_global_nonempty_local_param_refs": optimizer_global_nonempty_local_param_refs,
        },
    )
    tracked_candidates = _build_tracked_candidate_params(student_model, trainable_params)
    if not tracked_candidates:
        tracked_candidates = [(f"trainable_param_{idx}", param) for idx, param in enumerate(trainable_params)]

    resume_state_started_at = _phase_start()
    resume_state = _load_training_state_checkpoint(
        student_model=student_model,
        optimizer=optimizer,
        student_checkpoint_path=student_model_path,
        device=device,
    )
    if resume_state is not None:
        resumed_from_step = int(resume_state.get("global_step", 0))
        manifest["resume_training_state_path"] = str(resume_state.get("training_state_path"))
        manifest["resumed_from_step"] = resumed_from_step
        manifest["resume_skip_steps"] = int(resume_state.get("resume_skip_steps", resumed_from_step))
        if dist_ctx["is_main_process"]:
            _save_json(manifest_path, manifest)
        _log_phase_timing(
            phase_log_path=phase_log_path,
            dist_ctx=dist_ctx,
            phase="resume_training_state_load",
            started_at=resume_state_started_at,
            extra={
                "resume_training_state_path": str(resume_state.get("training_state_path")),
                "resumed_from_step": resumed_from_step,
                "resume_skip_steps": int(resume_state.get("resume_skip_steps", resumed_from_step)),
            },
        )

    def _pick_local_tracked_param():
        for name, param in tracked_candidates:
            if param.numel() > 0:
                return name, param
        return (None, None)

    global_step = int(resume_state.get("global_step", 0)) if resume_state is not None else 0
    process_initial_global_step = int(global_step)
    resume_skip_steps = int(resume_state.get("resume_skip_steps", global_step)) if resume_state is not None else 0
    pending_checkpoint_step: int | None = None
    pending_checkpoint_metadata: dict[str, Any] | None = None
    last_exported_checkpoint_dir: Path | None = None
    start_time = time.time()
    first_batch_started_at = _phase_start()
    first_batch_timing_logged = False
    for batch in _build_batches(
        dataset_spec_json=dataset_spec_json,
        tokenizer=tokenizer,
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        target_total_rows=args.target_total_rows,
        append_eos=bool(args.append_eos),
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_prefetch_factor=int(args.dataloader_prefetch_factor),
        dataloader_pin_memory=bool(args.dataloader_pin_memory),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
    ):
        if not first_batch_timing_logged:
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="first_batch_build",
                started_at=first_batch_started_at,
                step=process_initial_global_step + 1,
                extra={
                    "seq_len": int(args.seq_len),
                    "batch_size": int(args.batch_size),
                    "dataloader_num_workers": int(args.dataloader_num_workers),
                    "dataloader_prefetch_factor": int(args.dataloader_prefetch_factor),
                },
            )
            first_batch_timing_logged = True
        if resume_skip_steps > 0:
            resume_skip_steps -= 1
            continue
        if pending_checkpoint_step is not None and pending_checkpoint_metadata is not None:
            checkpoint_dir = output_dir / f"checkpoint_step_{pending_checkpoint_step:06d}"
            _export_healed_checkpoint(
                student_model=student_model,
                student_checkpoint_path=student_model_path,
                output_dir=checkpoint_dir,
                copy_files=bool(args.copy_files),
                overwrite=True,
                metadata=pending_checkpoint_metadata,
            )
            _save_training_state_checkpoint(
                student_model=student_model,
                optimizer=optimizer,
                output_dir=checkpoint_dir,
                global_step=pending_checkpoint_step,
                manifest=pending_checkpoint_metadata,
                device=device,
            )
            last_exported_checkpoint_dir = checkpoint_dir
            if dist_ctx["is_main_process"]:
                print(f"[checkpoint] wrote {checkpoint_dir}", flush=True)
            pending_checkpoint_step = None
            pending_checkpoint_metadata = None

        if global_step >= int(args.max_steps):
            break

        global_step += 1
        is_first_process_step = global_step == (process_initial_global_step + 1)
        student_input_device = _infer_input_device(student_model)
        batch = batch.to(student_input_device)
        lr_scale = min(1.0, global_step / max(int(args.warmup_steps), 1))
        for group in optimizer.param_groups:
            group["lr"] = float(args.learning_rate) * lr_scale
        if args.expert_parallel:
            _reset_expert_parallel_microphase_stats(student_model)
        if hybrid_shared_backbone and float(args.kl_weight) > 0:
            _reset_hybrid_dual_microphase_stats(student_model)

        labels = batch.clone()
        kl_loss = None
        teacher_hidden_states = None
        if hybrid_shared_backbone and float(args.kl_weight) > 0:
            hybrid_dual_forward_started_at = _phase_start()
            dual_outputs = student_model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                labels=None,
                use_cache=False,
                return_hidden_states_only=True,
                route="dual",
            )
            if is_first_process_step:
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase="first_hybrid_dual_forward",
                    started_at=hybrid_dual_forward_started_at,
                    step=global_step,
                )
            student_hidden_states = dual_outputs.student_hidden_states
            teacher_hidden_states = dual_outputs.teacher_hidden_states
            del dual_outputs
        else:
            student_forward_started_at = _phase_start()
            student_outputs = student_model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                labels=None,
                use_cache=False,
                return_hidden_states_only=True,
            )
            if is_first_process_step:
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase="first_student_forward",
                    started_at=student_forward_started_at,
                    step=global_step,
                )
            student_hidden_states = student_outputs.hidden_states
            del student_outputs

            if float(args.kl_weight) > 0 and teacher_model is not None:
                teacher_forward_started_at = _phase_start()
                teacher_input_device = _infer_input_device(teacher_model)
                teacher_batch = batch.to(teacher_input_device)
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=teacher_batch,
                        attention_mask=torch.ones_like(teacher_batch),
                        use_cache=False,
                        output_hidden_states=True,
                    )
                teacher_hidden_states = teacher_outputs.hidden_states[-1].to(student_hidden_states.device)
                del teacher_outputs
                if is_first_process_step:
                    _log_phase_timing(
                        phase_log_path=phase_log_path,
                        dist_ctx=dist_ctx,
                        phase="first_teacher_forward",
                        started_at=teacher_forward_started_at,
                        step=global_step,
                        extra={"hybrid_shared_backbone": bool(hybrid_shared_backbone)},
                    )

        if args.expert_parallel and is_first_process_step:
            microphase_stats = _consume_expert_parallel_microphase_stats(student_model)
            for microphase_name, duration_s in microphase_stats["durations"].items():
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase=f"first_expert_parallel_{microphase_name}",
                    started_at=time.perf_counter() - float(duration_s),
                    step=global_step,
                    extra={
                        "expert_parallel_overlap_mode": args.expert_parallel_overlap_mode,
                        "expert_partition_mode": args.expert_partition_mode,
                        "module_count": int(microphase_stats["module_count"]),
                    },
                )
            occupancy_stats = microphase_stats.get("occupancy", {})
            if int(occupancy_stats.get("sample_count", 0)) > 0:
                occupancy_extra = {
                    "expert_parallel_overlap_mode": args.expert_parallel_overlap_mode,
                    "local_mxfp4_routing_mode": args.local_mxfp4_routing_mode,
                    "expert_partition_mode": args.expert_partition_mode,
                    "module_count": int(microphase_stats["module_count"]),
                    "sample_count": int(occupancy_stats["sample_count"]),
                }
                for prefix, payload in (
                    ("mean", occupancy_stats.get("mean", {})),
                    ("max", occupancy_stats.get("max", {})),
                ):
                    for key, value in payload.items():
                        occupancy_extra[f"{key}_{prefix}"] = float(value)
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase="first_expert_parallel_occupancy",
                    started_at=time.perf_counter(),
                    step=global_step,
                    extra=occupancy_extra,
                )
        if hybrid_shared_backbone and float(args.kl_weight) > 0 and is_first_process_step:
            hybrid_microphase_stats = _consume_hybrid_dual_microphase_stats(student_model)
            for microphase_name, duration_s in hybrid_microphase_stats["durations"].items():
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase=f"first_hybrid_dual_{microphase_name}",
                    started_at=time.perf_counter() - float(duration_s),
                    step=global_step,
                    extra={"module_count": int(hybrid_microphase_stats["module_count"])},
                )

        _maybe_empty_cuda_cache(student_hidden_states, teacher_hidden_states, force=True)

        optimizer.zero_grad(set_to_none=True)
        tracked_param_name, tracked_param = _pick_local_tracked_param()
        tracked_slice_name = tracked_param_name
        tracked_slice_before = None
        if tracked_param is not None:
            tracked_slice_before = tracked_param.detach().view(-1)[:1024].float().cpu().clone()
        base_model_for_loss = _unwrap_model(student_model)
        lm_head = base_model_for_loss.lm_head
        lm_head_weight = _unwrap_model(lm_head).weight
        lm_head_bias = _unwrap_model(lm_head).bias
        target_tokens = labels[:, 1:].to(student_hidden_states.device)
        del labels
        del batch
        total_target_tokens = max(int(target_tokens.numel()), 1)
        ce_loss = student_hidden_states.new_zeros((), dtype=torch.float32)
        kl_accum = student_hidden_states.new_zeros((), dtype=torch.float32)
        grad_hidden_states = torch.zeros_like(student_hidden_states)
        chunk_tokens = max(int(args.distill_chunk_tokens), 1)
        vocab_chunk = max(int(args.distill_vocab_chunk), 1)
        vocab_size = int(lm_head_weight.shape[0])
        inv_total_target_tokens = 1.0 / float(total_target_tokens)
        batch_scale = 1.0 / max(int(target_tokens.shape[0]), 1)
        temperature = float(args.temperature)
        exact_loss_started_at = _phase_start()
        for start, stop in _iter_token_chunks(target_tokens.shape[1], chunk_tokens):
            student_chunk_hidden = student_hidden_states[:, start:stop]
            target_chunk = target_tokens[:, start:stop]
            target_chunk_flat = target_chunk.reshape(-1)
            flat_rows = torch.arange(target_chunk_flat.numel(), device=target_chunk_flat.device)
            student_ce_lse = None
            student_kl_lse = None
            teacher_kl_lse = None
            student_target_logits = torch.empty_like(target_chunk, dtype=torch.float32)
            teacher_target_logits = None
            if teacher_hidden_states is not None:
                teacher_target_logits = torch.empty_like(target_chunk, dtype=torch.float32)
            teacher_chunk_hidden = teacher_hidden_states[:, start:stop] if teacher_hidden_states is not None else None

            temperature_is_one = temperature == 1.0

            for vocab_start in range(0, vocab_size, vocab_chunk):
                vocab_stop = min(vocab_size, vocab_start + vocab_chunk)
                weight_chunk = lm_head_weight[vocab_start:vocab_stop]
                bias_chunk = lm_head_bias[vocab_start:vocab_stop] if lm_head_bias is not None else None
                student_vocab_logits = F.linear(student_chunk_hidden, weight_chunk, bias_chunk).float()
                ce_part_lse = torch.logsumexp(student_vocab_logits, dim=-1)
                student_ce_lse = ce_part_lse if student_ce_lse is None else torch.logaddexp(student_ce_lse, ce_part_lse)
                student_scaled_logits = student_vocab_logits if temperature_is_one else (student_vocab_logits / temperature)
                kl_part_lse = torch.logsumexp(student_scaled_logits, dim=-1)
                student_kl_lse = kl_part_lse if student_kl_lse is None else torch.logaddexp(student_kl_lse, kl_part_lse)

                flat_mask = (target_chunk_flat >= vocab_start) & (target_chunk_flat < vocab_stop)
                if flat_mask.any():
                    local_rows = flat_rows[flat_mask]
                    local_cols = (target_chunk_flat[flat_mask] - vocab_start).long()
                    student_target_logits.view(-1)[local_rows] = student_vocab_logits.reshape(-1, vocab_stop - vocab_start)[
                        local_rows, local_cols
                    ]

                if teacher_chunk_hidden is not None:
                    with torch.no_grad():
                        teacher_vocab_logits = F.linear(teacher_chunk_hidden, weight_chunk, bias_chunk).float()
                    teacher_scaled_logits = teacher_vocab_logits if temperature_is_one else (teacher_vocab_logits / temperature)
                    teacher_part_lse = torch.logsumexp(teacher_scaled_logits, dim=-1)
                    teacher_kl_lse = (
                        teacher_part_lse
                        if teacher_kl_lse is None
                        else torch.logaddexp(teacher_kl_lse, teacher_part_lse)
                    )
                    if flat_mask.any():
                        local_rows = flat_rows[flat_mask]
                        local_cols = (target_chunk_flat[flat_mask] - vocab_start).long()
                        teacher_target_logits.view(-1)[local_rows] = teacher_vocab_logits.reshape(
                            -1, vocab_stop - vocab_start
                        )[local_rows, local_cols]

                _maybe_empty_cuda_cache(
                    student_vocab_logits,
                    ce_part_lse,
                    student_scaled_logits,
                    kl_part_lse,
                    teacher_vocab_logits if teacher_chunk_hidden is not None else None,
                    teacher_scaled_logits if teacher_chunk_hidden is not None else None,
                    teacher_part_lse if teacher_chunk_hidden is not None else None,
                )
                del student_vocab_logits
                del ce_part_lse
                if not temperature_is_one:
                    del student_scaled_logits
                del kl_part_lse
                if teacher_chunk_hidden is not None:
                    del teacher_vocab_logits
                    if not temperature_is_one:
                        del teacher_scaled_logits
                    del teacher_part_lse
                del weight_chunk
                del bias_chunk

            _maybe_empty_cuda_cache(student_chunk_hidden, teacher_chunk_hidden)

            ce_chunk = (student_ce_lse - student_target_logits).sum() * inv_total_target_tokens
            ce_loss = ce_loss + ce_chunk.detach()
            grad_hidden_chunk = torch.zeros_like(student_chunk_hidden)

            for vocab_start in range(0, vocab_size, vocab_chunk):
                vocab_stop = min(vocab_size, vocab_start + vocab_chunk)
                weight_chunk = lm_head_weight[vocab_start:vocab_stop]
                bias_chunk = lm_head_bias[vocab_start:vocab_stop] if lm_head_bias is not None else None
                student_vocab_logits = F.linear(student_chunk_hidden, weight_chunk, bias_chunk).float()
                student_probs = torch.exp(student_vocab_logits - student_ce_lse.unsqueeze(-1))
                grad_logits = student_probs.mul(float(args.ce_weight) * inv_total_target_tokens)

                flat_mask = (target_chunk_flat >= vocab_start) & (target_chunk_flat < vocab_stop)
                if flat_mask.any():
                    local_rows = flat_rows[flat_mask]
                    local_cols = (target_chunk_flat[flat_mask] - vocab_start).long()
                    grad_logits_flat = grad_logits.reshape(-1, vocab_stop - vocab_start)
                    grad_logits_flat[local_rows, local_cols] -= float(args.ce_weight) * inv_total_target_tokens

                if teacher_chunk_hidden is not None:
                    with torch.no_grad():
                        teacher_vocab_logits = F.linear(teacher_chunk_hidden, weight_chunk, bias_chunk).float()
                    student_scaled_logits = (
                        student_vocab_logits if temperature_is_one else (student_vocab_logits / temperature)
                    )
                    student_temp_log_probs = student_scaled_logits - student_kl_lse.unsqueeze(-1)
                    teacher_scaled_logits = (
                        teacher_vocab_logits if temperature_is_one else (teacher_vocab_logits / temperature)
                    )
                    teacher_temp_log_probs = teacher_scaled_logits - teacher_kl_lse.unsqueeze(-1)
                    teacher_probs = teacher_temp_log_probs.exp()
                    kl_chunk = (
                        teacher_probs * (teacher_temp_log_probs - student_temp_log_probs)
                    ).sum() * ((temperature**2) * batch_scale)
                    kl_accum = kl_accum + kl_chunk.detach()
                    student_temp_probs = student_temp_log_probs.exp()
                    grad_logits.add_(
                        (student_temp_probs - teacher_probs).mul(float(args.kl_weight) * temperature * batch_scale)
                    )

                if weight_chunk.dtype == student_chunk_hidden.dtype:
                    weight_chunk_cast = weight_chunk
                else:
                    weight_chunk_cast = weight_chunk.to(dtype=student_chunk_hidden.dtype)
                grad_hidden_chunk.add_(
                    torch.matmul(
                        grad_logits.to(dtype=student_chunk_hidden.dtype),
                        weight_chunk_cast,
                    )
                )
                _maybe_empty_cuda_cache(
                    student_vocab_logits,
                    student_probs,
                    grad_logits,
                    teacher_vocab_logits if teacher_chunk_hidden is not None else None,
                    student_scaled_logits if teacher_chunk_hidden is not None else None,
                    student_temp_log_probs if teacher_chunk_hidden is not None else None,
                    teacher_scaled_logits if teacher_chunk_hidden is not None else None,
                    teacher_temp_log_probs if teacher_chunk_hidden is not None else None,
                    teacher_probs if teacher_chunk_hidden is not None else None,
                    student_temp_probs if teacher_chunk_hidden is not None else None,
                    weight_chunk_cast,
                )
                del student_vocab_logits
                del student_probs
                del grad_logits
                if weight_chunk_cast is not weight_chunk:
                    del weight_chunk_cast
                if teacher_chunk_hidden is not None:
                    del teacher_vocab_logits
                    if not temperature_is_one:
                        del student_scaled_logits
                    del student_temp_log_probs
                    if not temperature_is_one:
                        del teacher_scaled_logits
                    del teacher_temp_log_probs
                    del teacher_probs
                    del student_temp_probs
                del weight_chunk
                del bias_chunk

            grad_hidden_states[:, start:stop].add_(grad_hidden_chunk)
            _maybe_empty_cuda_cache(grad_hidden_chunk, student_chunk_hidden)
        del student_chunk_hidden
        del target_chunk
        del target_chunk_flat
        del flat_rows
        del student_ce_lse
        del student_kl_lse
        del teacher_kl_lse
        del student_target_logits
        del teacher_target_logits
        del teacher_chunk_hidden
        del ce_chunk
        del grad_hidden_chunk
        del target_tokens
        has_teacher_hidden = teacher_hidden_states is not None
        if has_teacher_hidden:
            _maybe_empty_cuda_cache(teacher_hidden_states)
            del teacher_hidden_states
        _maybe_empty_cuda_cache(grad_hidden_states, student_hidden_states, force=True)
        if grad_hidden_states.shape != student_hidden_states.shape:
            if grad_hidden_states.numel() != student_hidden_states.numel():
                raise RuntimeError(
                    f"Exact hidden-state gradient shape mismatch: grad={tuple(grad_hidden_states.shape)} "
                    f"student={tuple(student_hidden_states.shape)}"
                )
            grad_hidden_states = grad_hidden_states.reshape(student_hidden_states.shape)
        if is_first_process_step:
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="first_exact_loss_pass",
                started_at=exact_loss_started_at,
                step=global_step,
                extra={
                    "distill_chunk_tokens": int(args.distill_chunk_tokens),
                    "distill_vocab_chunk": int(args.distill_vocab_chunk),
                    "vocab_size": int(vocab_size),
                },
            )
        backward_started_at = _phase_start()
        student_hidden_states.backward(grad_hidden_states)
        if is_first_process_step:
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="first_backward",
                started_at=backward_started_at,
                step=global_step,
            )
        if hybrid_shared_backbone and float(args.kl_weight) > 0 and is_first_process_step:
            hybrid_backward_microphase_stats = _consume_hybrid_dual_microphase_stats(student_model)
            for microphase_name, duration_s in hybrid_backward_microphase_stats["durations"].items():
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase=f"first_backward_recompute_hybrid_dual_{microphase_name}",
                    started_at=time.perf_counter() - float(duration_s),
                    step=global_step,
                    extra={"module_count": int(hybrid_backward_microphase_stats["module_count"])},
                )
        if args.expert_parallel:
            grad_sync_started_at = _phase_start()
            _all_reduce_trainable_grads(trainable_params, world_size=int(dist_ctx["world_size"]))
            if is_first_process_step:
                _log_phase_timing(
                    phase_log_path=phase_log_path,
                    dist_ctx=dist_ctx,
                    phase="first_manual_grad_sync",
                    started_at=grad_sync_started_at,
                    step=global_step,
                    extra={"grad_sync_mode": "all_reduce_trainable_params"},
                )
        kl_loss = kl_accum if has_teacher_hidden else None
        total_loss = ce_loss.detach() * float(args.ce_weight)
        if kl_loss is not None:
            total_loss = total_loss + kl_loss.detach() * float(args.kl_weight)
        tracked_grad_norm = None
        if tracked_param is not None and tracked_param.grad is not None:
            tracked_grad_norm = float(tracked_param.grad.detach().float().norm().item())
        else:
            for candidate_name, candidate_param in tracked_candidates:
                if candidate_param.numel() == 0 or candidate_param.grad is None:
                    continue
                tracked_param_name = candidate_name
                tracked_param = candidate_param
                tracked_grad_norm = float(candidate_param.grad.detach().float().norm().item())
                break
        optimizer_step_started_at = _phase_start()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        if is_first_process_step:
            _log_phase_timing(
                phase_log_path=phase_log_path,
                dist_ctx=dist_ctx,
                phase="first_optimizer_step",
                started_at=optimizer_step_started_at,
                step=global_step,
            )
        tracked_delta_max = None
        tracked_delta_mean = None
        if (
            tracked_param is not None
            and tracked_slice_before is not None
            and tracked_param_name == tracked_slice_name
        ):
            tracked_slice_after = tracked_param.detach().view(-1)[:1024].float().cpu()
            tracked_delta = (tracked_slice_after - tracked_slice_before).abs()
            if tracked_delta.numel() > 0:
                tracked_delta_max = float(tracked_delta.max().item())
                tracked_delta_mean = float(tracked_delta.mean().item())

        record = {
            "step": global_step,
            "rank": dist_ctx["rank"],
            "ce_loss": float(ce_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()) if kl_loss is not None else None,
            "total_loss": float(total_loss.detach().cpu()),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.time() - start_time,
            "tracked_param_name": tracked_param_name,
            "tracked_grad_norm": tracked_grad_norm,
            "tracked_delta_max": tracked_delta_max,
            "tracked_delta_mean": tracked_delta_mean,
        }
        if dist_ctx["is_main_process"]:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if dist_ctx["is_main_process"] and global_step % max(int(args.log_every), 1) == 0:
            print(json.dumps(record), flush=True)

        if global_step % max(int(args.save_every), 1) == 0:
            pending_checkpoint_step = global_step
            pending_checkpoint_metadata = {
                **manifest,
                "completed_steps": global_step,
                "full_resume_available": True,
                "training_state_filename": _training_state_checkpoint_path(Path(".")).name,
            }

    if global_step == 0:
        raise RuntimeError(
            "Healing run completed zero optimization steps. "
            "The dataset/sequence configuration did not yield any full token windows. "
            "Use a smaller --seq-len or provide more tokenized text."
        )

    _barrier_if_needed(dist_ctx)
    if pending_checkpoint_step is not None and pending_checkpoint_metadata is not None:
        checkpoint_dir = output_dir / f"checkpoint_step_{pending_checkpoint_step:06d}"
        _export_healed_checkpoint(
            student_model=student_model,
            student_checkpoint_path=student_model_path,
            output_dir=checkpoint_dir,
            copy_files=bool(args.copy_files),
            overwrite=True,
            metadata=pending_checkpoint_metadata,
        )
        _save_training_state_checkpoint(
            student_model=student_model,
            optimizer=optimizer,
            output_dir=checkpoint_dir,
            global_step=pending_checkpoint_step,
            manifest=pending_checkpoint_metadata,
            device=device,
        )
        last_exported_checkpoint_dir = checkpoint_dir
        if dist_ctx["is_main_process"]:
            print(f"[checkpoint] wrote {checkpoint_dir}", flush=True)
        pending_checkpoint_step = None
        pending_checkpoint_metadata = None

    final_dir = output_dir / "healed_checkpoint"
    if (
        last_exported_checkpoint_dir is not None
        and last_exported_checkpoint_dir.exists()
        and last_exported_checkpoint_dir.name == f"checkpoint_step_{global_step:06d}"
    ):
        if dist_ctx["is_main_process"]:
            if final_dir.exists():
                if not bool(args.overwrite):
                    raise FileExistsError(f"{final_dir} already exists; pass --overwrite to replace it.")
                shutil.rmtree(final_dir)
            shutil.copytree(last_exported_checkpoint_dir, final_dir)
    else:
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
        _save_training_state_checkpoint(
            student_model=student_model,
            optimizer=optimizer,
            output_dir=final_dir,
            global_step=global_step,
            manifest={
                **manifest,
                "completed_steps": global_step,
                "full_resume_available": True,
                "training_state_filename": _training_state_checkpoint_path(Path(".")).name,
            },
            device=device,
        )
    if dist_ctx["is_main_process"]:
        print(f"[done] wrote healed checkpoint to {final_dir}", flush=True)
    _barrier_if_needed(dist_ctx)
    if dist_ctx["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
