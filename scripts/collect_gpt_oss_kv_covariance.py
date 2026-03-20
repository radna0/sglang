#!/usr/bin/env python3

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_TEXT_KEYS = [
    "text",
    "prompt",
    "response",
    "completion",
    "chosen",
    "answer",
    "question",
    "instruction",
    "input",
    "content",
    "messages",
    "message",
    "conversation",
    "conversations",
]


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep).strip("._") or "source"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CARE-style per-layer KV input covariance for GPT-OSS."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-spec-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--save-every-batches", type=int, default=32)
    parser.add_argument("--target-total-rows", type=int, default=None)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--save-per-source-covariance", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dp-world-size", type=int, default=None)
    parser.add_argument("--dp-rank", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--replica-device-map", default="auto")
    parser.add_argument(
        "--mxfp4-preswizzle-dir",
        default=os.environ.get("GPTOSS_MXFP4_PRESWIZZLE_DIR", ""),
    )
    parser.add_argument("--merge-timeout-s", type=int, default=1800)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _resolve_modelscope_env() -> None:
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.ai")
    for key in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "MS_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            os.environ.setdefault("MODELSCOPE_API_TOKEN", value)
            break


def _iter_jsonl_files(files: list[str]) -> Iterator[dict[str, Any]]:
    for file_path in files:
        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _iter_packed_torch_filelist(
    source: dict[str, Any], start_file_index: int = 0, start_row_index: int = 0
) -> Iterator[tuple[int, int, torch.Tensor]]:
    with Path(source["filelist"]).open("r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]
    for file_index, file_path in enumerate(files[start_file_index:], start=start_file_index):
        payload = torch.load(Path(file_path), map_location="cpu")
        input_ids = payload["input_ids"] if isinstance(payload, dict) else payload
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        row_start = start_row_index if file_index == start_file_index else 0
        for row_index, row in enumerate(input_ids[row_start:], start=row_start):
            yield file_index, row_index, row


def _iter_rows_hf(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        source["path"],
        source.get("config_name"),
        split=source.get("split", "train"),
        streaming=bool(source.get("streaming", False)),
    )
    try:
        for idx in range(len(dataset)):
            yield dataset[idx]
    except TypeError:
        for row in dataset:
            yield row


def _iter_rows_modelscope(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    _resolve_modelscope_env()
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as exc:
        raise RuntimeError(
            "ModelScope dataset loading is unavailable. Install the missing "
            "modelscope dataset extras or materialize the dataset to local JSONL first."
        ) from exc

    dataset = MsDataset.load(source["path"], split=source.get("split", "train"))
    try:
        for idx in range(len(dataset)):
            yield dataset[idx]
    except TypeError:
        for row in dataset:
            yield row


def _iter_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    kind = source["kind"]
    if kind == "packed_torch_filelist":
        raise ValueError(
            "packed_torch_filelist should be consumed via the packed-sequence path, not _iter_rows()."
        )
    if kind == "hf":
        yield from _iter_rows_hf(source)
        return
    if kind == "modelscope":
        yield from _iter_rows_modelscope(source)
        return
    if kind == "jsonl_filelist":
        with Path(source["filelist"]).open("r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
        yield from _iter_jsonl_files(files)
        return
    if kind == "jsonl_files":
        yield from _iter_jsonl_files(list(source["files"]))
        return
    raise ValueError(f"Unsupported source kind: {kind}")


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_stringify_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "content" in value:
            return _stringify_value(value["content"])
        if "text" in value:
            return _stringify_value(value["text"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _row_to_text(row: dict[str, Any], source: dict[str, Any]) -> str:
    if "concat_fields" in source:
        parts = []
        for key in source["concat_fields"]:
            if key in row:
                part = _stringify_value(row[key]).strip()
                if part:
                    parts.append(part)
        return "\n\n".join(parts).strip()

    fields = list(source.get("text_fields") or [])
    if source.get("text_field"):
        fields.insert(0, source["text_field"])
    fields.extend(COMMON_TEXT_KEYS)

    seen = set()
    for key in fields:
        if key in seen:
            continue
        seen.add(key)
        if key not in row:
            continue
        text = _stringify_value(row[key]).strip()
        if text:
            return text

    for value in row.values():
        text = _stringify_value(value).strip()
        if text:
            return text
    return ""


def _compute_row_budgets(
    spec: dict[str, Any], target_total_rows: Optional[int]
) -> list[int]:
    sources = spec["sources"]
    if target_total_rows is None:
        budgets = []
        for src in sources:
            if src.get("max_rows"):
                budgets.append(int(src["max_rows"]))
                continue
            if src.get("max_sequences"):
                budgets.append(0)
                continue
            if not src.get("max_rows"):
                raise ValueError(
                    "Each source needs max_rows or max_sequences when target_total_rows is not set."
                )
        return budgets

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
    for idx, (src, weight) in enumerate(zip(sources, weights)):
        if explicit[idx] > 0:
            continue
        quota = int(round(remaining * weight / total_weight))
        budgets[idx] = quota
        assigned += quota
    if assigned != remaining:
        for idx, src in enumerate(sources):
            if explicit[idx] == 0:
                budgets[idx] += remaining - assigned
                break
    return budgets


def _resolve_dist_value(explicit: Optional[int], env_name: str, default: int) -> int:
    if explicit is not None:
        return int(explicit)
    env_value = os.environ.get(env_name)
    if env_value is None or env_value == "":
        return int(default)
    return int(env_value)


def _distributed_context(args: argparse.Namespace) -> dict[str, int | bool]:
    world_size = _resolve_dist_value(args.dp_world_size, "WORLD_SIZE", 1)
    rank = _resolve_dist_value(args.dp_rank, "RANK", 0)
    local_rank = _resolve_dist_value(args.local_rank, "LOCAL_RANK", rank)
    return {
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "enabled": bool(int(world_size) > 1),
    }


def _aggregate_source_reports(source_report_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for reports in source_report_lists:
        for report in reports:
            name = str(report["name"])
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "kind": report.get("kind"),
                    "row_budget": int(report.get("row_budget", 0) or 0),
                    "rows_used": 0,
                    "sequences_emitted": 0,
                }
                order.append(name)
            merged[name]["rows_used"] += int(report.get("rows_used", 0) or 0)
            merged[name]["sequences_emitted"] += int(report.get("sequences_emitted", 0) or 0)
    return [merged[name] for name in order]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_has_any_data(manifest: dict[str, Any]) -> bool:
    for layer_meta in manifest.get("layers", {}).values():
        try:
            if int(layer_meta.get("num_tokens", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _merge_covariance_dirs(
    input_dirs: list[Path],
    output_dir: Path,
    hidden_size: int,
    layer_ids: list[int],
    metadata: dict[str, Any],
) -> None:
    accumulator = CovarianceAccumulator(hidden_size, layer_ids)
    for input_dir in input_dirs:
        manifest_path = input_dir / "covariance_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_manifest(manifest_path)
        for layer_id in layer_ids:
            layer_meta = manifest["layers"].get(str(layer_id))
            if not layer_meta:
                continue
            payload = torch.load(Path(layer_meta["path"]), map_location="cpu")
            count = int(payload.get("xtx_num_tokens", 0) or 0)
            if count <= 0:
                continue
            covariance = payload["covariance"].to(dtype=torch.float64, device="cpu")
            accumulator.xtx[layer_id].add_(covariance * float(count))
            accumulator.counts[layer_id] += count
    accumulator.save(output_dir, metadata=metadata)


def _load_covariance_accumulator(
    input_dir: Path, hidden_size: int, layer_ids: list[int]
) -> tuple["CovarianceAccumulator | None", dict[str, Any] | None]:
    manifest_path = input_dir / "covariance_manifest.json"
    if not manifest_path.exists():
        return None, None
    manifest = _load_manifest(manifest_path)
    accumulator = CovarianceAccumulator(hidden_size, layer_ids)
    found_any = False
    for layer_id in layer_ids:
        layer_meta = manifest.get("layers", {}).get(str(layer_id))
        if not layer_meta:
            continue
        payload = torch.load(Path(layer_meta["path"]), map_location="cpu")
        count = int(payload.get("xtx_num_tokens", 0) or 0)
        if count <= 0:
            continue
        covariance = payload["covariance"].to(dtype=torch.float64, device="cpu")
        accumulator.xtx[layer_id].copy_(covariance * float(count))
        accumulator.counts[layer_id] = count
        found_any = True
    if not found_any:
        return None, manifest
    return accumulator, manifest


def _build_default_source_state(
    source: dict[str, Any],
    row_budget: int,
    source_sequence_budget: int,
    local_sequence_budget: int,
) -> dict[str, Any]:
    return {
        "name": source.get("name", source.get("path", source["kind"])),
        "kind": source["kind"],
        "row_budget": int(row_budget),
        "sequence_budget": int(source_sequence_budget),
        "local_sequence_budget": int(local_sequence_budget),
        "rows_used": 0,
        "sequences_emitted": 0,
        "global_valid_rows": 0,
        "buffer": [],
        "packed_cursor": {"file_index": 0, "row_index": 0, "chunk_index": 0},
        "completed": False,
    }


def _source_report_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": state["name"],
        "kind": state["kind"],
        "rows_used": int(state.get("rows_used", 0) or 0),
        "row_budget": int(state.get("row_budget", 0) or 0),
        "sequence_budget": int(state.get("sequence_budget", 0) or 0),
        "local_sequence_budget": int(state.get("local_sequence_budget", 0) or 0),
        "sequences_emitted": int(state.get("sequences_emitted", 0) or 0),
        "covariance_dir": state.get("covariance_dir"),
    }


def _resume_state_is_compatible(
    resume_state: dict[str, Any] | None,
    spec: dict[str, Any],
    row_budgets: list[int],
    world_size: int,
) -> bool:
    if not resume_state:
        return False
    source_states = list(resume_state.get("source_states") or [])
    if len(source_states) != len(spec["sources"]):
        return False
    for source_state, source, row_budget in zip(source_states, spec["sources"], row_budgets):
        if str(source_state.get("name")) != str(source.get("name", source.get("path", source["kind"]))):
            return False
        if str(source_state.get("kind")) != str(source["kind"]):
            return False
        source_sequence_budget = int(source.get("max_sequences", 0) or 0)
        local_sequence_budget = source_sequence_budget
        if source_sequence_budget > 0 and world_size > 1:
            local_sequence_budget = source_sequence_budget // world_size
            if resume_state.get("rank", 0) < (source_sequence_budget % world_size):
                local_sequence_budget += 1
        if int(source_state.get("row_budget", 0) or 0) != int(row_budget):
            return False
        if int(source_state.get("sequence_budget", 0) or 0) != int(source_sequence_budget):
            return False
        if int(source_state.get("local_sequence_budget", 0) or 0) != int(local_sequence_budget):
            return False
    return True


def _resolve_model_kwargs(
    args: argparse.Namespace, dist_ctx: dict[str, int | bool]
) -> tuple[dict[str, Any], str, str | None]:
    device_map_value: Any = args.device_map
    load_mode = "auto_sharded"
    forward_device: str | None = None
    if dist_ctx["enabled"] or args.replica_device_map == "single_gpu":
        local_rank = int(dist_ctx["local_rank"])
        if not torch.cuda.is_available():
            raise RuntimeError("Replica-device loading requires CUDA.")
        torch.cuda.set_device(local_rank)
        device_map_value = {"": f"cuda:{local_rank}"}
        load_mode = "single_gpu_replica"
        forward_device = f"cuda:{local_rank}"
    elif isinstance(args.device_map, str) and args.device_map.startswith("cuda:"):
        device_map_value = {"": args.device_map}
        load_mode = "single_gpu_explicit"
        forward_device = args.device_map
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(args.dtype),
        "device_map": device_map_value,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    return model_kwargs, load_mode, forward_device


def _fixed_deserialize_serialized_triton_tensor(state: dict[str, Any], device: Any):
    import transformers.integrations.mxfp4 as hf_mxfp4

    hub = hf_mxfp4.triton_kernels_hub
    FP4 = hub.tensor.FP4
    Storage = hub.tensor.Storage
    Tensor = hub.tensor.Tensor
    layout = hub.tensor_details.layout

    data = state["data"]
    if getattr(data, "device", None) != device:
        data = data.to(device=device, non_blocking=False)
    if state["kind"] == "mxfp4_weight":
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        storage_layout = value_layout(tuple(data.shape), **value_layout_opts)
        dtype = FP4
    elif state["kind"] == "strided":
        storage_layout = layout.StridedLayout(tuple(data.shape))
        dtype = data.dtype
    else:
        raise ValueError(f"Unsupported serialized Triton tensor kind: {state['kind']}")
    storage = Storage(data, layout=storage_layout)
    return Tensor(
        storage=storage,
        dtype=dtype,
        shape=list(state["shape"]),
        shape_max=list(state["shape_max"]),
    )


def _cache_prefix_from_param_name(param_name: str) -> str | None:
    parts = param_name.split(".")
    if len(parts) < 6:
        return None
    return "__".join(parts[:5] + ["shared"])


def _install_mxfp4_preswizzle_cache(cache_dir: str | Path | None) -> bool:
    if not cache_dir:
        return False
    cache_path = Path(cache_dir).resolve()
    if not cache_path.exists():
        return False

    import transformers.integrations.mxfp4 as hf_mxfp4

    original_loader = hf_mxfp4.load_and_swizzle_mxfp4
    if getattr(original_loader, "_gptoss_preswizzle_patched", False):
        return True

    def _cached_load_and_swizzle_mxfp4(
        module, param_name, param_value, target_device, triton_kernels_hub, **kwargs
    ):
        device_mesh = kwargs.get("device_mesh")
        if device_mesh is not None:
            return original_loader(
                module,
                param_name,
                param_value,
                target_device,
                triton_kernels_hub,
                **kwargs,
            )
        setattr(module, param_name.rsplit(".", 1)[1], torch.nn.Parameter(param_value, requires_grad=False))
        if "blocks" in param_name:
            proj = param_name.split(".")[-1].split("_blocks")[0]
        elif "scales" in param_name:
            proj = param_name.split(".")[-1].split("_scales")[0]
        else:
            return

        blocks_attr = f"{proj}_blocks"
        scales_attr = f"{proj}_scales"
        blocks = getattr(module, blocks_attr)
        scales = getattr(module, scales_attr)
        if blocks.device.type == "meta" or scales.device.type == "meta":
            return

        local_experts = int(blocks.size(0))
        if proj == "gate_up_proj":
            expected_shape = [local_experts, int(module.hidden_size), int(module.intermediate_size) * 2]
        else:
            expected_shape = [local_experts, int(module.intermediate_size), int(module.hidden_size)]

        cache_prefix = _cache_prefix_from_param_name(param_name)
        cache_hit = None
        if cache_prefix is not None:
            for candidate in sorted(cache_path.glob(f"{cache_prefix}__*.pt")):
                payload = torch.load(candidate, map_location="cpu")
                if list(payload.get("weight", {}).get("shape", [])) == expected_shape:
                    cache_hit = payload
                    break

        if cache_hit is None:
            return original_loader(
                module,
                param_name,
                param_value,
                target_device,
                triton_kernels_hub,
                **kwargs,
            )

        if getattr(target_device, "type", target_device) == "cpu":
            target_device = "cuda"
        triton_weight_tensor = _fixed_deserialize_serialized_triton_tensor(
            cache_hit["weight"], target_device
        )
        weight_scale = _fixed_deserialize_serialized_triton_tensor(
            cache_hit["weight_scale"], target_device
        )
        setattr(module, proj, triton_weight_tensor)
        PrecisionConfig = triton_kernels_hub.matmul_ogs.PrecisionConfig
        FlexCtx = triton_kernels_hub.matmul_ogs.FlexCtx
        InFlexData = triton_kernels_hub.matmul_ogs.InFlexData
        setattr(
            module,
            f"{proj}_precision_config",
            PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
        )
        delattr(module, scales_attr)
        delattr(module, blocks_attr)
        del blocks
        del scales
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _cached_load_and_swizzle_mxfp4._gptoss_preswizzle_patched = True
    hf_mxfp4.load_and_swizzle_mxfp4 = _cached_load_and_swizzle_mxfp4
    return True


class CovarianceAccumulator:
    def __init__(self, hidden_size: int, layer_ids: Iterable[int]):
        self.hidden_size = int(hidden_size)
        self.layer_ids = list(layer_ids)
        self.xtx = {
            layer_id: torch.zeros((hidden_size, hidden_size), dtype=torch.float64)
            for layer_id in self.layer_ids
        }
        self.counts = {layer_id: 0 for layer_id in self.layer_ids}

    def accumulate(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).to(dtype=torch.float32)
        gram = hidden_2d.transpose(0, 1) @ hidden_2d
        self.xtx[layer_id].add_(gram.to(dtype=torch.float64, device="cpu"))
        self.counts[layer_id] += int(hidden_2d.shape[0])

    def save(self, output_dir: Path, metadata: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"layers": {}, **metadata}
        for layer_id in self.layer_ids:
            count = max(int(self.counts[layer_id]), 1)
            payload = {
                "covariance": (self.xtx[layer_id] / float(count)).to(dtype=torch.float32),
                "xtx_num_tokens": int(self.counts[layer_id]),
                "hidden_size": self.hidden_size,
                "layer_id": int(layer_id),
            }
            path = output_dir / f"layer_{layer_id:02d}.pt"
            torch.save(payload, path)
            manifest["layers"][str(layer_id)] = {
                "path": str(path),
                "num_tokens": int(self.counts[layer_id]),
            }
        _save_json(output_dir / "covariance_manifest.json", manifest)


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path).resolve()
    dataset_spec_path = Path(args.dataset_spec_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    spec = _load_json(dataset_spec_path)
    dist_ctx = _distributed_context(args)
    world_size = int(dist_ctx["world_size"])
    rank = int(dist_ctx["rank"])
    local_rank = int(dist_ctx["local_rank"])
    partial_root = output_dir / "partials"
    rank_output_dir = output_dir if world_size == 1 else partial_root / f"rank_{rank:02d}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    row_budgets = _compute_row_budgets(spec, args.target_total_rows)

    plan = []
    for source, row_budget in zip(spec["sources"], row_budgets):
        source_ref = source.get("path", source.get("filelist", "<local-files>"))
        plan.append(
            {
                "name": source.get("name", source_ref),
                "kind": source["kind"],
                "path": source_ref,
                "row_budget": int(row_budget),
            }
        )

    config = _load_json(model_path / "config.json")
    layer_end = int(args.layer_end) if args.layer_end is not None else int(config["num_hidden_layers"])
    if not (0 <= int(args.layer_start) < layer_end <= int(config["num_hidden_layers"])):
        raise ValueError(
            f"Invalid layer range [{args.layer_start}, {layer_end}) for {config['num_hidden_layers']} layers."
        )
    layer_ids = list(range(int(args.layer_start), layer_end))
    hidden_size = int(config["hidden_size"])

    source_state_defaults = []
    for source, row_budget in zip(spec["sources"], row_budgets):
        source_sequence_budget = int(source.get("max_sequences", 0) or 0)
        local_sequence_budget = source_sequence_budget
        if source_sequence_budget > 0 and world_size > 1:
            local_sequence_budget = source_sequence_budget // world_size
            if rank < (source_sequence_budget % world_size):
                local_sequence_budget += 1
        source_state_defaults.append(
            _build_default_source_state(
                source=source,
                row_budget=int(row_budget),
                source_sequence_budget=source_sequence_budget,
                local_sequence_budget=local_sequence_budget,
            )
        )

    if args.plan_only:
        print(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "dataset_spec_json": str(dataset_spec_path),
                    "output_dir": str(output_dir),
                    "seq_len": int(args.seq_len),
                    "batch_size": int(args.batch_size),
                    "layer_ids": layer_ids,
                    "world_size": world_size,
                    "rank": rank,
                    "sources": plan,
                },
                indent=2,
            )
        )
        return

    resume_accumulator: CovarianceAccumulator | None = None
    resume_manifest: dict[str, Any] | None = None
    resume_state: dict[str, Any] | None = None
    if args.resume:
        resume_accumulator, resume_manifest = _load_covariance_accumulator(
            rank_output_dir, hidden_size, layer_ids
        )
        candidate_resume_state = None
        if resume_manifest is not None:
            candidate_resume_state = resume_manifest.get("resume_state")
            has_data = _manifest_has_any_data(resume_manifest)
            compatible = _resume_state_is_compatible(
                candidate_resume_state, spec, row_budgets, world_size
            )
            if candidate_resume_state and compatible and has_data:
                resume_state = candidate_resume_state
            elif candidate_resume_state and not compatible:
                print(
                    f"[resume] rank={rank} ignoring incompatible resume state from {rank_output_dir}",
                    flush=True,
                )
            elif candidate_resume_state and not has_data:
                print(
                    f"[resume] rank={rank} ignoring zero-data resume state from {rank_output_dir}",
                    flush=True,
                )
            elif not candidate_resume_state:
                print(
                    f"[resume] rank={rank} manifest present at {rank_output_dir} has no resume_state; running from scratch",
                    flush=True,
                )
        else:
            print(
                f"[resume] rank={rank} no manifest found at {rank_output_dir}; running from scratch",
                flush=True,
            )

    model_kwargs, load_mode, forward_device = _resolve_model_kwargs(args, dist_ctx)
    cache_enabled = _install_mxfp4_preswizzle_cache(args.mxfp4_preswizzle_dir)

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "dataset_spec_json": str(dataset_spec_path),
                "output_dir": str(output_dir),
                "seq_len": int(args.seq_len),
                "batch_size": int(args.batch_size),
                "device_map": model_kwargs["device_map"],
                "load_mode": load_mode,
                "forward_device": forward_device,
                "mxfp4_preswizzle_dir": str(Path(args.mxfp4_preswizzle_dir).resolve())
                if args.mxfp4_preswizzle_dir
                else None,
                "mxfp4_preswizzle_cache_enabled": bool(cache_enabled),
                "dtype": args.dtype,
                "world_size": world_size,
                "rank": rank,
                "local_rank": local_rank,
                "sources": plan,
            },
            indent=2,
        ),
        flush=True,
    )

    source_states = [dict(state) for state in source_state_defaults]
    processed_batches = 0
    total_sequences = 0
    source_reports = []
    packed_batch: list[list[int]] = []
    all_sources_completed = False
    if resume_state is not None:
        source_states = [dict(state) for state in resume_state.get("source_states", source_state_defaults)]
        processed_batches = int(resume_state.get("processed_batches", 0) or 0)
        total_sequences = int(resume_state.get("total_sequences", 0) or 0)
        packed_batch = [
            [int(token_id) for token_id in sequence]
            for sequence in list(resume_state.get("packed_batch") or [])
        ]
        all_sources_completed = bool(resume_state.get("all_sources_completed", False))
        source_reports = [
            _source_report_from_state(state)
            for state in source_states
            if bool(state.get("completed"))
        ]
        print(
            f"[resume] rank={rank} batches={processed_batches} sequences={total_sequences} "
            f"completed_sources={len(source_reports)} packed_batch={len(packed_batch)}",
            flush=True,
        )

    accumulator = resume_accumulator or CovarianceAccumulator(hidden_size, layer_ids)
    model = None
    if not all_sources_completed:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()

    active_source_accumulator: CovarianceAccumulator | None = None
    handles = []
    if model is not None:
        for layer_id in layer_ids:
            def _hook(_module, inputs, layer_id=layer_id):
                hidden_states = inputs[0].detach()
                accumulator.accumulate(layer_id, hidden_states)
                if active_source_accumulator is not None:
                    active_source_accumulator.accumulate(layer_id, hidden_states)

            handle = model.model.layers[layer_id].self_attn.k_proj.register_forward_pre_hook(_hook)
            handles.append(handle)

    def _current_resume_state() -> dict[str, Any]:
        return {
            "version": 1,
            "processed_batches": int(processed_batches),
            "total_sequences": int(total_sequences),
            "packed_batch": packed_batch,
            "source_states": source_states,
            "rank": rank,
            "world_size": world_size,
            "all_sources_completed": all(bool(state.get("completed")) for state in source_states),
        }

    def _flush_batch() -> None:
        nonlocal processed_batches, total_sequences, packed_batch
        if not packed_batch:
            return
        batch_tensor = torch.tensor(packed_batch, dtype=torch.long)
        attention_mask = torch.ones_like(batch_tensor)
        if forward_device is not None:
            batch_tensor = batch_tensor.to(device=forward_device, non_blocking=False)
            attention_mask = attention_mask.to(device=forward_device, non_blocking=False)
        with torch.inference_mode():
            assert model is not None
            model(input_ids=batch_tensor, attention_mask=attention_mask, use_cache=False)
        processed_batches += 1
        total_sequences += len(packed_batch)
        packed_batch = []
        if processed_batches % max(int(args.save_every_batches), 1) == 0:
            accumulator.save(
                rank_output_dir if world_size > 1 else output_dir,
                metadata={
                    "model_path": str(model_path),
                    "dataset_spec_json": str(dataset_spec_path),
                    "processed_batches": processed_batches,
                    "total_sequences": total_sequences,
                    "world_size": world_size,
                    "rank": rank,
                    "sources": source_reports,
                    "resume_state": _current_resume_state(),
                },
            )
            print(
                f"[checkpoint] rank={rank} batches={processed_batches} sequences={total_sequences}",
                flush=True,
            )

    for source_index, (source, row_budget) in enumerate(zip(spec["sources"], row_budgets)):
        state = source_states[source_index]
        source_name = source.get("name", source.get("path", source["kind"]))
        source_sequence_budget = int(source.get("max_sequences", 0) or 0)
        local_sequence_budget = int(state.get("local_sequence_budget", 0) or 0)
        if bool(state.get("completed")):
            print(f"[resume] rank={rank} skipping completed source {source_name}", flush=True)
            continue

        buffer = [int(token_id) for token_id in list(state.get("buffer") or [])]
        rows_used = int(state.get("rows_used", 0) or 0)
        sequences_emitted = int(state.get("sequences_emitted", 0) or 0)
        global_valid_rows = int(state.get("global_valid_rows", 0) or 0)
        resume_valid_rows = int(state.get("global_valid_rows", 0) or 0)
        packed_cursor = dict(state.get("packed_cursor") or {"file_index": 0, "row_index": 0})
        packed_cursor.setdefault("chunk_index", 0)

        source_output_dir = output_dir / "by_source" / _safe_name(source_name)
        source_partial_output_dir = rank_output_dir / "by_source" / _safe_name(source_name)
        source_accumulator = None
        if args.save_per_source_covariance:
            source_accumulator, _ = _load_covariance_accumulator(
                source_partial_output_dir if world_size > 1 else source_output_dir,
                hidden_size,
                layer_ids,
            )
            if source_accumulator is None:
                source_accumulator = CovarianceAccumulator(hidden_size, layer_ids)
        active_source_accumulator = source_accumulator
        print(
            f"[source] {source_name} rows={row_budget} max_sequences={source_sequence_budget} "
            f"local_max_sequences={local_sequence_budget} resume_rows={rows_used} "
            f"resume_sequences={sequences_emitted}",
            flush=True,
        )

        if source["kind"] == "packed_torch_filelist":
            start_file_index = int(packed_cursor.get("file_index", 0) or 0)
            start_row_index = int(packed_cursor.get("row_index", 0) or 0)
            start_chunk_index = int(packed_cursor.get("chunk_index", 0) or 0)
            for file_index, row_index, token_ids_tensor in _iter_packed_torch_filelist(
                source, start_file_index=start_file_index, start_row_index=start_row_index
            ):
                if local_sequence_budget and sequences_emitted >= local_sequence_budget:
                    break
                if source_sequence_budget and global_valid_rows >= source_sequence_budget:
                    break
                if row_budget and rows_used >= row_budget:
                    break

                if isinstance(token_ids_tensor, torch.Tensor):
                    token_ids_list = token_ids_tensor.reshape(-1).tolist()
                else:
                    token_ids_list = [int(token_id) for token_id in token_ids_tensor]

                if len(token_ids_list) < int(args.seq_len):
                    rows_used += 1
                    state["rows_used"] = int(rows_used)
                    packed_cursor["file_index"] = int(file_index)
                    packed_cursor["row_index"] = int(row_index) + 1
                    packed_cursor["chunk_index"] = 0
                    state["global_valid_rows"] = int(global_valid_rows)
                    state["packed_cursor"] = dict(packed_cursor)
                    continue

                total_chunks = len(token_ids_list) // int(args.seq_len)
                if total_chunks <= 0:
                    rows_used += 1
                    state["rows_used"] = int(rows_used)
                    packed_cursor["file_index"] = int(file_index)
                    packed_cursor["row_index"] = int(row_index) + 1
                    packed_cursor["chunk_index"] = 0
                    state["global_valid_rows"] = int(global_valid_rows)
                    state["packed_cursor"] = dict(packed_cursor)
                    continue

                row_start_chunk = 0
                if int(file_index) == start_file_index and int(row_index) == start_row_index:
                    row_start_chunk = max(0, start_chunk_index)
                    if row_start_chunk >= total_chunks:
                        rows_used += 1
                        state["rows_used"] = int(rows_used)
                        packed_cursor["file_index"] = int(file_index)
                        packed_cursor["row_index"] = int(row_index) + 1
                        packed_cursor["chunk_index"] = 0
                        state["global_valid_rows"] = int(global_valid_rows)
                        state["packed_cursor"] = dict(packed_cursor)
                        continue

                for chunk_index in range(row_start_chunk, total_chunks):
                    if local_sequence_budget and sequences_emitted >= local_sequence_budget:
                        break
                    if source_sequence_budget and global_valid_rows >= source_sequence_budget:
                        break

                    take_this_rank = (global_valid_rows % world_size) == rank
                    global_valid_rows += 1
                    packed_cursor["file_index"] = int(file_index)
                    packed_cursor["row_index"] = int(row_index)
                    packed_cursor["chunk_index"] = int(chunk_index)
                    state["packed_cursor"] = dict(packed_cursor)
                    if source_sequence_budget and global_valid_rows > source_sequence_budget:
                        break

                    if take_this_rank:
                        chunk_start = int(chunk_index * int(args.seq_len))
                        token_ids = token_ids_list[chunk_start : chunk_start + int(args.seq_len)]
                        packed_batch.append(token_ids)
                        sequences_emitted += 1
                        state["sequences_emitted"] = int(sequences_emitted)
                        if len(packed_batch) >= int(args.batch_size):
                            _flush_batch()

                rows_used += 1
                state["rows_used"] = int(rows_used)

                packed_cursor["file_index"] = int(file_index)
                packed_cursor["row_index"] = int(row_index) + 1
                packed_cursor["chunk_index"] = 0
                state["global_valid_rows"] = int(global_valid_rows)
                state["packed_cursor"] = dict(packed_cursor)
        else:
            for row in _iter_rows(source):
                if row_budget and rows_used >= row_budget:
                    break
                if local_sequence_budget and sequences_emitted >= local_sequence_budget:
                    break
                text = _row_to_text(row, source)
                if not text:
                    continue
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                if row_budget and global_valid_rows >= row_budget:
                    break
                if global_valid_rows < resume_valid_rows:
                    global_valid_rows += 1
                    continue
                take_this_rank = (global_valid_rows % world_size) == rank
                global_valid_rows += 1
                state["global_valid_rows"] = int(global_valid_rows)
                if not take_this_rank:
                    continue
                if args.append_eos and eos_token_id is not None:
                    token_ids = list(token_ids) + [int(eos_token_id)]
                buffer.extend(int(x) for x in token_ids)
                rows_used += 1
                state["rows_used"] = int(rows_used)

                while len(buffer) >= int(args.seq_len):
                    if local_sequence_budget and sequences_emitted >= local_sequence_budget:
                        break
                    packed_batch.append(buffer[: int(args.seq_len)])
                    buffer = buffer[int(args.seq_len) :]
                    sequences_emitted += 1
                    state["sequences_emitted"] = int(sequences_emitted)
                    state["buffer"] = list(buffer)
                    if len(packed_batch) >= int(args.batch_size):
                        _flush_batch()
                state["buffer"] = list(buffer)

        state["rows_used"] = int(rows_used)
        state["sequences_emitted"] = int(sequences_emitted)
        state["global_valid_rows"] = int(global_valid_rows)
        state["buffer"] = list(buffer)
        state["packed_cursor"] = dict(packed_cursor)

        _flush_batch()

        state["completed"] = True
        state["covariance_dir"] = str(source_output_dir) if source_accumulator else None
        source_reports = [
            _source_report_from_state(report_state)
            for report_state in source_states
            if bool(report_state.get("completed"))
        ]
        accumulator.save(
            rank_output_dir if world_size > 1 else output_dir,
            metadata={
                "model_path": str(model_path),
                "dataset_spec_json": str(dataset_spec_path),
                "processed_batches": processed_batches,
                "total_sequences": total_sequences,
                "world_size": world_size,
                "rank": rank,
                "sources": source_reports,
                "resume_state": _current_resume_state(),
            },
        )
        if source_accumulator is not None:
            source_accumulator.save(
                source_partial_output_dir if world_size > 1 else source_output_dir,
                metadata={
                    "model_path": str(model_path),
                    "dataset_spec_json": str(dataset_spec_path),
                    "source": {
                        "name": source_name,
                        "kind": source["kind"],
                        "path": source.get("path", source.get("filelist", "<local-files>")),
                    },
                    "rows_used": rows_used,
                    "row_budget": int(row_budget),
                    "sequence_budget": source_sequence_budget,
                    "local_sequence_budget": local_sequence_budget,
                    "sequences_emitted": sequences_emitted,
                    "seq_len": int(args.seq_len),
                    "batch_size": int(args.batch_size),
                    "world_size": world_size,
                    "rank": rank,
                },
            )
        print(
            f"[source_done] rank={rank} {source_name} rows_used={rows_used} sequences={sequences_emitted}",
            flush=True,
        )
        active_source_accumulator = None

    _flush_batch()

    for handle in handles:
        handle.remove()

    source_reports = [
        _source_report_from_state(state)
        for state in source_states
        if bool(state.get("completed"))
    ]
    final_metadata = {
        "model_path": str(model_path),
        "dataset_spec_json": str(dataset_spec_path),
        "processed_batches": processed_batches,
        "total_sequences": total_sequences,
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "world_size": world_size,
        "rank": rank,
        "sources": source_reports,
        "resume_state": _current_resume_state(),
    }
    if world_size > 1:
        accumulator.save(rank_output_dir, metadata=final_metadata)
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        torch.distributed.barrier()
        if rank == 0:
            rank_dirs = [partial_root / f"rank_{worker_rank:02d}" for worker_rank in range(world_size)]
            rank_manifests = [
                _load_manifest(rank_dir / "covariance_manifest.json")
                for rank_dir in rank_dirs
            ]
            merged_metadata = {
                "model_path": str(model_path),
                "dataset_spec_json": str(dataset_spec_path),
                "processed_batches": sum(int(m.get("processed_batches", 0) or 0) for m in rank_manifests),
                "total_sequences": sum(int(m.get("total_sequences", 0) or 0) for m in rank_manifests),
                "seq_len": int(args.seq_len),
                "batch_size": int(args.batch_size),
                "world_size": world_size,
                "sources": _aggregate_source_reports(
                    [list(m.get("sources", [])) for m in rank_manifests]
                ),
                "merged_from": [str(rank_dir) for rank_dir in rank_dirs],
            }
            _merge_covariance_dirs(
                input_dirs=rank_dirs,
                output_dir=output_dir,
                hidden_size=int(config["hidden_size"]),
                layer_ids=layer_ids,
                metadata=merged_metadata,
            )
            if args.save_per_source_covariance:
                source_dir_names = sorted(
                    {
                        source_dir.name
                        for rank_dir in rank_dirs
                        for source_dir in (rank_dir / "by_source").glob("*")
                        if source_dir.is_dir()
                    }
                )
                for source_dir_name in source_dir_names:
                    input_dirs = [
                        rank_dir / "by_source" / source_dir_name
                        for rank_dir in rank_dirs
                        if (rank_dir / "by_source" / source_dir_name / "covariance_manifest.json").exists()
                    ]
                    if not input_dirs:
                        continue
                    source_manifests = [
                        _load_manifest(source_dir / "covariance_manifest.json")
                        for source_dir in input_dirs
                    ]
                    _merge_covariance_dirs(
                        input_dirs=input_dirs,
                        output_dir=output_dir / "by_source" / source_dir_name,
                        hidden_size=int(config["hidden_size"]),
                        layer_ids=layer_ids,
                        metadata={
                            "model_path": str(model_path),
                            "dataset_spec_json": str(dataset_spec_path),
                            "seq_len": int(args.seq_len),
                            "batch_size": int(args.batch_size),
                            "world_size": world_size,
                            "rows_used": sum(int(m.get("rows_used", 0) or 0) for m in source_manifests),
                            "sequences_emitted": sum(int(m.get("sequences_emitted", 0) or 0) for m in source_manifests),
                            "merged_from": [str(path) for path in input_dirs],
                            "source": source_manifests[0].get("source"),
                        },
                    )
            print(
                f"[done] merged covariance stats to {output_dir} world_size={world_size}",
                flush=True,
            )
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    else:
        accumulator.save(output_dir, metadata=final_metadata)
        print(
            f"[done] wrote covariance stats to {output_dir} batches={processed_batches} sequences={total_sequences}",
            flush=True,
        )


if __name__ == "__main__":
    main()
