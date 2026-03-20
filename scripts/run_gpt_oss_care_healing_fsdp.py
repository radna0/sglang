#!/usr/bin/env python3

import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _load_pyc_module():
    script_path = Path(__file__).resolve()
    pyc_name = (
        f"{script_path.stem}.original.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    )
    pyc_path = script_path.parent / "__pycache__" / pyc_name
    if not pyc_path.exists():
        raise FileNotFoundError(f"Missing cached bytecode module: {pyc_path}")
    sys.path.insert(0, str(script_path.parent))
    loader = importlib.machinery.SourcelessFileLoader(
        "_gpt_oss_care_healing_fsdp_pyc",
        str(pyc_path),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Could not create import spec for {pyc_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def _fixed_deserialize_serialized_triton_tensor(state, device):  # noqa: ANN001
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


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _safe_cuda_mem() -> dict[str, int | str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": "unavailable"}
        device = torch.cuda.current_device()
        return {
            "device": int(device),
            "allocated": int(torch.cuda.memory_allocated(device)),
            "reserved": int(torch.cuda.memory_reserved(device)),
            "max_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception as exc:  # pragma: no cover - debug only
        return {"cuda_mem_error": repr(exc)}


def _trace_event(event: str, **payload) -> None:
    if not _env_enabled("GPTOSS_CARE_FSDP_TRACE"):
        return
    record = {
        "ts": time.time(),
        "event": event,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "mem": _safe_cuda_mem(),
    }
    if payload:
        record.update(payload)
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    trace_path = os.environ.get("GPTOSS_CARE_FSDP_TRACE_FILE", "").strip()
    if trace_path:
        with Path(trace_path).open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _wrap_with_trace(name: str) -> None:
    fn = getattr(_MODULE, name, None)
    if fn is None:
        return

    def _wrapped(*args, **kwargs):  # noqa: ANN202
        _trace_event(f"{name}:start")
        started = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - debug only
            _trace_event(
                f"{name}:error",
                elapsed_s=round(time.time() - started, 3),
                error=repr(exc),
                traceback=traceback.format_exc(limit=8),
            )
            raise
        _trace_event(f"{name}:done", elapsed_s=round(time.time() - started, 3))
        return result

    setattr(_MODULE, name, _wrapped)
    globals()[name] = _wrapped


_MODULE = _load_pyc_module()
_MODULE.__file__ = str(Path(__file__).resolve())
_MODULE.json = json
_MODULE._deserialize_serialized_triton_tensor = _fixed_deserialize_serialized_triton_tensor

for _trace_name in (
    "_maybe_patch_hf_mxfp4_preswizzle_cache",
    "_maybe_enable_quantized_mxfp4_experts",
    "_load_model_from_checkpoint",
    "_load_teacher_topk_cache",
    "_export_healed_absorbed_checkpoint",
):
    _wrap_with_trace(_trace_name)

_original_main = _MODULE.main


def _traced_main() -> None:
    _trace_event("main:start", argv=sys.argv[1:])
    started = time.time()
    try:
        _original_main()
    except Exception as exc:  # pragma: no cover - debug only
        _trace_event(
            "main:error",
            elapsed_s=round(time.time() - started, 3),
            error=repr(exc),
            traceback=traceback.format_exc(limit=12),
        )
        raise
    _trace_event("main:done", elapsed_s=round(time.time() - started, 3))


_MODULE.main = _traced_main

for _name in dir(_MODULE):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_MODULE, _name))

main = _MODULE.main


if __name__ == "__main__":
    main()
