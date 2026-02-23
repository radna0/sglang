"""Utilities to dump FP8 KV-cache per-layer scaling factors to a JSON file.

This is a small helper for generating a `--quantization-param-path` file
("QuantParamSchema") in environments where the model does not ship KV scales.

Usage (typically set in the server environment):
  - SGLANG_KV_SCALE_DUMP_PATH=/path/to/kv_scales.json
  - SGLANG_KV_SCALE_DUMP_MODEL_TYPE=gpt_oss
  - SGLANG_KV_SCALE_DUMP_NUM_LAYERS=36
  - SGLANG_KV_SCALE_DUMP_TP_SIZE=1  (optional; default=1)

At runtime, call `record_kv_scale(layer_id, scale_float)` whenever a scale is
computed (e.g. via best-effort auto-calibration). On process exit, the JSON is
written if all required env vars are present.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Dict


_OUT_PATH = (os.environ.get("SGLANG_KV_SCALE_DUMP_PATH") or "").strip()
_MODEL_TYPE = (os.environ.get("SGLANG_KV_SCALE_DUMP_MODEL_TYPE") or "").strip()
_NUM_LAYERS = int((os.environ.get("SGLANG_KV_SCALE_DUMP_NUM_LAYERS") or "0").strip() or 0)
_TP_SIZE = int((os.environ.get("SGLANG_KV_SCALE_DUMP_TP_SIZE") or "1").strip() or 1)

# layer_id -> scale (Python float)
_SCALES: Dict[int, float] = {}


def record_kv_scale(layer_id: int, scale: float) -> None:
    """Record a KV-cache scale for a layer (best-effort; no-ops if disabled)."""
    if not _OUT_PATH:
        return
    try:
        _SCALES[int(layer_id)] = float(scale)
    except Exception:
        # Never let debug dumping affect inference.
        return


def _write_schema() -> None:
    if not _OUT_PATH:
        return
    if not _MODEL_TYPE or _NUM_LAYERS <= 0:
        # Misconfigured; do not write an invalid schema.
        return
    if _TP_SIZE <= 0:
        return

    # Schema requires all TP ranks and all layers to be present. For now, we
    # only generate rank-0 scales; other ranks (if any) reuse rank-0.
    rank0 = {i: float(_SCALES.get(i, 1.0)) for i in range(_NUM_LAYERS)}
    scaling_factor = {r: rank0 for r in range(_TP_SIZE)}

    schema = {
        "model_type": _MODEL_TYPE,
        "kv_cache": {
            "dtype": "float8_e4m3fn",
            "scaling_factor": scaling_factor,
        },
    }
    Path(_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(_OUT_PATH).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if _OUT_PATH:
    atexit.register(_write_schema)

