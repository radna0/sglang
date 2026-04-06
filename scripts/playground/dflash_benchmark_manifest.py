from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class DFlashServerContract:
    mode: str
    attention_backend: str
    draft_attention_backend: str
    moe_runner_backend: str
    sampling_backend: str
    kv_cache_dtype: str
    draft_kv_cache_dtype: str
    page_size: int
    draft_page_size: int
    block_size: int
    num_steps: int
    topk: int
    num_verify_tokens: int
    cuda_graph_mode: str
    disable_overlap_schedule: bool
    share_pools: Optional[bool] = None


@dataclass(frozen=True)
class DFlashBenchmarkManifest:
    run_name: str
    lane: str
    target_model_path: str
    draft_model_path: str
    reference_source: str
    server_contract: DFlashServerContract
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    host: str = field(default_factory=socket.gethostname)


def write_manifest(path: str | Path, manifest: DFlashBenchmarkManifest) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def append_result_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
    return out_path
