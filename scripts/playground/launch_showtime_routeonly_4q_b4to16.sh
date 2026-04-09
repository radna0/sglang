#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/showtime_routeonly_b4to16_20260404
SHOWTIME_ROOT=/workspace/showtime_exact_local_20260404
SHOWTIME_PY="$SHOWTIME_ROOT/showtime.py"
PYTHON=/workspace/venv-dflash/bin/python
REF_SRC=/workspace/30_03_DFLASH/root/reference.csv
MODEL=/workspace/30_03_DFLASH/workspace/offload_root/gpt-oss-120b
DRAFT=/root/cuda-dflash/linux_b200_setup/runs/aimo3_offline_production_official/ckpts/epoch_213_step_77040
REPO_PY=/workspace/sglang-dflash-pagesize-fix-old/python

QIDS=(86e8e5 dd7f5e a295e9 9c1c5f)

mkdir -p "$ROOT/refs"

"$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/workspace/showtime_routeonly_b4to16_20260404")
ref_src = Path("/workspace/30_03_DFLASH/root/reference.csv")
qids = ["86e8e5", "dd7f5e", "a295e9", "9c1c5f"]
df = pd.read_csv(ref_src)
for qid in qids:
    out = root / "refs" / f"{qid}.csv"
    row = df[df["id"] == qid].copy()
    row.to_csv(out, index=False)
    print(f"wrote {out}")
PY

launch_job() {
  local gpu="$1"
  local qid="$2"
  local port="$3"

  local name="gpu${gpu}_route_${qid}"
  local work_root="$ROOT/$name"
  local log="$work_root/run.log"
  mkdir -p "$work_root"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO_PY" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    SHOWTIME_ALLOW_LOCAL_NON_ALICE=1 \
    SHOWTIME_REFERENCE_CSV="$ROOT/refs/${qid}.csv" \
    SHOWTIME_WORK_ROOT="$work_root" \
    SHOWTIME_SUBMISSION_PATH="$work_root/submission.parquet" \
    SHOWTIME_PACORE_TRACE_DIR="$work_root/pacore_traces" \
    SHOWTIME_PACORE_TRACE_JSONL="$work_root/pacore_trace.jsonl" \
    SHOWTIME_MODEL_PATH="$MODEL" \
    SHOWTIME_DFLASH_DRAFT_MODEL_PATH="$DRAFT" \
    SHOWTIME_PORT="$port" \
    SHOWTIME_ENABLE_DFLASH=1 \
    SHOWTIME_DISABLE_OVERLAP_SCHEDULE=1 \
    SHOWTIME_KV_CACHE_DTYPE=fp8_e4m3 \
    SHOWTIME_DFLASH_DRAFT_KV_CACHE_DTYPE=bfloat16 \
    SHOWTIME_TEMPERATURE=0 \
    SHOWTIME_TOP_P=1.0 \
    SHOWTIME_TOP_K=1 \
    SHOWTIME_MIN_P=0.0 \
    SHOWTIME_ENABLE_ROUTE_EXPLORE=1 \
    SHOWTIME_ROUTE_EXPLORATION_WIDTH=32 \
    SHOWTIME_ROUTE_PROMOTE_WIDTH=8 \
    SHOWTIME_ROUTE_EXPLORATION_BLOCK_SIZE=4 \
    SHOWTIME_ROUTE_CONTINUATION_BLOCK_SIZE=16 \
    "$PYTHON" -u "$SHOWTIME_PY" >"$log" 2>&1 &

  local pid=$!
  echo "$pid" >"$work_root/pid"
  echo "$name pid=$pid log=$log"
}

for p in 33100 33101 33102 33103; do
  fuser -k "${p}/tcp" 2>/dev/null || true
done

launch_job 0 86e8e5 33100
launch_job 1 dd7f5e 33101
launch_job 2 a295e9 33102
launch_job 3 9c1c5f 33103

echo "root=$ROOT"
