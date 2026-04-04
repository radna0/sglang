#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/showtime_reference_matrix_20260404
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

root = Path("/workspace/showtime_reference_matrix_20260404")
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
  local mode="$2"
  local qid="$3"
  local port="$4"

  local name="gpu${gpu}_${mode}_${qid}"
  local work_root="$ROOT/$name"
  local log="$work_root/run.log"
  mkdir -p "$work_root"

  local pacore_widths=""
  local route_enable="0"
  if [[ "$mode" == "route" ]]; then
    route_enable="1"
  else
    pacore_widths="16,8,4"
  fi

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO_PY" \
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
    SHOWTIME_ENABLE_ROUTE_EXPLORE="$route_enable" \
    SHOWTIME_ROUTE_EXPLORATION_WIDTH=32 \
    SHOWTIME_ROUTE_PROMOTE_WIDTH=8 \
    SHOWTIME_ROUTE_EXPLORATION_BLOCK_SIZE=4 \
    SHOWTIME_ROUTE_CONTINUATION_BLOCK_SIZE=8 \
    SHOWTIME_PACORE_WIDTHS="$pacore_widths" \
    "$PYTHON" -u "$SHOWTIME_PY" >"$log" 2>&1 &

  local pid=$!
  echo "$pid" >"$work_root/pid"
  echo "$name pid=$pid log=$log"
}

launch_job 0 route 86e8e5 33000
launch_job 1 route dd7f5e 33001
launch_job 2 route a295e9 33002
launch_job 3 route 9c1c5f 33003
launch_job 4 pacore 86e8e5 33004
launch_job 5 pacore dd7f5e 33005
launch_job 6 pacore a295e9 33006
launch_job 7 pacore 9c1c5f 33007

echo "root=$ROOT"
