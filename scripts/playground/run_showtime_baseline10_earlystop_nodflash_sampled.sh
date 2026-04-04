#!/usr/bin/env bash
set -euo pipefail

# Prepared but not auto-started.
# Baseline showtime-harmony tool-calling run without DFLASH, using sampled decode
# instead of the greedy baseline.

ROOT_DIR="/workspace/showtime_baseline10_20260328_earlystop_nodflash_sampled"
PORT="${PORT:-23243}"

mkdir -p "${ROOT_DIR}"
export PYTHONPATH="/workspace/sglang-dflash-line/python"

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/sweep_showtime_harmony_reference.py \
  --out-dir "${ROOT_DIR}" \
  --attempts 8 \
  --early-stop 4 \
  --pacore-widths '' \
  --context-length 65536 \
  --max-running-requests 8 \
  --cuda-graph-max-bs 8 \
  --timeout-s 1800 \
  --max-turn-output-tokens 96 \
  --turns 128 \
  --port "${PORT}" \
  --disable-dflash \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 50 \
  --min-p 0.02

