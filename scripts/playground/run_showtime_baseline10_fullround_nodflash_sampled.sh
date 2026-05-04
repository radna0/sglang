#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/workspace/showtime_baseline10_20260328_fullround_nodflash_sampled}"
PORT="${PORT:-23264}"

mkdir -p "$OUT_DIR"

PYTHONPATH=/workspace/sglang-dflash-line/python \
/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/sweep_showtime_harmony_reference.py \
  --out-dir "$OUT_DIR" \
  --attempts 8 \
  --early-stop 4 \
  --full-round \
  --pacore-widths '' \
  --context-length 65536 \
  --max-running-requests 8 \
  --cuda-graph-max-bs 8 \
  --timeout-s 1800 \
  --max-turn-output-tokens 96 \
  --turns 128 \
  --port "$PORT" \
  --disable-dflash \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 50 \
  --min-p 0.02
