#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/root/gpt-oss-120b}"
OUT_ROOT="${2:-/root/out/gpt-oss-120b-care-full-r512_$(date -u +%Y%m%d_%H%M%S)}"
DATASET_SPEC="${3:-/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_corpus_mix.example.json}"
ROUND_MULTIPLE="${ROUND_MULTIPLE:-1}"

mkdir -p "${OUT_ROOT}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

/venv/main/bin/python /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_pipeline.py \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${DATASET_SPEC}" \
  --out-root "${OUT_ROOT}" \
  --target-total-rows 400000 \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device-map auto \
  --target-rank 512 \
  --min-rank 128 \
  --round-multiple "${ROUND_MULTIPLE}" \
  --qk-rope-head-dim 32 \
  2>&1 | tee "${OUT_ROOT}/pipeline.log"
