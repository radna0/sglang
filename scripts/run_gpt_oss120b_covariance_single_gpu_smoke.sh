#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
OUT_DIR="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_covariance_single_gpu_smoke_$(date -u +%Y%m%d_%H%M%S)}"
REPO_ROOT="/root/sglang-gpt-oss-care-mla"
SPEC_JSON="${REPO_ROOT}/docs/gpt_oss120b_covariance_smoke.local.json"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"

/venv/main/bin/python "${REPO_ROOT}/scripts/collect_gpt_oss_kv_covariance.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${SPEC_JSON}" \
  --output-dir "${OUT_DIR}" \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device-map auto \
  --replica-device-map single_gpu \
  --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
  --save-every-batches 1 \
  --save-per-source-covariance \
  2>&1 | tee "${OUT_DIR}/collector.log"
