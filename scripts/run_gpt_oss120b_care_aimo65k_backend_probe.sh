#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
RUN_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_aimo65k_backend_probe_$(date -u +%Y%m%d_%H%M%S)}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
DATASET_SPEC_JSON="${DATASET_SPEC_JSON:-${REPO_ROOT}/docs/gpt_oss120b_care_aimo65k_backend_probe.small.json}"
SEQ_LEN="${SEQ_LEN:-65536}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

mkdir -p "${RUN_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"

mkdir -p "${TMPDIR}"

echo "[run-root] ${RUN_ROOT}"
echo "[dataset-spec-json] ${DATASET_SPEC_JSON}"
echo "[seq-len] ${SEQ_LEN}"
echo "[batch-size] ${BATCH_SIZE}"
echo "[attn-implementation] ${ATTN_IMPLEMENTATION}"

/venv/main/bin/python "${REPO_ROOT}/scripts/collect_gpt_oss_kv_covariance.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${DATASET_SPEC_JSON}" \
  --output-dir "${RUN_ROOT}/covariance" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --device-map cuda:0 \
  --replica-device-map single_gpu \
  --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  2>&1 | tee "${RUN_ROOT}/probe.log"
