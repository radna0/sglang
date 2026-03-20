#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?model path required}"
TASKS="${2:?task list required}"
OUT_JSON="${3:?output json path required}"
LOG_PATH="${4:?log path required}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29620}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/workspace/offload_root}"

mkdir -p "$(dirname "${OUT_JSON}")" "$(dirname "${LOG_PATH}")"
OFFLOAD_DIR="${OFFLOAD_ROOT}/$(basename "${OUT_JSON}" .json)_offload"
mkdir -p "${OFFLOAD_DIR}"

exec accelerate launch \
  --num_processes "${NPROC}" \
  --multi_gpu \
  --main_process_port "${PORT}" \
  /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_lm_eval.py \
    --model-path "${MODEL_PATH}" \
    --tasks "${TASKS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --no-parallelize \
    --offload-folder "${OFFLOAD_DIR}" \
    --output-path "${OUT_JSON}" \
    > "${LOG_PATH}" 2>&1
