#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
OUT_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_$(date -u +%Y%m%d_%H%M%S)}"

DATASET_SPEC_JSON="${DATASET_SPEC_JSON:-${REPO_ROOT}/docs/gpt_oss120b_care_repro_alpaca_128x2048.json}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TARGET_RANK="${TARGET_RANK:-512}"
MIN_RANK="${MIN_RANK:-128}"
ROUND_MULTIPLE="${ROUND_MULTIPLE:-1}"
QK_ROPE_HEAD_DIM="${QK_ROPE_HEAD_DIM:-32}"
CONVERSION_DTYPE="${CONVERSION_DTYPE:-bfloat16}"
PASSKEY_CONTEXTS="${PASSKEY_CONTEXTS:-4096,8192,16384,32768}"
OFFLOAD_DIR="${OFFLOAD_DIR:-${OUT_ROOT}/offload}"
PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
RUN_EVAL="${RUN_EVAL:-1}"
COVARIANCE_WORLD_SIZE="${COVARIANCE_WORLD_SIZE:-8}"
COVARIANCE_REPLICA_DEVICE_MAP="${COVARIANCE_REPLICA_DEVICE_MAP:-single_gpu}"

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/launcher.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

NUM_LAYERS="$(${PYTHON_BIN} - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${MODEL_PATH}/config.json").read_text())
print(int(cfg["num_hidden_layers"]))
PY
)"
TARGET_TOTAL_RANK="${TARGET_TOTAL_RANK:-$(( NUM_LAYERS * TARGET_RANK ))}"

echo "[repo-root] ${REPO_ROOT}"
echo "[model-path] ${MODEL_PATH}"
echo "[dataset-spec-json] ${DATASET_SPEC_JSON}"
echo "[out-root] ${OUT_ROOT}"
echo "[seq-len] ${SEQ_LEN}"
echo "[target-rank] ${TARGET_RANK}"
echo "[target-total-rank] ${TARGET_TOTAL_RANK}"
echo "[min-rank] ${MIN_RANK}"
echo "[round-multiple] ${ROUND_MULTIPLE}"
echo "[qk-rope-head-dim] ${QK_ROPE_HEAD_DIM}"
echo "[preswizzle-dir] ${PRESWIZZLE_DIR}"
echo "[run-eval] ${RUN_EVAL}"
echo "[covariance-world-size] ${COVARIANCE_WORLD_SIZE}"
echo "[covariance-replica-device-map] ${COVARIANCE_REPLICA_DEVICE_MAP}"

${PYTHON_BIN} "${REPO_ROOT}/scripts/run_gpt_oss120b_care_pipeline.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${DATASET_SPEC_JSON}" \
  --out-root "${OUT_ROOT}/conversion" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --dtype "${CONVERSION_DTYPE}" \
  --device-map auto \
  --covariance-world-size "${COVARIANCE_WORLD_SIZE}" \
  --covariance-replica-device-map "${COVARIANCE_REPLICA_DEVICE_MAP}" \
  --mxfp4-preswizzle-dir "${PRESWIZZLE_DIR}" \
  --target-rank "${TARGET_RANK}" \
  --target-total-rank "${TARGET_TOTAL_RANK}" \
  --min-rank "${MIN_RANK}" \
  --round-multiple "${ROUND_MULTIPLE}" \
  --qk-rope-head-dim "${QK_ROPE_HEAD_DIM}"

CONVERTED_CHECKPOINT="${OUT_ROOT}/conversion/converted_checkpoint"
ZERO_SHOT_EVAL_DIR="${OUT_ROOT}/zero_shot_eval"

if [[ "${RUN_EVAL}" == "1" ]]; then
  bash "${REPO_ROOT}/scripts/run_gpt_oss120b_hf_baselines.sh" \
    "${CONVERTED_CHECKPOINT}" \
    "${ZERO_SHOT_EVAL_DIR}"

  ${PYTHON_BIN} "${REPO_ROOT}/scripts/run_gpt_oss_hf_passkey_eval.py" \
    --model-path "${CONVERTED_CHECKPOINT}" \
    --out-json "${ZERO_SHOT_EVAL_DIR}/hf_passkey.json" \
    --contexts "${PASSKEY_CONTEXTS}" \
    --positions start,middle,end \
    --num-samples 16 \
    --dtype bfloat16 \
    --device cuda \
    --max-memory-per-gpu 78GiB \
    --max-cpu-memory 512GiB \
    --offload-folder "${OFFLOAD_DIR}" \
    --max-new-tokens 16 \
    > "${ZERO_SHOT_EVAL_DIR}/hf_passkey.log" 2>&1
fi

echo "[done] ${OUT_ROOT}"
