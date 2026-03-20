#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
RUN_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_full_run_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${3:-convert_only}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
CONVERSION_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_corpus_mix.full.json"
GENERAL_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_general_phase.full.json"
CALIB_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_calib_phase.full.json"
AIMO_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_aimo_phase.full.json"

CONVERT_ROOT="${RUN_ROOT}/conversion"
HEAL_ROOT="${RUN_ROOT}/healing"

mkdir -p "${RUN_ROOT}" "${CONVERT_ROOT}" "${HEAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GPTOSS_MXFP4_PRESWIZZLE_WAIT_S="${GPTOSS_MXFP4_PRESWIZZLE_WAIT_S:-7200}"
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"
export CONVERSION_DEVICE_MAP="${CONVERSION_DEVICE_MAP:-cuda:0}"
export CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP="${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP:-single_gpu}"
export ROUND_MULTIPLE="${ROUND_MULTIPLE:-1}"
export TARGET_RANK="${TARGET_RANK:-512}"
export MIN_RANK="${MIN_RANK:-128}"

NUM_LAYERS="$(
  /venv/main/bin/python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${MODEL_PATH}/config.json").read_text())
print(int(cfg["num_hidden_layers"]))
PY
)"
export TARGET_TOTAL_RANK="${TARGET_TOTAL_RANK:-$(( NUM_LAYERS * TARGET_RANK ))}"

mkdir -p "${TMPDIR}"

echo "[run-root] ${RUN_ROOT}"
echo "[mode] ${MODE}"
echo "[conversion-device-map] ${CONVERSION_DEVICE_MAP}"
echo "[conversion-covariance-replica-device-map] ${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP}"
echo "[target-rank] ${TARGET_RANK}"
echo "[target-total-rank] ${TARGET_TOTAL_RANK}"
echo "[min-rank] ${MIN_RANK}"
echo "[round-multiple] ${ROUND_MULTIPLE}"

/venv/main/bin/python "${REPO_ROOT}/scripts/run_gpt_oss120b_care_pipeline.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${CONVERSION_SPEC}" \
  --out-root "${CONVERT_ROOT}" \
  --target-total-rows 400000 \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device-map "${CONVERSION_DEVICE_MAP}" \
  --covariance-world-size 8 \
  --covariance-replica-device-map "${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP}" \
  --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
  --target-rank "${TARGET_RANK}" \
  --target-total-rank "${TARGET_TOTAL_RANK}" \
  --min-rank "${MIN_RANK}" \
  --round-multiple "${ROUND_MULTIPLE}" \
  --qk-rope-head-dim 32 \
  --rank-source-fusion \
  2>&1 | tee "${CONVERT_ROOT}/pipeline.log"

if [[ "${MODE}" == "convert_only" ]]; then
  exit 0
fi

if [[ "${MODE}" != "full" ]]; then
  echo "unsupported mode: ${MODE}" >&2
  exit 1
fi

/venv/main/bin/python "${REPO_ROOT}/scripts/run_gpt_oss120b_care_healing_pipeline.py" \
  --runtime fsdp \
  --fsdp-use-teacher-topk-cache \
  --teacher-topk-cache-device-map auto \
  --student-model-path "${CONVERT_ROOT}/converted_checkpoint" \
  --teacher-model-path "${MODEL_PATH}" \
  --general-dataset-spec-json "${GENERAL_SPEC}" \
  --calib-dataset-spec-json "${CALIB_SPEC}" \
  --aimo-dataset-spec-json "${AIMO_SPEC}" \
  --output-root "${HEAL_ROOT}" \
  --torchrun-nproc-per-node 8 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --gradient-checkpointing \
  --seq-len 512 \
  --general-seq-len 512 \
  --calib-seq-len 2048 \
  --aimo-seq-len 32768 \
  --batch-size 1 \
  --learning-rate 1e-6 \
  --warmup-steps 100 \
  --kl-weight 0.1 \
  --distill-topk 64 \
  --general-steps 2000 \
  --calib-steps 1000 \
  --aimo-steps 500 \
  --general-subset all_mla \
  --calib-subset rope_only \
  --aimo-subset all_mla_plus_o \
  --overwrite \
  2>&1 | tee "${HEAL_ROOT}/pipeline.log"
