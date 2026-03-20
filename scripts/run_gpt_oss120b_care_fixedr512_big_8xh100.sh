#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
RUN_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_$(date -u +%Y%m%d_%H%M%S)}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
DATASET_SPEC_JSON="${DATASET_SPEC_JSON:-${REPO_ROOT}/docs/gpt_oss120b_care_corpus_mix.full.json}"
PREPACKED_SPEC_JSON="${PREPACKED_SPEC_JSON:-}"
TARGET_TOTAL_ROWS="${TARGET_TOTAL_ROWS:-400000}"
SEQ_LEN="${SEQ_LEN:-2048}"
TARGET_RANK="${TARGET_RANK:-512}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-2}"

OUT_ROOT="${RUN_ROOT}/conversion"
mkdir -p "${RUN_ROOT}" "${OUT_ROOT}"
PIPELINE_LOG="${OUT_ROOT}/pipeline.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CONVERSION_DEVICE_MAP="${CONVERSION_DEVICE_MAP:-cuda:0}"
export CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP="${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP:-single_gpu}"
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"

mkdir -p "${TMPDIR}"

echo "[run-root] ${RUN_ROOT}"
if [[ -n "${PREPACKED_SPEC_JSON}" ]]; then
  ACTIVE_SPEC_JSON="${PREPACKED_SPEC_JSON}"
else
  ACTIVE_SPEC_JSON="${DATASET_SPEC_JSON}"
fi
echo "[dataset-spec-json] ${ACTIVE_SPEC_JSON}"
echo "[target-total-rows] ${TARGET_TOTAL_ROWS}"
echo "[seq-len] ${SEQ_LEN}"
echo "[target-rank] ${TARGET_RANK}"
echo "[batch-size] ${BATCH_SIZE}"
echo "[conversion-device-map] ${CONVERSION_DEVICE_MAP}"
echo "[conversion-covariance-replica-device-map] ${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP}"

if [[ -f "${PIPELINE_LOG}" && -s "${PIPELINE_LOG}" ]]; then
  ROTATED_LOG="${OUT_ROOT}/pipeline.restart_$(date -u +%Y%m%d_%H%M%S).log"
  mv "${PIPELINE_LOG}" "${ROTATED_LOG}"
  echo "[rotated-pipeline-log] ${ROTATED_LOG}"
fi

PASS_TARGET_TOTAL_ROWS=1
if [[ -n "${PREPACKED_SPEC_JSON}" ]]; then
  if /venv/main/bin/python - "${ACTIVE_SPEC_JSON}" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1], "r", encoding="utf-8"))
sources = spec.get("sources", [])
all_packed = bool(sources) and all(src.get("kind") == "packed_torch_filelist" for src in sources)
sys.exit(0 if all_packed else 1)
PY
  then
    PASS_TARGET_TOTAL_ROWS=0
  fi
fi

CMD=(
  /venv/main/bin/python "${REPO_ROOT}/scripts/run_gpt_oss120b_care_pipeline.py"
  --model-path "${MODEL_PATH}"
  --dataset-spec-json "${ACTIVE_SPEC_JSON}"
  --out-root "${OUT_ROOT}"
  --seq-len "${SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device-map "${CONVERSION_DEVICE_MAP}"
  --covariance-world-size 8
  --covariance-replica-device-map "${CONVERSION_COVARIANCE_REPLICA_DEVICE_MAP}"
  --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}"
  --target-rank "${TARGET_RANK}"
  --uniform-rank
  --qk-rope-head-dim 32
)

if [[ "${PASS_TARGET_TOTAL_ROWS}" == "1" ]]; then
  CMD+=(--target-total-rows "${TARGET_TOTAL_ROWS}")
fi

"${CMD[@]}" 2>&1 | tee "${PIPELINE_LOG}"
