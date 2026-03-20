#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
RUN_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_aimo3_toolintegrated_$(date -u +%Y%m%d_%H%M%S)}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
RAW_SPEC_JSON="${REPO_ROOT}/docs/gpt_oss120b_care_aimo3_toolintegrated_phase.full.json"
PREPACK_SPEC_JSON="${PREPACK_SPEC_JSON:-}"

SEQ_LEN="${SEQ_LEN:-65536}"
COLLECT_SEQ_LEN="${COLLECT_SEQ_LEN:-${SEQ_LEN}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORLD_SIZE="${WORLD_SIZE:-8}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-1024}"
PREPACK="${PREPACK:-1}"
MATERIALIZE_DATA_CSV="${MATERIALIZE_DATA_CSV:-}"
DATAROOT="${DATAROOT:-/workspace/data/jsonl_shards_kaggle}"
DP_REPLICA_DEVICE_MAP="${DP_REPLICA_DEVICE_MAP:-single_gpu}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
LIMIT_ROWS="${LIMIT_ROWS:-}"

CONVERSION_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"

mkdir -p "${RUN_ROOT}/prepack" "${RUN_ROOT}/conversion"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${TMPDIR}"

echo "[run-root] ${RUN_ROOT}"
echo "[model-path] ${MODEL_PATH}"
echo "[seq-len] ${SEQ_LEN}"
echo "[collect-seq-len] ${COLLECT_SEQ_LEN}"
echo "[batch-size] ${BATCH_SIZE}"
echo "[dp-world-size] ${WORLD_SIZE}"
echo "[cuda-visible-devices] ${CUDA_VISIBLE_DEVICES}"
exported_count=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
if [[ "${WORLD_SIZE}" -gt "${exported_count}" ]]; then
  echo "[warning] world size ${WORLD_SIZE} exceeds visible devices ${exported_count}." >&2
  echo "[warning] check CUDA_VISIBLE_DEVICES or set WORLD_SIZE manually." >&2
  exit 1
fi

if [[ -n "${MATERIALIZE_DATA_CSV}" ]]; then
  echo "[materialize] input_csv=${MATERIALIZE_DATA_CSV}"
  /venv/main/bin/python "${REPO_ROOT}/scripts/materialize_aimo3_tool_integrated_csv.py" \
    --csv-path "${MATERIALIZE_DATA_CSV}" \
    --out-root "${DATAROOT}" \
    --dataset-slug "jeannkouagou/aimo3-tool-integrated-reasoning" \
    --rows-per-shard "${ROWS_PER_SHARD}" \
    --split-name train \
    ${LIMIT_ROWS:+--limit-rows "${LIMIT_ROWS}"}
fi

ACTIVE_SPEC_JSON="${RUN_ROOT}/prepack/prepacked_dataset_spec.json"
if [[ "${PREPACK}" == "1" && -n "${PREPACK_SPEC_JSON}" ]]; then
  echo "[prepack] using provided packed spec: ${PREPACK_SPEC_JSON}"
  if [[ ! -f "${PREPACK_SPEC_JSON}" ]]; then
    echo "[prepack] missing provided packed spec: ${PREPACK_SPEC_JSON}" >&2
    exit 1
  fi
  ACTIVE_SPEC_JSON="${PREPACK_SPEC_JSON}"
elif [[ "${PREPACK}" == "1" ]]; then
  echo "[prepack] start"
  PREPACK_LOG="${RUN_ROOT}/prepack/prepack.log"
  /venv/main/bin/python "${REPO_ROOT}/scripts/prepack_gpt_oss_corpus.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-spec-json "${RAW_SPEC_JSON}" \
    --output-dir "${RUN_ROOT}/prepack" \
    --seq-len "${SEQ_LEN}" \
    --shard-sequences "${SHARD_SEQUENCES:-2048}" \
    2>&1 | tee "${PREPACK_LOG}"

  # For safety, fail fast if prepack did not emit the expected spec.
  if [[ ! -f "${ACTIVE_SPEC_JSON}" ]]; then
    echo "[prepack] missing spec: ${ACTIVE_SPEC_JSON}" >&2
    exit 1
  fi
else
  # Fallback to raw spec if prepack disabled for smoke/debug.
  ACTIVE_SPEC_JSON="${RAW_SPEC_JSON}"
fi

echo "[active-spec-json] ${ACTIVE_SPEC_JSON}"

COVARIANCE_LOG="${RUN_ROOT}/conversion/covariance.log"
if (( WORLD_SIZE > 1 )); then
  echo "[covariance] launch torchrun"
  torchrun \
    --standalone \
    --nproc_per_node "${WORLD_SIZE}" \
    --master_port "${MASTER_PORT:-29600}" \
    "${REPO_ROOT}/scripts/collect_gpt_oss_kv_covariance.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-spec-json "${ACTIVE_SPEC_JSON}" \
    --output-dir "${RUN_ROOT}/conversion/covariance" \
    --seq-len "${COLLECT_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --replica-device-map "${DP_REPLICA_DEVICE_MAP}" \
    --dp-world-size "${WORLD_SIZE}" \
    ${ATTN_IMPLEMENTATION:+--attn-implementation "${ATTN_IMPLEMENTATION}"} \
    --mxfp4-preswizzle-dir "${CONVERSION_PRESWIZZLE_DIR}" \
    --save-per-source-covariance \
    --save-every-batches 8 \
    2>&1 | tee "${COVARIANCE_LOG}"
else
  echo "[covariance] launch single process"
  /venv/main/bin/python "${REPO_ROOT}/scripts/collect_gpt_oss_kv_covariance.py" \
    --model-path "${MODEL_PATH}" \
    --dataset-spec-json "${ACTIVE_SPEC_JSON}" \
    --output-dir "${RUN_ROOT}/conversion/covariance" \
    --seq-len "${COLLECT_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --replica-device-map "${DP_REPLICA_DEVICE_MAP}" \
    --dp-world-size "${WORLD_SIZE}" \
    ${ATTN_IMPLEMENTATION:+--attn-implementation "${ATTN_IMPLEMENTATION}"} \
    --mxfp4-preswizzle-dir "${CONVERSION_PRESWIZZLE_DIR}" \
    --save-per-source-covariance \
    --save-every-batches 8 \
    2>&1 | tee "${COVARIANCE_LOG}"
fi

echo "[done] logs at ${RUN_ROOT}/prepack/prepack.log and ${COVARIANCE_LOG}"
