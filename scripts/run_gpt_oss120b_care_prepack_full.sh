#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
OUTPUT_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_prepack_$(date -u +%Y%m%d_%H%M%S)}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
RAW_SPEC_JSON="${RAW_SPEC_JSON:-${REPO_ROOT}/docs/gpt_oss120b_care_corpus_mix.full.json}"
SEQ_LEN="${SEQ_LEN:-2048}"
TARGET_TOTAL_ROWS="${TARGET_TOTAL_ROWS:-400000}"
SHARD_SEQUENCES="${SHARD_SEQUENCES:-1024}"

mkdir -p "${OUTPUT_ROOT}"

echo "[output-root] ${OUTPUT_ROOT}"
echo "[raw-spec-json] ${RAW_SPEC_JSON}"
echo "[seq-len] ${SEQ_LEN}"
echo "[target-total-rows] ${TARGET_TOTAL_ROWS}"
echo "[shard-sequences] ${SHARD_SEQUENCES}"

/venv/main/bin/python "${REPO_ROOT}/scripts/prepack_gpt_oss_corpus.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-spec-json "${RAW_SPEC_JSON}" \
  --output-dir "${OUTPUT_ROOT}" \
  --seq-len "${SEQ_LEN}" \
  --target-total-rows "${TARGET_TOTAL_ROWS}" \
  --shard-sequences "${SHARD_SEQUENCES}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepack.log"
