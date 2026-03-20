#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 STUDENT_MODEL_PATH GENERAL_SPEC_JSON CALIB_SPEC_JSON AIMO_SPEC_JSON [extra args...]" >&2
  exit 1
fi

STUDENT_MODEL_PATH="$1"
GENERAL_SPEC_JSON="$2"
CALIB_SPEC_JSON="$3"
AIMO_SPEC_JSON="$4"
shift 4

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"
export GPTOSS_TEACHER_TOPK_CACHE_BATCH_SIZE="${GPTOSS_TEACHER_TOPK_CACHE_BATCH_SIZE:-4}"

python "$(dirname "$0")/run_gpt_oss120b_care_healing_pipeline.py" \
  --runtime fsdp \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --general-dataset-spec-json "${GENERAL_SPEC_JSON}" \
  --calib-dataset-spec-json "${CALIB_SPEC_JSON}" \
  --aimo-dataset-spec-json "${AIMO_SPEC_JSON}" \
  --device cuda \
  --torchrun-nproc-per-node 8 \
  --dtype bfloat16 \
  --teacher-topk-cache-batch-size "${GPTOSS_TEACHER_TOPK_CACHE_BATCH_SIZE}" \
  --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
  --quantized-expert-layout replicated \
  --gradient-checkpointing \
  "$@"
