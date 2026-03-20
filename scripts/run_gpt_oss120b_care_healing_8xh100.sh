#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <student_model_path> <general_spec_json> <calib_spec_json> <aimo_spec_json> [extra args...]"
  exit 1
fi

STUDENT_MODEL_PATH="$1"
GENERAL_SPEC_JSON="$2"
CALIB_SPEC_JSON="$3"
AIMO_SPEC_JSON="$4"
shift 4

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python "${SCRIPT_DIR}/run_gpt_oss120b_care_healing_pipeline.py" \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --general-dataset-spec-json "${GENERAL_SPEC_JSON}" \
  --calib-dataset-spec-json "${CALIB_SPEC_JSON}" \
  --aimo-dataset-spec-json "${AIMO_SPEC_JSON}" \
  --device cuda \
  --device-map auto \
  --gradient-checkpointing \
  "$@"
