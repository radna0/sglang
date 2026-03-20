#!/usr/bin/env bash
set -euo pipefail

CONVERT_ROOT="${1:?usage: $0 CONVERT_ROOT HEAL_ROOT}"
HEAL_ROOT="${2:?usage: $0 CONVERT_ROOT HEAL_ROOT}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
MODEL_PATH="/workspace/offload_root/gpt-oss-120b"
GENERAL_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_general_phase.full.json"
CALIB_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_calib_phase.full.json"
AIMO_SPEC="${REPO_ROOT}/docs/gpt_oss120b_care_aimo_phase.full.json"

STUDENT_MODEL_PATH="${CONVERT_ROOT}/converted_checkpoint"

mkdir -p "${HEAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export GPTOSS_MXFP4_PRESWIZZLE_WAIT_S="${GPTOSS_MXFP4_PRESWIZZLE_WAIT_S:-7200}"

mkdir -p "${TMPDIR}"

/venv/main/bin/python "${REPO_ROOT}/scripts/run_gpt_oss120b_care_healing_pipeline.py" \
  --runtime fsdp \
  --fsdp-use-teacher-topk-cache \
  --teacher-topk-cache-device-map auto \
  --student-model-path "${STUDENT_MODEL_PATH}" \
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
