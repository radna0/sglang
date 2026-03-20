#!/usr/bin/env bash
set -euo pipefail

HF_MODEL_PATH="${1:?usage: $0 HF_MODEL_PATH OUT_ROOT [SGLANG_MODEL_PATH]}"
OUT_ROOT="${2:?usage: $0 HF_MODEL_PATH OUT_ROOT [SGLANG_MODEL_PATH]}"
SGLANG_MODEL_PATH="${3:-}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"

mkdir -p "${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="${TEMP:-/workspace/tmp}"
export TMP="${TMP:-/workspace/tmp}"

CMD=(
  /venv/main/bin/python
  "${REPO_ROOT}/scripts/run_gpt_oss_care_benchmark_suite.py"
  --hf-model-path "${HF_MODEL_PATH}"
  --out-root "${OUT_ROOT}"
  --contexts "2048,4096,8192,16384"
  --ppl-task "wikitext"
  --batch-size "auto"
  --max-batch-size 4
  --dtype bfloat16
  --max-memory-per-gpu "78GiB"
  --max-cpu-memory "512GiB"
  --offload-folder "${OUT_ROOT}/offload"
  --run-passkey
  --passkey-contexts "4096,8192,16384"
  --passkey-positions "start,middle,end"
  --passkey-num-samples 16
  --run-longbench-v2
  --longbench-v2-num-examples 100
)

if [[ -n "${SGLANG_MODEL_PATH}" ]]; then
  CMD+=(--run-sglang --sglang-model-path "${SGLANG_MODEL_PATH}")
fi

"${CMD[@]}" 2>&1 | tee "${OUT_ROOT}/benchmark_suite.log"
