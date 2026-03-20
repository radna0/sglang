#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?checkpoint path required}"
OUT_DIR="${2:?output dir required}"
TASKS="${3:-arc_easy,hellaswag,mmlu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_LAUNCHER="${SCRIPT_DIR}/run_gpt_oss_hf_lm_eval_accelerate.sh"

mkdir -p "${OUT_DIR}"

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for raw_task in "${TASK_ARRAY[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [[ -n "${task}" ]] || continue
  out_json="${OUT_DIR}/${task}.json"
  log_path="${OUT_DIR}/${task}.log"
  echo "[compare] task=${task} checkpoint=${CHECKPOINT_PATH}"
  bash "${EVAL_LAUNCHER}" "${CHECKPOINT_PATH}" "${task}" "${out_json}" "${log_path}"
done

