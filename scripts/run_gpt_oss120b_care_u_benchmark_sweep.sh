#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${1:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_u_convert_only_sweep_latest}"
OUT_ROOT="${2:-${SWEEP_ROOT}/benchmark_sweep}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
LM_EVAL_SH="${REPO_ROOT}/scripts/run_gpt_oss_hf_lm_eval_accelerate.sh"
FIXED_PACK_PY="${REPO_ROOT}/scripts/run_gpt_oss_hf_fixed_pack_ppl.py"
PASSKEY_PY="${REPO_ROOT}/scripts/run_gpt_oss_hf_passkey_eval.py"

RANKS="${RANKS:-1024,512,448,384,320,256,128}"
FIXED_PACK_MANIFEST="${FIXED_PACK_MANIFEST:-/workspace/sglang_gpt_oss_care_runs/fixed_domain_eval_packs_v1/manifest.json}"
FIXED_PACK_DIR="${FIXED_PACK_DIR:-/workspace/sglang_gpt_oss_care_runs/fixed_domain_eval_packs_v1}"
PASSKEY_LENGTHS="${PASSKEY_LENGTHS:-2048,4096}"
TASKS="${TASKS:-arc_easy,arc_challenge,hellaswag,piqa,mmlu,openbookqa,race,winogrande}"
NPROC="${NPROC:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

mkdir -p "${OUT_ROOT}"

echo "[sweep-root] ${SWEEP_ROOT}"
echo "[out-root] ${OUT_ROOT}"
echo "[ranks] ${RANKS}"
echo "[tasks] ${TASKS}"
echo "[nproc] ${NPROC}"
echo "[batch-size] ${BATCH_SIZE}"
echo "[max-batch-size] ${MAX_BATCH_SIZE}"
echo "[max-length] ${MAX_LENGTH}"

IFS=',' read -r -a rank_list <<< "${RANKS}"
for rank in "${rank_list[@]}"; do
  rank="$(echo "${rank}" | xargs)"
  ckpt="${SWEEP_ROOT}/care_u_r${rank}/conversion/converted_checkpoint"
  rank_out="${OUT_ROOT}/care_u_r${rank}"
  mkdir -p "${rank_out}/tasks" "${rank_out}/fixed_pack" "${rank_out}/passkey"

  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "[skip] missing checkpoint ${ckpt}" >&2
    continue
  fi

  echo "[bench][care-u] rank=${rank} checkpoint=${ckpt}"

  IFS=',' read -r -a task_list <<< "${TASKS}"
  for task in "${task_list[@]}"; do
    task="$(echo "${task}" | xargs)"
    NPROC="${NPROC}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    bash "${LM_EVAL_SH}" \
      "${ckpt}" \
      "${task}" \
      "${rank_out}/tasks/${task}.json" \
      "${rank_out}/tasks/${task}.log"
  done

  /venv/main/bin/python "${PASSKEY_PY}" \
    --model-path "${ckpt}" \
    --out-json "${rank_out}/passkey/passkey.json" \
    --contexts "${PASSKEY_LENGTHS}"

  /venv/main/bin/python "${FIXED_PACK_PY}" \
    --model-path "${ckpt}" \
    --pack-jsonl "${FIXED_PACK_DIR}/combined_fixed_eval_pack.jsonl" \
    --output-path "${rank_out}/fixed_pack/combined_fixed_eval_pack.json" \
    --contexts "2048,8192"

  /venv/main/bin/python "${FIXED_PACK_PY}" \
    --model-path "${ckpt}" \
    --pack-jsonl "${FIXED_PACK_DIR}/aimo3_long.jsonl" \
    --output-path "${rank_out}/fixed_pack/aimo3_long.json" \
    --contexts "2048,8192"
done
