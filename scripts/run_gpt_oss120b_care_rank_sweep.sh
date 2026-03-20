#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
OUT_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_rank_sweep_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${3:-both}"

RANKS_CARE_U="${RANKS_CARE_U:-1024,512,448,384,320,256,128}"
RANKS_CARE_E="${RANKS_CARE_E:-1024,512,448,384,320,256,128}"
FIXED_LAUNCHER="${REPO_ROOT}/scripts/run_gpt_oss120b_care_fixedr512_big_8xh100.sh"
CAREE_LAUNCHER="${REPO_ROOT}/scripts/run_gpt_oss120b_care_big_8xh100.sh"

mkdir -p "${OUT_ROOT}"

echo "[model-path] ${MODEL_PATH}"
echo "[out-root] ${OUT_ROOT}"
echo "[mode] ${MODE}"
echo "[care-u-ranks] ${RANKS_CARE_U}"
echo "[care-e-ranks] ${RANKS_CARE_E}"

run_care_u() {
  local rank="$1"
  local run_root="${OUT_ROOT}/care_u_r${rank}"
  echo "[launch][care-u] rank=${rank} run_root=${run_root}"
  TARGET_RANK="${rank}" \
  bash "${FIXED_LAUNCHER}" "${MODEL_PATH}" "${run_root}"
}

run_care_e() {
  local rank="$1"
  local run_root="${OUT_ROOT}/care_e_r${rank}"
  echo "[launch][care-e] rank=${rank} run_root=${run_root}"
  TARGET_RANK="${rank}" \
  bash "${CAREE_LAUNCHER}" "${MODEL_PATH}" "${run_root}" convert_only
}

case "${MODE}" in
  care_u)
    IFS=',' read -r -a ranks <<< "${RANKS_CARE_U}"
    for rank in "${ranks[@]}"; do
      run_care_u "$(echo "${rank}" | xargs)"
    done
    ;;
  care_e)
    IFS=',' read -r -a ranks <<< "${RANKS_CARE_E}"
    for rank in "${ranks[@]}"; do
      run_care_e "$(echo "${rank}" | xargs)"
    done
    ;;
  both)
    IFS=',' read -r -a ranks_u <<< "${RANKS_CARE_U}"
    for rank in "${ranks_u[@]}"; do
      run_care_u "$(echo "${rank}" | xargs)"
    done
    IFS=',' read -r -a ranks_e <<< "${RANKS_CARE_E}"
    for rank in "${ranks_e[@]}"; do
      run_care_e "$(echo "${rank}" | xargs)"
    done
    ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac
