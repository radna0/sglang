#!/usr/bin/env bash
set -euo pipefail

WAIT_PATTERN="${WAIT_PATTERN:-run_dflash_tree_overlap_single_request_best.sh /workspace/dflash_tree_overlap_single_request_best_20260330_block16_debug}"
ROOT_DIR="${ROOT_DIR:-/workspace/dflash_tree_overlap_config_sweep_20260330}"
LOG_DIR="${LOG_DIR:-/workspace/dflash_tree_overlap_config_sweep_20260330}"
mkdir -p "${LOG_DIR}"

echo "[tree-overlap-queue] waiting for current overlap run to finish: ${WAIT_PATTERN}" | tee -a "${LOG_DIR}/queue.log"
while pgrep -af "${WAIT_PATTERN}" >/dev/null 2>&1; do
  sleep 30
done

echo "[tree-overlap-queue] launching overlap config sweep" | tee -a "${LOG_DIR}/queue.log"
/workspace/sglang-dflash-line/scripts/playground/run_dflash_tree_overlap_config_sweep.sh "${ROOT_DIR}" \
  > "${LOG_DIR}/launch.log" 2>&1
