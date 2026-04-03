#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/raw_target_block16_ablation_20260403}"
mkdir -p "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/venv-dflash/bin/python}"
SCRIPT="${SCRIPT:-/workspace/sglang-dflash-tree-active/scripts/playground/bench_reference_dflash.py}"
MODEL_PATH="${MODEL_PATH:-/root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/30_03_DFLASH/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/workspace/30_03_DFLASH/root/reference.csv}"

export PYTHONPATH="${PYTHONPATH:-/workspace/sglang-dflash-tree-active/python}"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --draft-model-path "${DRAFT_MODEL_PATH}"
  --reference-csv "${REFERENCE_CSV}"
  --question-ids 92ba6a
  --context-length 8192
  --decode-len 2048
  --decode-to-context-limit
  --buffer-tokens 512
  --concurrency 1
  --num-prompts 1
  --page-size 1
  --cuda-graph-max-bs 1
  --max-running-requests 1
  --piecewise-cuda-graph-max-tokens 8192
  --kv-cache-dtype fp8_e4m3
  --draft-kv-cache-dtype bfloat16
  --attention-backend fa3
  --moe-runner-backend triton_kernel
  --draft-attention-backend fa3
  --speculative-moe-runner-backend triton_kernel
  --mem-fraction-static 0.80
  --speculative-draft-mem-fraction-static 0.97
  --temperature 0.0
  --top-p 1.0
  --top-k 1
  --min-p 0.0
  --disable-overlap-schedule
  --skip-baseline
  --disable-stream
  --speculative-algorithm DFLASH_TREE
  --speculative-dflash-block-size 16
  --speculative-num-steps 8
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 9
)

launch_variant() {
  local gpu="$1"
  local name="$2"
  shift 2
  local out_json="${ROOT_DIR}/${name}.json"
  local log_path="${ROOT_DIR}/${name}.log"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "$@" \
    "${PYTHON_BIN}" "${SCRIPT}" "${COMMON_ARGS[@]}" \
    --baseline-port "$((26000 + gpu * 10))" \
    --dflash-port "$((26001 + gpu * 10))" \
    --out-json "${out_json}" >"${log_path}" 2>&1 &
  echo "${name} $! ${log_path}"
}

launch_variant 2 baseline_safe \
  SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH=1 \
  SGLANG_DFLASH_TREE_USE_SAFE_GREEDY_VERIFY=1 \
  SGLANG_DFLASH_TREE_SAFE_GREEDY_FROM_HIDDEN=1 \
  SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1

launch_variant 5 no_explicit_embeds \
  SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH=1 \
  SGLANG_DFLASH_TREE_USE_SAFE_GREEDY_VERIFY=1 \
  SGLANG_DFLASH_TREE_SAFE_GREEDY_FROM_HIDDEN=1

launch_variant 6 no_safe_greedy \
  SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH=1

launch_variant 7 fastpath_on
