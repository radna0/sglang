#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/raw_target_block16_explicit_tree_speed_sweep_20260403}"
mkdir -p "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/venv-dflash/bin/python}"
REPO_ROOT="${REPO_ROOT:-/workspace/sglang-dflash-tree-active}"
SCRIPT="${REPO_ROOT}/scripts/playground/dflash/bench_reference.py"
PYTHONPATH_ROOT="${REPO_ROOT}/python"

MODEL_PATH="${MODEL_PATH:-/root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/30_03_DFLASH/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/workspace/30_03_DFLASH/root/reference.csv}"
QUESTION_ID="${QUESTION_ID:-92ba6a}"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --draft-model-path "$DRAFT_MODEL_PATH"
  --reference-csv "$REFERENCE_CSV"
  --question-ids "$QUESTION_ID"
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
)

run_case() {
  local name="$1"
  local gpu="$2"
  local port_base="$3"
  local steps="$4"
  local topk="$5"
  local vt="$6"

  local out_json="${ROOT}/${name}.json"
  local log_file="${ROOT}/${name}.log"

  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PYTHONPATH_ROOT}" \
    SGLANG_DFLASH_DRAFT_SHARE_POOLS=1 \
    SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1 \
    "${PYTHON_BIN}" "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --speculative-num-steps "${steps}" \
    --speculative-eagle-topk "${topk}" \
    --speculative-num-draft-tokens "${vt}" \
    --baseline-port "${port_base}" \
    --dflash-port "$((port_base + 1))" \
    --out-json "${out_json}" \
    >"${log_file}" 2>&1
}

run_case steps8_topk1_vt9 4 26400 8 1 9 &
pid1=$!
run_case steps7_topk2_vt9 5 26410 7 2 9 &
pid2=$!
run_case steps5_topk4_vt9 6 26420 5 4 9 &
pid3=$!
run_case steps9_topk2_vt16 7 26430 9 2 16 &
pid4=$!

wait "$pid1" "$pid2" "$pid3" "$pid4"

echo "done: ${ROOT}"
