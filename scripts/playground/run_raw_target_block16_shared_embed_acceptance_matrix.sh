#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/raw_target_block16_shared_embed_acceptance_matrix_20260403}"
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
  --decode-len 512
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

run_case() {
  local name="$1"
  local gpu="$2"
  local port_base="$3"
  local embed_mode="$4"
  local graph_mode="$5"
  shift 5

  local out_json="${ROOT}/${name}.json"
  local log_file="${ROOT}/${name}.log"

  local -a env_prefix=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    PYTHONPATH="${PYTHONPATH_ROOT}"
    SGLANG_DFLASH_DRAFT_SHARE_POOLS=1
    SGLANG_DFLASH_TREE_DISABLE_DRAFT_FASTPATH=1
  )
  if [[ "${embed_mode}" == "explicit" ]]; then
    env_prefix+=(SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1)
  fi

  local -a mode_args=()
  case "${graph_mode}" in
    eager)
      mode_args+=(--disable-cuda-graph)
      ;;
    graph)
      mode_args+=(--disable-piecewise-cuda-graph)
      ;;
    graph_piecewise)
      ;;
    *)
      echo "unknown graph mode: ${graph_mode}" >&2
      return 1
      ;;
  esac

  "${env_prefix[@]}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    "${mode_args[@]}" \
    --baseline-port "${port_base}" \
    --dflash-port "$((port_base + 1))" \
    --out-json "${out_json}" \
    >"${log_file}" 2>&1
}

run_case explicit_eager 4 26300 explicit eager &
pid1=$!
run_case shared_eager 5 26310 shared eager &
pid2=$!
run_case shared_graph 6 26320 shared graph &
pid3=$!
run_case shared_graph_piecewise 7 26330 shared graph_piecewise &
pid4=$!

wait "$pid1" "$pid2" "$pid3" "$pid4"

echo "done: ${ROOT}"
