#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/workspace/raw_target_block16_plain_target_single_request_20260403}"
mkdir -p "$ROOT"
mkdir -p /workspace/tmp

PYTHON_BIN="${PYTHON_BIN:-/workspace/venv-dflash/bin/python}"
REPO_ROOT="${REPO_ROOT:-/workspace/sglang-dflash-tree-active}"
SCRIPT="${REPO_ROOT}/scripts/playground/bench_reference_dflash.py"
PYTHONPATH_ROOT="${REPO_ROOT}/python"

MODEL_PATH="${MODEL_PATH:-/root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/30_03_DFLASH/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/workspace/30_03_DFLASH/root/reference.csv}"
QUESTION_ID="${QUESTION_ID:-92ba6a}"
MODE="${MODE:-greedy}"

SAMPLING_TEMP="0.0"
TOP_P="1.0"
TOP_K="1"
MIN_P="0.0"
if [[ "$MODE" == "sampled" ]]; then
  SAMPLING_TEMP="1.0"
  TOP_P="1.0"
  TOP_K="50"
  MIN_P="0.02"
fi

export PYTHONPATH="$PYTHONPATH_ROOT"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1
export SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1
export SGLANG_DFLASH_DISABLE_AUX_CAPTURE=1
export SGLANG_DFLASH_PREFILL_PLAIN_TARGET=1
export SGLANG_DFLASH_TARGET_RUNTIME_PLAIN=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_CACHE_DIR=/workspace/sglang_cache
export TRITON_CACHE_DIR=/workspace/triton_cache
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp

exec "$PYTHON_BIN" "$SCRIPT" \
  --model-path "$MODEL_PATH" \
  --draft-model-path "$DRAFT_MODEL_PATH" \
  --reference-csv "$REFERENCE_CSV" \
  --question-ids "$QUESTION_ID" \
  --context-length 65536 \
  --decode-len 65536 \
  --decode-to-context-limit \
  --buffer-tokens 512 \
  --concurrency 1 \
  --num-prompts 1 \
  --page-size 256 \
  --draft-page-size 1 \
  --cuda-graph-max-bs 1 \
  --max-running-requests 1 \
  --piecewise-cuda-graph-max-tokens 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --mem-fraction-static 0.93 \
  --speculative-draft-mem-fraction-static 0.97 \
  --temperature "$SAMPLING_TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --min-p "$MIN_P" \
  --disable-overlap-schedule \
  --disable-stream \
  --speculative-algorithm DFLASH_TREE \
  --speculative-dflash-block-size 16 \
  --speculative-num-steps 8 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 9 \
  --baseline-port 28500 \
  --dflash-port 28501 \
  --out-json "$ROOT/${MODE}.json"
