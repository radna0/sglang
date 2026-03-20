#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_clean}"
OUT_DIR="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/sglang_smoke}"
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
TP_SIZE="${TP_SIZE:-8}"
PAGE_SIZE="${PAGE_SIZE:-64}"
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
PORT="${PORT:-30020}"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="/root/sglang-gpt-oss-care-mla/python:${PYTHONPATH:-}"
exec /usr/bin/python3 /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_care_sglang_smoke.py \
  --model-path "${MODEL_PATH}" \
  --tp-size "${TP_SIZE}" \
  --page-size "${PAGE_SIZE}" \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --attention-backend "${ATTN_BACKEND}" \
  --port "${PORT}" \
  --server-log "${OUT_DIR}/server_${ATTN_BACKEND}.log" \
  --out-json "${OUT_DIR}/smoke_${ATTN_BACKEND}.json"
