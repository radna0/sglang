#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/route5_explore32_route8_block8_sampled_20260329}"
OUT_JSON="${ROOT_DIR}/result.json"
PORT_EXPLORE="${PORT_EXPLORE:-23521}"
PORT_CONTINUE="${PORT_CONTINUE:-23522}"

mkdir -p "${ROOT_DIR}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/dflash/route_reference.py \
  --reference-csv /root/reference.csv \
  --question-ids "86e8e5,dd7f5e,a295e9,9c1c5f,92ba6a" \
  --out-json "${OUT_JSON}" \
  --exploration-port "${PORT_EXPLORE}" \
  --continuation-port "${PORT_CONTINUE}" \
  --final-context-length 65536 \
  --exploration-decode-len 8192 \
  --exploration-concurrency 32 \
  --exploration-num-prompts 32 \
  --buffer-tokens 512 \
  --mem-fraction-static 0.90 \
  --dflash-block-size 8 \
  --promotion-mode strict \
  --promote-total-k 8 \
  --min-keep-per-qid 1 \
  --exploration-round-len 2048 \
  --exploration-min-rounds 2 \
  --exploration-stop-accept-le 3.25 \
  --exploration-stop-selected-mean-accept-ge 3.5 \
  --exploration-stop-selected-margin-ge 0.10 \
  --green-accept-ge 6.0 \
  --hard-accept-lt 3.0 \
  --conflict-accept-ge 3.5 \
  --conflict-accept-lt 6.0 \
  --conflict-q-entropy-le 0.7 \
  --conflict-q-max-ge 0.85 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 50 \
  --min-p 0.02 \
  --disable-stream
