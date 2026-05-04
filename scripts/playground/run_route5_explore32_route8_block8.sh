#!/usr/bin/env bash
set -euo pipefail

# Prepared but not auto-started.
# First focused route-study before the full 10-problem DFlash route sweep.
#
# Purpose:
# - test whether explore32 -> route8 is actually helping on a difficulty ladder
# - keep the physical DFlash block at 8 for this pass
# - do NOT enable adaptive/failfast continuation here yet
# - do NOT assume we will find 8 good branches; measure what the router actually finds
#
# Problem set:
# - hardest:      86e8e5
# - harder:       dd7f5e
# - decently hard:a295e9
# - medium:       9c1c5f
# - easiest:      92ba6a

ROOT_DIR="/workspace/route5_explore32_route8_block8_20260328"
OUT_JSON="${ROOT_DIR}/result.json"
PORT_EXPLORE="${PORT_EXPLORE:-23251}"
PORT_CONTINUE="${PORT_CONTINUE:-23252}"

mkdir -p "${ROOT_DIR}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/route_reference_dflash.py \
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
  --disable-stream

