#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/route5_explore32_route8_block4to8_tree_sampled_20260403}"
OUT_JSON="${ROOT_DIR}/result.json"
PORT_EXPLORE="${PORT_EXPLORE:-23641}"
PORT_CONTINUE="${PORT_CONTINUE:-23642}"
MODEL_PATH="${MODEL_PATH:-/workspace/30_03_DFLASH/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/30_03_DFLASH/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/workspace/30_03_DFLASH/root/reference.csv}"

mkdir -p "${ROOT_DIR}"

export PYTHONPATH="/workspace/sglang-dflash-pagesize-fix-old/python"
export SGLANG_DFLASH_USE_DIRECT_REQ_TO_TOKEN_WRITE=1

/workspace/venv-dflash/bin/python /workspace/sglang-dflash-pagesize-fix-old/scripts/playground/route_reference_dflash.py \
  --model-path "${MODEL_PATH}" \
  --draft-model-path "${DRAFT_MODEL_PATH}" \
  --reference-csv "${REFERENCE_CSV}" \
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
  --exploration-dflash-block-size 4 \
  --continuation-dflash-block-size 8 \
  --speculative-algorithm DFLASH_TREE \
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
