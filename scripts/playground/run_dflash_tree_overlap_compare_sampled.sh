#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_overlap_compare_sampled_20260329}"
mkdir -p "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/epoch_65_step_23760}"
QUESTION_IDS="${QUESTION_IDS:-92ba6a}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
CONCURRENCY="${CONCURRENCY:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
PAGE_SIZE="${PAGE_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-7}"
TOPK="${TOPK:-4}"
VERIFY_TOKENS="${VERIFY_TOKENS:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1
export SGLANG_ENABLE_DFLASH_TREE_OVERLAP_EXPERIMENTAL=1

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --draft-model-path "${DRAFT_MODEL_PATH}"
  --reference-csv /root/reference.csv
  --question-ids "${QUESTION_IDS}"
  --context-length "${CONTEXT_LENGTH}"
  --decode-len "${CONTEXT_LENGTH}"
  --decode-to-context-limit
  --buffer-tokens 512
  --concurrency "${CONCURRENCY}"
  --num-prompts "${NUM_PROMPTS}"
  --page-size "${PAGE_SIZE}"
  --draft-page-size "${PAGE_SIZE}"
  --cuda-graph-max-bs "${CONCURRENCY}"
  --max-running-requests "${CONCURRENCY}"
  --piecewise-cuda-graph-max-tokens 8192
  --kv-cache-dtype fp8_e4m3
  --draft-kv-cache-dtype bfloat16
  --attention-backend fa3
  --draft-attention-backend fa3
  --moe-runner-backend triton_kernel
  --speculative-moe-runner-backend triton_kernel
  --speculative-algorithm DFLASH_TREE
  --speculative-dflash-block-size "${BLOCK_SIZE}"
  --speculative-num-steps "${NUM_STEPS}"
  --speculative-eagle-topk "${TOPK}"
  --speculative-num-draft-tokens "${VERIFY_TOKENS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --temperature 1.0
  --top-p 1.0
  --top-k 50
  --min-p 0.02
  --disable-stream
  --skip-baseline
)

NON_OVERLAP_JSON="${ROOT_DIR}/non_overlap.json"
OVERLAP_JSON="${ROOT_DIR}/overlap.json"

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py \
  "${COMMON_ARGS[@]}" \
  --dflash-port 23520 \
  --disable-overlap-schedule \
  --out-json "${NON_OVERLAP_JSON}" \
  > "${ROOT_DIR}/non_overlap.log" 2>&1

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py \
  "${COMMON_ARGS[@]}" \
  --dflash-port 23521 \
  --out-json "${OVERLAP_JSON}" \
  > "${ROOT_DIR}/overlap.log" 2>&1

/venv/main/bin/python - <<'PY' "${NON_OVERLAP_JSON}" "${OVERLAP_JSON}"
import json
import sys
from pathlib import Path

non_path = Path(sys.argv[1])
ov_path = Path(sys.argv[2])
non = json.loads(non_path.read_text(encoding="utf-8"))
ov = json.loads(ov_path.read_text(encoding="utf-8"))

def row(payload):
    d = payload["dflash"]
    return {
        "wall_tok_s": d["wall_tok_s"],
        "accept_length": d["accept_length"],
        "verify_ct_sum": d["verify_ct_sum"],
        "avg_output_tokens": d["avg_output_tokens"],
    }

summary = {
    "non_overlap": row(non),
    "overlap": row(ov),
    "speedup_wall_tok_s": (
        float(ov["dflash"]["wall_tok_s"]) / float(non["dflash"]["wall_tok_s"])
        if float(non["dflash"]["wall_tok_s"]) > 0
        else None
    ),
}
out_path = non_path.parent / "summary.json"
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
