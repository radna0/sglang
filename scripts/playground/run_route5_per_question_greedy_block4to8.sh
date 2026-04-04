#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/route5_per_question_greedy_block4to8_20260328}"
mkdir -p "${ROOT_DIR}"
export ROOT_DIR

export PYTHONPATH="/workspace/sglang-dflash-line/python"

QIDS=(
  "86e8e5"
  "dd7f5e"
  "a295e9"
  "9c1c5f"
  "92ba6a"
)

PORT_EXPLORE_BASE="${PORT_EXPLORE_BASE:-23310}"
PORT_CONTINUE_BASE="${PORT_CONTINUE_BASE:-23360}"

i=0
for qid in "${QIDS[@]}"; do
  explore_port=$((PORT_EXPLORE_BASE + i))
  continue_port=$((PORT_CONTINUE_BASE + i))
  out_json="${ROOT_DIR}/${qid}.json"

  /venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/route_reference_dflash.py \
    --reference-csv /root/reference.csv \
    --question-ids "${qid}" \
    --out-json "${out_json}" \
    --exploration-port "${explore_port}" \
    --continuation-port "${continue_port}" \
    --final-context-length 65536 \
    --exploration-decode-len 8192 \
    --exploration-concurrency 32 \
    --exploration-num-prompts 32 \
    --buffer-tokens 512 \
    --mem-fraction-static 0.90 \
    --exploration-dflash-block-size 4 \
    --continuation-dflash-block-size 8 \
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
    --temperature 0.0 \
    --top-p 1.0 \
    --top-k 1 \
    --min-p 0.0 \
    --disable-stream \
    > "${ROOT_DIR}/${qid}.log" 2>&1

  i=$((i + 1))
done

/venv/main/bin/python - <<'PY'
import glob, json, os
root = os.environ["ROOT_DIR"] if "ROOT_DIR" in os.environ else ""
if not root:
    raise SystemExit("ROOT_DIR missing")
rows = []
for path in sorted(glob.glob(os.path.join(root, "*.json"))):
    data = json.load(open(path))
    qid = os.path.basename(path).removesuffix(".json")
    cont = data.get("continuation", {})
    maj = cont.get("majority_vote", {})
    summ = cont.get("summary", {})
    rows.append({
        "question_id": qid,
        "selected_count": data.get("selection", {}).get("selected_count"),
        "selected_by_label": data.get("selection", {}).get("selected_by_label"),
        "majority_answer": maj.get("majority_answer"),
        "majority_support": maj.get("majority_support"),
        "expected_answer": maj.get("expected_answer"),
        "is_correct_majority": maj.get("is_correct_majority"),
        "wall_tok_s": summ.get("wall_tok_s"),
        "accept_length": summ.get("accept_length"),
        "verify_ct_sum": summ.get("verify_ct_sum"),
    })
summary = {
    "rows": rows,
    "final_correct": sum(1 for r in rows if r.get("is_correct_majority")),
    "count": len(rows),
}
out = os.path.join(root, "summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY
