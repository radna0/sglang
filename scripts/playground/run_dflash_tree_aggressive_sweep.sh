#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_aggressive_sweep_20260330}"
mkdir -p "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/root/reference.csv}"
QUESTION_IDS="${QUESTION_IDS:-92ba6a}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
DECODE_LEN="${DECODE_LEN:-2048}"
CONCURRENCY="${CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
PAGE_SIZE="${PAGE_SIZE:-1}"
DRAFT_PAGE_SIZE="${DRAFT_PAGE_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
BASELINE_PORT_BASE="${BASELINE_PORT_BASE:-24380}"
TREE_PORT_BASE="${TREE_PORT_BASE:-24480}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0.0}"
BLOCK_SIZES_RAW="${BLOCK_SIZES_RAW:-8,16}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1

read -r -a BLOCK_SIZES <<< "$(echo "${BLOCK_SIZES_RAW}" | tr ',;' '  ')"

BASELINE_DIR="${ROOT_DIR}/linear_baselines"
TREE_DIR="${ROOT_DIR}/tree_sweep"
mkdir -p "${BASELINE_DIR}" "${TREE_DIR}"

MANIFEST="${ROOT_DIR}/tree_manifest.jsonl"
BEST_JSON="${ROOT_DIR}/best_by_block_size.json"
BEST_MD="${ROOT_DIR}/best_by_block_size.md"
SUMMARY_MD="${ROOT_DIR}/summary.md"

: > "${MANIFEST}"

run_bench() {
  local out_json="$1"
  local port="$2"
  local speculative_algorithm="$3"
  local block_size="$4"
  local speculative_num_steps="$5"
  local speculative_eagle_topk="$6"
  local speculative_num_draft_tokens="$7"
  local baseline_json_in="${8:-}"
  local skip_baseline="${9:-0}"

  local cmd=(
    /venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py
    --model-path "${MODEL_PATH}"
    --draft-model-path "${DRAFT_MODEL_PATH}"
    --reference-csv "${REFERENCE_CSV}"
    --question-ids "${QUESTION_IDS}"
    --context-length "${CONTEXT_LENGTH}"
    --decode-len "${DECODE_LEN}"
    --decode-to-context-limit
    --buffer-tokens 512
    --concurrency "${CONCURRENCY}"
    --num-prompts "${NUM_PROMPTS}"
    --page-size "${PAGE_SIZE}"
    --draft-page-size "${DRAFT_PAGE_SIZE}"
    --cuda-graph-max-bs "${CONCURRENCY}"
    --max-running-requests "${CONCURRENCY}"
    --piecewise-cuda-graph-max-tokens 8192
    --kv-cache-dtype fp8_e4m3
    --draft-kv-cache-dtype bfloat16
    --attention-backend fa3
    --moe-runner-backend triton_kernel
    --draft-attention-backend fa3
    --speculative-moe-runner-backend triton_kernel
    --speculative-algorithm "${speculative_algorithm}"
    --speculative-dflash-block-size "${block_size}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --min-p "${MIN_P}"
    --disable-stream
    --out-json "${out_json}"
    --baseline-port "${port}"
    --dflash-port "$((port + 1))"
  )

  if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    cmd+=(--disable-cuda-graph)
  fi

  if [[ "${DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
    cmd+=(--disable-overlap-schedule)
  fi

  if [[ -n "${baseline_json_in}" ]]; then
    cmd+=(--baseline-json-in "${baseline_json_in}")
  fi

  if [[ "${skip_baseline}" == "1" ]]; then
    cmd+=(--skip-baseline)
  fi

  if [[ "${speculative_algorithm}" == "DFLASH_TREE" ]]; then
    cmd+=(--speculative-num-steps "${speculative_num_steps}")
    cmd+=(--speculative-eagle-topk "${speculative_eagle_topk}")
    cmd+=(--speculative-num-draft-tokens "${speculative_num_draft_tokens}")
  fi

  "${cmd[@]}"
}

linear_to_baseline_json() {
  local linear_report="$1"
  local baseline_json="$2"
  /venv/main/bin/python - <<'PY' "${linear_report}" "${baseline_json}"
import json, sys
from pathlib import Path

linear_report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baseline = {
    "baseline": linear_report["dflash"],
    "baseline_request_metrics": linear_report.get("dflash_request_metrics") or [],
    "baseline_request_metric_aggregate": linear_report.get("dflash_request_metric_aggregate") or {},
}
Path(sys.argv[2]).write_text(json.dumps(baseline, indent=2), encoding="utf-8")
PY
}

emit_presets_for_block() {
  local block_size="$1"
  case "${block_size}" in
    4)
      cat <<'EOF'
3 4 4
3 4 8
EOF
      ;;
    8)
      cat <<'EOF'
4 2 6
5 4 8
5 4 12
6 4 16
7 2 16
EOF
      ;;
    16)
      cat <<'EOF'
8 1 9
5 4 16
5 4 24
7 4 32
9 4 48
9 8 48
9 8 64
11 4 64
13 2 64
EOF
      ;;
    *)
      # fallback: strong conservative anchor + a wider point
      local step_a=$((block_size / 2))
      if (( step_a < 1 )); then
        step_a=1
      fi
      local step_b=$((block_size - 1))
      printf "%s %s %s\n" "${step_a}" 4 "${block_size}"
      printf "%s %s %s\n" "${step_b}" 2 "$((block_size * 2))"
      ;;
  esac
}

max_verify_tokens_for_cfg() {
  local spec_steps="$1"
  local topk="$2"
  echo $((1 + topk + (spec_steps - 1) * topk * topk))
}

block_index=0
for block_size in "${BLOCK_SIZES[@]}"; do
  block_size="${block_size// /}"
  [[ -z "${block_size}" ]] && continue

  linear_dir="${BASELINE_DIR}/block_${block_size}"
  mkdir -p "${linear_dir}"
  linear_report="${linear_dir}/linear_report.json"
  linear_baseline="${linear_dir}/linear_baseline.json"
  linear_port="$((BASELINE_PORT_BASE + block_index * 10))"

  echo "[tree-aggressive] linear baseline for block_size=${block_size}"
  run_bench \
    "${linear_report}" \
    "${linear_port}" \
    "DFLASH" \
    "${block_size}" \
    "0" \
    "0" \
    "0" \
    "" \
    "1"

  linear_to_baseline_json "${linear_report}" "${linear_baseline}"

  while read -r spec_steps topk verify_tokens; do
    [[ -z "${spec_steps:-}" ]] && continue
    if (( spec_steps <= 0 || spec_steps >= block_size )); then
      continue
    fi
    if (( topk <= 0 )); then
      continue
    fi
    max_verify_tokens="$(max_verify_tokens_for_cfg "${spec_steps}" "${topk}")"
    if (( verify_tokens > max_verify_tokens )); then
      verify_tokens="${max_verify_tokens}"
    fi
    if (( verify_tokens <= 0 )); then
      continue
    fi

    cfg_dir="${TREE_DIR}/block_${block_size}"
    mkdir -p "${cfg_dir}"
    cfg_name="steps_${spec_steps}_topk_${topk}_vt_${verify_tokens}"
    cfg_report="${cfg_dir}/${cfg_name}.json"
    cfg_port="$((TREE_PORT_BASE + block_index * 100 + spec_steps * 10 + topk))"

    echo "[tree-aggressive] tree cfg block=${block_size} steps=${spec_steps} topk=${topk} vt=${verify_tokens}"
    run_bench \
      "${cfg_report}" \
      "${cfg_port}" \
      "DFLASH_TREE" \
      "${block_size}" \
      "${spec_steps}" \
      "${topk}" \
      "${verify_tokens}" \
      "${linear_baseline}" \
      "0"

    /venv/main/bin/python - <<'PY' "${cfg_report}" "${MANIFEST}" "${block_size}" "${spec_steps}" "${topk}" "${verify_tokens}"
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest_path = Path(sys.argv[2])
entry = {
    "block_size": int(sys.argv[3]),
    "spec_steps": int(sys.argv[4]),
    "tree_topk": int(sys.argv[5]),
    "num_verify_tokens": int(sys.argv[6]),
    "path": sys.argv[1],
    "linear_wall_tok_s": (report.get("baseline") or {}).get("wall_tok_s"),
    "tree_wall_tok_s": (report.get("dflash") or {}).get("wall_tok_s"),
    "tree_accept_length": (report.get("dflash") or {}).get("accept_length"),
    "tree_correct_boxed_rate": (report.get("dflash_request_metric_aggregate") or {}).get("correct_boxed_rate"),
    "tree_speedup_wall_tok_s": report.get("speedup_wall_tok_s"),
    "tree_speedup_req_s": report.get("speedup_req_s"),
}
with manifest_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
PY
  done < <(emit_presets_for_block "${block_size}")

  block_index=$((block_index + 1))
done

/venv/main/bin/python - <<'PY' "${MANIFEST}" "${BEST_JSON}" "${BEST_MD}" "${SUMMARY_MD}"
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
best_json_path = Path(sys.argv[2])
best_md_path = Path(sys.argv[3])
summary_md_path = Path(sys.argv[4])
entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not entries:
    raise SystemExit("no tree aggressive entries collected")

best = {}
for entry in entries:
    if (entry.get("tree_correct_boxed_rate") or 0.0) < 1.0:
        continue
    block = str(entry["block_size"])
    cur = best.get(block)
    if cur is None or (entry.get("tree_wall_tok_s") or 0.0) > (cur.get("tree_wall_tok_s") or 0.0):
        best[block] = entry

best_json_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

best_lines = [
    "# DFlash Tree Aggressive Sweep Best By Block Size",
    "",
    "| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Path |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for block in sorted(best, key=lambda x: int(x)):
    entry = best[block]
    best_lines.append(
        f"| {block} | {entry['spec_steps']} | {entry['tree_topk']} | {entry['num_verify_tokens']} | "
        f"{entry['tree_wall_tok_s']:.3f} | {entry['linear_wall_tok_s']:.3f} | {entry['tree_speedup_wall_tok_s']:.3f} | "
        f"{entry['tree_accept_length']:.3f} | {entry['path']} |"
    )
best_md_path.write_text("\n".join(best_lines) + "\n", encoding="utf-8")

summary_lines = [
    "# DFlash Tree Aggressive Sweep Summary",
    "",
    "| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for entry in entries:
    summary_lines.append(
        f"| {entry['block_size']} | {entry['spec_steps']} | {entry['tree_topk']} | {entry['num_verify_tokens']} | "
        f"{entry['tree_wall_tok_s']:.3f} | {entry['linear_wall_tok_s']:.3f} | {entry['tree_speedup_wall_tok_s']:.3f} | "
        f"{entry['tree_accept_length']:.3f} | {entry['tree_correct_boxed_rate']:.3f} | {entry['path']} |"
    )
summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n".join(best_lines))
print()
print("\n".join(summary_lines))
PY
