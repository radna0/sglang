#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_config_sweep_20260330}"
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
BASELINE_PORT_BASE="${BASELINE_PORT_BASE:-23680}"
TREE_PORT_BASE="${TREE_PORT_BASE:-23780}"
BLOCK_SIZES_RAW="${BLOCK_SIZES_RAW:-4,8,16}"
TREE_TOPKS_RAW="${TREE_TOPKS_RAW:-1,2,4}"
TREE_SPEC_STEPS_MODE="${TREE_SPEC_STEPS_MODE:-compact}"
TREE_VERIFY_TOKENS_RAW="${TREE_VERIFY_TOKENS_RAW:-block}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0.0}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1

read -r -a BLOCK_SIZES <<< "$(echo "${BLOCK_SIZES_RAW}" | tr ',;' '  ')"
read -r -a TREE_TOPKS <<< "$(echo "${TREE_TOPKS_RAW}" | tr ',;' '  ')"

BASELINE_DIR="${ROOT_DIR}/linear_baselines"
TREE_DIR="${ROOT_DIR}/tree_sweep"
mkdir -p "${BASELINE_DIR}" "${TREE_DIR}"

TREE_MANIFEST="${ROOT_DIR}/tree_manifest.jsonl"
BEST_JSON="${ROOT_DIR}/best_by_block_size.json"
BEST_MD="${ROOT_DIR}/best_by_block_size.md"
SUMMARY_MD="${ROOT_DIR}/summary.md"

: > "${TREE_MANIFEST}"

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

  if [[ "${speculative_algorithm}" == "DFLASH" || "${speculative_algorithm}" == "DFLASH_TREE" ]]; then
    cmd+=(--draft-attention-backend fa3)
    cmd+=(--draft-kv-cache-dtype bfloat16)
    cmd+=(--draft-page-size "${DRAFT_PAGE_SIZE}")
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

choose_spec_steps() {
  local block_size="$1"
  if [[ "${TREE_SPEC_STEPS_MODE}" == "full" ]]; then
    seq 1 $((block_size - 1))
    return
  fi
  local mid=$((block_size / 2))
  if [[ "${mid}" -lt 1 ]]; then
    mid=1
  fi
  printf "%s\n" 1 "${mid}" $((block_size - 1)) | awk '!seen[$0]++ && $1 > 0'
}

choose_verify_tokens() {
  local block_size="$1"
  if [[ "${TREE_VERIFY_TOKENS_RAW}" == "block" ]]; then
    echo "derived"
    return
  fi
  echo "${TREE_VERIFY_TOKENS_RAW}" | tr ',;' '  ' | awk '{for (i=1;i<=NF;i++) print $i}'
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

  echo "[tree-sweep] linear baseline for block_size=${block_size}"
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

  mapfile -t spec_steps_candidates < <(choose_spec_steps "${block_size}")
  mapfile -t verify_tokens_candidates < <(choose_verify_tokens "${block_size}")

  for spec_steps in "${spec_steps_candidates[@]}"; do
    [[ -z "${spec_steps}" ]] && continue
    for topk in "${TREE_TOPKS[@]}"; do
      topk="${topk// /}"
      [[ -z "${topk}" ]] && continue
      if (( topk <= 0 || topk > block_size )); then
        continue
      fi
      for verify_tokens in "${verify_tokens_candidates[@]}"; do
        verify_tokens="${verify_tokens// /}"
        [[ -z "${verify_tokens}" ]] && continue
        if [[ "${verify_tokens}" == "derived" ]]; then
          verify_tokens=$((topk + spec_steps))
          if (( verify_tokens > block_size )); then
            verify_tokens="${block_size}"
          fi
        fi
        if (( verify_tokens <= 0 )); then
          continue
        fi
        if (( verify_tokens > topk + spec_steps )); then
          continue
        fi

        cfg_dir="${TREE_DIR}/block_${block_size}"
        mkdir -p "${cfg_dir}"
        cfg_name="steps_${spec_steps}_topk_${topk}_vt_${verify_tokens}"
        cfg_report="${cfg_dir}/${cfg_name}.json"
        cfg_port="$((TREE_PORT_BASE + block_index * 100 + spec_steps * 10 + topk))"

        echo "[tree-sweep] tree cfg block=${block_size} steps=${spec_steps} topk=${topk} vt=${verify_tokens}"
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

        /venv/main/bin/python - <<'PY' "${cfg_report}" "${TREE_MANIFEST}" "${block_size}" "${spec_steps}" "${topk}" "${verify_tokens}"
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest_path = Path(sys.argv[2])
block_size = int(sys.argv[3])
spec_steps = int(sys.argv[4])
topk = int(sys.argv[5])
verify_tokens = int(sys.argv[6])
entry = {
    "block_size": block_size,
    "spec_steps": spec_steps,
    "tree_topk": topk,
    "num_verify_tokens": verify_tokens,
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
      done
    done
  done

  block_index=$((block_index + 1))
done

/venv/main/bin/python - <<'PY' "${TREE_MANIFEST}" "${BEST_JSON}" "${BEST_MD}" "${SUMMARY_MD}"
import json, sys
from pathlib import Path

tree_manifest = Path(sys.argv[1])
best_json = Path(sys.argv[2])
best_md = Path(sys.argv[3])
summary_md = Path(sys.argv[4])

records = []
if tree_manifest.exists():
    with tree_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

best_by_block = {}
for rec in records:
    if rec.get("tree_correct_boxed_rate") not in (1.0, 1):
        continue
    block = str(rec["block_size"])
    cur = best_by_block.get(block)
    if cur is None or float(rec.get("tree_wall_tok_s") or 0.0) > float(cur.get("tree_wall_tok_s") or 0.0):
        best_by_block[block] = rec

best_json.write_text(json.dumps(best_by_block, indent=2), encoding="utf-8")

best_lines = []
best_lines.append("# Best Tree Config Per Block Size")
best_lines.append("")
best_lines.append("| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Path |")
best_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for block in sorted(best_by_block.keys(), key=lambda x: int(x)):
    best = best_by_block[block]
    best_lines.append(
        f"| {block} | {best['spec_steps']} | {best['tree_topk']} | {best['num_verify_tokens']} | "
        f"{float(best.get('tree_wall_tok_s') or 0.0):.3f} | {float(best.get('linear_wall_tok_s') or 0.0):.3f} | "
        f"{float(best.get('tree_speedup_wall_tok_s') or 0.0):.3f} | "
        f"{float(best.get('tree_accept_length') or 0.0):.3f} | {best['path']} |"
    )
best_md.write_text("\n".join(best_lines) + "\n", encoding="utf-8")

lines = []
lines.append("# DFlash Tree Sweep Summary")
lines.append("")
lines.append("Single-request tree verify tuning sweep against a matching linear DFlash baseline.")
lines.append("")
lines.append("| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for block in sorted({int(r["block_size"]) for r in records}):
    candidates = [r for r in records if int(r["block_size"]) == block]
    if not candidates:
        continue
    for r in candidates:
        lines.append(
            f"| {block} | {r['spec_steps']} | {r['tree_topk']} | {r['num_verify_tokens']} | "
            f"{float(r.get('tree_wall_tok_s') or 0.0):.3f} | {float(r.get('linear_wall_tok_s') or 0.0):.3f} | "
            f"{float(r.get('tree_speedup_wall_tok_s') or 0.0):.3f} | "
            f"{float(r.get('tree_accept_length') or 0.0):.3f} | "
            f"{r.get('tree_correct_boxed_rate')} | {r['path']} |"
        )
    best = best_by_block.get(str(block))
    if best is not None:
        lines.append(
            f"\nBest block {block}: steps={best['spec_steps']} topk={best['tree_topk']} "
            f"verify_tokens={best['num_verify_tokens']} tree_tok/s={float(best.get('tree_wall_tok_s') or 0.0):.3f} "
            f"speedup={float(best.get('tree_speedup_wall_tok_s') or 0.0):.3f}"
        )
        lines.append("")

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({
    "best_by_block_size": best_by_block,
    "summary_md": str(summary_md),
    "best_json": str(best_json),
}, indent=2))
PY

echo "[tree-sweep] summary written to ${SUMMARY_MD}"
echo "[tree-sweep] best configs written to ${BEST_JSON}"
