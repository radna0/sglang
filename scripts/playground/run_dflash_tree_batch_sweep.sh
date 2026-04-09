#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_batch_sweep_20260330}"
BEST_JSON="${BEST_JSON:-/workspace/dflash_tree_config_sweep_20260330/best_by_block_size.json}"
mkdir -p "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/root/reference.csv}"
QUESTION_IDS="${QUESTION_IDS:-92ba6a}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
DECODE_LEN="${DECODE_LEN:-2048}"
BLOCK_SIZES_RAW="${BLOCK_SIZES_RAW:-4,8,16}"
CONCURRENCY_RAW="${CONCURRENCY_RAW:-1,4,8,16}"
PROMPTS_MULTIPLIER="${PROMPTS_MULTIPLIER:-4}"
PAGE_SIZE="${PAGE_SIZE:-1}"
DRAFT_PAGE_SIZE="${DRAFT_PAGE_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
BASELINE_PORT_BASE="${BASELINE_PORT_BASE:-24180}"
TREE_PORT_BASE="${TREE_PORT_BASE:-24280}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0.0}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1

read -r -a BLOCK_SIZES <<< "$(echo "${BLOCK_SIZES_RAW}" | tr ',;' '  ')"
read -r -a CONCURRENCIES <<< "$(echo "${CONCURRENCY_RAW}" | tr ',;' '  ')"

LINEAR_DIR="${ROOT_DIR}/linear"
TREE_DIR="${ROOT_DIR}/tree"
mkdir -p "${LINEAR_DIR}" "${TREE_DIR}"

MANIFEST="${ROOT_DIR}/manifest.jsonl"
BEST_MD="${ROOT_DIR}/summary.md"
: > "${MANIFEST}"

run_bench() {
  local out_json="$1"
  local port="$2"
  local speculative_algorithm="$3"
  local block_size="$4"
  local speculative_num_steps="$5"
  local speculative_eagle_topk="$6"
  local speculative_num_draft_tokens="$7"
  local concurrency="$8"
  local num_prompts="$9"
  local baseline_json_in="${10:-}"
  local skip_baseline="${11:-0}"

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
    --concurrency "${concurrency}"
    --num-prompts "${num_prompts}"
    --page-size "${PAGE_SIZE}"
    --draft-page-size "${DRAFT_PAGE_SIZE}"
    --cuda-graph-max-bs "${concurrency}"
    --max-running-requests "${concurrency}"
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

block_index=0
for block_size in "${BLOCK_SIZES[@]}"; do
  block_size="${block_size// /}"
  [[ -z "${block_size}" ]] && continue

  cfg_json="$(jq -c --arg block "${block_size}" '.[$block]' "${BEST_JSON}")"
  if [[ "${cfg_json}" == "null" || -z "${cfg_json}" ]]; then
    echo "[tree-batch-sweep] missing best config for block=${block_size}" >&2
    exit 1
  fi

  spec_steps="$(jq -r '.spec_steps' <<< "${cfg_json}")"
  topk="$(jq -r '.tree_topk' <<< "${cfg_json}")"
  verify_tokens="$(jq -r '.num_verify_tokens' <<< "${cfg_json}")"

  for concurrency in "${CONCURRENCIES[@]}"; do
    concurrency="${concurrency// /}"
    [[ -z "${concurrency}" ]] && continue
    num_prompts=$((concurrency * PROMPTS_MULTIPLIER))

    linear_dir="${LINEAR_DIR}/block_${block_size}/c_${concurrency}"
    tree_dir="${TREE_DIR}/block_${block_size}/c_${concurrency}"
    mkdir -p "${linear_dir}" "${tree_dir}"

    linear_report="${linear_dir}/linear_report.json"
    linear_baseline="${linear_dir}/linear_baseline.json"
    tree_report="${tree_dir}/tree_report.json"

    linear_port="$((BASELINE_PORT_BASE + block_index * 100 + concurrency))"
    tree_port="$((TREE_PORT_BASE + block_index * 100 + concurrency))"

    echo "[tree-batch-sweep] linear block=${block_size} concurrency=${concurrency}"
    run_bench \
      "${linear_report}" \
      "${linear_port}" \
      "DFLASH" \
      "${block_size}" \
      "0" \
      "0" \
      "0" \
      "${concurrency}" \
      "${num_prompts}" \
      "" \
      "1"

    linear_to_baseline_json "${linear_report}" "${linear_baseline}"

    echo "[tree-batch-sweep] tree block=${block_size} concurrency=${concurrency} steps=${spec_steps} topk=${topk} vt=${verify_tokens}"
    run_bench \
      "${tree_report}" \
      "${tree_port}" \
      "DFLASH_TREE" \
      "${block_size}" \
      "${spec_steps}" \
      "${topk}" \
      "${verify_tokens}" \
      "${concurrency}" \
      "${num_prompts}" \
      "${linear_baseline}" \
      "0"

    /venv/main/bin/python - <<'PY' "${tree_report}" "${MANIFEST}" "${block_size}" "${concurrency}" "${spec_steps}" "${topk}" "${verify_tokens}"
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest_path = Path(sys.argv[2])
entry = {
    "block_size": int(sys.argv[3]),
    "concurrency": int(sys.argv[4]),
    "spec_steps": int(sys.argv[5]),
    "tree_topk": int(sys.argv[6]),
    "num_verify_tokens": int(sys.argv[7]),
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

  block_index=$((block_index + 1))
done

/venv/main/bin/python - <<'PY' "${MANIFEST}" "${BEST_MD}"
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_md_path = Path(sys.argv[2])
entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not entries:
    raise SystemExit("no tree batch entries collected")

lines = [
    "# DFlash Tree Batch Sweep Summary",
    "",
    "| Block | Concurrency | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for entry in entries:
    lines.append(
        f"| {entry['block_size']} | {entry['concurrency']} | {entry['spec_steps']} | {entry['tree_topk']} | {entry['num_verify_tokens']} | "
        f"{entry['tree_wall_tok_s']:.3f} | {entry['linear_wall_tok_s']:.3f} | {entry['tree_speedup_wall_tok_s']:.3f} | "
        f"{entry['tree_accept_length']:.3f} | {entry['tree_correct_boxed_rate']:.3f} | {entry['path']} |"
    )
summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
