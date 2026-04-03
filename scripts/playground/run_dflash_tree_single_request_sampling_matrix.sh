#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_single_request_sampling_matrix_20260330}"
BEST_JSON="${BEST_JSON:-/workspace/dflash_tree_config_sweep_20260330/best_by_block_size.json}"
mkdir -p "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/root/reference.csv}"
QUESTION_IDS="${QUESTION_IDS:-92ba6a}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
DECODE_LEN="${DECODE_LEN:-2048}"
BLOCK_SIZES_RAW="${BLOCK_SIZES_RAW:-4,8,16}"
PAGE_SIZE="${PAGE_SIZE:-1}"
DRAFT_PAGE_SIZE="${DRAFT_PAGE_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
BASELINE_PORT_BASE="${BASELINE_PORT_BASE:-24680}"
TREE_PORT_BASE="${TREE_PORT_BASE:-24780}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1

read -r -a BLOCK_SIZES <<< "$(echo "${BLOCK_SIZES_RAW}" | tr ',;' '  ')"

LINEAR_DIR="${ROOT_DIR}/linear"
TREE_DIR="${ROOT_DIR}/tree"
mkdir -p "${LINEAR_DIR}" "${TREE_DIR}"

MANIFEST="${ROOT_DIR}/manifest.jsonl"
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
  local temperature="$8"
  local top_p="$9"
  local top_k="${10}"
  local min_p="${11}"
  local baseline_json_in="${12:-}"
  local skip_baseline="${13:-0}"
  local draft_branch_mode="${14:-topk}"

  local cmd=(
    /venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/dflash/bench_reference.py
    --model-path "${MODEL_PATH}"
    --draft-model-path "${DRAFT_MODEL_PATH}"
    --reference-csv "${REFERENCE_CSV}"
    --question-ids "${QUESTION_IDS}"
    --context-length "${CONTEXT_LENGTH}"
    --decode-len "${DECODE_LEN}"
    --decode-to-context-limit
    --buffer-tokens 512
    --concurrency 1
    --num-prompts 1
    --page-size "${PAGE_SIZE}"
    --draft-page-size "${DRAFT_PAGE_SIZE}"
    --cuda-graph-max-bs 1
    --max-running-requests 1
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
    --temperature "${temperature}"
    --top-p "${top_p}"
    --top-k "${top_k}"
    --min-p "${min_p}"
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

  if [[ "${speculative_algorithm}" == "DFLASH_TREE" ]]; then
    SGLANG_DFLASH_TREE_DRAFT_BRANCH_MODE="${draft_branch_mode}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
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

sample_branch_cfg_for_block() {
  local block_size="$1"
  case "${block_size}" in
    4)  echo "3 4 4" ;;
    8)  echo "4 2 6" ;;
    16) echo "5 4 16" ;;
    *)  echo "$((block_size-1)) 2 $((block_size*2))" ;;
  esac
}

append_manifest() {
  local report="$1"
  local block_size="$2"
  local target_mode="$3"
  local draft_branch_mode="$4"
  local spec_steps="$5"
  local topk="$6"
  local verify_tokens="$7"
  /venv/main/bin/python - <<'PY' "${report}" "${MANIFEST}" "${block_size}" "${target_mode}" "${draft_branch_mode}" "${spec_steps}" "${topk}" "${verify_tokens}"
import json, sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = {
    "block_size": int(sys.argv[3]),
    "target_mode": sys.argv[4],
    "draft_branch_mode": sys.argv[5],
    "spec_steps": int(sys.argv[6]),
    "tree_topk": int(sys.argv[7]),
    "num_verify_tokens": int(sys.argv[8]),
    "path": sys.argv[1],
    "linear_wall_tok_s": (report.get("baseline") or {}).get("wall_tok_s"),
    "tree_wall_tok_s": (report.get("dflash") or {}).get("wall_tok_s"),
    "tree_accept_length": (report.get("dflash") or {}).get("accept_length"),
    "tree_correct_boxed_rate": (report.get("dflash_request_metric_aggregate") or {}).get("correct_boxed_rate"),
    "tree_speedup_wall_tok_s": report.get("speedup_wall_tok_s"),
    "tree_speedup_req_s": report.get("speedup_req_s"),
}
with Path(sys.argv[2]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\n")
PY
}

run_if_missing() {
  local out_json="$1"
  shift
  if [[ -s "${out_json}" ]]; then
    echo "[tree-single-matrix] skip existing ${out_json}"
    return 0
  fi
  "$@"
}

block_index=0
for block_size in "${BLOCK_SIZES[@]}"; do
  block_size="${block_size// /}"
  [[ -z "${block_size}" ]] && continue

  cfg_json="$(jq -c --arg block "${block_size}" '.[$block]' "${BEST_JSON}")"
  if [[ "${cfg_json}" == "null" || -z "${cfg_json}" ]]; then
    echo "[tree-single-matrix] missing best config for block=${block_size}" >&2
    exit 1
  fi

  best_steps="$(jq -r '.spec_steps' <<< "${cfg_json}")"
  best_topk="$(jq -r '.tree_topk' <<< "${cfg_json}")"
  best_vt="$(jq -r '.num_verify_tokens' <<< "${cfg_json}")"
  read -r sample_steps sample_topk sample_vt <<< "$(sample_branch_cfg_for_block "${block_size}")"

  greedy_linear_dir="${LINEAR_DIR}/block_${block_size}/greedy"
  sampled_linear_dir="${LINEAR_DIR}/block_${block_size}/sampled"
  mkdir -p "${greedy_linear_dir}" "${sampled_linear_dir}"
  greedy_linear_report="${greedy_linear_dir}/linear_report.json"
  greedy_linear_baseline="${greedy_linear_dir}/linear_baseline.json"
  sampled_linear_report="${sampled_linear_dir}/linear_report.json"
  sampled_linear_baseline="${sampled_linear_dir}/linear_baseline.json"

  greedy_port="$((BASELINE_PORT_BASE + block_index * 20))"
  sampled_port="$((BASELINE_PORT_BASE + block_index * 20 + 2))"

  echo "[tree-single-matrix] linear greedy block=${block_size}"
  run_if_missing "${greedy_linear_report}" run_bench "${greedy_linear_report}" "${greedy_port}" "DFLASH" "${block_size}" 0 0 0 0.0 1.0 1 0.0 "" 1
  if [[ ! -s "${greedy_linear_baseline}" ]]; then
    linear_to_baseline_json "${greedy_linear_report}" "${greedy_linear_baseline}"
  fi

  echo "[tree-single-matrix] linear sampled block=${block_size}"
  run_if_missing "${sampled_linear_report}" run_bench "${sampled_linear_report}" "${sampled_port}" "DFLASH" "${block_size}" 0 0 0 1.0 1.0 50 0.02 "" 1
  if [[ ! -s "${sampled_linear_baseline}" ]]; then
    linear_to_baseline_json "${sampled_linear_report}" "${sampled_linear_baseline}"
  fi

  tree_block_dir="${TREE_DIR}/block_${block_size}"
  mkdir -p "${tree_block_dir}"

  greedy_tree_report="${tree_block_dir}/greedy_target__draft_topk.json"
  echo "[tree-single-matrix] tree greedy-target draft-topk block=${block_size} steps=${best_steps} topk=${best_topk} vt=${best_vt}"
  run_if_missing "${greedy_tree_report}" run_bench "${greedy_tree_report}" "$((TREE_PORT_BASE + block_index * 30))" "DFLASH_TREE" "${block_size}" "${best_steps}" "${best_topk}" "${best_vt}" 0.0 1.0 1 0.0 "${greedy_linear_baseline}" 0 "topk"
  append_manifest "${greedy_tree_report}" "${block_size}" "greedy" "topk" "${best_steps}" "${best_topk}" "${best_vt}"

  sampled_target_topk_report="${tree_block_dir}/sampled_target__draft_topk.json"
  echo "[tree-single-matrix] tree sampled-target draft-topk block=${block_size} steps=${best_steps} topk=${best_topk} vt=${best_vt}"
  run_if_missing "${sampled_target_topk_report}" run_bench "${sampled_target_topk_report}" "$((TREE_PORT_BASE + block_index * 30 + 2))" "DFLASH_TREE" "${block_size}" "${best_steps}" "${best_topk}" "${best_vt}" 1.0 1.0 50 0.02 "${sampled_linear_baseline}" 0 "topk"
  append_manifest "${sampled_target_topk_report}" "${block_size}" "sampled" "topk" "${best_steps}" "${best_topk}" "${best_vt}"

  sampled_target_sample_report="${tree_block_dir}/sampled_target__draft_sample.json"
  echo "[tree-single-matrix] tree sampled-target draft-sample block=${block_size} steps=${sample_steps} topk=${sample_topk} vt=${sample_vt}"
  run_if_missing "${sampled_target_sample_report}" run_bench "${sampled_target_sample_report}" "$((TREE_PORT_BASE + block_index * 30 + 4))" "DFLASH_TREE" "${block_size}" "${sample_steps}" "${sample_topk}" "${sample_vt}" 1.0 1.0 50 0.02 "${sampled_linear_baseline}" 0 "sample"
  append_manifest "${sampled_target_sample_report}" "${block_size}" "sampled" "sample" "${sample_steps}" "${sample_topk}" "${sample_vt}"

  block_index=$((block_index + 1))
done

/venv/main/bin/python - <<'PY' "${MANIFEST}" "${SUMMARY_MD}"
import json, sys
from pathlib import Path

entries = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
if not entries:
    raise SystemExit("no entries collected")

lines = [
    "# DFlash Tree Single-Request Sampling Matrix",
    "",
    "| Block | Target | Draft branches | Steps | Topk | VT | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for entry in entries:
    lines.append(
        "| {block_size} | {target_mode} | {draft_branch_mode} | {spec_steps} | {tree_topk} | {num_verify_tokens} | {tree_wall_tok_s:.3f} | {linear_wall_tok_s:.3f} | {tree_speedup_wall_tok_s:.3f} | {tree_accept_length:.3f} | {tree_correct_boxed_rate:.3f} | `{path}` |".format(
            **entry
        )
    )

Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[tree-single-matrix] done -> ${ROOT_DIR}"
