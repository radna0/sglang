#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/dflash_tree_overlap_config_sweep_20260330}"
LINEAR_SWEEP_ROOT="${LINEAR_SWEEP_ROOT:-/workspace/dflash_tree_config_sweep_20260330}"
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
BASELINE_PORT_BASE="${BASELINE_PORT_BASE:-23880}"
TREE_PORT_BASE="${TREE_PORT_BASE:-23980}"
BLOCK_SIZES_RAW="${BLOCK_SIZES_RAW:-4,8,16}"
TREE_TOPKS_RAW="${TREE_TOPKS_RAW:-1,2,4}"
TREE_SPEC_STEPS_MODE="${TREE_SPEC_STEPS_MODE:-compact}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
MIN_P="${MIN_P:-0.0}"

export PYTHONPATH="/workspace/sglang-dflash-line/python"
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1
export SGLANG_ENABLE_DFLASH_TREE_OVERLAP_EXPERIMENTAL=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

read -r -a BLOCK_SIZES <<< "$(echo "${BLOCK_SIZES_RAW}" | tr ',;' '  ')"
read -r -a TREE_TOPKS <<< "$(echo "${TREE_TOPKS_RAW}" | tr ',;' '  ')"

TREE_DIR="${ROOT_DIR}/tree_overlap_sweep"
mkdir -p "${TREE_DIR}"

TREE_MANIFEST="${ROOT_DIR}/tree_overlap_manifest.jsonl"
BEST_JSON="${ROOT_DIR}/best_by_block_size.json"
BEST_MD="${ROOT_DIR}/best_by_block_size.md"
SUMMARY_MD="${ROOT_DIR}/summary.md"

: > "${TREE_MANIFEST}"

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

priority_config_for_block() {
  local block_size="$1"
  case "${block_size}" in
    4)
      printf "3 4\n"
      ;;
    8)
      printf "4 2\n"
      ;;
    16)
      printf "8 1\n"
      ;;
    *)
      return 1
      ;;
  esac
}

run_bench() {
  local out_json="$1"
  local port="$2"
  local block_size="$3"
  local speculative_num_steps="$4"
  local speculative_eagle_topk="$5"
  local speculative_num_draft_tokens="$6"
  local baseline_json_in="$7"

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
    --speculative-algorithm DFLASH_TREE
    --speculative-dflash-block-size "${block_size}"
    --speculative-num-steps "${speculative_num_steps}"
    --speculative-eagle-topk "${speculative_eagle_topk}"
    --speculative-num-draft-tokens "${speculative_num_draft_tokens}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --min-p "${MIN_P}"
    --disable-stream
    --baseline-json-in "${baseline_json_in}"
    --dflash-port "$((port + 1))"
    --out-json "${out_json}"
  )

  if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    cmd+=(--disable-cuda-graph)
  fi

  "${cmd[@]}"
}

block_index=0
for block_size in "${BLOCK_SIZES[@]}"; do
  block_size="${block_size// /}"
  [[ -z "${block_size}" ]] && continue

  baseline_json="${LINEAR_SWEEP_ROOT}/linear_baselines/block_${block_size}/linear_baseline.json"
  if [[ ! -f "${baseline_json}" ]]; then
    echo "[tree-overlap-sweep] missing linear baseline: ${baseline_json}" >&2
    exit 1
  fi

  mapfile -t spec_steps_candidates < <(choose_spec_steps "${block_size}")
  read -r priority_spec_steps priority_topk < <(priority_config_for_block "${block_size}" || printf " \n")
  priority_config_done=0

  if [[ -n "${priority_spec_steps:-}" && -n "${priority_topk:-}" ]]; then
    if (( priority_spec_steps > 0 && priority_topk > 0 && priority_topk <= block_size )); then
      verify_tokens=$((priority_topk + priority_spec_steps))
      if (( verify_tokens > block_size )); then
        verify_tokens="${block_size}"
      fi
      if (( verify_tokens > 0 )); then
        cfg_dir="${TREE_DIR}/block_${block_size}"
        mkdir -p "${cfg_dir}"
        cfg_name="steps_${priority_spec_steps}_topk_${priority_topk}_vt_${verify_tokens}"
        cfg_report="${cfg_dir}/${cfg_name}.json"
        cfg_port="$((TREE_PORT_BASE + block_index * 100 + priority_spec_steps * 10 + priority_topk))"

        echo "[tree-overlap-sweep] block=${block_size} steps=${priority_spec_steps} topk=${priority_topk} vt=${verify_tokens} (priority)"
        run_bench \
          "${cfg_report}" \
          "${cfg_port}" \
          "${block_size}" \
          "${priority_spec_steps}" \
          "${priority_topk}" \
          "${verify_tokens}" \
          "${baseline_json}"

        /venv/main/bin/python - <<'PY' "${cfg_report}" "${TREE_MANIFEST}" "${block_size}" "${priority_spec_steps}" "${priority_topk}" "${verify_tokens}"
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
        priority_config_done=1
      fi
    fi
  fi

  for spec_steps in "${spec_steps_candidates[@]}"; do
    [[ -z "${spec_steps}" ]] && continue
    for topk in "${TREE_TOPKS[@]}"; do
      topk="${topk// /}"
      [[ -z "${topk}" ]] && continue
      if [[ "${priority_config_done}" == "1" && "${spec_steps}" == "${priority_spec_steps}" && "${topk}" == "${priority_topk}" ]]; then
        continue
      fi
      if (( topk <= 0 || topk > block_size )); then
        continue
      fi

      verify_tokens=$((topk + spec_steps))
      if (( verify_tokens > block_size )); then
        verify_tokens="${block_size}"
      fi
      if (( verify_tokens <= 0 )); then
        continue
      fi

      cfg_dir="${TREE_DIR}/block_${block_size}"
      mkdir -p "${cfg_dir}"
      cfg_name="steps_${spec_steps}_topk_${topk}_vt_${verify_tokens}"
      cfg_report="${cfg_dir}/${cfg_name}.json"
      cfg_port="$((TREE_PORT_BASE + block_index * 100 + spec_steps * 10 + topk))"

      echo "[tree-overlap-sweep] block=${block_size} steps=${spec_steps} topk=${topk} vt=${verify_tokens}"
      run_bench \
        "${cfg_report}" \
        "${cfg_port}" \
        "${block_size}" \
        "${spec_steps}" \
        "${topk}" \
        "${verify_tokens}" \
        "${baseline_json}"

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
  block_index=$((block_index + 1))
done

/venv/main/bin/python - <<'PY' "${TREE_MANIFEST}" "${BEST_JSON}" "${BEST_MD}" "${SUMMARY_MD}"
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
best_json_path = Path(sys.argv[2])
best_md_path = Path(sys.argv[3])
summary_md_path = Path(sys.argv[4])

entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not entries:
    raise SystemExit("no overlap tree entries collected")

best: dict[int, dict] = {}
rows: list[str] = []
for entry in entries:
    rows.append(
        f"| {entry['block_size']} | {entry['spec_steps']} | {entry['tree_topk']} | {entry['num_verify_tokens']} | "
        f"{entry['tree_wall_tok_s']:.3f} | {entry['linear_wall_tok_s']:.3f} | "
        f"{entry['tree_speedup_wall_tok_s']:.3f} | {entry['tree_accept_length']:.3f} | "
        f"{entry['tree_correct_boxed_rate']:.3f} | {entry['path']} |"
    )
    block = int(entry["block_size"])
    cur = best.get(block)
    if cur is None or float(entry["tree_wall_tok_s"]) > float(cur["tree_wall_tok_s"]):
        best[block] = entry

best_rows = []
for block in sorted(best):
    entry = best[block]
    best_rows.append(
        {
            "block_size": block,
            "spec_steps": entry["spec_steps"],
            "tree_topk": entry["tree_topk"],
            "num_verify_tokens": entry["num_verify_tokens"],
            "tree_wall_tok_s": entry["tree_wall_tok_s"],
            "linear_wall_tok_s": entry["linear_wall_tok_s"],
            "tree_speedup_wall_tok_s": entry["tree_speedup_wall_tok_s"],
            "tree_accept_length": entry["tree_accept_length"],
            "tree_correct_boxed_rate": entry["tree_correct_boxed_rate"],
            "path": entry["path"],
        }
    )

best_json_path.write_text(json.dumps(best_rows, indent=2), encoding="utf-8")

md = [
    "# DFlash Tree Overlap Sweep Summary",
    "",
    "Overlap-enabled single-request tree sweep against the linear baselines from `dflash_tree_config_sweep_20260330`.",
    "",
    "| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
md.extend(rows)
md.append("")
md.append("# Best By Block Size")
md.append("")
md.append("| Block | Steps | Topk | Verify tokens | Tree tok/s | Linear tok/s | Speedup | Accept len | Correct | Path |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for row in best_rows:
    md.append(
        f"| {row['block_size']} | {row['spec_steps']} | {row['tree_topk']} | {row['num_verify_tokens']} | "
        f"{row['tree_wall_tok_s']:.3f} | {row['linear_wall_tok_s']:.3f} | "
        f"{row['tree_speedup_wall_tok_s']:.3f} | {row['tree_accept_length']:.3f} | "
        f"{row['tree_correct_boxed_rate']:.3f} | {row['path']} |"
    )
best_md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
summary_md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(best_rows, indent=2))
PY
