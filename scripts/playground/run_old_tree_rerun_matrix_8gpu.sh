#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/old_tree_rerun_matrix_20260403}"
mkdir -p "${ROOT_DIR}"

OLD_REPO="/workspace/sglang-dflash-pagesize-fix-old"
PYTHON_BIN="/workspace/venv-dflash/bin/python"
BENCH="${OLD_REPO}/scripts/playground/bench_reference_dflash.py"
ROUTE_GREEDY="${OLD_REPO}/scripts/playground/run_route5_explore32_route8_block4to8_tree_greedy.sh"
ROUTE_SAMPLED="${OLD_REPO}/scripts/playground/run_route5_explore32_route8_block4to8_tree_sampled.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/30_03_DFLASH/workspace/offload_root/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/workspace/30_03_DFLASH/root/epoch_65_step_23760}"
REFERENCE_CSV="${REFERENCE_CSV:-/workspace/30_03_DFLASH/root/reference.csv}"
QIDS_FIVE="${QIDS_FIVE:-86e8e5,dd7f5e,a295e9,9c1c5f,92ba6a}"
QID_SINGLE="${QID_SINGLE:-92ba6a}"

export PYTHONPATH="${OLD_REPO}/python"

launch_job_script() {
  local gpu="$1"
  local log="$2"
  local script_path="$3"
  shift 3
  cat >"${script_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${OLD_REPO}/python"
$*
EOF
  chmod +x "${script_path}"
  nohup bash "${script_path}" >"${log}" 2>&1 < /dev/null &
  echo $!
}

bench_cmd() {
  local out_json="$1"
  local base_port="$2"
  local algo="$3"
  local qids="$4"
  local concurrency="$5"
  local num_prompts="$6"
  local temperature="$7"
  local top_p="$8"
  local top_k="$9"
  local min_p="${10}"
  shift 10

  local cmd=(
    "${PYTHON_BIN}" "${BENCH}"
    --model-path "${MODEL_PATH}"
    --draft-model-path "${DRAFT_MODEL_PATH}"
    --reference-csv "${REFERENCE_CSV}"
    --question-ids "${qids}"
    --out-json "${out_json}"
    --skip-baseline
    --context-length 65536
    --decode-len 65536
    --decode-to-context-limit
    --buffer-tokens 512
    --concurrency "${concurrency}"
    --num-prompts "${num_prompts}"
    --page-size 1
    --draft-page-size 1
    --cuda-graph-max-bs "${concurrency}"
    --max-running-requests "${concurrency}"
    --piecewise-cuda-graph-max-tokens 8192
    --kv-cache-dtype fp8_e4m3
    --draft-kv-cache-dtype bfloat16
    --attention-backend fa3
    --moe-runner-backend triton_kernel
    --draft-attention-backend fa3
    --speculative-moe-runner-backend triton_kernel
    --speculative-algorithm "${algo}"
    --speculative-dflash-block-size 4
    --mem-fraction-static 0.90
    --temperature "${temperature}"
    --top-p "${top_p}"
    --top-k "${top_k}"
    --min-p "${min_p}"
    --disable-overlap-schedule
    --disable-stream
    --baseline-port "${base_port}"
    --dflash-port "$((base_port + 1))"
  )

  if [[ "${algo}" == "DFLASH_TREE" ]]; then
    cmd+=(
      --speculative-num-steps 3
      --speculative-eagle-topk 4
      --speculative-num-draft-tokens 4
    )
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

run_dir_route_greedy="${ROOT_DIR}/route_tree_greedy"
run_dir_route_sampled="${ROOT_DIR}/route_tree_sampled"
jobs_dir="${ROOT_DIR}/jobs"
mkdir -p "${run_dir_route_greedy}" "${run_dir_route_sampled}" "${jobs_dir}"

declare -a pids

pids+=("$(launch_job_script 0 "${run_dir_route_greedy}/run.log" "${jobs_dir}/route_tree_greedy.sh" "MODEL_PATH=${MODEL_PATH@Q} DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH@Q} REFERENCE_CSV=${REFERENCE_CSV@Q} bash ${ROUTE_GREEDY@Q} ${run_dir_route_greedy@Q}")")
pids+=("$(launch_job_script 1 "${run_dir_route_sampled}/run.log" "${jobs_dir}/route_tree_sampled.sh" "MODEL_PATH=${MODEL_PATH@Q} DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH@Q} REFERENCE_CSV=${REFERENCE_CSV@Q} bash ${ROUTE_SAMPLED@Q} ${run_dir_route_sampled@Q}")")

single_linear_json="${ROOT_DIR}/single_linear_block4_greedy.json"
single_tree_json="${ROOT_DIR}/single_tree_block4_greedy.json"
batch_linear_greedy_json="${ROOT_DIR}/batch_linear_block4_greedy.json"
batch_tree_greedy_json="${ROOT_DIR}/batch_tree_block4_greedy.json"
batch_linear_sampled_json="${ROOT_DIR}/batch_linear_block4_sampled.json"
batch_tree_sampled_json="${ROOT_DIR}/batch_tree_block4_sampled.json"

single_linear_cmd="$(bench_cmd "${single_linear_json}" 26020 DFLASH "${QID_SINGLE}" 1 1 0.0 1.0 1 0.0)"
single_tree_cmd="$(bench_cmd "${single_tree_json}" 26030 DFLASH_TREE "${QID_SINGLE}" 1 1 0.0 1.0 1 0.0)"
batch_linear_greedy_cmd="$(bench_cmd "${batch_linear_greedy_json}" 26040 DFLASH "${QIDS_FIVE}" 8 8 0.0 1.0 1 0.0)"
batch_tree_greedy_cmd="$(bench_cmd "${batch_tree_greedy_json}" 26050 DFLASH_TREE "${QIDS_FIVE}" 8 8 0.0 1.0 1 0.0)"
batch_linear_sampled_cmd="$(bench_cmd "${batch_linear_sampled_json}" 26060 DFLASH "${QIDS_FIVE}" 8 8 1.0 1.0 50 0.02)"
batch_tree_sampled_cmd="$(bench_cmd "${batch_tree_sampled_json}" 26070 DFLASH_TREE "${QIDS_FIVE}" 8 8 1.0 1.0 50 0.02)"

pids+=("$(launch_job_script 2 "${ROOT_DIR}/single_linear_block4_greedy.log" "${jobs_dir}/single_linear_block4_greedy.sh" "${single_linear_cmd}")")
pids+=("$(launch_job_script 3 "${ROOT_DIR}/single_tree_block4_greedy.log" "${jobs_dir}/single_tree_block4_greedy.sh" "${single_tree_cmd}")")
pids+=("$(launch_job_script 4 "${ROOT_DIR}/batch_linear_block4_greedy.log" "${jobs_dir}/batch_linear_block4_greedy.sh" "${batch_linear_greedy_cmd}")")
pids+=("$(launch_job_script 5 "${ROOT_DIR}/batch_tree_block4_greedy.log" "${jobs_dir}/batch_tree_block4_greedy.sh" "${batch_tree_greedy_cmd}")")
pids+=("$(launch_job_script 6 "${ROOT_DIR}/batch_linear_block4_sampled.log" "${jobs_dir}/batch_linear_block4_sampled.sh" "${batch_linear_sampled_cmd}")")
pids+=("$(launch_job_script 7 "${ROOT_DIR}/batch_tree_block4_sampled.log" "${jobs_dir}/batch_tree_block4_sampled.sh" "${batch_tree_sampled_cmd}")")

printf '%s\n' "${pids[@]}" > "${ROOT_DIR}/pids.txt"
echo "launched_root=${ROOT_DIR}"
echo "pids_file=${ROOT_DIR}/pids.txt"
