#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/root/gpt-oss-120b}"
OUT_DIR="${2:-/root/sglang-gpt-oss-care-mla/logs/hf_eval/full_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/ppl" "${OUT_DIR}/tasks" "${OUT_DIR}/offload"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export GPTOSS_MXFP4_PRESWIZZLE_DIR="${GPTOSS_MXFP4_PRESWIZZLE_DIR:-/workspace/gptoss_mxfp4_preswizzle_shared_v2}"

RUNNER="/venv/main/bin/python /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_lm_eval.py"
PPL_RUNNER="/venv/main/bin/python /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_ppl_sharded.py"
TORCHRUN="/venv/main/bin/torchrun"
MERGER="/venv/main/bin/python /root/sglang-gpt-oss-care-mla/scripts/merge_gpt_oss_eval_jsons.py"
TABLE_RUNNER="/venv/main/bin/python /root/sglang-gpt-oss-care-mla/scripts/render_gpt_oss_baseline_table.py"

PPL_TASKS=(wikitext c4)
BENCHMARK_TASKS=(arc_easy arc_challenge hellaswag piqa mmlu openbookqa race winogrande)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=(0 1 2 3 4 5 6 7)
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPUs available." >&2
  exit 1
fi

echo "[out-dir] ${OUT_DIR}"
echo "[gpus] ${GPU_IDS[*]}"

run_task() {
  local gpu_id="$1"
  local task_name="$2"
  local output_path="$3"
  local log_path="$4"
  local max_length="$5"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    ${RUNNER} \
      --model-path "${MODEL_PATH}" \
      --tasks "${task_name}" \
      --batch-size 1 \
      --max-batch-size 1 \
      --max-length "${max_length}" \
      --device cuda:0 \
      --no-parallelize \
      --offload-folder "${OUT_DIR}/offload/${task_name}" \
      --output-path "${output_path}" \
      > "${log_path}" 2>&1
}

echo "[stage] ppl:wikitext"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
  ${PPL_RUNNER} \
    --model-path "${MODEL_PATH}" \
    --task wikitext \
    --dtype bfloat16 \
    --max-length 2048 \
    --device cuda \
    --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
    --output-path "${OUT_DIR}/ppl/wikitext.json" \
    > "${OUT_DIR}/ppl/wikitext.log" 2>&1

echo "[stage] ppl:c4_dp${#GPU_IDS[@]}"
GPU_JOINED="$(IFS=,; echo "${GPU_IDS[*]}")"
CUDA_VISIBLE_DEVICES="${GPU_JOINED}" \
  ${TORCHRUN} --standalone --nproc_per_node "${#GPU_IDS[@]}" \
    /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_ppl_sharded.py \
      --model-path "${MODEL_PATH}" \
      --task c4 \
      --dtype bfloat16 \
      --max-length 2048 \
      --device cuda \
      --progress-every-docs 128 \
      --mxfp4-preswizzle-dir "${GPTOSS_MXFP4_PRESWIZZLE_DIR}" \
      --output-path "${OUT_DIR}/ppl/c4.json" \
      > "${OUT_DIR}/ppl/c4.log" 2>&1

${MERGER} \
  --inputs "${OUT_DIR}/ppl/wikitext.json" "${OUT_DIR}/ppl/c4.json" \
  --output "${OUT_DIR}/ppl.json" \
  > "${OUT_DIR}/ppl_merge.log" 2>&1

echo "[stage] benchmark"
task_pids=()
for idx in "${!BENCHMARK_TASKS[@]}"; do
  task_name="${BENCHMARK_TASKS[$idx]}"
  gpu_id="${GPU_IDS[$(( idx % ${#GPU_IDS[@]} ))]}"
  run_task \
    "${gpu_id}" \
    "${task_name}" \
    "${OUT_DIR}/tasks/${task_name}.json" \
    "${OUT_DIR}/tasks/${task_name}.log" \
    2048 &
  task_pids+=("$!")
done

for pid in "${task_pids[@]}"; do
  wait "${pid}"
done

${MERGER} \
  --inputs \
    "${OUT_DIR}/tasks/arc_easy.json" \
    "${OUT_DIR}/tasks/arc_challenge.json" \
    "${OUT_DIR}/tasks/hellaswag.json" \
    "${OUT_DIR}/tasks/piqa.json" \
    "${OUT_DIR}/tasks/mmlu.json" \
    "${OUT_DIR}/tasks/openbookqa.json" \
    "${OUT_DIR}/tasks/race.json" \
    "${OUT_DIR}/tasks/winogrande.json" \
  --output "${OUT_DIR}/benchmarks.json" \
  > "${OUT_DIR}/benchmarks_merge.log" 2>&1

${TABLE_RUNNER} \
  --run-dir "${OUT_DIR}" \
  --markdown-out "${OUT_DIR}/baseline_table.md" \
  --csv-out "${OUT_DIR}/baseline_table.csv" \
  --json-out "${OUT_DIR}/baseline_table.json" \
  > "${OUT_DIR}/baseline_table.log" 2>&1

echo "[done] ${OUT_DIR}"
