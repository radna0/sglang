#!/usr/bin/env bash
set -uo pipefail

ROOT="/workspace/reference5_tree_correctness_matrix_20260403"
REPO="/workspace/sglang-dflash-tree-active"
PYTHON="/workspace/venv-dflash/bin/python"
MODEL="/root/gpt-oss-120b"
DRAFT="/workspace/30_03_DFLASH/root/epoch_65_step_23760"
REFERENCE_CSV="/workspace/30_03_DFLASH/root/reference.csv"

mkdir -p "${ROOT}"

run_one() {
  local gpu="$1"
  local problem="$2"
  local mode="$3"
  local base_port="$4"
  local out_json="${ROOT}/${problem}_${mode}.json"
  local log_file="${ROOT}/${problem}_${mode}.log"

  local temp top_p top_k min_p
  if [[ "${mode}" == "greedy" ]]; then
    temp="0.0"
    top_p="1.0"
    top_k="1"
    min_p="0.0"
  else
    temp="1.0"
    top_p="1.0"
    top_k="50"
    min_p="0.02"
  fi

  echo "launch gpu=${gpu} problem=${problem} mode=${mode} out=${out_json}" >&2
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${REPO}/python"
    export SGLANG_DFLASH_DRAFT_SHARE_POOLS=1
    export SGLANG_DFLASH_TREE_FORCE_EXPLICIT_INPUT_EMBEDS=1
    exec "${PYTHON}" "${REPO}/scripts/playground/dflash/bench_reference.py" \
      --model-path "${MODEL}" \
      --draft-model-path "${DRAFT}" \
      --reference-csv "${REFERENCE_CSV}" \
      --question-ids "${problem}" \
      --context-length 65536 \
      --decode-len 65536 \
      --decode-to-context-limit \
      --buffer-tokens 512 \
      --concurrency 8 \
      --num-prompts 8 \
      --page-size 1 \
      --cuda-graph-max-bs 8 \
      --max-running-requests 8 \
      --piecewise-cuda-graph-max-tokens 65536 \
      --kv-cache-dtype fp8_e4m3 \
      --draft-kv-cache-dtype bfloat16 \
      --attention-backend fa3 \
      --moe-runner-backend triton_kernel \
      --draft-attention-backend fa3 \
      --speculative-moe-runner-backend triton_kernel \
      --mem-fraction-static 0.80 \
      --speculative-draft-mem-fraction-static 0.97 \
      --temperature "${temp}" \
      --top-p "${top_p}" \
      --top-k "${top_k}" \
      --min-p "${min_p}" \
      --disable-overlap-schedule \
      --skip-baseline \
      --disable-stream \
      --speculative-algorithm DFLASH_TREE \
      --speculative-dflash-block-size 16 \
      --speculative-num-steps 8 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 9 \
      --baseline-port "${base_port}" \
      --dflash-port "$((base_port + 1))" \
      --out-json "${out_json}"
  ) >"${log_file}" 2>&1 &
  LAST_PID=$!
}

LAST_PID=""
run_one 0 86e8e5 greedy 27200
pid_g0=$LAST_PID
run_one 1 dd7f5e greedy 27210
pid_g1=$LAST_PID
run_one 2 a295e9 greedy 27220
pid_g2=$LAST_PID
run_one 3 9c1c5f greedy 27230
pid_g3=$LAST_PID
run_one 4 92ba6a greedy 27240
pid_g4=$LAST_PID
run_one 5 86e8e5 sampled 27250
pid_s0=$LAST_PID
run_one 6 dd7f5e sampled 27260
pid_s1=$LAST_PID
run_one 7 a295e9 sampled 27270
pid_s2=$LAST_PID

wait "${pid_g3}" || true
run_one 3 9c1c5f sampled 27280
pid_s3=$LAST_PID

wait "${pid_g4}" || true
run_one 4 92ba6a sampled 27290
pid_s4=$LAST_PID

wait "${pid_g0}" || true
wait "${pid_g1}" || true
wait "${pid_g2}" || true
wait "${pid_s0}" || true
wait "${pid_s1}" || true
wait "${pid_s2}" || true
wait "${pid_s3}" || true
wait "${pid_s4}" || true
