#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/workspace/offload_root/gpt-oss-120b}"
BASE_RUN_ROOT="${2:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210}"
OUT_ROOT="${3:-/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_u_convert_only_sweep_$(date -u +%Y%m%d_%H%M%S)}"

REPO_ROOT="/root/sglang-gpt-oss-care-mla"
CONVERT_SCRIPT="${REPO_ROOT}/scripts/convert_gpt_oss_to_care_mla.py"
RANKS="${RANKS:-1024,512,448,384,320,256,128}"
QK_ROPE_HEAD_DIM="${QK_ROPE_HEAD_DIM:-32}"
QK_NOPE_HEAD_DIM="${QK_NOPE_HEAD_DIM:-32}"
DTYPE="${DTYPE:-bfloat16}"
CONVERSION_DEVICE_MAP="${CONVERSION_DEVICE_MAP:-cuda:0}"
NUM_KV_HEADS="$(
  /venv/main/bin/python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${MODEL_PATH}/config.json").read_text())
print(int(cfg.get("num_key_value_heads", 1) or 1))
PY
)"
MLA_ROPE_NUM_KV_HEADS="${MLA_ROPE_NUM_KV_HEADS:-${NUM_KV_HEADS}}"

BASE_CONV_ROOT="${BASE_RUN_ROOT}/conversion"
COVARIANCE_DIR="${BASE_CONV_ROOT}/covariance"
DATASET_SPEC_JSON="${DATASET_SPEC_JSON:-}"
if [[ -z "${DATASET_SPEC_JSON}" ]]; then
  DATASET_SPEC_JSON="$(python - <<'PY'
import json, pathlib
p = pathlib.Path('/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/pipeline_manifest.json')
if not p.exists():
    raise SystemExit(1)
print(json.loads(p.read_text())["dataset_spec_json"])
PY
)"
fi

if [[ ! -f "${COVARIANCE_DIR}/covariance_manifest.json" ]]; then
  echo "missing covariance manifest at ${COVARIANCE_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "[model-path] ${MODEL_PATH}"
echo "[base-run-root] ${BASE_RUN_ROOT}"
echo "[covariance-dir] ${COVARIANCE_DIR}"
echo "[dataset-spec-json] ${DATASET_SPEC_JSON}"
echo "[out-root] ${OUT_ROOT}"
echo "[ranks] ${RANKS}"
echo "[mla-rope-num-kv-heads] ${MLA_ROPE_NUM_KV_HEADS}"

IFS=',' read -r -a rank_list <<< "${RANKS}"
for rank in "${rank_list[@]}"; do
  rank="$(echo "${rank}" | xargs)"
  run_root="${OUT_ROOT}/care_u_r${rank}"
  out_root="${run_root}/conversion"
  mkdir -p "${out_root}"
  log_path="${out_root}/pipeline.log"
  converted_dir="${out_root}/converted_checkpoint"
  pipeline_manifest="${out_root}/pipeline_manifest.json"

  echo "[launch][care-u-convert-only] rank=${rank} run_root=${run_root}"
  cat > "${pipeline_manifest}" <<JSON
{
  "model_path": "${MODEL_PATH}",
  "dataset_spec_json": "${DATASET_SPEC_JSON}",
  "out_root": "${out_root}",
  "covariance_dir": "${COVARIANCE_DIR}",
  "rank_schedule_json": null,
  "converted_dir": "${converted_dir}",
  "target_rank": ${rank},
  "target_total_rank": null,
  "uniform_rank": true,
  "qk_rope_head_dim": ${QK_ROPE_HEAD_DIM},
  "qk_nope_head_dim": ${QK_NOPE_HEAD_DIM},
  "mla_rope_num_kv_heads": ${MLA_ROPE_NUM_KV_HEADS},
  "decoupled_rope_dim": 0,
  "decoupled_rope_init": "mean",
  "rank_source_fusion": false
}
JSON

  /venv/main/bin/python "${CONVERT_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${converted_dir}" \
    --device auto \
    --kv-lora-rank "${rank}" \
    --qk-rope-head-dim "${QK_ROPE_HEAD_DIM}" \
    --qk-nope-head-dim "${QK_NOPE_HEAD_DIM}" \
    --mla-rope-num-kv-heads "${MLA_ROPE_NUM_KV_HEADS}" \
    --decoupled-rope-dim 0 \
    --decoupled-rope-init mean \
    --covariance-dir "${COVARIANCE_DIR}" \
    --covariance-shrinkage 0.0001 \
    --overwrite \
    2>&1 | tee "${log_path}"
done
