#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SM="${1:-sm90}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SM}" in
  sm90)
    export FLASHINFER_CUDA_ARCH_LIST="9.0a"
    export FLASH_MLA_DISABLE_SM100=1
    ;;
  sm100)
    export FLASHINFER_CUDA_ARCH_LIST="10.0a"
    ;;
  *)
    echo "Unsupported SM target: ${SM}. Use sm90 or sm100." >&2
    exit 1
    ;;
esac

export MAX_JOBS="${MAX_JOBS:-16}"

cd "${ROOT_DIR}"
git submodule update --init 3rdparty/flashmla 3rdparty/flashmla-fp8 3rdparty/flashmla-swap

install_pkg() {
  local pkg_name="$1"
  local pkg_dir="$2"
  shift 2
  "${PYTHON_BIN}" -m pip uninstall -y "${pkg_name}" >/dev/null 2>&1 || true
  (
    cd "${pkg_dir}"
    env "$@" "${PYTHON_BIN}" -m pip install --no-build-isolation -v .
  )
}

install_pkg flash_mla 3rdparty/flashmla
install_pkg flash_mla_swap 3rdparty/flashmla-swap FLASH_MLA_DISABLE_FP16=1
install_pkg flash_mla_fp8 3rdparty/flashmla-fp8 FLASH_MLA_DISABLE_FP16=1

echo "Installed FluentLLM-pinned FlashMLA kernels for ${SM}."
