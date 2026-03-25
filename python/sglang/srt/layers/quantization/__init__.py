# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.bitsandbytes import BitsAndBytesConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config
from sglang.srt.layers.quantization.petit import PetitNvFp4Config
from sglang.srt.layers.quantization.qoq import QoQConfig
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.utils import is_cuda, is_hip, is_npu, mxfp_supported


def _is_optional_quant_import_error(exc: ModuleNotFoundError) -> bool:
    return exc.name in {"tvm_ffi", "flashinfer", "gguf"}


try:
    from sglang.srt.layers.quantization.gguf import GGUFConfig
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    GGUFConfig = None

try:
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    CompressedTensorsConfig = None

try:
    from sglang.srt.layers.quantization.fp8 import Fp8Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    Fp8Config = None

try:
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4Config,
        ModelOptFp8Config,
    )
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    ModelOptFp4Config = None
    ModelOptFp8Config = None

try:
    from sglang.srt.layers.quantization.fpgemm_fp8 import FBGEMMFp8Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    FBGEMMFp8Config = None

try:
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    Mxfp4Config = None

try:
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    W4AFp8Config = None

try:
    from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    W8A8Fp8Config = None

try:
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    QuarkConfig = None

try:
    from sglang.srt.layers.quantization.quark_int4fp8_moe import QuarkInt4Fp8Config
except ModuleNotFoundError as exc:
    if not _is_optional_quant_import_error(exc):
        raise
    QuarkInt4Fp8Config = None

_is_mxfp_supported = mxfp_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "blockwise_int8": BlockInt8Config,
    "w8a8_int8": W8A8Int8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "qoq": QoQConfig,
    "petit_nvfp4": PetitNvFp4Config,
    "auto-round": AutoRoundConfig,
    "modelslim": ModelSlimConfig,
}

if GGUFConfig is not None:
    BASE_QUANTIZATION_METHODS["gguf"] = GGUFConfig

if Fp8Config is not None:
    BASE_QUANTIZATION_METHODS.update(
        {
            "fp8": Fp8Config,
            "mxfp8": Fp8Config,
        }
    )

if ModelOptFp8Config is not None:
    BASE_QUANTIZATION_METHODS.update(
        {
            "modelopt": ModelOptFp8Config,
            "modelopt_fp8": ModelOptFp8Config,
        }
    )

if ModelOptFp4Config is not None:
    BASE_QUANTIZATION_METHODS["modelopt_fp4"] = ModelOptFp4Config

if CompressedTensorsConfig is not None:
    BASE_QUANTIZATION_METHODS["compressed-tensors"] = CompressedTensorsConfig

if W4AFp8Config is not None:
    BASE_QUANTIZATION_METHODS["w4afp8"] = W4AFp8Config

if W8A8Fp8Config is not None:
    BASE_QUANTIZATION_METHODS["w8a8_fp8"] = W8A8Fp8Config

if FBGEMMFp8Config is not None:
    BASE_QUANTIZATION_METHODS["fbgemm_fp8"] = FBGEMMFp8Config

if QuarkInt4Fp8Config is not None:
    BASE_QUANTIZATION_METHODS["quark_int4fp8_moe"] = QuarkInt4Fp8Config

if QuarkConfig is not None:
    BASE_QUANTIZATION_METHODS["quark"] = QuarkConfig


if (is_cuda() or (_is_mxfp_supported and is_hip())) and Mxfp4Config is not None:
    BASE_QUANTIZATION_METHODS.update(
        {
            "mxfp4": Mxfp4Config,
        }
    )

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
