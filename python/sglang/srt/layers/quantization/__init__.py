# Adapted from vLLM/SGLang quantization registry, but made lazy to keep imports
# lightweight (especially on notebook environments like Kaggle that may have
# optional deps with binary incompatibilities).

from __future__ import annotations

import builtins
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, Optional, Tuple, Type

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import is_cuda, is_hip, is_npu, mxfp_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class _LazySpec:
    module: str
    attr: str


def _specs() -> Dict[str, _LazySpec]:
    # Keep keys stable (used for CLI validation / help text). Values load lazily.
    out: Dict[str, _LazySpec] = {
        "fp8": _LazySpec("sglang.srt.layers.quantization.fp8", "Fp8Config"),
        "blockwise_int8": _LazySpec(
            "sglang.srt.layers.quantization.blockwise_int8", "BlockInt8Config"
        ),
        "modelopt": _LazySpec(
            "sglang.srt.layers.quantization.modelopt_quant", "ModelOptFp8Config"
        ),
        "modelopt_fp8": _LazySpec(
            "sglang.srt.layers.quantization.modelopt_quant", "ModelOptFp8Config"
        ),
        "modelopt_fp4": _LazySpec(
            "sglang.srt.layers.quantization.modelopt_quant", "ModelOptFp4Config"
        ),
        "w8a8_int8": _LazySpec(
            "sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8Config"
        ),
        "w8a8_fp8": _LazySpec(
            "sglang.srt.layers.quantization.w8a8_fp8", "W8A8Fp8Config"
        ),
        "awq": _LazySpec("sglang.srt.layers.quantization.awq", "AWQConfig"),
        "awq_marlin": _LazySpec("sglang.srt.layers.quantization.awq", "AWQMarlinConfig"),
        "gguf": _LazySpec("sglang.srt.layers.quantization.gguf", "GGUFConfig"),
        "gptq": _LazySpec("sglang.srt.layers.quantization.gptq", "GPTQConfig"),
        "gptq_marlin": _LazySpec(
            "sglang.srt.layers.quantization.gptq", "GPTQMarlinConfig"
        ),
        "moe_wna16": _LazySpec(
            "sglang.srt.layers.quantization.moe_wna16", "MoeWNA16Config"
        ),
        "compressed-tensors": _LazySpec(
            "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
            "CompressedTensorsConfig",
        ),
        "qoq": _LazySpec("sglang.srt.layers.quantization.qoq", "QoQConfig"),
        "w4afp8": _LazySpec("sglang.srt.layers.quantization.w4afp8", "W4AFp8Config"),
        "petit_nvfp4": _LazySpec(
            "sglang.srt.layers.quantization.petit", "PetitNvFp4Config"
        ),
        "fbgemm_fp8": _LazySpec(
            "sglang.srt.layers.quantization.fpgemm_fp8", "FBGEMMFp8Config"
        ),
        "quark": _LazySpec(
            "sglang.srt.layers.quantization.quark.quark", "QuarkConfig"
        ),
        "auto-round": _LazySpec(
            "sglang.srt.layers.quantization.auto_round", "AutoRoundConfig"
        ),
        "modelslim": _LazySpec(
            "sglang.srt.layers.quantization.modelslim.modelslim", "ModelSlimConfig"
        ),
        "quark_int4fp8_moe": _LazySpec(
            "sglang.srt.layers.quantization.quark_int4fp8_moe", "QuarkInt4Fp8Config"
        ),
    }
    if is_cuda() or (mxfp_supported() and is_hip()):
        out["mxfp4"] = _LazySpec("sglang.srt.layers.quantization.mxfp4", "Mxfp4Config")
    return out


class _LazyQuantizationMethods(Dict[str, Type[QuantizationConfig]]):
    def __init__(self, specs: Dict[str, _LazySpec]):
        super().__init__()
        self._specs = dict(specs)

    def _load_one(self, key: str) -> Optional[Type[QuantizationConfig]]:
        # IMPORTANT: our `__contains__` is overridden to reflect spec availability,
        # not whether we've already loaded/cached the class. Use the base dict
        # membership check here.
        if super().__contains__(key):
            return super().__getitem__(key)
        spec = self._specs.get(key)
        if spec is None:
            return None
        try:
            mod = importlib.import_module(spec.module)
            cls = getattr(mod, spec.attr)
        except Exception:
            # Some optional quantization methods may fail to import in certain envs.
            # Preserve behavior by returning a DummyConfig for those methods, but
            # only for the compressed-tensors key (others should error when used).
            if key == "compressed-tensors":
                cls = DummyConfig  # type: ignore[assignment]
            else:
                raise
        super().__setitem__(key, cls)
        return cls

    def __contains__(self, key: object) -> bool:
        return bool(isinstance(key, str) and key in self._specs)

    def __getitem__(self, key: str):
        cls = self._load_one(key)
        if cls is None:
            raise KeyError(key)
        return cls

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs.keys())

    def keys(self) -> Iterable[str]:  # type: ignore[override]
        return self._specs.keys()

    def items(self) -> Iterable[Tuple[str, Type[QuantizationConfig]]]:  # type: ignore[override]
        for k in self._specs.keys():
            yield k, self[k]

    def values(self) -> Iterable[Type[QuantizationConfig]]:  # type: ignore[override]
        for k in self._specs.keys():
            yield self[k]

    def __len__(self) -> int:
        return len(self._specs)


QUANTIZATION_METHODS = _LazyQuantizationMethods(_specs())


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[quantization]
