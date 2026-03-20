"""Scheme registry for compressed_tensors support.

The module keeps optional imports defensive so environments with partial
flashinfer builds can still boot SGLang for unsupported MXINT4 MoE schemes.
"""

from __future__ import annotations

import logging

from .compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
)
from .compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4
from .compressed_tensors_w4a4_nvfp4_moe import CompressedTensorsW4A4Nvfp4MoE
from .compressed_tensors_w4a8_int8_moe import NPUCompressedTensorsW4A8Int8DynamicMoE
from .compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from .compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE
from .compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
    NPUCompressedTensorsW8A8Int8,
)
from .compressed_tensors_w8a8_int8_moe import NPUCompressedTensorsW8A8Int8DynamicMoE
from .compressed_tensors_w8a16_fp8 import CompressedTensorsW8A16Fp8
from .compressed_tensors_wNa16 import WNA16_SUPPORTED_BITS, CompressedTensorsWNA16
from .compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
    NPUCompressedTensorsW4A16Int4DynamicMoE,
)

_logger = logging.getLogger(__name__)

try:
    from .compressed_tensors_w4a4_mxint4_moe import CompressedTensorsMxInt4MoE
except Exception as exc:  # pragma: no cover - environment dependent
    _logger.warning(
        "Skipping CompressedTensorsMxInt4MoE import because required flashinfer symbols are unavailable: %s",
        exc,
    )

    class CompressedTensorsMxInt4MoE(CompressedTensorsMoEScheme):
        """Fallback when MXINT4 MoE backend is unavailable in this runtime."""

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "CompressedTensorsMxInt4MoE is unavailable because flashinfer fused_moe "
                "is missing mxint4 symbols. Falling back to unquantized MoE is required "
                "to continue serving this checkpoint in this environment."
            )


__all__ = [
    "CompressedTensorsLinearScheme",
    "CompressedTensorsMoEScheme",
    "CompressedTensorsW8A8Fp8",
    "CompressedTensorsW8A8Fp8MoE",
    "CompressedTensorsW8A16Fp8",
    "CompressedTensorsW8A8Int8",
    "NPUCompressedTensorsW8A8Int8",
    "NPUCompressedTensorsW8A8Int8DynamicMoE",
    "CompressedTensorsWNA16",
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
    "NPUCompressedTensorsW4A16Int4DynamicMoE",
    "WNA16_SUPPORTED_BITS",
    "CompressedTensorsW4A4Fp4",
    "CompressedTensorsW4A4Nvfp4MoE",
    "NPUCompressedTensorsW4A8Int8DynamicMoE",
    "CompressedTensorsMxInt4MoE",
]
