# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch

from sglang.srt.environ import envs

E2M1_MAX = 6.0
# Put constants directly on CUDA if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=_device
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=_device
)


class KVFP4QuantizeUtil:
    """Utility class for MXFP4 quantization and dequantization operations."""

    @staticmethod
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if envs.SGLANG_EXPERIMENTAL_NVFP4.get():
            return KVFP4QuantizeUtil._batched_quantize_fouroversix(tensor)
        return KVFP4QuantizeUtil._batched_quantize_standard(tensor)

    @staticmethod
    @torch.compile
    def _batched_quantize_standard(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to KVFP4 format
        Args:
            tensor: Input tensor of shape [B, M, N]

        Returns:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
        """
        b, m, n = tensor.shape

        # Reshape to [B, M*N/16, 16] for block-wise quantization
        reshaped = tensor.view(b, m * n // 16, 16)

        # Compute scale factors per block
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values
        scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
        scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)

        # Apply scaling
        scaled = reshaped / torch.exp2(scale_exp)

        # Quantize to FP4
        sign_bits = (scaled < 0).to(torch.uint8) << 3
        abs_vals = scaled.abs()

        # Pure tensor version (CUDA Graph safe)
        magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)

        # Combine sign and magnitude
        fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

        # Pack two FP4 values into one uint8
        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

        return packed, scale_factors

    @staticmethod
    @torch.compile
    def _batched_quantize_fouroversix(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Experimental: FourOverSix-inspired per-block scale selection for KV FP4.

        Keeps the **same** KV FP4 storage format (packed E2M1 nibbles + per-block
        uint8 exponent scales), but chooses between scaling a block to max=6 (default)
        vs max=4 (tighter) by comparing per-block MSE after quantize+dequant.
        """
        b, m, n = tensor.shape

        reshaped = tensor.view(b, m * n // 16, 16)
        block_max = reshaped.abs().max(dim=-1, keepdim=True).values

        def _quantize_with_scale(
            scale_exp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            scaled = reshaped / torch.exp2(scale_exp)
            sign_bits = (scaled < 0).to(torch.uint8) << 3
            abs_vals = scaled.abs()
            magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
            fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)  # [B, blocks, 16]

            sign_mask = (fp4_vals & 0x08) != 0
            magnitude_idx = fp4_vals & 0x07
            float_vals = E2M1_VALUES[magnitude_idx.long()]
            float_vals = torch.where(sign_mask, -float_vals, float_vals)
            dequant = float_vals * torch.exp2(scale_exp)

            mse = ((dequant - reshaped) ** 2).mean(dim=-1)  # [B, blocks]
            return fp4_vals, mse, scale_exp

        scale_exp_6 = torch.ceil(
            torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10))
        )
        scale_exp_4 = torch.ceil(
            torch.log2(torch.clamp(block_max / 4.0, min=1e-10))
        )

        fp4_vals_6, mse_6, _ = _quantize_with_scale(scale_exp_6)
        fp4_vals_4, mse_4, _ = _quantize_with_scale(scale_exp_4)

        select_4 = mse_4 < mse_6  # [B, blocks]
        fp4_vals = torch.where(select_4.unsqueeze(-1), fp4_vals_4, fp4_vals_6)
        scale_exp = torch.where(select_4.unsqueeze(-1), scale_exp_4, scale_exp_6)

        scale_factors = (scale_exp.squeeze(-1) + 127).to(torch.uint8)

        fp4_reshaped = fp4_vals.view(b, m, n)
        packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]
        return packed, scale_factors

    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize KVFP4 tensor
        Args:
            quant_tensor: Quantized tensor of shape [B, M, N/2]
            scale_factors: Scale factors of shape [B, M*N/16]
            dtype: Target dtype for output

        Returns:
            Dequantized tensor of shape [B, M, N]
        """
        b, m, n_half = quant_tensor.shape
        n = n_half * 2

        # More efficient unpacking using bit operations
        fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=quant_tensor.device)
        fp4_vals[..., 0::2] = quant_tensor & 0x0F
        fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

        # Extract sign and magnitude
        sign_mask = (fp4_vals & 0x08) != 0
        magnitude_idx = fp4_vals & 0x07

        # Convert to float values
        float_vals = E2M1_VALUES[magnitude_idx.long()]
        float_vals = torch.where(sign_mask, -float_vals, float_vals)

        # Reshape for block-wise scaling
        reshaped = float_vals.view(b, m * n // 16, 16)

        # Apply scale factors
        scale_exp = scale_factors.float() - 127
        scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

        return scaled.view(b, m, n).to(dtype)
