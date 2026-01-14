#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

void scatter_quant_kv_fp8_cuda(torch::Tensor loc,
                               torch::Tensor k_in,
                               torch::Tensor v_in,
                               torch::Tensor k_out,
                               torch::Tensor v_out,
                               double k_scale,
                               double v_scale,
                               double fp8_max);

void scatter_quant_kv_fp8(torch::Tensor loc,
                          torch::Tensor k_in,
                          torch::Tensor v_in,
                          torch::Tensor k_out,
                          torch::Tensor v_out,
                          double k_scale,
                          double v_scale,
                          double fp8_max) {
  TORCH_CHECK(loc.is_cuda(), "loc must be CUDA");
  TORCH_CHECK(k_in.is_cuda(), "k_in must be CUDA");
  TORCH_CHECK(v_in.is_cuda(), "v_in must be CUDA");
  TORCH_CHECK(k_out.is_cuda(), "k_out must be CUDA");
  TORCH_CHECK(v_out.is_cuda(), "v_out must be CUDA");

  TORCH_CHECK(loc.dim() == 1, "loc must be 1D");
  TORCH_CHECK(k_in.dim() == 2 || k_in.dim() == 3, "k_in must be 2D or 3D");
  TORCH_CHECK(v_in.dim() == 2 || v_in.dim() == 3, "v_in must be 2D or 3D");
  TORCH_CHECK(k_out.dim() == 2, "k_out must be 2D");
  TORCH_CHECK(v_out.dim() == 2, "v_out must be 2D");

  TORCH_CHECK(k_in.scalar_type() == at::ScalarType::Half ||
                  k_in.scalar_type() == at::ScalarType::BFloat16 ||
                  k_in.scalar_type() == at::ScalarType::Float,
              "k_in must be fp16/bf16/fp32");
  TORCH_CHECK(v_in.scalar_type() == at::ScalarType::Half ||
                  v_in.scalar_type() == at::ScalarType::BFloat16 ||
                  v_in.scalar_type() == at::ScalarType::Float,
              "v_in must be fp16/bf16/fp32");

  TORCH_CHECK(loc.scalar_type() == at::ScalarType::Int ||
                  loc.scalar_type() == at::ScalarType::Long,
              "loc must be int32/int64");

  TORCH_CHECK(k_out.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                  k_out.scalar_type() == at::ScalarType::Float8_e5m2,
              "k_out must be float8 (e4m3fn/e5m2)");
  TORCH_CHECK(v_out.scalar_type() == k_out.scalar_type(),
              "v_out dtype must match k_out dtype");

  // k_in/v_in must match batch size and flattened size.
  const int64_t B = loc.numel();
  TORCH_CHECK(B > 0, "loc must be non-empty");

  int64_t N = 0;
  if (k_in.dim() == 2) {
    TORCH_CHECK(k_in.size(0) == B, "k_in batch mismatch");
    N = k_in.size(1);
  } else {
    TORCH_CHECK(k_in.size(0) == B, "k_in batch mismatch");
    N = k_in.size(1) * k_in.size(2);
  }
  int64_t N_v = 0;
  if (v_in.dim() == 2) {
    TORCH_CHECK(v_in.size(0) == B, "v_in batch mismatch");
    N_v = v_in.size(1);
  } else {
    TORCH_CHECK(v_in.size(0) == B, "v_in batch mismatch");
    N_v = v_in.size(1) * v_in.size(2);
  }
  TORCH_CHECK(N > 0, "k_in has invalid flattened size");
  TORCH_CHECK(N_v == N, "v_in flattened size mismatch");

  TORCH_CHECK(k_out.size(1) == N, "k_out width mismatch");
  TORCH_CHECK(v_out.size(1) == N, "v_out width mismatch");

  at::cuda::CUDAGuard device_guard(k_in.device());
  scatter_quant_kv_fp8_cuda(loc, k_in, v_in, k_out, v_out, k_scale, v_scale, fp8_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_quant_kv_fp8",
        &scatter_quant_kv_fp8,
        "FA3 FP8 KV scatter-quantize (K/V together) (CUDA)");
}

