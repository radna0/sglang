#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

void quantize_q_fp8_fixed_stride_cuda(torch::Tensor q,
                                     torch::Tensor out_q,
                                     torch::Tensor out_descale,
                                     int64_t max_seqlen_q,
                                     int64_t q_heads,
                                     int64_t kv_heads,
                                     double fp8_max);

void quantize_q_fp8_fixed_stride(torch::Tensor q,
                                torch::Tensor out_q,
                                torch::Tensor out_descale,
                                int64_t max_seqlen_q,
                                int64_t q_heads,
                                int64_t kv_heads,
                                double fp8_max) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(out_q.is_cuda(), "out_q must be CUDA");
  TORCH_CHECK(out_descale.is_cuda(), "out_descale must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
  TORCH_CHECK(out_descale.is_contiguous(), "out_descale must be contiguous");

  TORCH_CHECK(q.dim() == 3, "q must be (total_q, q_heads, head_dim)");
  TORCH_CHECK(out_q.sizes() == q.sizes(), "out_q must match q shape");

  TORCH_CHECK(out_descale.dim() == 2, "out_descale must be (batch, kv_heads)");
  TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q must be > 0");
  TORCH_CHECK(q_heads > 0 && kv_heads > 0, "q_heads/kv_heads must be > 0");
  TORCH_CHECK(q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads");

  const auto out_q_type = out_q.scalar_type();
  TORCH_CHECK(out_q_type == at::ScalarType::Float8_e4m3fn ||
                  out_q_type == at::ScalarType::Float8_e5m2,
              "out_q must be float8 (e4m3fn/e5m2)");
  TORCH_CHECK(out_descale.scalar_type() == at::ScalarType::Float,
              "out_descale must be float32");

  const auto q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 ||
                  q_type == at::ScalarType::Float,
              "q must be fp16/bf16/fp32");

  at::cuda::CUDAGuard device_guard(q.device());
  quantize_q_fp8_fixed_stride_cuda(
      q, out_q, out_descale, max_seqlen_q, q_heads, kv_heads, fp8_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_q_fp8_fixed_stride",
        &quantize_q_fp8_fixed_stride,
        "FA3 FP8 Q quantize (fixed stride) (CUDA)");
}
