#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__device__ __forceinline__ float _to_float(float v) { return v; }
__device__ __forceinline__ float _to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float _to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T _fp8_cast(float v);

template <>
__device__ __forceinline__ __nv_fp8_e4m3 _fp8_cast<__nv_fp8_e4m3>(float v) {
  return static_cast<__nv_fp8_e4m3>(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 _fp8_cast<__nv_fp8_e5m2>(float v) {
  return static_cast<__nv_fp8_e5m2>(v);
}

template <typename QType, typename Fp8Type>
__global__ void quantize_q_fp8_fixed_stride_kernel(const QType* __restrict__ q,
                                                   Fp8Type* __restrict__ out_q,
                                                   float* __restrict__ out_descale,
                                                   int32_t batch,
                                                   int32_t max_seqlen_q,
                                                   int32_t q_heads,
                                                   int32_t kv_heads,
                                                   int32_t head_dim,
                                                   float fp8_max) {
  const int32_t block = static_cast<int32_t>(blockIdx.x);
  const int32_t hk = block % kv_heads;
  const int32_t b = block / kv_heads;
  if (b >= batch) return;

  const int32_t group = q_heads / kv_heads;
  const int32_t elems_per_hk = max_seqlen_q * group * head_dim;

  float local_max = 0.f;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < elems_per_hk; i += blockDim.x) {
    const int32_t token = i / (group * head_dim);
    const int32_t rem = i - token * group * head_dim;
    const int32_t g = rem / head_dim;
    const int32_t d = rem - g * head_dim;
    const int32_t q_head = hk * group + g;
    const int32_t t = b * max_seqlen_q + token;
    const int64_t idx = (static_cast<int64_t>(t) * q_heads + q_head) * head_dim + d;
    const float v = fabsf(_to_float(q[idx]));
    local_max = fmaxf(local_max, v);
  }

  __shared__ float sdata[256];
  sdata[threadIdx.x] = local_max;
  __syncthreads();

  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float maxabs = sdata[0];
  const float descale = (maxabs > 0.f) ? (maxabs / fp8_max) : 0.f;
  if (threadIdx.x == 0) {
    out_descale[b * kv_heads + hk] = descale;
  }
  __syncthreads();

  const float inv_descale = (descale > 0.f) ? (1.f / descale) : 0.f;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < elems_per_hk; i += blockDim.x) {
    const int32_t token = i / (group * head_dim);
    const int32_t rem = i - token * group * head_dim;
    const int32_t g = rem / head_dim;
    const int32_t d = rem - g * head_dim;
    const int32_t q_head = hk * group + g;
    const int32_t t = b * max_seqlen_q + token;
    const int64_t idx = (static_cast<int64_t>(t) * q_heads + q_head) * head_dim + d;

    float v = _to_float(q[idx]) * inv_descale;
    v = fmaxf(-fp8_max, fminf(v, fp8_max));
    out_q[idx] = _fp8_cast<Fp8Type>(v);
  }
}

}  // namespace

void quantize_q_fp8_fixed_stride_cuda(torch::Tensor q,
                                     torch::Tensor out_q,
                                     torch::Tensor out_descale,
                                     int64_t max_seqlen_q,
                                     int64_t q_heads,
                                     int64_t kv_heads,
                                     double fp8_max) {
  const auto stream = c10::cuda::getDefaultCUDAStream(q.device().index());
  const int64_t total_q = q.size(0);
  const int64_t head_dim = q.size(2);
  TORCH_CHECK(head_dim > 0 && head_dim <= 256, "head_dim must be in (0,256]");
  TORCH_CHECK(total_q % max_seqlen_q == 0, "total_q must be divisible by max_seqlen_q");

  const int32_t batch = static_cast<int32_t>(total_q / max_seqlen_q);
  TORCH_CHECK(out_descale.size(0) == batch, "out_descale batch mismatch");
  TORCH_CHECK(out_descale.size(1) == kv_heads, "out_descale kv_heads mismatch");

  const int threads = 256;
  const int blocks = batch * static_cast<int32_t>(kv_heads);
  const float fp8_max_f = static_cast<float>(fp8_max);

  const auto q_type = q.scalar_type();
  const auto out_q_type = out_q.scalar_type();

  if (q_type == at::ScalarType::BFloat16) {
    const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    if (out_q_type == at::ScalarType::Float8_e4m3fn) {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<__nv_bfloat16, __nv_fp8_e4m3>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    } else {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e5m2*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<__nv_bfloat16, __nv_fp8_e5m2>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    }
  } else if (q_type == at::ScalarType::Half) {
    const auto* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
    if (out_q_type == at::ScalarType::Float8_e4m3fn) {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<__half, __nv_fp8_e4m3>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    } else {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e5m2*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<__half, __nv_fp8_e5m2>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    }
  } else if (q_type == at::ScalarType::Float) {
    const auto* q_ptr = static_cast<const float*>(q.data_ptr());
    if (out_q_type == at::ScalarType::Float8_e4m3fn) {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<float, __nv_fp8_e4m3>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    } else {
      auto* out_ptr = reinterpret_cast<__nv_fp8_e5m2*>(out_q.data_ptr());
      quantize_q_fp8_fixed_stride_kernel<float, __nv_fp8_e5m2>
          <<<blocks, threads, 0, stream>>>(
              q_ptr,
              out_ptr,
              static_cast<float*>(out_descale.data_ptr()),
              batch,
              static_cast<int32_t>(max_seqlen_q),
              static_cast<int32_t>(q_heads),
              static_cast<int32_t>(kv_heads),
              static_cast<int32_t>(head_dim),
              fp8_max_f);
    }
  } else {
    TORCH_CHECK(false, "Unsupported q dtype");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
