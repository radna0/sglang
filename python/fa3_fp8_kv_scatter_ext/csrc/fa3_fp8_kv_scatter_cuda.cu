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

template <typename Fp8Type>
__device__ __forceinline__ Fp8Type _fp8_cast(float v);

template <>
__device__ __forceinline__ __nv_fp8_e4m3 _fp8_cast<__nv_fp8_e4m3>(float v) {
  return static_cast<__nv_fp8_e4m3>(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 _fp8_cast<__nv_fp8_e5m2>(float v) {
  return static_cast<__nv_fp8_e5m2>(v);
}

template <typename IdType>
__device__ __forceinline__ int64_t _load_loc(const IdType* __restrict__ loc,
                                             int64_t b,
                                             int64_t stride0) {
  return static_cast<int64_t>(loc[b * stride0]);
}

template <typename InType, typename OutFp8Type, typename IdType>
__global__ void scatter_quant_kv_fp8_kernel_2d_by_b(const IdType* __restrict__ loc,
                                                    int64_t loc_stride0,
                                                    const InType* __restrict__ k_in,
                                                    int64_t k_stride0,
                                                    int64_t k_stride1,
                                                    const InType* __restrict__ v_in,
                                                    int64_t v_stride0,
                                                    int64_t v_stride1,
                                                    OutFp8Type* __restrict__ k_out,
                                                    int64_t k_out_stride0,
                                                    int64_t k_out_stride1,
                                                    OutFp8Type* __restrict__ v_out,
                                                    int64_t v_out_stride0,
                                                    int64_t v_out_stride1,
                                                    int64_t B,
                                                    int64_t N,
                                                    float inv_k_scale,
                                                    float inv_v_scale,
                                                    float fp8_max) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  if (b >= B) return;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (j >= N) return;

  __shared__ int64_t s_row;
  if (threadIdx.x == 0) {
    s_row = _load_loc(loc, b, loc_stride0);
  }
  __syncthreads();
  const int64_t row = s_row;

  const int64_t k_src = b * k_stride0 + j * k_stride1;
  const int64_t v_src = b * v_stride0 + j * v_stride1;
  float k_val = _to_float(k_in[k_src]) * inv_k_scale;
  float v_val = _to_float(v_in[v_src]) * inv_v_scale;
  k_val = fmaxf(-fp8_max, fminf(k_val, fp8_max));
  v_val = fmaxf(-fp8_max, fminf(v_val, fp8_max));

  const int64_t k_dst = row * k_out_stride0 + j * k_out_stride1;
  const int64_t v_dst = row * v_out_stride0 + j * v_out_stride1;
  k_out[k_dst] = _fp8_cast<OutFp8Type>(k_val);
  v_out[v_dst] = _fp8_cast<OutFp8Type>(v_val);
}

template <typename InType, typename OutFp8Type, typename IdType>
__global__ void scatter_quant_kv_fp8_kernel_3d_by_b(const IdType* __restrict__ loc,
                                                    int64_t loc_stride0,
                                                    const InType* __restrict__ k_in,
                                                    int64_t k_stride0,
                                                    int64_t k_stride1,
                                                    int64_t k_stride2,
                                                    const InType* __restrict__ v_in,
                                                    int64_t v_stride0,
                                                    int64_t v_stride1,
                                                    int64_t v_stride2,
                                                    OutFp8Type* __restrict__ k_out,
                                                    int64_t k_out_stride0,
                                                    int64_t k_out_stride1,
                                                    OutFp8Type* __restrict__ v_out,
                                                    int64_t v_out_stride0,
                                                    int64_t v_out_stride1,
                                                    int64_t B,
                                                    int64_t H,
                                                    int64_t D,
                                                    float inv_k_scale,
                                                    float inv_v_scale,
                                                    float fp8_max) {
  const int64_t b = static_cast<int64_t>(blockIdx.x);
  if (b >= B) return;
  const int64_t N = H * D;
  const int64_t j = static_cast<int64_t>(blockIdx.y) * blockDim.x +
                    static_cast<int64_t>(threadIdx.x);
  if (j >= N) return;

  __shared__ int64_t s_row;
  if (threadIdx.x == 0) {
    s_row = _load_loc(loc, b, loc_stride0);
  }
  __syncthreads();
  const int64_t row = s_row;

  const int64_t h = j / D;
  const int64_t d = j - h * D;

  const int64_t k_src = b * k_stride0 + h * k_stride1 + d * k_stride2;
  const int64_t v_src = b * v_stride0 + h * v_stride1 + d * v_stride2;
  float k_val = _to_float(k_in[k_src]) * inv_k_scale;
  float v_val = _to_float(v_in[v_src]) * inv_v_scale;
  k_val = fmaxf(-fp8_max, fminf(k_val, fp8_max));
  v_val = fmaxf(-fp8_max, fminf(v_val, fp8_max));

  const int64_t k_dst = row * k_out_stride0 + j * k_out_stride1;
  const int64_t v_dst = row * v_out_stride0 + j * v_out_stride1;
  k_out[k_dst] = _fp8_cast<OutFp8Type>(k_val);
  v_out[v_dst] = _fp8_cast<OutFp8Type>(v_val);
}

}  // namespace

void scatter_quant_kv_fp8_cuda(torch::Tensor loc,
                               torch::Tensor k_in,
                               torch::Tensor v_in,
                               torch::Tensor k_out,
                               torch::Tensor v_out,
                               double k_scale,
                               double v_scale,
                               double fp8_max) {
  const auto stream = c10::cuda::getDefaultCUDAStream(k_in.device().index());

  const int64_t B = loc.numel();
  const int64_t out_dtype = static_cast<int64_t>(k_out.scalar_type());
  const float fp8_max_f = static_cast<float>(fp8_max);
  const float inv_k_scale =
      (k_scale > 0.0) ? (1.0f / static_cast<float>(k_scale)) : 0.0f;
  const float inv_v_scale =
      (v_scale > 0.0) ? (1.0f / static_cast<float>(v_scale)) : 0.0f;

  const int threads = 256;

  const int64_t loc_stride0 = loc.stride(0);

  const auto k_type = k_in.scalar_type();
  TORCH_CHECK(v_in.scalar_type() == k_type, "k_in/v_in dtype mismatch");

  auto launch = [&](auto dummy_in, auto dummy_fp8, auto dummy_id) {
    using InType = decltype(dummy_in);
    using OutFp8Type = decltype(dummy_fp8);
    using IdType = decltype(dummy_id);

    const IdType* loc_ptr = reinterpret_cast<const IdType*>(loc.data_ptr());
    const InType* k_ptr = reinterpret_cast<const InType*>(k_in.data_ptr());
    const InType* v_ptr = reinterpret_cast<const InType*>(v_in.data_ptr());
    OutFp8Type* k_out_ptr = reinterpret_cast<OutFp8Type*>(k_out.data_ptr());
    OutFp8Type* v_out_ptr = reinterpret_cast<OutFp8Type*>(v_out.data_ptr());

    const int64_t k_out_stride0 = k_out.stride(0);
    const int64_t k_out_stride1 = k_out.stride(1);
    const int64_t v_out_stride0 = v_out.stride(0);
    const int64_t v_out_stride1 = v_out.stride(1);

    if (k_in.dim() == 2) {
      const int64_t N = k_in.size(1);
      const int grid_y = static_cast<int>((N + threads - 1) / threads);
      const dim3 grid(static_cast<unsigned int>(B), static_cast<unsigned int>(grid_y), 1);
      scatter_quant_kv_fp8_kernel_2d_by_b<InType, OutFp8Type, IdType>
          <<<grid, threads, 0, stream>>>(
              loc_ptr,
              loc_stride0,
              k_ptr,
              k_in.stride(0),
              k_in.stride(1),
              v_ptr,
              v_in.stride(0),
              v_in.stride(1),
              k_out_ptr,
              k_out_stride0,
              k_out_stride1,
              v_out_ptr,
              v_out_stride0,
              v_out_stride1,
              B,
              N,
              inv_k_scale,
              inv_v_scale,
              fp8_max_f);
    } else {
      const int64_t H = k_in.size(1);
      const int64_t D = k_in.size(2);
      const int64_t N = H * D;
      const int grid_y = static_cast<int>((N + threads - 1) / threads);
      const dim3 grid(static_cast<unsigned int>(B), static_cast<unsigned int>(grid_y), 1);

      const bool flatten_ok = (k_in.stride(2) == 1) && (v_in.stride(2) == 1) &&
                              (k_in.stride(1) == D) && (v_in.stride(1) == D);
      if (flatten_ok) {
        // Fast path: contiguous (B,H,D) layout can be treated as (B, H*D), avoiding
        // per-element div/mod in the 3D kernel.
        scatter_quant_kv_fp8_kernel_2d_by_b<InType, OutFp8Type, IdType>
            <<<grid, threads, 0, stream>>>(
                loc_ptr,
                loc_stride0,
                k_ptr,
                k_in.stride(0),
                /*k_stride1=*/1,
                v_ptr,
                v_in.stride(0),
                /*v_stride1=*/1,
                k_out_ptr,
                k_out_stride0,
                k_out_stride1,
                v_out_ptr,
                v_out_stride0,
                v_out_stride1,
                B,
                N,
                inv_k_scale,
                inv_v_scale,
                fp8_max_f);
      } else {
        scatter_quant_kv_fp8_kernel_3d_by_b<InType, OutFp8Type, IdType>
            <<<grid, threads, 0, stream>>>(
                loc_ptr,
                loc_stride0,
                k_ptr,
                k_in.stride(0),
                k_in.stride(1),
                k_in.stride(2),
                v_ptr,
                v_in.stride(0),
                v_in.stride(1),
                v_in.stride(2),
                k_out_ptr,
                k_out_stride0,
                k_out_stride1,
                v_out_ptr,
                v_out_stride0,
                v_out_stride1,
                B,
                H,
                D,
                inv_k_scale,
                inv_v_scale,
                fp8_max_f);
      }
    }
  };

  if (loc.scalar_type() == at::ScalarType::Int) {
    if (out_dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
      if (k_type == at::ScalarType::BFloat16) {
        launch(__nv_bfloat16{}, __nv_fp8_e4m3{}, int32_t{});
      } else if (k_type == at::ScalarType::Half) {
        launch(__half{}, __nv_fp8_e4m3{}, int32_t{});
      } else {
        launch(float{}, __nv_fp8_e4m3{}, int32_t{});
      }
    } else {
      if (k_type == at::ScalarType::BFloat16) {
        launch(__nv_bfloat16{}, __nv_fp8_e5m2{}, int32_t{});
      } else if (k_type == at::ScalarType::Half) {
        launch(__half{}, __nv_fp8_e5m2{}, int32_t{});
      } else {
        launch(float{}, __nv_fp8_e5m2{}, int32_t{});
      }
    }
  } else {
    if (out_dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
      if (k_type == at::ScalarType::BFloat16) {
        launch(__nv_bfloat16{}, __nv_fp8_e4m3{}, int64_t{});
      } else if (k_type == at::ScalarType::Half) {
        launch(__half{}, __nv_fp8_e4m3{}, int64_t{});
      } else {
        launch(float{}, __nv_fp8_e4m3{}, int64_t{});
      }
    } else {
      if (k_type == at::ScalarType::BFloat16) {
        launch(__nv_bfloat16{}, __nv_fp8_e5m2{}, int64_t{});
      } else if (k_type == at::ScalarType::Half) {
        launch(__half{}, __nv_fp8_e5m2{}, int64_t{});
      } else {
        launch(float{}, __nv_fp8_e5m2{}, int64_t{});
      }
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
