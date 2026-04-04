#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__global__ void update_token_table_kernel(
    const T* __restrict__ tokens,
    T* __restrict__ token_table,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ column_starts,
    const int32_t* __restrict__ req_lens,
    const T* __restrict__ ignore_tokens,
    const int ignore_tokens_len,
    const int64_t num_reqs) {
  const int req_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (req_idx >= num_reqs) {
    return;
  }
  const int row = row_indices[req_idx];
  const int start = column_starts[req_idx];
  const int len = req_lens[req_idx];
  for (int i = 0; i < len; ++i) {
    const T token = tokens[start + i];
    bool ignore = false;
    for (int j = 0; j < ignore_tokens_len; ++j) {
      if (token == ignore_tokens[j]) {
        ignore = true;
        break;
      }
    }
    token_table[row * len + i] = ignore ? T(-1) : token;
  }
}

template <typename T>
void update_token_table(
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView ne_token_table,
    tvm::ffi::TensorView row_indices,
    tvm::ffi::TensorView column_starts,
    tvm::ffi::TensorView req_lens,
    tvm::ffi::TensorView ignore_tokens) {
  const int64_t num_reqs = row_indices.shape(0);
  const int ignore_tokens_len = ignore_tokens.shape(0);
  const int block = 128;
  const int grid = (num_reqs + block - 1) / block;
  LaunchKernel(grid, block, tokens.device())(
      update_token_table_kernel<T>,
      static_cast<const T*>(tokens.data_ptr()),
      static_cast<T*>(ne_token_table.data_ptr()),
      static_cast<const int32_t*>(row_indices.data_ptr()),
      static_cast<const int32_t*>(column_starts.data_ptr()),
      static_cast<const int32_t*>(req_lens.data_ptr()),
      static_cast<const T*>(ignore_tokens.data_ptr()),
      ignore_tokens_len,
      num_reqs);
}

}  // namespace
