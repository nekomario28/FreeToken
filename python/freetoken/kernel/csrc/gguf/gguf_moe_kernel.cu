// ROCm operation-split binding for GGUF grouped large-batch MoE kernels.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "dispatch.h"
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "quantize_q8_1.cuh"

torch::Tensor ggml_moe_a8(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    using Fn = void (*)(
        const void*, const void*, scalar_t*, const int*, const int*, const int*,
        int, int, int, int, int, int, int, int, cudaStream_t);
    Fn fn = nullptr;
    switch (type) {
      case 2: fn = &ggml_moe_q4_0_q8_1_cuda<scalar_t>; break;
      case 3: fn = &ggml_moe_q4_1_q8_1_cuda<scalar_t>; break;
      case 6: fn = &ggml_moe_q5_0_q8_1_cuda<scalar_t>; break;
      case 7: fn = &ggml_moe_q5_1_q8_1_cuda<scalar_t>; break;
      case 8: fn = &ggml_moe_q8_0_q8_1_cuda<scalar_t>; break;
      case 10: fn = &ggml_moe_q2_K_q8_1_cuda<scalar_t>; break;
      case 11: fn = &ggml_moe_q3_K_q8_1_cuda<scalar_t>; break;
      case 12: fn = &ggml_moe_q4_K_q8_1_cuda<scalar_t>; break;
      case 13: fn = &ggml_moe_q5_K_q8_1_cuda<scalar_t>; break;
      case 14: fn = &ggml_moe_q6_K_q8_1_cuda<scalar_t>; break;
      default:
        TORCH_CHECK(false, "ggml_moe_a8: unsupported GGUF quant type ", type,
                    " (MMQ kernels exist only for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q6_K; "
                    "I-quants must route through ggml_dequantize)");
    }
    fn(quant_X.data_ptr(), W.data_ptr(), (scalar_t*)Y.data_ptr(),
       (int*)sorted_token_ids.data_ptr(), (int*)expert_ids.data_ptr(),
       (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row, tokens,
       padded, row, top_k, sorted_token_ids.sizes()[0], stream);
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2: return MOE_X_Q4_0;
    case 3: return MOE_X_Q4_1;
    case 6: return MOE_X_Q5_0;
    case 7: return MOE_X_Q5_1;
    case 8: return MOE_X_Q8_0;
    case 10: return MOE_X_Q2_K;
    case 11: return MOE_X_Q3_K;
    case 12: return MOE_X_Q4_K;
    case 13: return MOE_X_Q5_K;
    case 14: return MOE_X_Q6_K;
    default:
      TORCH_CHECK(false, "ggml_moe_get_block_size: unsupported GGUF quant type ", type,
                  " (MMQ kernels exist only for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q6_K; "
                  "I-quants must route through ggml_dequantize)");
      return 0;
  }
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_moe_a8", &ggml_moe_a8, "");
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size, "");
}
