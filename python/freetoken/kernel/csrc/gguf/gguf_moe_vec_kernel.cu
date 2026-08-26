// ROCm operation-split binding for GGUF routed small-batch MoE vector kernels.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "dispatch.h"
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "moe_vec.cuh"
#include "quantize_q8_1.cuh"

torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), stream);
        break;
      default:
        TORCH_CHECK(false, "ggml_moe_a8_vec: unsupported GGUF quant type ", type,
                    " (MMVQ kernels exist for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q6_K/IQ2_XXS/IQ2_XS/"
                    "IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS/IQ1_M)");
    }
  });
  return Y;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_moe_a8_vec", &ggml_moe_a8_vec, "");
}
