// ROCm operation-split binding for GGUF large-batch MMQ.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "dispatch.h"
#include "ggml-common.h"
#include "vecdotq.cuh"
#include "mmq.cuh"
#include "quantize_q8_1.cuh"

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,
    torch::Tensor X,
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, batch, stream);
    using Fn = void (*)(const void*, const void*, scalar_t*, int, int, int, int, int, cudaStream_t);
    Fn fn = nullptr;
    switch (type) {
      case 2: fn = &ggml_mul_mat_q4_0_q8_1_cuda<scalar_t>; break;
      case 3: fn = &ggml_mul_mat_q4_1_q8_1_cuda<scalar_t>; break;
      case 6: fn = &ggml_mul_mat_q5_0_q8_1_cuda<scalar_t>; break;
      case 7: fn = &ggml_mul_mat_q5_1_q8_1_cuda<scalar_t>; break;
      case 8: fn = &ggml_mul_mat_q8_0_q8_1_cuda<scalar_t>; break;
      case 10: fn = &ggml_mul_mat_q2_K_q8_1_cuda<scalar_t>; break;
      case 11: fn = &ggml_mul_mat_q3_K_q8_1_cuda<scalar_t>; break;
      case 12: fn = &ggml_mul_mat_q4_K_q8_1_cuda<scalar_t>; break;
      case 13: fn = &ggml_mul_mat_q5_K_q8_1_cuda<scalar_t>; break;
      case 14: fn = &ggml_mul_mat_q6_K_q8_1_cuda<scalar_t>; break;
      default:
        TORCH_CHECK(false, "ggml_mul_mat_a8: unsupported GGUF quant type ", type,
                    " (MMQ kernels exist only for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q6_K; "
                    "I-quants must route through ggml_dequantize)");
    }
    fn(W.data_ptr(), quant_X.data_ptr(), (scalar_t*)Y.data_ptr(),
       col, row, batch, padded, row, stream);
  });
  return Y;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "");
}
