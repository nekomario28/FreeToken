// ROCm operation-split binding for the vendored GGUF dequant kernels.
// Kernel implementations remain in the sgl-kernel/llama.cpp-derived headers.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "dispatch.h"
#include "ggml-common.h"
#include "dequantize.cuh"

torch::Tensor ggml_dequantize(
    torch::Tensor W,
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    TORCH_CHECK(
        to_cuda != nullptr,
        "ggml_dequantize: unsupported GGUF quant type ", type,
        " (dequant kernels exist for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K-Q6_K/IQ2_XXS/"
        "IQ2_XS/IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS/IQ1_M)");
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "");
}
