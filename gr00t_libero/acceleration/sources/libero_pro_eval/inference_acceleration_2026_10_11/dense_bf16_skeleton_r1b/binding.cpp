#include <torch/extension.h>
torch::Tensor dense_bf16(torch::Tensor a, torch::Tensor w, torch::Tensor bias, int64_t cfg);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense", &dense_bf16, "Diagnostic BF16 CUTLASS dense-only GEMM");
}
