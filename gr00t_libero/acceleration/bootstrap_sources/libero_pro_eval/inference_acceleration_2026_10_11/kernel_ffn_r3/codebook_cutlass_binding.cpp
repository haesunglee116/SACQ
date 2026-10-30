#include <torch/extension.h>

torch::Tensor codebook_cutlass_gemm_cuda(
    torch::Tensor x,
    torch::Tensor packed_indices,
    torch::Tensor codebook,
    torch::Tensor scales,
    torch::Tensor shift,
    torch::Tensor bias,
    int64_t index_bits,
    int64_t index_window_bytes);

torch::Tensor codebook_cutlass_scalarstore_gemm_cuda(
    torch::Tensor x,
    torch::Tensor packed_indices,
    torch::Tensor codebook,
    torch::Tensor scales,
    torch::Tensor shift,
    torch::Tensor bias,
    int64_t index_bits,
    int64_t index_window_bytes);

void set_ffn_launch_config(int64_t, int64_t, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_ffn_launch_config", &set_ffn_launch_config);
  m.def(
      "codebook_gemm",
      &codebook_cutlass_gemm_cuda,
      "Packed d4 codebook CuTe SM80 GEMM (CUDA)");
  m.def(
      "codebook_gemm_scalarstore",
      &codebook_cutlass_scalarstore_gemm_cuda,
      "Scalar-shared-store packed d4 codebook CuTe SM80 GEMM control (CUDA)");
}
