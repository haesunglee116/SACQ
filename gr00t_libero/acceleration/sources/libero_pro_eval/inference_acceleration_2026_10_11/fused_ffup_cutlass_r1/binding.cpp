#include <torch/extension.h>
torch::Tensor fused_ffup(torch::Tensor a, torch::Tensor packed, torch::Tensor cb,
 torch::Tensor scales, torch::Tensor shift, torch::Tensor bias, torch::Tensor dense,
 int64_t bits, int64_t cfg);
torch::Tensor decode_ffup(torch::Tensor packed, torch::Tensor cb, torch::Tensor scales,
 torch::Tensor shift, int64_t bits);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("fused",&fused_ffup);
 m.def("decode_check",&decode_ffup);
}
