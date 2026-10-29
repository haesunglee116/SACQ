#include <torch/extension.h>
std::vector<torch::Tensor> norm_modulate(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("norm_modulate",&norm_modulate);}
