// Uses the public CUTLASS device::Gemm API; no quantization or dequantization.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace sacq_dense_skeleton {
using E=cutlass::bfloat16_t;
template<int Elements> using Output=cutlass::epilogue::thread::LinearCombination<E,Elements,float,float>;
template<int BM,int BN,int BK,int WM,int WN,int Stages>
using Gemm=cutlass::gemm::device::Gemm<
  E,cutlass::layout::RowMajor,E,cutlass::layout::ColumnMajor,E,cutlass::layout::RowMajor,
  float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<BM,BN,BK>,cutlass::gemm::GemmShape<WM,WN,BK>,
  cutlass::gemm::GemmShape<16,8,16>,Output<(BN==32?4:8)>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,Stages,8,8>;

template<class G>
void launch(torch::Tensor a,torch::Tensor w,torch::Tensor bias,torch::Tensor y,cudaStream_t stream) {
  const int M=a.size(0),N=w.size(0),K=a.size(1);
  const bool has_bias=bias.numel()!=0;
  E* source=has_bias?reinterpret_cast<E*>(bias.data_ptr()):reinterpret_cast<E*>(y.data_ptr());
  typename G::Arguments args({M,N,K},
    {reinterpret_cast<E*>(a.data_ptr()),K},{reinterpret_cast<E*>(w.data_ptr()),K},
    {source,has_bias?0:N},{reinterpret_cast<E*>(y.data_ptr()),N},
    {1.f,has_bias?1.f:0.f},1);
  G op;
  auto status=op.can_implement(args);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS can_implement: ",cutlassGetStatusString(status));
  TORCH_CHECK(G::get_workspace_size(args)==0,"Unexpected dense skeleton workspace");
  status=op(args,nullptr,stream);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS run: ",cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

torch::Tensor dense_bf16(torch::Tensor a,torch::Tensor w,torch::Tensor bias,int64_t cfg) {
  using namespace sacq_dense_skeleton;
  c10::cuda::CUDAGuard guard(a.device());
  for(auto t:{a,w,bias})
    TORCH_CHECK(t.is_cuda()&&t.device()==a.device()&&t.is_contiguous(),"CUDA contiguous same-device operands required");
  TORCH_CHECK(a.scalar_type()==at::kBFloat16&&w.scalar_type()==at::kBFloat16,
              "Original BF16 operands only; no precision conversion");
  TORCH_CHECK(a.dim()==2&&w.dim()==2&&a.size(0)==41&&a.size(1)==1536&&w.size(0)==6144&&w.size(1)==1536,
              "Only approved actual M41,N6144,K1536 geometry");
  TORCH_CHECK(bias.numel()==0||(bias.scalar_type()==at::kBFloat16&&bias.numel()==w.size(0)),"Invalid BF16 bias");
  TORCH_CHECK(cfg>=0&&cfg<=7,"Invalid dense configuration");
  auto y=torch::empty({a.size(0),w.size(0)},a.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  switch(cfg) {
    case 0:launch<Gemm<64,64,32,32,32,6>>(a,w,bias,y,stream);break;
    case 1:launch<Gemm<64,64,32,32,32,3>>(a,w,bias,y,stream);break;
    case 2:launch<Gemm<64,128,32,32,64,3>>(a,w,bias,y,stream);break;
    case 3:launch<Gemm<64,32,32,32,16,3>>(a,w,bias,y,stream);break;
    case 4:launch<Gemm<64,64,64,32,32,3>>(a,w,bias,y,stream);break;
    case 5:launch<Gemm<32,128,32,16,64,3>>(a,w,bias,y,stream);break;
    case 6:launch<Gemm<32,64,64,16,32,3>>(a,w,bias,y,stream);break;
    case 7:launch<cutlass::gemm::device::Gemm<E,cutlass::layout::RowMajor,E,cutlass::layout::ColumnMajor,
               E,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80>>
               (a,w,bias,y,stream);break;
  }
  return y;
}
