// One-layer diagnostic: original packed SACQ -> shared BF16 -> CUTLASS MMA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/gemm.h>
#include "sacq_mma_multistage.h"

namespace sacq_fused_ffup {
using E=cutlass::bfloat16_t;
constexpr int M=41,N=6144,K=1536;

// Same paired BF16 rounding as the frozen selected decoder. +0 shift is
// verified on the immutable module before this specialization is executed.
__device__ __forceinline__ uint64_t scale_code(uint64_t code,uint16_t scale) {
 uint32_t a,b,ss=uint32_t(scale)*0x10001u;
 asm volatile("{ .reg .b32 z; mov.b32 z,0;\n"
   " fma.rn.bf16x2 %0,%2,%4,z;\n"
   " fma.rn.bf16x2 %1,%3,%4,z;\n}"
   : "=r"(a),"=r"(b):"r"(uint32_t(code)),"r"(uint32_t(code>>32)),"r"(ss));
 return uint64_t(a)|(uint64_t(b)<<32);
}

struct Payload {
 const uint32_t* packed=nullptr;
 const uint64_t* cb=nullptr;
 const uint16_t* scales=nullptr;
 const E* dense=nullptr;
 int bits=0;
};

__device__ __forceinline__ unsigned index_at(Payload const& p,int g) {
 unsigned bit=unsigned(g)*unsigned(p.bits),off=bit&31;
 uint32_t low=p.packed[bit>>5],high=0;
 if(off+unsigned(p.bits)>32) high=p.packed[(bit>>5)+1];
 return __funnelshift_r(low,high,off)&((1u<<p.bits)-1);
}

// The original CUTLASS B thread map is reused directly in logical K,N
// coordinates. No out-of-allocation or fake dense pointer arithmetic.
template<class Original,int Mode>
class CodebookIterator {
 public:
 using Shape=typename Original::Shape;
 using Element=typename Original::Element;
 using Layout=typename Original::Layout;
 using ThreadMap=typename Original::ThreadMap;
 using AccessType=typename Original::AccessType;
 using TensorRef=typename Original::TensorRef;
 using TensorCoord=typename Original::TensorCoord;
 static int const kAccessesPerVector=Original::kAccessesPerVector;
 struct Params {
   Payload payload;
   CUTLASS_HOST_DEVICE Params()=default;
   CUTLASS_HOST_DEVICE Params(Layout const&) {}
 };
 private:
 Payload p_;
 TensorCoord tile_,extent_;
 int initial_k_,initial_n_,iteration_=0;
 bool enabled_=true;
 uint64_t table0_=0,table1_=0;
 public:
 CUTLASS_DEVICE CodebookIterator(Params const& params,Element*,TensorCoord extent,
     int thread,TensorCoord tile,int const* =nullptr)
   :p_(params.payload),tile_(tile),extent_(extent) {
   static_assert(AccessType::kElements==8,"One access is two original d4 codewords");
   static_assert(K%Shape::kRow==0 && N%Shape::kColumn==0,"Approved whole tiles only");
   auto initial=ThreadMap::initial_offset(thread);
   initial_k_=initial.contiguous();initial_n_=initial.strided();
   if constexpr(Mode==2) {
     int lane=thread&31;
     table0_=p_.cb[lane];
     if(p_.bits==6) table1_=p_.cb[lane+32];
   }
 }
 CUTLASS_DEVICE void set_iteration_index(int i) {iteration_=i;}
 CUTLASS_DEVICE CodebookIterator& operator++() {++iteration_;return *this;}
 CUTLASS_DEVICE void add_tile_offset(TensorCoord off) {
   tile_.row()+=off.row()*Shape::kRow;
   tile_.column()+=off.column()*Shape::kColumn;
 }
 CUTLASS_DEVICE void clear_mask(bool clear=true) {if(clear) enabled_=false;}
 CUTLASS_DEVICE uint64_t lookup(unsigned index) const {
   if constexpr(Mode==2) {
     uint32_t lo=__shfl_sync(0xffffffff,uint32_t(table0_),index&31);
     uint32_t hi=__shfl_sync(0xffffffff,uint32_t(table0_>>32),index&31);
     if(p_.bits==6) {
       uint32_t lo1=__shfl_sync(0xffffffff,uint32_t(table1_),index&31);
       uint32_t hi1=__shfl_sync(0xffffffff,uint32_t(table1_>>32),index&31);
       if(index&32) {lo=lo1;hi=hi1;}
     }
     return uint64_t(lo)|(uint64_t(hi)<<32);
   } else return p_.cb[index];
 }
 CUTLASS_DEVICE void store(AccessType* destination) const {
   int v=iteration_%kAccessesPerVector;
   int residual=iteration_/kAccessesPerVector;
   int c=residual%ThreadMap::Iterations::kContiguous;
   int s=residual/ThreadMap::Iterations::kContiguous;
   int k=tile_.row()+initial_k_+c*ThreadMap::Delta::kContiguous+v*AccessType::kElements;
   int n=tile_.column()+initial_n_+s*ThreadMap::Delta::kStrided;
   bool valid=enabled_ && k<extent_.row() && n<extent_.column();
   AccessType fragment;
   if constexpr(Mode==0) {
     fragment.clear();
     if(valid) fragment=*reinterpret_cast<AccessType const*>(p_.dense+n*K+k);
   } else {
     int g=(n*K+k)/4;
     unsigned i0=0,i1=0;uint16_t scale=0;
     if(valid) {i0=index_at(p_,g);i1=index_at(p_,g+1);scale=p_.scales[g>>4];}
     // All lanes must reach register LUT shuffles, including masked stages.
     auto lo=scale_code(lookup(i0),scale),hi=scale_code(lookup(i1),scale);
     auto words=reinterpret_cast<uint64_t*>(&fragment);
     words[0]=valid?lo:0;words[1]=valid?hi:0;
   }
   *destination=fragment;
 }
};

template<int BN,int BK,int WN,int Stages,int Mode>
void launch(torch::Tensor a,torch::Tensor bias,torch::Tensor y,Payload payload,cudaStream_t stream) {
 using Output=cutlass::epilogue::thread::LinearCombination<E,(BN==32?4:8),float,float>;
 using Dense=cutlass::gemm::device::Gemm<E,cutlass::layout::RowMajor,E,cutlass::layout::ColumnMajor,
   E,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
   cutlass::gemm::GemmShape<64,BN,BK>,cutlass::gemm::GemmShape<32,WN,BK>,
   cutlass::gemm::GemmShape<16,8,16>,Output,
   cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,Stages,8,8>;
 using Original=typename Dense::GemmKernel::Mma;
 using B=CodebookIterator<typename Original::IteratorB,Mode>;
 using Mma=cutlass::gemm::threadblock::SacqMmaMultistage<typename Original::Shape,
   typename Original::IteratorA,typename Original::SmemIteratorA,Original::kCacheOpA,
   B,typename Original::SmemIteratorB,Original::kCacheOpB,float,cutlass::layout::RowMajor,
   typename Original::Policy,Stages>;
 using Kernel=cutlass::gemm::kernel::Gemm<Mma,typename Dense::GemmKernel::Epilogue,
   typename Dense::ThreadblockSwizzle,false>;
 bool has_bias=bias.numel()!=0;
 E* source=has_bias?reinterpret_cast<E*>(bias.data_ptr()):reinterpret_cast<E*>(y.data_ptr());
 typename Kernel::Params params({M,N,K},{1,N/BN,1},
   {reinterpret_cast<E*>(a.data_ptr()),K},
   {nullptr,K},{source,has_bias?0:N},{reinterpret_cast<E*>(y.data_ptr()),N},
   {1.f,has_bias?1.f:0.f});
 params.params_B.payload=payload;
 int shared=int(sizeof(typename Kernel::SharedStorage));
 if(shared>=48*1024) C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
   cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
 cutlass::Kernel<Kernel><<<dim3(1,N/BN,1),Kernel::kThreadCount,shared,stream>>>(params);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void check_decode(Payload p,uint64_t* out) {
 int g=blockIdx.x*blockDim.x+threadIdx.x;
 if(g<N*K/4) out[g]=scale_code(p.cb[index_at(p,g)],p.scales[g>>4]);
}

Payload payload(torch::Tensor packed,torch::Tensor cb,torch::Tensor scales,int64_t bits) {
 TORCH_CHECK(packed.is_cuda()&&cb.is_cuda()&&scales.is_cuda());
 TORCH_CHECK(packed.device()==cb.device()&&cb.device()==scales.device());
 TORCH_CHECK(packed.is_contiguous()&&cb.is_contiguous()&&scales.is_contiguous());
 TORCH_CHECK(packed.scalar_type()==at::kByte&&cb.scalar_type()==at::kBFloat16&&scales.scalar_type()==at::kBFloat16);
 TORCH_CHECK(bits>0&&bits<=16&&cb.numel()==(int64_t(1)<<bits)*4);
 TORCH_CHECK(packed.numel()>=int64_t(N)*K/4*bits/8&&scales.numel()==N*K/64);
 return {reinterpret_cast<uint32_t const*>(packed.data_ptr()),reinterpret_cast<uint64_t const*>(cb.data_ptr()),
   reinterpret_cast<uint16_t const*>(scales.data_ptr()),nullptr,int(bits)};
}
}

torch::Tensor fused_ffup(torch::Tensor a,torch::Tensor packed,torch::Tensor cb,torch::Tensor scales,
 torch::Tensor shift,torch::Tensor bias,torch::Tensor dense,int64_t bits,int64_t cfg) {
 using namespace sacq_fused_ffup;
 c10::cuda::CUDAGuard guard(a.device());
 auto p=payload(packed,cb,scales,bits);
 TORCH_CHECK(a.is_cuda()&&a.device()==cb.device()&&a.is_contiguous()&&a.scalar_type()==at::kBFloat16);
 TORCH_CHECK(a.dim()==2&&a.size(0)==M&&a.size(1)==K);
 TORCH_CHECK(shift.is_cuda()&&shift.device()==a.device()&&shift.scalar_type()==at::kBFloat16&&shift.numel()==1);
 TORCH_CHECK(bias.is_cuda()&&bias.device()==a.device()&&bias.is_contiguous()&&bias.scalar_type()==at::kBFloat16&&bias.numel()==N);
 TORCH_CHECK(cfg>=0&&cfg<=7);
 if(cfg==0) {
   TORCH_CHECK(dense.is_cuda()&&dense.device()==a.device()&&dense.is_contiguous()&&dense.scalar_type()==at::kBFloat16&&dense.numel()==N*K);
   p.dense=reinterpret_cast<E const*>(dense.data_ptr());
 }
 if(cfg==2||cfg==4||cfg==5||cfg==7) TORCH_CHECK(bits==5||bits==6,"Register LUT supports32/64entries only");
 auto y=torch::empty({M,N},a.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 switch(cfg) {
   case 0:launch<64,32,32,6,0>(a,bias,y,p,stream);break;
   case 1:launch<64,32,32,6,1>(a,bias,y,p,stream);break;
   case 2:launch<64,32,32,6,2>(a,bias,y,p,stream);break;
   case 3:launch<64,32,32,3,1>(a,bias,y,p,stream);break;
   case 4:launch<64,32,32,3,2>(a,bias,y,p,stream);break;
   case 5:launch<64,64,32,3,2>(a,bias,y,p,stream);break;
   case 6:launch<32,64,16,3,1>(a,bias,y,p,stream);break;
   case 7:launch<32,64,16,3,2>(a,bias,y,p,stream);break;
 }
 return y;
}

torch::Tensor decode_ffup(torch::Tensor packed,torch::Tensor cb,torch::Tensor scales,torch::Tensor shift,int64_t bits) {
 using namespace sacq_fused_ffup;
 c10::cuda::CUDAGuard guard(cb.device());
 auto p=payload(packed,cb,scales,bits);
 auto out=torch::empty({N,K},cb.options());
 check_decode<<<(N*K/4+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(p,reinterpret_cast<uint64_t*>(out.data_ptr()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
