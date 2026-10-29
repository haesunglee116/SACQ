// Welford implementation adapted from PyTorch v2.11.0, BSD-3-Clause.
// See upstream LICENSE copied beside this source and pinned primary-source receipt.
#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/thread_constants.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace codex_adaln_sharedmax {
using at::acc_type;
constexpr int vec_size=4;
template<typename scalar_t,int V>struct alignas(sizeof(scalar_t)*V) aligned_vector{scalar_t val[V];};
struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U, bool rms_norm> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  if constexpr (!rms_norm){
    U delta = val - curr_sum.mean;
    U new_count = curr_sum.count + 1.f;
//Due to low CU count, we run into accuracy issues on gfx90a with `__builtin_amdgcn_rcpf`
#if defined(USE_ROCM) && !defined(__gfx90a__) && defined(USE_LAYERNORM_FAST_RECIPROCAL)
    U new_mean = curr_sum.mean + delta * __builtin_amdgcn_rcpf(new_count);
#else
    U new_mean = curr_sum.mean + delta * (1.f/new_count); //proper division is slow, this is less accurate but noticeably faster
#endif
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
  } else{
    return {0.f, curr_sum.sigma2 + val * val, 0};
  }
}

template<bool rms_norm> __device__
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA
) {
  if constexpr (!rms_norm){
    using U = decltype(dataB.count);
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    U mean, sigma2;
    if (count > decltype(dataB.count){0}) {
//Due to low CU count, we run into accuracy issues on gfx90a with `__builtin_amdgcn_rcpf`
#if defined(USE_ROCM) && !defined(__gfx90a__) && defined(USE_LAYERNORM_FAST_RECIPROCAL)
      auto coef = __builtin_amdgcn_rcpf(count);
#else
      auto coef = 1.f/count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
#endif
      auto nA = dataA.count * coef;
      auto nB = dataB.count * coef;
      mean = nA*dataA.mean + nB*dataB.mean;
      sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
      mean = U(0);
      sigma2 = U(0);
    }
    return {mean, sigma2, count};
  } else {
    return {0.f, dataB.sigma2 + dataA.sigma2, 0};
  }
}

template<typename T, bool rms_norm = false>
__device__ WelfordDataLN compute_stats(
  const T*  __restrict__ X,
  const int N,
  float * buf
  ) {
    //X points to the row to read
    using vec_t = aligned_vector<T, vec_size>;
    using acc_t = acc_type<T, true>;
    const vec_t * X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N/vec_size;
    WelfordDataLN wd(0.f, 0.f, 0.f);
    //no tail, we check that N is multiple of vec_size
    for (int i = thrx; i < n_vec_to_read; i += numx) {
      vec_t data = X_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++){
        wd = cuWelfordOnlineSum<acc_t, rms_norm>(static_cast<acc_t>(data.val[ii]), wd);
      }
    }
    // intra-warp reduction
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset), WARP_SHFL_DOWN(wd.sigma2, offset), WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine<rms_norm>(wd, wdB);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float * meansigmabuf = buf;
      float * countbuf = buf + blockDim.y;
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          meansigmabuf[2*wrt_y] = wd.mean;
          meansigmabuf[2*wrt_y+1] = wd.sigma2;
          countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          WelfordDataLN wdB{meansigmabuf[2*threadIdx.y],
                          meansigmabuf[2*threadIdx.y+1],
                          countbuf[threadIdx.y]};
          wd = cuWelfordCombine<rms_norm>(wd, wdB);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0 && threadIdx.y ==0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2/float(N);
      }
      __syncthreads();
      return WelfordDataLN{meansigmabuf[0], meansigmabuf[1],0.f};

    } else {
      return WelfordDataLN{WARP_SHFL(wd.mean,0), WARP_SHFL(wd.sigma2,0)/float(N), 0.f};
    }
}




template<bool DebugNorm>
__global__ void kernel(const at::BFloat16* X,const at::BFloat16* G,const at::BFloat16* B,
 const at::BFloat16* Mod,at::BFloat16* Y,at::BFloat16* Native,int8_t* Q,float* S,float eps){
 extern __shared__ float scratch[];
 constexpr int N=1536;int row=blockIdx.x,thread=threadIdx.x+32*threadIdx.y;
 if(row>=41){
  for(int col=thread;col<N;col+=blockDim.x*blockDim.y)Q[row*N+col]=0;
  if(thread==0)S[row]=1.f;return;
 }
 auto wd=compute_stats<at::BFloat16,false>(X+row*N,N,scratch);
 float rs=c10::cuda::compat::rsqrt(wd.sigma2+eps);
 using V=aligned_vector<at::BFloat16,4>;
 auto xv=reinterpret_cast<V const*>(X+row*N);
 auto gv=reinterpret_cast<V const*>(G);auto bv=reinterpret_cast<V const*>(B);
 auto sv=reinterpret_cast<V const*>(Mod);auto tv=reinterpret_cast<V const*>(Mod+N);
 auto yv=reinterpret_cast<V*>(Y+row*N);
 auto nv=reinterpret_cast<V*>(Native ? Native+row*N : nullptr);
 float maximum=0.f;
 for(int i=thread;i<N/4;i+=blockDim.x*blockDim.y){
  V data=xv[i],out,native,scales=sv[i],shifts=tv[i];
  #pragma unroll
  for(int j=0;j<4;j++){
   float value=rs*(float(data.val[j])-wd.mean);
   if(G&&B)native.val[j]=float(gv[i].val[j])*value+float(bv[i].val[j]);
   else if(G)native.val[j]=float(gv[i].val[j])*value;
   else if(B)native.val[j]=value+float(bv[i].val[j]);
   else native.val[j]=value;
   // Keep each original eager BF16 rounding; explicit rn intrinsics forbid
   // folding multiplication/addition across the rounded intermediates.
   at::BFloat16 factor=__fadd_rn(1.f,float(scales.val[j]));
   at::BFloat16 product=__fmul_rn(float(native.val[j]),float(factor));
   out.val[j]=__fadd_rn(float(product),float(shifts.val[j]));
  }
  yv[i]=out;
  #pragma unroll
  for(int j=0;j<4;j++){float v=float(out.val[j]);scratch[32+i*4+j]=v;maximum=fmaxf(maximum,fabsf(v));}
  if constexpr(DebugNorm)nv[i]=native;
 }
 __syncthreads(); // Maximum is accumulated from the same BF16 outputs; shared payload is read only after later barriers.
 for(int offset=16;offset>0;offset/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
 if(threadIdx.x==0)scratch[threadIdx.y]=maximum;
 __syncthreads();
 if(threadIdx.y==0){
  maximum=threadIdx.x<blockDim.y?scratch[threadIdx.x]:0.f;
  for(int offset=16;offset>0;offset/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
  if(threadIdx.x==0)scratch[0]=maximum>0.f?__fmul_rn(maximum,float(1.0/127.0)):1.f;
 }
 __syncthreads();
 const float scale=scratch[0];if(thread==0)S[row]=scale;
 for(int col=thread;col<N;col+=blockDim.x*blockDim.y){
  float normalized=fminf(fmaxf(__fdiv_rn(scratch[32+col],scale),-127.f),127.f);
  Q[row*N+col]=static_cast<int8_t>(__float2int_rn(normalized));
 }
}
}
std::vector<torch::Tensor> norm_modulate(torch::Tensor x,torch::Tensor gamma,torch::Tensor beta,
 torch::Tensor modulation,double eps,bool debug){
 c10::cuda::CUDAGuard guard(x.device());
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.is_contiguous()&&x.dim()==3&&x.size(0)==1&&x.size(1)==41&&x.size(2)==1536);
 TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%8==0);
 for(auto t:{gamma,beta})TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==at::kBFloat16&&(t.numel()==0||t.numel()==1536));
 TORCH_CHECK(modulation.is_cuda()&&modulation.device()==x.device()&&modulation.is_contiguous()&&modulation.scalar_type()==at::kBFloat16&&modulation.dim()==2&&modulation.size(0)==1&&modulation.size(1)==3072);
 auto y=torch::empty_like(x);auto native=debug?torch::empty_like(x):torch::empty({0},x.options());
 const at::BFloat16* gp=gamma.numel()?gamma.data_ptr<at::BFloat16>():nullptr;
 const at::BFloat16* bp=beta.numel()?beta.data_ptr<at::BFloat16>():nullptr;
 const int warp=at::cuda::warp_size();TORCH_CHECK(warp==32);dim3 block(warp,num_threads()/warp,1);
 const int shared=(32+1536)*sizeof(float);
 auto q=torch::empty({48,1536},x.options().dtype(torch::kInt8));auto scales=torch::empty({48,1},x.options().dtype(torch::kFloat32));
 auto stream=at::cuda::getCurrentCUDAStream();
 if(debug)codex_adaln_sharedmax::kernel<true><<<48,block,shared,stream>>>(x.data_ptr<at::BFloat16>(),gp,bp,modulation.data_ptr<at::BFloat16>(),y.data_ptr<at::BFloat16>(),native.data_ptr<at::BFloat16>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),float(eps));
 else codex_adaln_sharedmax::kernel<false><<<48,block,shared,stream>>>(x.data_ptr<at::BFloat16>(),gp,bp,modulation.data_ptr<at::BFloat16>(),y.data_ptr<at::BFloat16>(),nullptr,q.data_ptr<int8_t>(),scales.data_ptr<float>(),float(eps));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,native,q,scales};
}
