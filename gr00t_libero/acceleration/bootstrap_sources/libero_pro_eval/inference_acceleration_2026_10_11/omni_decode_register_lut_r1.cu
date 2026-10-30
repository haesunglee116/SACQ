#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <climits>

__device__ __forceinline__ uint64_t affine_bf16(uint64_t code, uint16_t scale, uint16_t shift) {
    uint32_t lo=uint32_t(code), hi=uint32_t(code>>32), a, b;
    uint32_t ss=uint32_t(scale)*0x10001u;
    uint32_t hh=uint32_t(shift)*0x10001u;
    asm volatile(
        "{ .reg .b32 negative_zero, one;\n"
        "  mov.b32 negative_zero, 0x80008000;\n"
        "  mov.b32 one, 0x3f803f80;\n"
        "  fma.rn.bf16x2 %0, %2, %4, negative_zero;\n"
        "  fma.rn.bf16x2 %0, %0, one, %5;\n"
        "  fma.rn.bf16x2 %1, %3, %4, negative_zero;\n"
        "  fma.rn.bf16x2 %1, %1, one, %5;\n"
        "}\n" : "=r"(a), "=r"(b) : "r"(lo), "r"(hi), "r"(ss), "r"(hh));
    return uint64_t(a)|(uint64_t(b)<<32);
}

template<int STATIC_BITS, int VPT, bool REGISTER_LUT>
__global__ void decode(const uint32_t* __restrict__ packed,
                       const uint64_t* __restrict__ cb,
                       const uint16_t* __restrict__ scales,
                       const uint16_t* __restrict__ shift,
                       uint64_t* __restrict__ out,
                       int count,int dynamic_bits) {
    constexpr int THREADS=128;
    int bits=STATIC_BITS ? STATIC_BITS : dynamic_bits;
    unsigned mask=(1u<<bits)-1u;
    int lane=threadIdx.x&31;
    uint64_t table0=0,table1=0;
    if constexpr (REGISTER_LUT) {
        static_assert(STATIC_BITS==5 || STATIC_BITS==6,"Register LUT is only32/64 entries");
        table0=cb[lane];
        if constexpr (STATIC_BITS==6) table1=cb[lane+32];
    }
    uint16_t h=shift[0];
    // CTA-uniform loop condition keeps every warp lane active for shuffles,
    // including masked output lanes in the final chunk.
    for (int base=blockIdx.x*THREADS*VPT;base<count;base+=gridDim.x*THREADS*VPT) {
        #pragma unroll
        for (int item=0;item<VPT;item++) {
            int g=base+threadIdx.x+item*THREADS;
            bool valid=g<count;
            unsigned index=0;
            uint16_t s=0;
            if (valid) {
                unsigned bit=unsigned(g)*unsigned(bits);
                unsigned pos=bit&31u;
                uint32_t low=packed[bit>>5];
                uint32_t high=pos+unsigned(bits)>32u ? packed[(bit>>5)+1] : 0;
                index=((low>>pos)|(high<<((32u-pos)&31u)))&mask;
                s=scales[g>>4]; // group64/d4 is16 codebook vectors.
            }
            uint64_t code;
            if constexpr (REGISTER_LUT) {
                uint32_t lo=__shfl_sync(0xffffffffu,uint32_t(table0),index&31u);
                uint32_t hi=__shfl_sync(0xffffffffu,uint32_t(table0>>32),index&31u);
                if constexpr (STATIC_BITS==6) {
                    uint32_t lo1=__shfl_sync(0xffffffffu,uint32_t(table1),index&31u);
                    uint32_t hi1=__shfl_sync(0xffffffffu,uint32_t(table1>>32),index&31u);
                    if (index&32u) {lo=lo1;hi=hi1;}
                }
                code=uint64_t(lo)|(uint64_t(hi)<<32);
            } else {
                code=cb[index];
            }
            uint64_t result=affine_bf16(code,s,h);
            if (valid) out[g]=result;
        }
    }
}

template<int VPT>
void launch(const uint32_t* p,const uint64_t* cb,const uint16_t* s,const uint16_t* h,
            uint64_t* out,int count,int bits,int blocks,bool use_lut,cudaStream_t stream) {
    if (bits==5) {
        if (use_lut) decode<5,VPT,true><<<blocks,128,0,stream>>>(p,cb,s,h,out,count,bits);
        else decode<5,VPT,false><<<blocks,128,0,stream>>>(p,cb,s,h,out,count,bits);
    } else if (bits==6) {
        if (use_lut) decode<6,VPT,true><<<blocks,128,0,stream>>>(p,cb,s,h,out,count,bits);
        else decode<6,VPT,false><<<blocks,128,0,stream>>>(p,cb,s,h,out,count,bits);
    } else {
        // Explicit global-LUT fallback for the same higher-bit payloads.
        decode<0,VPT,false><<<blocks,128,0,stream>>>(p,cb,s,h,out,count,bits);
    }
}

torch::Tensor decompress(torch::Tensor packed,torch::Tensor cb,torch::Tensor scales,
                         torch::Tensor shift,int64_t n,int64_t k,int64_t bits,
                         int64_t vpt,int64_t blocks_per_sm,bool use_lut) {
    TORCH_CHECK(packed.is_cuda() && cb.is_cuda() && scales.is_cuda() && shift.is_cuda());
    TORCH_CHECK(packed.device()==cb.device() && scales.device()==cb.device() && shift.device()==cb.device());
    TORCH_CHECK(packed.scalar_type()==torch::kUInt8 && cb.scalar_type()==torch::kBFloat16);
    TORCH_CHECK(scales.scalar_type()==torch::kBFloat16 && shift.scalar_type()==torch::kBFloat16);
    TORCH_CHECK(packed.is_contiguous() && cb.is_contiguous() && scales.is_contiguous());
    TORCH_CHECK(n>0 && k>0 && k%64==0 && bits>0 && bits<=16 && shift.numel()==1);
    int64_t count64=n*k/4;
    TORCH_CHECK(count64<=INT_MAX/bits && (count64*bits)%32==0);
    TORCH_CHECK(packed.numel()>=count64*bits/8 && scales.numel()==n*k/64);
    TORCH_CHECK(cb.numel()==(int64_t(1)<<bits)*4);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr())%4==0);
    TORCH_CHECK(vpt==2 || vpt==4 || vpt==8);
    TORCH_CHECK(blocks_per_sm==0 || blocks_per_sm==4 || blocks_per_sm==8 || blocks_per_sm==16);
    c10::cuda::CUDAGuard guard(cb.device());
    auto out=torch::empty({n,k},cb.options());
    int count=int(count64);
    int blocks=(count+128*int(vpt)-1)/(128*int(vpt));
    if (blocks_per_sm) blocks=std::min(blocks,int(blocks_per_sm)*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    auto stream=at::cuda::getCurrentCUDAStream();
    auto p=reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>());
    auto c=reinterpret_cast<const uint64_t*>(cb.data_ptr<c10::BFloat16>());
    auto s=reinterpret_cast<const uint16_t*>(scales.data_ptr<c10::BFloat16>());
    auto h=reinterpret_cast<const uint16_t*>(shift.data_ptr<c10::BFloat16>());
    auto dest=reinterpret_cast<uint64_t*>(out.data_ptr<c10::BFloat16>());
    if (vpt==2) launch<2>(p,c,s,h,dest,count,int(bits),blocks,use_lut,stream);
    else if (vpt==4) launch<4>(p,c,s,h,dest,count,int(bits),blocks,use_lut,stream);
    else launch<8>(p,c,s,h,dest,count,int(bits),blocks,use_lut,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("decompress",&decompress);}
