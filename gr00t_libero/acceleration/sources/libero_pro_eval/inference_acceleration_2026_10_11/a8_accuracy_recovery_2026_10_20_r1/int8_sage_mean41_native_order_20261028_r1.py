"""Length41 BF16 mean with native four-accumulator order; actual-byte gated."""
import torch
import triton
import triton.language as tl

@triton.jit
def _mean41_native_order(K,O,S1:tl.constexpr,S2:tl.constexpr,H:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 h=i//D;d=i%D;valid=i<H*D
 v0=tl.full((B,),0,tl.float32);v1=tl.full((B,),0,tl.float32)
 v2=tl.full((B,),0,tl.float32);v3=tl.full((B,),0,tl.float32)
 for j in range(11):
  base=h*S1+d+j*4*S2
  v0=v0+tl.load(K+base,valid&(j*4<41),0).to(tl.float32)
  v1=v1+tl.load(K+base+S2,valid&(j*4+1<41),0).to(tl.float32)
  v2=v2+tl.load(K+base+2*S2,valid&(j*4+2<41),0).to(tl.float32)
  v3=v3+tl.load(K+base+3*S2,valid&(j*4+3<41),0).to(tl.float32)
 value=((v0+v1)+v2)+v3
 tl.store(O+i,(value*(1.0/41.0)).to(tl.bfloat16),valid)

def mean(k):
 if k.shape[2]!=41:return k.mean(dim=2,keepdim=True)
 b,h,n,d=k.shape
 assert b==1 and d in (48,64,128) and k.stride(-1)==1 and k.dtype==torch.bfloat16
 # Native output vectorization4, block_y4 and41<64 imply step_input1.
 assert h*d>=128 and d%4==0
 out=torch.empty((b,h,1,d),dtype=k.dtype,device=k.device)
 _mean41_native_order[(triton.cdiv(h*d,32),)](k,out,k.stride(1),k.stride(2),h,d,32,num_warps=4,enable_fp_fusion=False)
 return out
