"""Parallel four-way native-order BF16 mean; actual Sage operand byte-gated."""
import torch,triton
import triton.language as tl
from int8_sage_mean41_native_order_20261028_r1 import mean as mean41
@triton.jit
def _mean_native_parallel_y(K,O,S1:tl.constexpr,S2:tl.constexpr,H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 y=tl.arange(0,4)[:,None]
 h=i//D;d=i%D;valid=i<H*D
 a0=tl.full((4,B),0,tl.float32);a1=tl.full((4,B),0,tl.float32)
 a2=tl.full((4,B),0,tl.float32);a3=tl.full((4,B),0,tl.float32)
 for j in range(triton.cdiv(N,16)):
  base=h[None,:]*S1+d[None,:]+(j*16+y)*S2
  a0=a0+tl.load(K+base,valid[None,:]&(j*16+y<N),0).to(tl.float32)
  a1=a1+tl.load(K+base+4*S2,valid[None,:]&(j*16+y+4<N),0).to(tl.float32)
  a2=a2+tl.load(K+base+8*S2,valid[None,:]&(j*16+y+8<N),0).to(tl.float32)
  a3=a3+tl.load(K+base+12*S2,valid[None,:]&(j*16+y+12<N),0).to(tl.float32)
 v=((a0+a1)+a2)+a3
 v0=tl.gather(v,tl.full((1,B),0,tl.int32),axis=0).reshape((B,))
 v1=tl.gather(v,tl.full((1,B),1,tl.int32),axis=0).reshape((B,))
 v2=tl.gather(v,tl.full((1,B),2,tl.int32),axis=0).reshape((B,))
 v3=tl.gather(v,tl.full((1,B),3,tl.int32),axis=0).reshape((B,))
 value=(v0+v2)+(v1+v3)
 tl.store(O+i,(value*(1.0/N)).to(tl.bfloat16),valid)
def mean(k):
 b,h,n,d=k.shape
 if n==41:return mean41(k)
 if not 64<=n<1024:return k.mean(dim=2,keepdim=True)
 assert b==1 and d in (48,64,128) and k.stride(-1)==1 and k.dtype==torch.bfloat16
 assert h*d>=128 and d%4==0
 out=torch.empty((b,h,1,d),dtype=k.dtype,device=k.device)
 _mean_native_parallel_y[(triton.cdiv(h*d,32),)](k,out,k.stride(1),k.stride(2),h,d,n,32,num_warps=4,enable_fp_fusion=False)
 return out
