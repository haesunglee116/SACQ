"""Preserve original BF16 projection output and fuse its residual add in split-K reduction."""
import torch,triton
import triton.language as tl
from int8_splitk_exact_20261028_r1 import _int8_splitk_partial
FRAME=None
STATE=None
SUMMARY=dict(calls=0,checks=0,validation_reference_calls=0)
@triton.jit
def _int8_splitk_reduce_residual(P,SA,SW,B,Y,H,Z,M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<M*N
    acc=tl.full((BLOCK,),0,tl.int32)
    for part in tl.static_range(SPLIT):acc+=tl.load(P+part*M*N+i,valid,0)
    sw=tl.load(SW+i%N,valid,1.0);sa=tl.load(SA+i//N,valid,1.0)
    scaled=acc.to(tl.float32)*sw
    if HAS_BIAS:
        bias=tl.load(B+i%N,valid,0).to(tl.float32)
        value=tl.fma(sa,scaled,bias)
    else:value=sa*scaled
    rounded=value.to(tl.bfloat16)
    tl.store(Y+i,rounded,valid)
    residual=tl.load(H+i,valid,0).to(tl.float32)
    result=(rounded.to(tl.float32)+residual).to(tl.bfloat16)
    tl.store(Z+i,result,valid)

def run_ff(ff,x,hidden):
 global FRAME
 assert FRAME is None and not ff.training
 assert hidden.shape==(1,41,1536) and hidden.dtype==torch.bfloat16 and hidden.is_contiguous()
 frame=dict(hidden=hidden,output=None)
 FRAME=frame
 try:y=ff(x)
 finally:FRAME=None
 assert frame['output'] is not None,'NoFusedFFdownResidual'
 out=frame['output'].reshape_as(hidden)
 if STATE['validation_enabled']:
  expected=y+hidden
  assert torch.equal(out.view(torch.int16),expected.view(torch.int16)),'FFdownResidualBF16Bytes'
  SUMMARY['checks']+=1
 return out

def enable():
 import int8_splitk_frozen_20261028_r1 as frozen
 import perrow_int8_fast_policy_r3 as fast
 previous_compute=frozen.compute
 previous_install=fast.install
 def compute(a,w,sa,sw,bias,m,cfg):
  if FRAME is None:return previous_compute(a,w,sa,sw,bias,m,cfg)
  if FRAME['output'] is not None:
   assert STATE['validation_enabled'],'UnexpectedSecondFFdownOutsideValidation'
   SUMMARY['validation_reference_calls']+=1
   return previous_compute(a,w,sa,sw,bias,m,cfg)
  n,k=w.shape;split,bm,bn,bk=cfg
  assert (m,n,k)==(41,1536,6144) and tuple(cfg)==(4,64,64,128)
  assert k*127*127<=torch.iinfo(torch.int32).max
  partial=torch.empty((split,m,n),device=a.device,dtype=torch.int32)
  y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
  z=torch.empty_like(y)
  _int8_splitk_partial[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](a,w,partial,m,n,k,split,bm,bn,bk,num_warps=4,num_stages=3,enable_fp_fusion=False)
  _int8_splitk_reduce_residual[(triton.cdiv(m*n,512),)](partial,sa,sw,bias if bias is not None else y,y,FRAME['hidden'],z,m,n,split,bias is not None,512,num_warps=4,enable_fp_fusion=False)
  FRAME['output']=z;SUMMARY['calls']+=1
  return y
 frozen.compute=compute
 def install(policy,names,contract):
  global STATE
  result=previous_install(policy,names,contract)
  STATE=policy.model._perrow_a8_candidate_state
  return result
 fast.install=install
 return SUMMARY
