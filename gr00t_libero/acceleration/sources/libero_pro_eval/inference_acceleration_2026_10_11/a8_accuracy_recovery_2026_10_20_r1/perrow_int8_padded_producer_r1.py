"""Fused producers for unchanged per-row INT8 RTN. Mode requires byte gates."""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _divide(x,s,MODE:tl.constexpr):
    if MODE==0:return tl.div_rn(x,s)
    elif MODE==1:
        return tl.inline_asm_elementwise('div.approx.ftz.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,s],dtype=tl.float32,is_pure=True,pack=1)
    elif MODE==2:return x*libdevice.rcp_rn(s)
    elif MODE==3:
        r=tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;',constraints='=f,f',args=[s],dtype=tl.float32,is_pure=True,pack=1)
        return x*r
    elif MODE==4:return libdevice.fast_dividef(x,s)
    else:return x/s

@triton.jit
def _activation(X,Q,S,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B)
    x=tl.load(X+row*K+col,(row<M)&(col<K),0).to(tl.float32)
    maximum=tl.max(tl.abs(x),0)
    inverse=tl.full((),1.0/127.0,tl.float32)
    scale=tl.where(maximum>0,maximum*inverse,1.0)
    normalized=tl.minimum(tl.maximum(_divide(x,scale,MODE),-127.0),127.0)
    encoded=libdevice.nearbyint(normalized).to(tl.int8)
    tl.store(Q+row*K+col,encoded,col<K);tl.store(S+row,scale)

@triton.jit
def _weight(W,S,Q,COUNT:tl.constexpr,K:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<COUNT
    x=tl.load(W+i,mask,0).to(tl.float32);scale=tl.load(S+i//K,mask,1)
    normalized=tl.minimum(tl.maximum(_divide(x,scale,MODE),-127.0),127.0)
    tl.store(Q+i,libdevice.nearbyint(normalized).to(tl.int8),mask)

def activation(x,mode):
    assert x.ndim==2 and x.dtype==torch.bfloat16 and x.is_contiguous() and mode in range(6)
    m,k=x.shape;mp=max(32,triton.cdiv(m,16)*16);q=torch.empty((mp,k),device=x.device,dtype=torch.int8);s=torch.empty((mp,1),device=x.device,dtype=torch.float32)
    _activation[(mp,)](x,q,s,m,k,triton.next_power_of_2(k),mode,num_warps=4,enable_fp_fusion=False)
    return q,s

def weight(w,s,mode):
    assert w.ndim==2 and w.dtype==torch.bfloat16 and w.is_contiguous() and s.dtype==torch.float32 and tuple(s.shape)==(w.shape[0],1)
    q=torch.empty_like(w,dtype=torch.int8)
    _weight[(triton.cdiv(w.numel(),1024),)](w,s,q,w.numel(),w.shape[1],1024,mode,num_warps=4,enable_fp_fusion=False)
    return q
