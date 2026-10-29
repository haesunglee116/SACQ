"""Split K in exact bounded INT32, then scale once in original native order."""
import torch
import triton
import triton.language as tl

@triton.jit
def _int8_splitk_partial(A,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                         SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    part=tl.program_id(2)
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    STEPS:tl.constexpr=triton.cdiv(K,BK*SPLIT)
    acc=tl.full((BM,BN),0,tl.int32)
    for step in range(STEPS):
        kk=(part*STEPS+step)*BK+rk
        a=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
        w=tl.load(W+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        acc=tl.dot(a,w,acc,out_dtype=tl.int32)
    tl.store(P+part*M*N+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))

@triton.jit
def _int8_splitk_reduce(P,SA,SW,B,Y,M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<M*N
    acc=tl.full((BLOCK,),0,tl.int32)
    for part in tl.static_range(SPLIT):acc+=tl.load(P+part*M*N+i,valid,0)
    sw=tl.load(SW+i%N,valid,1.0);sa=tl.load(SA+i//N,valid,1.0)
    scaled=acc.to(tl.float32)*sw
    if HAS_BIAS:
        bias=tl.load(B+i%N,valid,0).to(tl.float32)
        value=tl.fma(sa,scaled,bias)
    else:value=sa*scaled
    tl.store(Y+i,value.to(tl.bfloat16),valid)

def compute(a,w,sa,sw,bias,m,cfg):
    n,k=w.shape;split,bm,bn,bk=cfg
    assert k*127*127<=torch.iinfo(torch.int32).max
    partial=torch.empty((split,m,n),device=a.device,dtype=torch.int32)
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    _int8_splitk_partial[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](a,w,partial,m,n,k,split,bm,bn,bk,num_warps=4,num_stages=3,enable_fp_fusion=False)
    _int8_splitk_reduce[(triton.cdiv(m*n,512),)](partial,sa,sw,bias if bias is not None else y,y,m,n,split,bias is not None,512,num_warps=4,enable_fp_fusion=False)
    return y
