"""Native INT8->INT32 cuBLAS plus byte-gated per-row/per-column epilogue."""
import torch
import triton
import triton.language as tl

@triton.jit
def _epilogue(A,SA,SW,BIAS,Y,M:tl.constexpr,N:tl.constexpr,HAS_BIAS:tl.constexpr,FMA:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<M*N
    value=tl.load(A+i,mask,0).to(tl.float32)
    sa=tl.load(SA+i//N,mask,1);sw=tl.load(SW+i%N,mask,1)
    scaled=value*sa
    if HAS_BIAS:
        bias=tl.load(BIAS+i%N,mask,0).to(tl.float32)
        if FMA:value=tl.fma(scaled,sw,bias)
        else:value=scaled*sw+bias
    else:value=scaled*sw
    tl.store(Y+i,value.to(tl.bfloat16),mask)

def linear(a,w,sa,sw,bias,fma,actual_rows=None):
    assert a.dtype==w.dtype==torch.int8 and a.ndim==w.ndim==2
    acc=torch._int_mm(a,w.t());assert acc.dtype==torch.int32
    m=a.shape[0] if actual_rows is None else actual_rows
    assert 0<m<=a.shape[0]
    y=torch.empty((m,w.shape[0]),dtype=torch.bfloat16,device=a.device)
    _epilogue[(triton.cdiv(y.numel(),1024),)](acc,sa,sw,bias if bias is not None else y,y,m,w.shape[0],bias is not None,fma,1024,num_warps=4,enable_fp_fusion=False)
    return y
