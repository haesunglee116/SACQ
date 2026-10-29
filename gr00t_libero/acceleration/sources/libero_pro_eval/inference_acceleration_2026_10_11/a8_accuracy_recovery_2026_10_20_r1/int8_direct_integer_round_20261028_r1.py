"""Direct nearest-even FP32-to-INT8 conversion on already-clamped values.
Matches CUTLASS NumericConverter<int8_t,float,round_to_nearest> instruction.
"""
import triton
import triton.language as tl
@triton.jit
def round_int8(x):
    value=tl.inline_asm_elementwise('cvt.rni.sat.s8.f32 $0, $1;',constraints='=r,f',args=[x],dtype=tl.int32,is_pure=True,pack=1)
    return value.to(tl.int8)

@triton.jit
def _check_kernel(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0)
    tl.store(Y+i,round_int8(x),i<N)

def validate_edges():
    import torch
    half=torch.arange(-127,127,device='cuda',dtype=torch.float32)+0.5
    offsets=torch.arange(-64,65,device='cuda',dtype=torch.int32)
    values=(half.view(torch.int32)[:,None]+offsets[None,:]).flatten().view(torch.float32).clamp(-127,127)
    values=torch.cat([values,torch.tensor([-127.,127.,-0.,0.],device='cuda')])
    out=torch.empty_like(values,dtype=torch.int8)
    _check_kernel[(triton.cdiv(values.numel(),256),)](values,out,values.numel(),256)
    expected=values.round().to(torch.int8)
    assert torch.equal(out,expected),'DirectRNHalfBoundaryBytes'
    return dict(status='PASS',tested_values=values.numel(),half_integer_neighbors_ULP=64,includes_signed_zero_and_endpoints=True,no_rng_used=True)
