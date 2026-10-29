"""Dynamic tensor absmax/448 without full-sized abs/FP32 staging arrays.

Finite BF16 inputs have exactly representable FP32 magnitudes. Reduction does
not sum or change the approved per-tensor scale rule. Zero tensor scale is1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial_absmax(x,out,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(x+i,i<COUNT,0).to(tl.float32)
    maximum=tl.max(tl.abs(value),0)
    tl.store(out+tl.program_id(0),maximum)


@triton.jit
def _finish(partials,out,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    maximum=tl.max(tl.load(partials+i,i<COUNT,0),0)
    scale=tl.where(maximum>0,maximum*tl.full((),1.0/448.0,tl.float32),1.0)
    tl.store(out,scale)


@triton.jit
def _small_absmax(x,out,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    value=tl.load(x+i,i<COUNT,0).to(tl.float32)
    maximum=tl.max(tl.abs(value),0)
    tl.store(out,tl.where(maximum>0,maximum*tl.full((),1.0/448.0,tl.float32),1.0))


def dynamic_scale(a):
    if not a.is_cuda or a.dtype!=torch.bfloat16 or not a.is_contiguous():
        raise TypeError('Contiguous CUDA BF16 source activation required')
    count=a.numel()
    if not count:raise ValueError('Empty activation')
    scale=torch.empty((),dtype=torch.float32,device=a.device)
    if count<=8192:
        _small_absmax[(1,)](a,scale,count,triton.next_power_of_2(count),num_warps=4)
    else:
        parts=triton.cdiv(count,4096)
        partials=torch.empty((parts,),dtype=torch.float32,device=a.device)
        _partial_absmax[(parts,)](a,partials,count,4096,num_warps=4)
        _finish[(1,)](partials,scale,parts,triton.next_power_of_2(parts),num_warps=4)
    return scale
