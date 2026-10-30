"""Inference-only BF16 elementwise fusion with explicit eager rounding points.

No reduction, weight reconstruction or approximate nonlinear function is changed.
Candidates must pass the full-model raw-bit and RNG gate before timing acceptance.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _modulate(x,mod,out,T:tl.constexpr,C:tl.constexpr,SIZE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    batch=i//(T*C)
    channel=i%C
    scale=tl.load(mod+batch*2*C+channel,mask=i<SIZE,other=0).to(tl.float32)
    shift=tl.load(mod+batch*2*C+C+channel,mask=i<SIZE,other=0).to(tl.float32)
    value=tl.load(x+i,mask=i<SIZE,other=0).to(tl.float32)
    factor=(1.0+scale).to(tl.bfloat16).to(tl.float32)
    product=(value*factor).to(tl.bfloat16).to(tl.float32)
    result=(product+shift).to(tl.bfloat16)
    tl.store(out+i,result,mask=i<SIZE)


def modulate(x,modulation):
    if x.dtype != torch.bfloat16 or modulation.dtype != x.dtype or x.ndim!=3:
        raise ValueError('BF16 [batch,tokens,channels] only')
    if not x.is_contiguous() or not modulation.is_contiguous():
        raise ValueError('Contiguous layout required')
    batch,tokens,channels=x.shape
    if modulation.shape!=(batch,2*channels):
        raise ValueError('Modulation layout mismatch')
    y=torch.empty_like(x)
    _modulate[(triton.cdiv(x.numel(),256),)](x,modulation,y,T=tokens,C=channels,SIZE=x.numel(),B=256,enable_fp_fusion=False)
    return y


@triton.jit
def _euler(action,pred,strength,SIZE:tl.constexpr,DT:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(action+i,mask=i<SIZE,other=0).to(tl.float32)
    p=tl.load(pred+i,mask=i<SIZE,other=0).to(tl.float32)
    s=tl.load(strength+i,mask=i<SIZE,other=0).to(tl.float32)
    step=(DT*p).to(tl.bfloat16).to(tl.float32)
    weighted=(step*s).to(tl.bfloat16).to(tl.float32)
    tl.store(action+i,(a+weighted).to(tl.bfloat16),mask=i<SIZE)


def euler_update_(actions,pred,strength,dt):
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (actions,pred,strength)):
        raise ValueError('Contiguous BF16 Euler inputs required')
    if actions.shape!=pred.shape or actions.shape!=strength.shape:
        raise ValueError('Euler shapes differ')
    _euler[(triton.cdiv(actions.numel(),256),)](actions,pred,strength,SIZE=actions.numel(),DT=dt,B=256,enable_fp_fusion=False)
