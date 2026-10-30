"""Exact 4-bit dictionary decoder: one packed-byte load for two weights."""
import torch
import triton
import triton.language as tl
import exact_dictionary_decode as baseline
from bottleneck_attribution_r3 import graph_measure

ORIGINAL=baseline.reconstruct
SELECTED={}

@triton.jit
def _decode_nibbles(packed,levels,out,PAIRS:tl.constexpr,GROUP:tl.constexpr,BLOCK:tl.constexpr):
    p=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=p<PAIRS
    byte=tl.load(packed+p,valid,0).to(tl.int32)
    group=p//(GROUP//2)
    lo=tl.load(levels+group*16+(byte&15),valid,0)
    hi=tl.load(levels+group*16+(byte>>4),valid,0)
    values=tl.interleave(lo,hi)
    i=tl.program_id(0)*BLOCK*2+tl.arange(0,BLOCK*2)
    tl.store(out+i,values,i<PAIRS*2)


def key(m):return (m.out_features,m.in_features,m.group_size,m.bits)

def reconstruct(m,config=None):
    config=config or SELECTED.get(key(m),(0,0))
    if config[0]==0:return ORIGINAL(m)
    assert m.bits==4 and m.group_size%2==0
    count=m.in_features*m.out_features
    out=torch.empty((m.out_features,m.in_features),dtype=torch.bfloat16,device=m.packed.device)
    b,w=config
    _decode_nibbles[(triton.cdiv(count//2,b),)](m.packed,m.levels,out,count//2,m.group_size,b,num_warps=w)
    return out


def tune(model,c,emit):
    from exact_packed_linear import ExactPackedLinear
    rows=[];seen=set()
    for name,m in model.named_modules():
        if type(m) is not ExactPackedLinear or m.bits!=4 or key(m) in seen:continue
        seen.add(key(m))
        ref=m.reconstruct();results=[]
        for cfg in [(0,0),(256,4),(512,4),(1024,4),(2048,4),(2048,8)]:
            t,out=graph_measure(lambda:reconstruct(m,cfg),c)
            assert torch.equal(out.view(torch.int16),ref.view(torch.int16)),(name,cfg)
            results.append(dict(config=cfg,timing=t));del out
        best=min(results,key=lambda r:r['timing']['median_ms'])
        SELECTED[key(m)]=tuple(best['config'])
        rows.append(dict(name=name,geometry=key(m),selected=best['config'],candidates=results));emit(rows)
        print('DICTIONARY_SELECTED',key(m),best['config'],best['timing']['median_ms'],flush=True)
        del ref
    return rows


def activate():baseline.reconstruct=reconstruct
