"""Exact d4 decoder candidates; no repacking or dense weight cache.

Flatten vector IDs and load four adjacent codebook entries together. Word mode
extracts indices from aligned uint32 loads with an explicitly masked second word.
Both BF16 rounding points are retained; stock F.linear remains unchanged.
"""
import torch
import triton
import triton.language as tl
import decoder_tuning as baseline
from bottleneck_attribution_r3 import graph_measure

OLD_DECODE = baseline.dequantize
SELECTED = {}

@triton.jit
def _decode_vectors(packed, cb, scale, shift, out,
                    COUNT:tl.constexpr, BITS:tl.constexpr, WINDOW:tl.constexpr,
                    WORDS:tl.constexpr, WORD_MODE:tl.constexpr, GROUP:tl.constexpr,
                    BLOCK:tl.constexpr):
    v=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=v<COUNT
    bit=v*BITS
    if WORD_MODE:
        wi=bit//32
        off=bit%32
        lo=tl.load(packed+wi,valid & (wi<WORDS),0).to(tl.uint32)
        hi=tl.load(packed+wi+1,valid & (off+BITS>32) & (wi+1<WORDS),0).to(tl.uint32)
        idx=((lo>>off) | (hi << ((32-off)%32))) & ((1<<BITS)-1)
    else:
        byte=bit//8
        off=bit%8
        word=tl.load(packed+byte,valid,0).to(tl.uint32)
        if WINDOW>=2:
            word |= tl.load(packed+byte+1,valid,0).to(tl.uint32)<<8
        if WINDOW>=3:
            word |= tl.load(packed+byte+2,valid,0).to(tl.uint32)<<16
        if WINDOW>=4:
            word |= tl.load(packed+byte+3,valid,0).to(tl.uint32)<<24
        idx=(word>>off)&((1<<BITS)-1)
    s=tl.load(scale+v//(GROUP//4),valid,0)
    z=tl.load(shift)
    component=tl.arange(0,4)
    q=tl.load(cb+idx[:,None]*4+component[None,:],valid[:,None],0)
    w=((q*s[:,None]).to(tl.bfloat16)+z).to(tl.bfloat16)
    tl.store(out+v[:,None]*4+component[None,:],w,valid[:,None])


def decode(m,config=None):
    config=config or SELECTED.get(baseline.key(m),('old',0,0))
    if config[0]=='old':return OLD_DECODE(m)
    mode,block,warps=config
    assert m.vector_dim==4 and m.group_size%4==0
    p=m.packed_indices
    count=m.out_features*m.in_features//4
    if mode=='word':
        assert (count*m.index_bits)%32==0
        p=p[:p.numel()//4*4].view(torch.uint32)
    w=torch.empty((m.out_features,m.in_features),device=m.codebook.device,dtype=torch.bfloat16)
    _decode_vectors[(triton.cdiv(count,block),)](p,m.codebook,m.scales,m.shift,w,
        count,m.index_bits,m.index_window_bytes,p.numel() if mode=='word' else 0,
        mode=='word',m.group_size,block,num_warps=warps,enable_fp_fusion=False)
    return w


def tune(samples,c,emit):
    rows=[];seen=set()
    pressure=torch.empty(4*torch.cuda.get_device_properties(0).L2_cache_size,device='cuda',dtype=torch.uint8)
    for _,sample in sorted(samples.items(),key=lambda a:str(a[0])):
        m=sample['module']
        if not hasattr(m,'packed_indices'):continue
        key=baseline.key(m)
        if key in seen:continue
        seen.add(key)
        x=sample['x_cpu'].to('cuda');w=m.dequantize_weight()
        ref=torch.nn.functional.linear(x,w,m.bias)
        candidates=[('old',0,0)]+[(mode,b,4) for mode in ['byte','word'] for b in [128,256,512]]
        results=[]
        for cfg in candidates:
            actual=decode(m,cfg)
            exact=torch.equal(actual.view(torch.int16),w.view(torch.int16));del actual
            if not exact:raise RuntimeError(f'Weight mismatch: {key}/{cfg}')
            fn=lambda cfg=cfg:torch.nn.functional.linear(x,decode(m,cfg),m.bias)
            t,out=graph_measure(fn,c)
            assert torch.equal(out.view(torch.int16),ref.view(torch.int16)),(key,cfg)
            del out
            results.append(dict(config=cfg,repeated=t))
        finalists=sorted(results,key=lambda r:r['repeated']['median_ms'])[:2]
        old=results[0]
        if old not in finalists:finalists.append(old)
        for row in finalists:
            fn=lambda cfg=row['config']:torch.nn.functional.linear(x,decode(m,cfg),m.bias)
            t,out=graph_measure(fn,c,pressure)
            assert torch.equal(out.view(torch.int16),ref.view(torch.int16));del out
            row['cache_pressure']=t
            row['score_ms']=(row['repeated']['median_ms']+t['median_ms'])/2
        winner=min(finalists,key=lambda r:r['score_ms'])
        SELECTED[key]=tuple(winner['config'])
        rows.append(dict(geometry=key,name=sample['name'],selected=winner['config'],candidates=results))
        emit(rows)
        print('DECODER_SELECTED',key,winner['config'],'combined_us',winner['score_ms']*1000,flush=True)
        del x,w,ref
    del pressure
    return rows


def activate():
    baseline.dequantize=decode
