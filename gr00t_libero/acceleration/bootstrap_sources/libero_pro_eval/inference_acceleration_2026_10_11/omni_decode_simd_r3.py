"""Exact d4 reconstruction using paired BF16 arithmetic and 64-bit stores.

SM89 lacks standalone BF16x2 mul/add but supports BF16x2 FMA. Two rounded
FMAs retain the original multiply-round then add-round sequence. Negative
zero as the product addend preserves the sign of a zero multiplication.
All actual immutable weight bits must pass before any timing/adoption.
"""
import torch
import triton
import triton.language as tl

CONFIGS=[(8,64,4),(16,64,4),(32,32,4),(32,64,4),(32,128,8),(64,64,8)]
COMPILED={}


@triton.jit
def _decode_simd(packed,cb64,scales,shift,out64,
                 N:tl.constexpr,K:tl.constexpr,BITS:tl.constexpr,WINDOW:tl.constexpr,
                 BN:tl.constexpr,BV:tl.constexpr,NATIVE:tl.constexpr,INDEX32:tl.constexpr):
    ns=tl.program_id(0)*BN+tl.arange(0,BN)
    vs=tl.program_id(1)*BV+tl.arange(0,BV)
    valid=(ns[:,None]<N)&(vs[None,:]<K//4)
    bit=(ns[:,None]*(K//4)+vs[None,:])*BITS
    if INDEX32:
        words=tl.cast(packed,tl.pointer_type(tl.uint32))
        position=bit%32
        offset=bit//32
        low=tl.load(words+offset,valid,0)
        high=tl.load(words+offset+1,valid&((position+BITS)>32),0)
        index=((low>>position)|(high<<((32-position)&31)))&((1<<BITS)-1)
    else:
        byte=bit//8
        word=tl.load(packed+byte,valid,0).to(tl.uint32)
        if WINDOW>=2:word|=tl.load(packed+byte+1,valid,0).to(tl.uint32)<<8
        if WINDOW>=3:word|=tl.load(packed+byte+2,valid,0).to(tl.uint32)<<16
        if WINDOW>=4:word|=tl.load(packed+byte+3,valid,0).to(tl.uint32)<<24
        index=(word>>(bit%8))&((1<<BITS)-1)
    vector=tl.load(cb64+index,valid,0).to(tl.uint64)
    scale=tl.load(scales+ns[:,None]*(K//64)+vs[None,:]//16,valid,0)
    s=tl.load(shift)
    if NATIVE:
        lo=(vector&0xffffffff).to(tl.uint32)
        hi=(vector>>32).to(tl.uint32)
        sb=scale.to(tl.uint16,bitcast=True).to(tl.uint32)
        hb=s.to(tl.uint16,bitcast=True).to(tl.uint32)
        pair_scale=sb|(sb<<16);pair_shift=hb|(hb<<16)
        low,high=tl.inline_asm_elementwise(
            """{
                .reg .b32 negative_zero, one;
                mov.b32 negative_zero, 0x80008000;
                mov.b32 one, 0x3f803f80;
                fma.rn.bf16x2 $0, $2, $4, negative_zero;
                fma.rn.bf16x2 $0, $0, one, $5;
                fma.rn.bf16x2 $1, $3, $4, negative_zero;
                fma.rn.bf16x2 $1, $1, one, $5;
            }""",constraints="=r,=r,r,r,r,r",args=[lo,hi,pair_scale,pair_shift],
            dtype=(tl.uint32,tl.uint32),is_pure=True,pack=1)
        output=low.to(tl.uint64)|(high.to(tl.uint64)<<32)
    else:
        q0=(vector&65535).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        q1=((vector>>16)&65535).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        q2=((vector>>32)&65535).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        q3=((vector>>48)&65535).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        w0=((q0*scale).to(tl.bfloat16)+s).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
        w1=((q1*scale).to(tl.bfloat16)+s).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
        w2=((q2*scale).to(tl.bfloat16)+s).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
        w3=((q3*scale).to(tl.bfloat16)+s).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
        output=w0|(w1<<16)|(w2<<32)|(w3<<48)
    tl.store(out64+ns[:,None]*(K//4)+vs[None,:],output,valid)


def dequantize(m,config=(8,64,4),native=True,index32=True):
    assert m.vector_dim==4 and m.group_size==64 and m.index_window_bytes<=4
    assert m.codebook.dtype==m.scales.dtype==m.shift.dtype==torch.bfloat16
    assert m.codebook.is_contiguous() and m.shift.numel()==1
    n,k=m.out_features,m.in_features;bn,bv,nw=config
    assert (n*(k//4)*m.index_bits)%32==0,'Aligned index word reads require complete final words'
    output=torch.empty((n,k),dtype=torch.bfloat16,device=m.codebook.device)
    kernel=_decode_simd[(triton.cdiv(n,bn),triton.cdiv(k//4,bv))](
        m.packed_indices,m.codebook.view(torch.int64),m.scales,m.shift,output.view(torch.int64),
        N=n,K=k,BITS=m.index_bits,WINDOW=m.index_window_bytes,
        BN=bn,BV=bv,NATIVE=native,INDEX32=index32,num_warps=nw,enable_fp_fusion=False)
    COMPILED[(n,k,m.index_bits,m.index_window_bytes,tuple(config),native,index32)]=kernel
    return output


def linear(x,m,config=(8,64,4),native=True,index32=True):
    return torch.nn.functional.linear(x,dequantize(m,config,native,index32),m.bias)


def install(policy,selection):
    import decoder_tuning as decoder
    names={id(m):n for n,m in policy.model.named_modules()}
    original=decoder.linear
    checked=0
    for m in policy.model.modules():
        config=selection.get(names[id(m)])
        if config is None:continue
        expected=m.dequantize_weight();actual=dequantize(m,tuple(config['config']),config['native'],config['index32'])
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),names[id(m)]
        del expected,actual
        checked+=1
    assert checked==len(selection)>0
    def selected(x,m):
        config=selection.get(names[id(m)])
        if config is None:return original(x,m)
        return linear(x,m,tuple(config['config']),config['native'],config['index32'])
    decoder.linear=selected
    return dict(selected_layers=checked,weights_exact=True,permanent_decoded_weight_cache=False,
                temporary_weight_shape_unchanged=True,output_store_bits=64,
                rounding='BF16 multiply-round then add-round, full actual weights exact-gated')
