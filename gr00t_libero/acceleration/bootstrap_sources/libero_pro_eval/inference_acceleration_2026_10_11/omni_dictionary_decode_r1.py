"""Exact frozen scalar level dictionaries, larger tiles and packed64 stores."""
import torch
import triton
import triton.language as tl

CONFIGS=[('scalar',256,2),('scalar',256,4),('scalar',512,4),('scalar',1024,4),
         ('scalar',2048,4),('scalar',4096,4),('scalar',4096,8),('scalar',8192,8),
         ('vec4',128,4),('vec4',256,4),('vec4',512,4),('vec4',1024,4),('vec4',2048,4),('vec4',2048,8)]
COMPILED={}


@triton.jit
def _scalar(packed,levels,out,COUNT:tl.constexpr,GROUP:tl.constexpr,
            BITS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<COUNT;bit=i*BITS;byte=bit//8;position=bit%8
    word=tl.load(packed+byte,valid,0).to(tl.uint32)
    if BITS==3:
        word|=tl.load(packed+byte+1,valid&((position+BITS)>8),0).to(tl.uint32)<<8
    index=(word>>position)&((1<<BITS)-1)
    value=tl.load(levels+(i//GROUP)*(1<<BITS)+index,valid,0)
    tl.store(out+i,value,valid)


@triton.jit
def _vec4(packed,levels,out64,COUNT:tl.constexpr,GROUP:tl.constexpr,
          BITS:tl.constexpr,BLOCK:tl.constexpr):
    v=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=v<COUNT//4
    if BITS==4:
        source=tl.cast(packed,tl.pointer_type(tl.uint16))
        word=tl.load(source+v,valid,0).to(tl.uint32)
    elif BITS==2:
        word=tl.load(packed+v,valid,0).to(tl.uint32)
    else:
        source=tl.cast(packed,tl.pointer_type(tl.uint32))
        bit=v*4*BITS;offset=bit//32;position=bit%32
        low=tl.load(source+offset,valid,0).to(tl.uint64)
        high=tl.load(source+offset+1,valid&((position+4*BITS)>32),0).to(tl.uint64)
        word=(low|(high<<32))>>position
    group=(v*4//GROUP)*(1<<BITS)
    i0=word&((1<<BITS)-1)
    i1=(word>>BITS)&((1<<BITS)-1)
    i2=(word>>(2*BITS))&((1<<BITS)-1)
    i3=(word>>(3*BITS))&((1<<BITS)-1)
    a=tl.load(levels+group+i0,valid,0).to(tl.uint16,bitcast=True).to(tl.uint64)
    b=tl.load(levels+group+i1,valid,0).to(tl.uint16,bitcast=True).to(tl.uint64)
    c=tl.load(levels+group+i2,valid,0).to(tl.uint16,bitcast=True).to(tl.uint64)
    d=tl.load(levels+group+i3,valid,0).to(tl.uint16,bitcast=True).to(tl.uint64)
    tl.store(out64+v,a|(b<<16)|(c<<32)|(d<<48),valid)


def reconstruct(m,config=('vec4',512,4)):
    from exact_packed_linear import ExactPackedLinear
    assert type(m) is ExactPackedLinear and not m.training and not torch.is_grad_enabled()
    assert m.packed.dtype==torch.uint8 and m.levels.dtype==torch.bfloat16
    assert m.packed.is_contiguous() and m.levels.is_contiguous()
    count=m.in_features*m.out_features;mode,block,warps=config
    assert m.bits in (2,3,4) and m.group_size%4==0 and count%m.group_size==0
    assert (count*m.bits)%32==0 and m.packed.data_ptr()%4==0
    assert tuple(m.levels.shape)==(count//m.group_size,1<<m.bits)
    out=torch.empty((m.out_features,m.in_features),device=m.levels.device,dtype=torch.bfloat16)
    if mode=='scalar':
        kernel=_scalar[(triton.cdiv(count,block),)](m.packed,m.levels,out,
            COUNT=count,GROUP=m.group_size,BITS=m.bits,BLOCK=block,num_warps=warps)
    else:
        assert mode=='vec4'
        kernel=_vec4[(triton.cdiv(count//4,block),)](m.packed,m.levels,out.view(torch.int64),
            COUNT=count,GROUP=m.group_size,BITS=m.bits,BLOCK=block,num_warps=warps)
    COMPILED[(m.out_features,m.in_features,m.bits,m.group_size,tuple(config))]=kernel
    return out


def install(policy,selection):
    from exact_packed_linear import ExactPackedLinear
    modules=dict(policy.model.named_modules());checked=0
    for name,cfg in selection.items():
        m=modules[name];assert type(m) is ExactPackedLinear
        expected=m._decode_original();actual=reconstruct(m,tuple(cfg['config']))
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),name
        del expected,actual
        m.reconstruct=lambda _m=m,_cfg=tuple(cfg['config']):reconstruct(_m,_cfg)
        checked+=1
    assert checked==len(selection)>0
    return dict(selected_scalar_layers=checked,weight_bits_exact=True,
        quantization_unchanged=True,level_dictionary_unchanged=True,
        permanent_expanded_weights=False,temporary_matrix_shape_unchanged=True)
