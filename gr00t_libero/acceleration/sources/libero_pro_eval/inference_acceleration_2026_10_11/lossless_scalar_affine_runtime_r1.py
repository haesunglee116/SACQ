"""Exact reversible affine codec of existing scalar BF16 checkpoint weights.

No nearest-weight mapping, precision change or permanent dense reconstruction.
Every original source weight bit must pass CPU and GPU before replacement.
"""
import numpy as np
import torch
from torch import nn
import triton
import triton.language as tl
from lossless_scalar_affine_r1 import encode

CONFIGS=[('scalar',512,4),('scalar',1024,4),('scalar',2048,4),
         ('scalar',4096,8),('vec4',256,4),('vec4',512,4),
         ('vec4',1024,4),('vec4',2048,8)]

def pack_codes(codes,bits):
    flat=codes.reshape(-1).astype(np.uint8)
    assert flat.size%64==0 and np.all(flat<(1<<bits))
    out=np.zeros(flat.size*bits//8+1,dtype=np.uint8)
    if bits==4:
        out[:-1]=flat[0::2]|(flat[1::2]<<4)
    elif bits==2:
        out[:-1]=flat[0::4]|(flat[1::4]<<2)|(flat[2::4]<<4)|(flat[3::4]<<6)
    else:
        assert bits==3
        bit=np.arange(flat.size,dtype=np.uint32)*bits
        value=flat.astype(np.uint32);position=bit%8
        np.bitwise_or.at(out,bit//8,(value<<position).astype(np.uint8))
        np.bitwise_or.at(out,bit//8+1,(value>>(8-position)).astype(np.uint8))
    return out


@triton.jit
def _decode_affine_scalar(packed,scales,offsets,out,COUNT:tl.constexpr,
                          BITS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<COUNT
    bit=i*BITS;position=bit%8
    word=tl.load(packed+bit//8,valid,0).to(tl.uint32)
    if BITS==3:word|=tl.load(packed+bit//8+1,valid&(position+BITS>8),0).to(tl.uint32)<<8
    q=((word>>position)&((1<<BITS)-1)).to(tl.int32)
    offset=tl.load(offsets+i//64,valid,0).to(tl.int32)
    scale=tl.load(scales+i//64,valid,0)
    value=((q+offset).to(tl.float32)*scale).to(tl.bfloat16)
    tl.store(out+i,value,valid)


@triton.jit
def _decode_affine_vec4(packed,scales,offsets,out64,COUNT:tl.constexpr,
                        BITS:tl.constexpr,BLOCK:tl.constexpr):
    v=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=v<COUNT//4
    if BITS==4:
        source=tl.cast(packed,tl.pointer_type(tl.uint16))
        word=tl.load(source+v,valid,0).to(tl.uint32)
    elif BITS==2:word=tl.load(packed+v,valid,0).to(tl.uint32)
    else:
        source=tl.cast(packed,tl.pointer_type(tl.uint32))
        bit=v*4*BITS;position=bit%32
        low=tl.load(source+bit//32,valid,0).to(tl.uint64)
        high=tl.load(source+bit//32+1,valid&(position+4*BITS>32),0).to(tl.uint64)
        word=(low|(high<<32))>>position
    scale=tl.load(scales+v//16,valid,0)
    offset=tl.load(offsets+v//16,valid,0).to(tl.int32)
    q0=(word&((1<<BITS)-1)).to(tl.int32)+offset
    q1=((word>>BITS)&((1<<BITS)-1)).to(tl.int32)+offset
    q2=((word>>(2*BITS))&((1<<BITS)-1)).to(tl.int32)+offset
    q3=((word>>(3*BITS))&((1<<BITS)-1)).to(tl.int32)+offset
    a=(q0.to(tl.float32)*scale).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
    b=(q1.to(tl.float32)*scale).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
    c=(q2.to(tl.float32)*scale).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
    d=(q3.to(tl.float32)*scale).to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint64)
    tl.store(out64+v,a|(b<<16)|(c<<32)|(d<<48),valid)


class LosslessAffineLinear(nn.Module):
    def __init__(self,original):
        super().__init__()
        from exact_packed_linear import ExactPackedLinear
        assert type(original) is ExactPackedLinear and original.group_size==64
        assert not original.training and not torch.is_grad_enabled()
        expected=original.reconstruct();cpu=expected.cpu().contiguous()
        encoded=encode(cpu,original.bits)
        if not encoded['whole_tensor_exact']:
            raise ValueError('Unsupported exact affine group; never approximate')
        self.in_features=original.in_features;self.out_features=original.out_features
        self.bits=original.bits;self.group_size=64
        self.register_buffer('packed',torch.from_numpy(pack_codes(encoded['codes'],self.bits)).to(expected.device))
        self.register_buffer('scales',torch.from_numpy(encoded['scales']).to(expected.device))
        self.register_buffer('offsets',torch.from_numpy(encoded['offsets']).to(expected.device))
        self.register_buffer('bias',original.bias)
        self.config=('vec4',512,4);self.eval().requires_grad_(False)
        actual=self.reconstruct()
        assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)), 'Affine GPU weightbits mismatch'
        del encoded,cpu,expected,actual

    def reconstruct(self):return reconstruct(self,self.config)

    def forward(self,x):return torch.nn.functional.linear(x,self.reconstruct(),self.bias)


def reconstruct(m,config=('vec4',512,4)):
    assert type(m) is LosslessAffineLinear and not m.training and not torch.is_grad_enabled()
    assert m.scales.dtype==torch.float32 and m.offsets.dtype==torch.int8 and m.packed.dtype==torch.uint8
    count=m.out_features*m.in_features;mode,block,warps=config
    assert m.group_size==64 and count%64==0 and m.packed.numel()==count*m.bits//8+1
    assert m.scales.numel()==m.offsets.numel()==count//64
    assert m.packed.data_ptr()%4==0
    out=torch.empty((m.out_features,m.in_features),dtype=torch.bfloat16,device=m.packed.device)
    if mode=='scalar':
        _decode_affine_scalar[(triton.cdiv(count,block),)](m.packed,m.scales,m.offsets,out,
            COUNT=count,BITS=m.bits,BLOCK=block,num_warps=warps,enable_fp_fusion=False)
    else:
        assert mode=='vec4'
        _decode_affine_vec4[(triton.cdiv(count//4,block),)](m.packed,m.scales,m.offsets,out.view(torch.int64),
            COUNT=count,BITS=m.bits,BLOCK=block,num_warps=warps,enable_fp_fusion=False)
    return out


def install(policy,selection):
    modules=dict(policy.model.named_modules());checked=0;before_bytes=0;after_bytes=0
    for name,cfg in selection.items():
        original=modules[name];candidate=LosslessAffineLinear(original)
        candidate.config=tuple(cfg['config'])
        expected=original.reconstruct();actual=candidate.reconstruct()
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),name
        before_bytes+=sum(t.numel()*t.element_size() for t in original.buffers())
        after_bytes+=sum(t.numel()*t.element_size() for t in candidate.buffers())
        parent,leaf=name.rsplit('.',1);setattr(policy.model.get_submodule(parent),leaf,candidate)
        checked+=1;del expected,actual,candidate,original
    assert checked==len(selection)>0
    return dict(selected_scalar_layers=checked,weight_bits_exact=True,
        requantization=False,permanent_expanded_weights=False,
        original_buffer_bytes=before_bytes,new_buffer_bytes=after_bytes,
        encoded_payload_change='Reversible integercodes/FP32scales/int8offsets, exact same BF16 weights',
        resident_memory_measured=False)
