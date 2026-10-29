"""Reversible encoding of existing BF16 scalar weights, never requantization.

Recover a float32 scale and integer offset only when every original BF16 bit
is reproduced. Unsupported groups remain unsupported; no approximate fallback.
CPU encoding is outside inference. This does not alter the source checkpoint.
"""
import numpy as np
import torch


def rounded_bf16_bits(values):
    bits=np.asarray(values,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)


def encode(weight,bits,group_size=64,batch_groups=1024):
    assert weight.device.type=='cpu' and weight.dtype==torch.bfloat16
    assert bits in (2,3,4) and group_size==64 and weight.ndim==2
    assert weight.shape[1]%group_size==0 and weight.is_contiguous()
    source_bits=weight.view(torch.int16).numpy().view(np.uint16).reshape(-1,64)
    values=weight.float().numpy().reshape(-1,64)
    total=values.shape[0];limit=(1<<bits)-1
    scales=np.zeros(total,dtype=np.float32)
    offsets=np.zeros(total,dtype=np.int8)
    codes=np.zeros(values.shape,dtype=np.uint8)
    supported=np.zeros(total,dtype=np.bool_)
    for start in range(0,total,batch_groups):
        end=min(start+batch_groups,total)
        raw=source_bits[start:end];v=values[start:end]
        magnitude=raw&np.uint16(0x7fff);nonzero=magnitude!=0
        finite=np.all(magnitude<0x7f80,axis=1)
        positive_zero=np.all((raw!=0x8000),axis=1)
        all_zero=np.all(raw==0,axis=1)
        found=all_zero.copy();scale_out=np.ones(end-start,dtype=np.float32)
        offset_out=np.zeros(end-start,dtype=np.int8)
        code_out=np.zeros(v.shape,dtype=np.uint8)
        # Exact BF16 rounding cells around every observed absolute value.
        abs_v=(magnitude.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        prev=((np.maximum(magnitude.astype(np.int32)-1,0).astype(np.uint32))<<16).view(np.float32).astype(np.float64)
        next_v=((magnitude.astype(np.uint32)+1)<<16).view(np.float32).astype(np.float64)
        lower=(prev+abs_v)*0.5;upper=(abs_v+next_v)*0.5
        smallest=np.min(np.where(nonzero,abs_v,np.inf),axis=1)
        smallest=np.where(np.isfinite(smallest),smallest,1.0)
        for divisor in range(1,limit+1):
            eligible=~found & finite & positive_zero
            if not eligible.any():break
            proposed=smallest/divisor
            integers=np.rint(v.astype(np.float64)/proposed[:,None])
            lo_i=np.minimum(integers.min(axis=1),0)
            hi_i=integers.max(axis=1)
            eligible&=(integers.min(axis=1)>=-limit)&(hi_i<=limit)
            eligible&=(hi_i-lo_i<=limit)&np.all(nonzero==(integers!=0),axis=1)
            denominator=np.abs(integers)
            l=np.full(v.shape,-np.inf,dtype=np.float64)
            u=np.full(v.shape,np.inf,dtype=np.float64)
            np.divide(lower,denominator,out=l,where=nonzero&(denominator>0))
            np.divide(upper,denominator,out=u,where=nonzero&(denominator>0))
            lower_scale=l.max(axis=1);upper_scale=u.min(axis=1)
            candidate=((lower_scale+upper_scale)*0.5).astype(np.float32)
            eligible&=(lower_scale<=upper_scale)&np.isfinite(candidate)&(candidate>0)
            # Float32 multiplication followed by BF16 RN is checked against
            # every source bit, including ties, underflow and signed zeros.
            rebuilt=(integers.astype(np.float32)*candidate[:,None]).astype(np.float32)
            eligible&=np.all(rounded_bf16_bits(rebuilt)==raw,axis=1)
            if eligible.any():
                scale_out[eligible]=candidate[eligible]
                offset_out[eligible]=lo_i[eligible].astype(np.int8)
                code_out[eligible]=(integers[eligible]-lo_i[eligible,None]).astype(np.uint8)
                found|=eligible
        supported[start:end]=found
        scales[start:end]=scale_out;offsets[start:end]=offset_out;codes[start:end]=code_out
    # Whole-tensor receipt is explicit; callers may not silently encode a
    # failed group using nearest levels or higher error tolerance.
    rebuilt=(codes.astype(np.float32)+offsets[:,None].astype(np.float32))*scales[:,None]
    exact=np.all(rounded_bf16_bits(rebuilt)==source_bits,axis=1)
    assert np.all(exact[supported])
    return dict(scales=scales,offsets=offsets,codes=codes,supported=supported,
                groups=total,exact_supported_groups=int(supported.sum()),
                whole_tensor_exact=bool(supported.all() and exact.all()),
                bits=bits,group_size=64,requantization=False)
