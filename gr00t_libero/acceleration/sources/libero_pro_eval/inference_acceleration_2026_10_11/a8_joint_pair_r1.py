"""FP8 inference candidate: compressed payload -> FP8 without dense BF16 staging.

Retains existing RTN semantics: exact decoded BF16 rounding, tensor absmax/448,
FP32 scale/accumulation, BF16 outputs. Fixed weight scales computed once, while
activations are dynamically scaled every call. No retained expanded weights.
"""
import types
import gc
import json
import hashlib
from pathlib import Path
import statistics
import torch
import triton
import triton.language as tl
from a8_rtn_r2 import Quantizer
from a8_dynamic_scale_r2 import dynamic_scale, _partial_absmax

@triton.jit
def _fp8_rtn(x):
    # Direct FP32 -> FP8 conversion. Avoid the compiler's intermediate FP16 RZ.
    bits=tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
        constraints="=h,f",args=[x],dtype=tl.uint16,is_pure=True,pack=1)
    return bits.to(tl.uint8).to(tl.float8e4nv,bitcast=True)


@triton.jit
def _fp8_pair(lo,hi):
    bits=tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        constraints="=h,f,f",args=[lo,hi],dtype=tl.uint16,is_pure=True,pack=1)
    a=bits.to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    b=(bits>>8).to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    return a,b


@triton.jit
def _decode_codebook_fp8(
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    weight_ptr,
    fp8_scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    RECIP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Decode one packed index and store all d4 BF16 weight components."""
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    total_vectors: tl.constexpr = K // VECTOR_DIM
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
    linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
    bit_positions = linear_index_ids * INDEX_BITS
    byte_ids = bit_positions // 8
    bit_offsets = bit_positions % 8
    words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
    if INDEX_WINDOW_BYTES >= 2:
        words = words | (
            tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
        )
    if INDEX_WINDOW_BYTES >= 3:
        words = words | (
            tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
        )
    if INDEX_WINDOW_BYTES >= 4:
        words = words | (
            tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
        )
    indices = (words >> bit_offsets) & ((1 << INDEX_BITS) - 1)
    group_ids = offs_v // vectors_per_group
    scale = tl.load(
        scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
        mask=valid_nv,
        other=0.0,
    )
    shift = tl.load(shift_ptr)
    # d4 codebook vectors are exactly8contiguous BF16bytes. One aligned
    # uint64 gather replaces4independent scalar BF16memory instructions.
    word=tl.load(codebook_ptr.to(tl.pointer_type(tl.uint64))+indices,mask=valid_nv,other=0)
    # Native BF16x2 mirrors the A16 CUDA decoder's __hmul2 then __hadd2.
    # Keep both BF16 roundings, never fuse multiplication and addition.
    ss=scale.to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint32)*65537
    zz=shift.to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint32)*65537
    lo=word.to(tl.uint32)
    hi=(word>>32).to(tl.uint32)
    lo=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x80008000; fma.rn.bf16x2 $0, $1, $2, c; }",constraints="=r,r,r",args=[lo,ss],dtype=tl.uint32,is_pure=True,pack=1)
    hi=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x80008000; fma.rn.bf16x2 $0, $1, $2, c; }",constraints="=r,r,r",args=[hi,ss],dtype=tl.uint32,is_pure=True,pack=1)
    lo=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x3f803f80; fma.rn.bf16x2 $0, $1, c, $2; }",constraints="=r,r,r",args=[lo,zz],dtype=tl.uint32,is_pure=True,pack=1)
    hi=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x3f803f80; fma.rn.bf16x2 $0, $1, c, $2; }",constraints="=r,r,r",args=[hi,zz],dtype=tl.uint32,is_pure=True,pack=1)
    w0=lo.to(tl.uint16).to(tl.bfloat16,bitcast=True)
    w1=(lo>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    w2=hi.to(tl.uint16).to(tl.bfloat16,bitcast=True)
    w3=(hi>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    sw=tl.load(fp8_scale_ptr)
    if RECIP:
        inv=tl.div_rn(1.0,sw)
        w0=w0.to(tl.float32)*inv;w1=w1.to(tl.float32)*inv
        w2=w2.to(tl.float32)*inv;w3=w3.to(tl.float32)*inv
    else:
        w0=tl.div_rn(w0.to(tl.float32),sw);w1=tl.div_rn(w1.to(tl.float32),sw)
        w2=tl.div_rn(w2.to(tl.float32),sw);w3=tl.div_rn(w3.to(tl.float32),sw)
    w0=tl.minimum(tl.maximum(w0,-448.0),448.0)
    w1=tl.minimum(tl.maximum(w1,-448.0),448.0)
    w2=tl.minimum(tl.maximum(w2,-448.0),448.0)
    w3=tl.minimum(tl.maximum(w3,-448.0),448.0)
    p0,p1=_fp8_pair(w0,w1);p2,p3=_fp8_pair(w2,w3)
    values=tl.interleave(tl.interleave(p0,p2),tl.interleave(p1,p3))
    offs_k = pid_v * BLOCK_V * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
    valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        values,
        mask=valid_nk,
    )

@triton.jit
def _decode_codebook_and_absmax(
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    weight_ptr,
    fp8_scale_ptr,
    a_ptr, partial_ptr, COUNT:tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    RECIP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid=tl.program_id(0)
    weight_blocks:tl.constexpr=triton.cdiv(N,BLOCK_N)*triton.cdiv(K//VECTOR_DIM,BLOCK_V)
    if pid < weight_blocks:
        pid_n = pid // triton.cdiv(K // VECTOR_DIM, BLOCK_V)
        pid_v = pid % triton.cdiv(K // VECTOR_DIM, BLOCK_V)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        total_vectors: tl.constexpr = K // VECTOR_DIM
        vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
        valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
        linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8
        words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            words = words | (
                tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
            )
        if INDEX_WINDOW_BYTES >= 3:
            words = words | (
                tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
            )
        if INDEX_WINDOW_BYTES >= 4:
            words = words | (
                tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
            )
        indices = (words >> bit_offsets) & ((1 << INDEX_BITS) - 1)
        group_ids = offs_v // vectors_per_group
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nv,
            other=0.0,
        )
        shift = tl.load(shift_ptr)
        # d4 codebook vectors are exactly8contiguous BF16bytes. One aligned
        # uint64 gather replaces4independent scalar BF16memory instructions.
        word=tl.load(codebook_ptr.to(tl.pointer_type(tl.uint64))+indices,mask=valid_nv,other=0)
        # Native BF16x2 mirrors the A16 CUDA decoder's __hmul2 then __hadd2.
        # Keep both BF16 roundings, never fuse multiplication and addition.
        ss=scale.to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint32)*65537
        zz=shift.to(tl.bfloat16).to(tl.uint16,bitcast=True).to(tl.uint32)*65537
        lo=word.to(tl.uint32)
        hi=(word>>32).to(tl.uint32)
        lo=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x80008000; fma.rn.bf16x2 $0, $1, $2, c; }",constraints="=r,r,r",args=[lo,ss],dtype=tl.uint32,is_pure=True,pack=1)
        hi=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x80008000; fma.rn.bf16x2 $0, $1, $2, c; }",constraints="=r,r,r",args=[hi,ss],dtype=tl.uint32,is_pure=True,pack=1)
        lo=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x3f803f80; fma.rn.bf16x2 $0, $1, c, $2; }",constraints="=r,r,r",args=[lo,zz],dtype=tl.uint32,is_pure=True,pack=1)
        hi=tl.inline_asm_elementwise("{ .reg .b32 c; mov.b32 c, 0x3f803f80; fma.rn.bf16x2 $0, $1, c, $2; }",constraints="=r,r,r",args=[hi,zz],dtype=tl.uint32,is_pure=True,pack=1)
        w0=lo.to(tl.uint16).to(tl.bfloat16,bitcast=True)
        w1=(lo>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        w2=hi.to(tl.uint16).to(tl.bfloat16,bitcast=True)
        w3=(hi>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        sw=tl.load(fp8_scale_ptr)
        if RECIP:
            inv=tl.div_rn(1.0,sw)
            w0=w0.to(tl.float32)*inv;w1=w1.to(tl.float32)*inv
            w2=w2.to(tl.float32)*inv;w3=w3.to(tl.float32)*inv
        else:
            w0=tl.div_rn(w0.to(tl.float32),sw);w1=tl.div_rn(w1.to(tl.float32),sw)
            w2=tl.div_rn(w2.to(tl.float32),sw);w3=tl.div_rn(w3.to(tl.float32),sw)
        w0=tl.minimum(tl.maximum(w0,-448.0),448.0)
        w1=tl.minimum(tl.maximum(w1,-448.0),448.0)
        w2=tl.minimum(tl.maximum(w2,-448.0),448.0)
        w3=tl.minimum(tl.maximum(w3,-448.0),448.0)
        p0,p1=_fp8_pair(w0,w1);p2,p3=_fp8_pair(w2,w3)
        values=tl.interleave(tl.interleave(p0,p2),tl.interleave(p1,p3))
        offs_k = pid_v * BLOCK_V * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
        valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            values,
            mask=valid_nk,
        )
    else:
        part=pid-weight_blocks
        offs=part*4096+tl.arange(0,4096)
        value=tl.load(a_ptr+offs,offs<COUNT,0).to(tl.float32)
        tl.store(partial_ptr+part,tl.max(tl.abs(value),0))

@triton.jit
def _decode_dictionary_fp8(packed, levels, out, scale_ptr,
                           COUNT:tl.constexpr,GROUP:tl.constexpr,RECIP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<COUNT
    byte=tl.load(packed+i//2,valid,0).to(tl.int32)
    index=(byte>>((i%2)*4))&15
    w=tl.load(levels+(i//GROUP)*16+index,valid,0).to(tl.float32)
    if RECIP:w=w*tl.div_rn(1.0,tl.load(scale_ptr))
    else:w=tl.div_rn(w,tl.load(scale_ptr))
    w=_fp8_rtn(tl.minimum(tl.maximum(w,-448.0),448.0))
    tl.store(out+i,w,valid)


@triton.jit
def _quantize_activation(a,q,scale_ptr,M:tl.constexpr,K:tl.constexpr,
                         MP:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(a+i,i<M*K,0).to(tl.float32)
    value=tl.div_rn(value,tl.load(scale_ptr))
    value=_fp8_rtn(tl.minimum(tl.maximum(value,-448.0),448.0))
    tl.store(q+i,value,i<MP*K)


@triton.jit
def _epilogue(y,b,out,M:tl.constexpr,N:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(y+i,i<M*N,0)
    if HAS_BIAS:v=v+tl.load(b+i%N,i<M*N,0).to(tl.float32)
    tl.store(out+i,v.to(tl.bfloat16),i<M*N)


DECODE_TILES={}
DECODE_TUNING=[]
DECODE_CONFIGS=((16,64,4),(8,64,4),(16,32,4),(8,128,4),(16,128,4),(16,64,8))


def _decode_launch(m,scale,q,tile):
    n,k=m.out_features,m.in_features
    bn,bv,warps=tile
    _decode_codebook_fp8[(triton.cdiv(n,bn),triton.cdiv(k//4,bv))](
        m.packed_indices,m.codebook,m.scales,m.shift,q,scale,
        N=n,K=k,NUM_GROUPS=k//m.group_size,GROUP_SIZE=m.group_size,
        VECTOR_DIM=4,INDEX_BITS=m.index_bits,INDEX_WINDOW_BYTES=m.index_window_bytes,
        RECIP=getattr(m,'_a8_fast_weight_div',True),BLOCK_N=bn,BLOCK_V=bv,
        num_warps=warps,enable_fp_fusion=False)


def _decode_key(m):
    return (m.out_features,m.in_features,m.group_size,m.index_bits,
            m.index_window_bytes,getattr(m,'_a8_fast_weight_div',True))


def tune_decode(m,scale,name):
    if _decode_key(m) not in DECODE_TILES:
        raise RuntimeError('Unrecorded frozen decode geometry: '+name)


def decode(m,scale,codebook):
    n,k=m.out_features,m.in_features
    if n%16 or k%16:raise ValueError('Unpadded weight geometry not16aligned')
    q=torch.empty((n,k),device=scale.device,dtype=torch.float8_e4m3fn)
    if codebook:
        if m.vector_dim!=4:raise ValueError('Existing d4 scope required')
        _decode_launch(m,scale,q,DECODE_TILES.get(_decode_key(m),DECODE_CONFIGS[0]))
    else:
        if m.bits!=4:raise ValueError('Only approved VisionHQQ4 dictionary')
        _decode_dictionary_fp8[(triton.cdiv(n*k,1024),)](
            m.packed,m.levels,q,scale,n*k,m.group_size,getattr(m,'_a8_fast_weight_div',True),1024,num_warps=4)
    return q


ACTIVATION_OBSERVER=None


@triton.jit
def _finish_quantize(a,partials,q,scale_out,COUNT:tl.constexpr,PADDED:tl.constexpr,
                     PARTS:tl.constexpr,REDUCE:tl.constexpr,BLOCK:tl.constexpr):
    j=tl.arange(0,REDUCE)
    maximum=tl.max(tl.load(partials+j,j<PARTS,0),0)
    scale=tl.where(maximum>0,maximum*tl.full((),1.0/448.0,tl.float32),1.0)
    if tl.program_id(0)==0:tl.store(scale_out,scale)
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(a+i,i<COUNT,0).to(tl.float32)
    value=tl.div_rn(value,scale)
    value=_fp8_rtn(tl.minimum(tl.maximum(value,-448.0),448.0))
    tl.store(q+i,value,i<PADDED)


def prepare_activation(a,mp):
    rows,k=a.shape
    aq=torch.empty((mp,k),dtype=torch.float8_e4m3fn,device=a.device)
    parts=triton.cdiv(a.numel(),4096)
    # Large tensors retain the already validated three-launch path: avoid a
    # large duplicated reduction in each quantization CTA.
    if a.numel()>8192 and parts<=256:
        partials=torch.empty((parts,),dtype=torch.float32,device=a.device)
        sa=torch.empty((),dtype=torch.float32,device=a.device)
        _partial_absmax[(parts,)](a,partials,a.numel(),4096,num_warps=4)
        _finish_quantize[(triton.cdiv(mp*k,1024),)](
            a,partials,aq,sa,a.numel(),mp*k,parts,triton.next_power_of_2(parts),1024,num_warps=4)
    else:
        sa=dynamic_scale(a)
        _quantize_activation[(triton.cdiv(mp*k,1024),)](a,aq,sa,rows,k,mp,1024,num_warps=4)
    if ACTIVATION_OBSERVER is not None:ACTIVATION_OBSERVER(a,aq,sa)
    return aq,sa


JOINT_WEIGHT_CHECKED=set()

def linear(x,m,scale,codebook):
    if x.dtype!=torch.bfloat16 or not x.is_cuda:raise TypeError('CUDA BF16 source required')
    k=m.in_features;n=m.out_features
    a=x.reshape(-1,k).contiguous(); rows=a.shape[0]; mp=triton.cdiv(rows,16)*16
    shared=getattr(m,'_a8_shared_activation',None)
    joint=(shared is None and codebook and rows==41 and 8192<a.numel()<=1048576)
    if joint:
        wq=torch.empty((n,k),device=a.device,dtype=torch.float8_e4m3fn)
        aq=torch.empty((mp,k),device=a.device,dtype=torch.float8_e4m3fn)
        parts=triton.cdiv(a.numel(),4096)
        partials=torch.empty(parts,device=a.device,dtype=torch.float32)
        sa=torch.empty((),device=a.device,dtype=torch.float32)
        bn,bv,warps=DECODE_TILES[_decode_key(m)]
        blocks=triton.cdiv(n,bn)*triton.cdiv(k//4,bv)
        _decode_codebook_and_absmax[(blocks+parts,)](
            m.packed_indices,m.codebook,m.scales,m.shift,wq,scale,a,partials,a.numel(),
            N=n,K=k,NUM_GROUPS=k//m.group_size,GROUP_SIZE=m.group_size,
            VECTOR_DIM=4,INDEX_BITS=m.index_bits,INDEX_WINDOW_BYTES=m.index_window_bytes,
            RECIP=getattr(m,'_a8_fast_weight_div',True),BLOCK_N=bn,BLOCK_V=bv,
            num_warps=warps,enable_fp_fusion=False)
        _finish_quantize[(triton.cdiv(mp*k,1024),)](
            a,partials,aq,sa,a.numel(),mp*k,parts,triton.next_power_of_2(parts),1024,num_warps=4)
        if ACTIVATION_OBSERVER is not None:
            ACTIVATION_OBSERVER(a,aq,sa)
            if id(m) not in JOINT_WEIGHT_CHECKED:
                reference=decode(m,scale,True)
                if not torch.equal(reference.view(torch.uint8),wq.view(torch.uint8)):
                    raise RuntimeError('Joint decode changes FP8 weight bytes')
                JOINT_WEIGHT_CHECKED.add(id(m))
    else:
        if shared is None:aq,sa=prepare_activation(a,mp)
        else:
            pointer,shape,aq,sa=shared
            if x.data_ptr()!=pointer or tuple(x.shape)!=shape:
                raise RuntimeError('Shared activation input mismatch')
        wq=decode(m,scale,codebook)
    y=torch._scaled_mm(aq,wq.t(),sa,scale,bias=m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
    if isinstance(y,tuple):y=y[0]
    if y.dtype!=torch.bfloat16:raise TypeError('Native FP8 GEMM BF16 output required')
    return y[:rows].reshape(*x.shape[:-1],n)


def install(model,names,vision_names,format,mode,selection):
    path=Path(selection['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=selection['sha256']:
        raise RuntimeError('Frozen decode selection SHA mismatch')
    data=json.loads(path.read_text())
    if data['scope']!=441:raise RuntimeError('Frozen scope mismatch')
    if DECODE_TILES or DECODE_TUNING:raise RuntimeError('Frozen installer already used')
    for row in data['groups']:
        key=tuple(row['key']);tile=tuple(row['selected']['tile'])
        if tile not in DECODE_CONFIGS or key in DECODE_TILES:
            raise RuntimeError('Invalid frozen decode selection')
        DECODE_TILES[key]=tile
    DECODE_TUNING.extend(data['groups'])
    import decoder_tuning as dec
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    from exact_packed_linear import ExactPackedLinear
    if format!='fp8_e4m3fn' or mode!='real_gemm':raise ValueError('Only realFP8 candidate')
    if len(names)!=337 or len(vision_names)!=104:raise ValueError('Approved scope441 required')
    allnames=list(names)+list(vision_names)
    if len(set(allnames))!=441:raise ValueError('Duplicate scope')
    layers={n:model.get_submodule(n) for n in allnames}
    if any(type(layers[n]) is not Packed for n in names):raise TypeError('Codebook scope mismatch')
    if any('.visual.' not in n or type(layers[n]) is not ExactPackedLinear or layers[n].bits!=4 for n in vision_names):raise TypeError('Vision scope mismatch')
    scales={};checks=[];orig={n:m.forward for n,m in layers.items()}
    q=Quantizer(format)
    for name,m in layers.items():
        is_codebook=name in names
        weight=dec.dequantize(m) if is_codebook else m.reconstruct()
        ref,scale=q.quantize(weight)
        candidate=decode(m,scale,is_codebook)
        if not torch.equal(candidate.view(torch.uint8),ref.view(torch.uint8)):
            # Exact immutable FP8 weight operands are mandatory. Retry precise
            # division for boundary cases; this never changes quantized bytes.
            m._a8_fast_weight_div=False
            del candidate
            candidate=decode(m,scale,is_codebook)
            if not torch.equal(candidate.view(torch.uint8),ref.view(torch.uint8)):
                raise RuntimeError('FP8 operand mismatch: '+name)
        else:m._a8_fast_weight_div=True
        if is_codebook:
            tune_decode(m,scale,name)
            tuned=decode(m,scale,True)
            if not torch.equal(tuned.view(torch.uint8),ref.view(torch.uint8)):
                raise RuntimeError('Selected tile changes immutable FP8 weights: '+name)
            del tuned
        scales[id(m)]=scale
        checks.append(name)
        del weight,ref,candidate
    # One packed FP32 allocation, no441 separately allocated scalars.
    scale_storage=torch.stack([scales[id(m)] for m in layers.values()])
    for i,m in enumerate(layers.values()):scales[id(m)]=scale_storage[i]
    model._a8_weight_scales_ref=scales
    old=dec.linear; codebook_ids={id(layers[n]) for n in names}
    def dispatch(x,m):
        if id(m) not in codebook_ids:return old(x,m)
        return linear(x,m,scales[id(m)],True)
    for name,m in layers.items():
        cb=name in names
        def forward(self,x,cb=cb):return linear(x,self,scales[id(self)],cb)
        m.forward=types.MethodType(forward,m)
        m._a8_prepare_activation=prepare_activation
        m._a8_minimum_rows=16
    dec.linear=dispatch
    def restore():
        dec.linear=old
        for name,m in layers.items():m.forward=orig[name]
    return dict(format=format,mode=mode,names=sorted(allnames),
        activation_scale='dynamic_per_tensor_absmax',weight_scale='per_tensor_absmax_precomputed_exact',
        dense_weight_cache=False,real_gemm=True,accumulation='fp32',
        exact_fp8_weight_operands=checks,fused_decode=True,decode_tile_tuning=DECODE_TUNING,
        vector64_codebook_gather=True,joint_weight_decode_absmax=True,native_bf16x2_decoder_arithmetic=True,frozen_decode_selection=True,temporary_pair_shared_activation=True,
        fused_scale_finish_activation=True,activation_operand_gate=True,
        native_bias_bf16_epilogue=True,packed_weight_scales=True,paired_fp8_codebook_conversion=True,
        weight_reciprocal_exact_layers=[n for n,m in layers.items() if m._a8_fast_weight_div],
        weight_precise_division_layers=[n for n,m in layers.items() if not m._a8_fast_weight_div],
        weight_scale_logical_bytes=sum(s.numel()*s.element_size() for s in scales.values())),restore
