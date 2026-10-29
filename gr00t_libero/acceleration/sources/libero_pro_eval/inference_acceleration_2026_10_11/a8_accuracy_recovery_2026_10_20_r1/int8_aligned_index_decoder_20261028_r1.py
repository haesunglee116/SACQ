"""Original packed BF16 reconstruction fused with exact per-row INT8 RTN."""
from int8_direct_integer_round_20261028_r1 import round_int8
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from int8_refined_divide_20261028_r1 import _divide
@triton.jit
def _int8_rtn(x):return round_int8(x)
@triton.jit
def _int8_pair(a,b):return _int8_rtn(a),_int8_rtn(b)


@triton.jit
def _affine_bf16x4(word,scale,shift):
    lo=word.to(tl.uint32);hi=(word>>32).to(tl.uint32)
    ss=scale.to(tl.uint16,bitcast=True).to(tl.uint32)*0x10001
    hh=shift.to(tl.uint16,bitcast=True).to(tl.uint32)*0x10001
    a,b=tl.inline_asm_elementwise(
        "{ .reg .b32 nz, one; mov.b32 nz, 0x80008000; mov.b32 one, 0x3f803f80; "
        "fma.rn.bf16x2 $0, $2, $4, nz; fma.rn.bf16x2 $0, $0, one, $5; "
        "fma.rn.bf16x2 $1, $3, $4, nz; fma.rn.bf16x2 $1, $1, one, $5; }",
        constraints='=r,=r,r,r,r,r',args=[lo,hi,ss,hh],dtype=(tl.uint32,tl.uint32),is_pure=True,pack=1)
    return a.to(tl.uint64)|(b.to(tl.uint64)<<32)

@triton.jit
def _decode_codebook_perrow_int8(
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    weight_ptr,
    int8_scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    MODE: tl.constexpr,
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
    # Packed rows have an integral number of uint32 words for approved shapes.
    tl.static_assert((N*(K//VECTOR_DIM)*INDEX_BITS)%32==0)
    word_ids=bit_positions//32
    bit_offsets=bit_positions%32
    TOTAL_WORDS:tl.constexpr=N*(K//VECTOR_DIM)*INDEX_BITS//32
    first=tl.load(packed_ptr.to(tl.pointer_type(tl.uint32))+word_ids,mask=valid_nv,other=0)
    crosses=(bit_offsets+INDEX_BITS)>32
    second=tl.load(packed_ptr.to(tl.pointer_type(tl.uint32))+word_ids+1,
                   mask=valid_nv&crosses&(word_ids+1<TOTAL_WORDS),other=0)
    words=(first>>bit_offsets)|(second<<(32-bit_offsets))
    indices=(words&((1<<INDEX_BITS)-1)).to(tl.int32)
    group_ids = offs_v // vectors_per_group
    scale = tl.load(
        scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
        mask=valid_nv,
        other=0.0,
    )
    shift = tl.load(shift_ptr)
    word=tl.load(codebook_ptr.to(tl.pointer_type(tl.uint64))+indices,mask=valid_nv,other=0)
    word=_affine_bf16x4(word,scale,shift)
    q0=word.to(tl.uint16).to(tl.bfloat16,bitcast=True)
    q1=(word>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    q2=(word>>32).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    q3=(word>>48).to(tl.uint16).to(tl.bfloat16,bitcast=True)
    w0 = q0
    w1 = q1
    w2 = q2
    w3 = q3
    sw=tl.load(int8_scale_ptr+offs_n,offs_n<N,1.0)[:,None]
    w0=_divide(w0.to(tl.float32),sw,MODE);w1=_divide(w1.to(tl.float32),sw,MODE)
    w2=_divide(w2.to(tl.float32),sw,MODE);w3=_divide(w3.to(tl.float32),sw,MODE)
    w0=tl.minimum(tl.maximum(w0,-127.0),127.0)
    w1=tl.minimum(tl.maximum(w1,-127.0),127.0)
    w2=tl.minimum(tl.maximum(w2,-127.0),127.0)
    w3=tl.minimum(tl.maximum(w3,-127.0),127.0)
    p0,p1=_int8_pair(w0,w1);p2,p3=_int8_pair(w2,w3)
    values=tl.interleave(tl.interleave(p0,p2),tl.interleave(p1,p3))
    offs_k = pid_v * BLOCK_V * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
    valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        values,
        mask=valid_nk,
    )

@triton.jit
def _decode_affine_perrow_int8(P,S,O,SW,Q,COUNT:tl.constexpr,K:tl.constexpr,
                              BITS:tl.constexpr,MODE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<COUNT
    bit=i*BITS;position=bit%8
    word=tl.load(P+bit//8,valid,0).to(tl.uint32)
    if BITS==3:word|=tl.load(P+bit//8+1,valid&(position+BITS>8),0).to(tl.uint32)<<8
    q=((word>>position)&((1<<BITS)-1)).to(tl.int32)
    offset=tl.load(O+i//64,valid,0).to(tl.int32);scale=tl.load(S+i//64,valid,0)
    w=((q+offset).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
    sw=tl.load(SW+i//K,valid,1)
    value=tl.minimum(tl.maximum(_divide(w,sw,MODE),-127.0),127.0)
    tl.store(Q+i,round_int8(value),valid)

def decode(m,mode):
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear as Affine
    n,k=m.out_features,m.in_features;sw=m._a8_perrow_weight_scale
    assert tuple(sw.shape)==(n,1) and sw.dtype==torch.float32
    out=torch.empty((n,k),device=sw.device,dtype=torch.int8)
    if type(m) is Packed:
        assert m.packed_indices.data_ptr()%4==0
        import decoder_tuning
        assert m.vector_dim==4 and m.codebook.dtype==torch.bfloat16 and m.codebook.data_ptr()%8==0
        bn,bv,warps=decoder_tuning.SELECTED.get(decoder_tuning.key(m),decoder_tuning.CONFIGS[0])
        _decode_codebook_perrow_int8[(triton.cdiv(n,bn),triton.cdiv(k//4,bv))](m.packed_indices,m.codebook,m.scales,m.shift,out,sw,
          n,k,k//m.group_size,m.group_size,4,m.index_bits,m.index_window_bytes,mode,bn,bv,num_warps=warps,enable_fp_fusion=False)
    else:
        assert type(m) is Affine and m.group_size==64 and m.bits in (2,3,4)
        _decode_affine_perrow_int8[(triton.cdiv(n*k,1024),)](m.packed,m.scales,m.offsets,sw,out,n*k,k,m.bits,mode,1024,num_warps=4,enable_fp_fusion=False)
    return out

