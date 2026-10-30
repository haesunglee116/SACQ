"""Original decoder bytes, combined native SiLU/mul/amax producer."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from a8_joint_pair_r1 import _fp8_pair

@triton.jit
def _decode_codebook_and_swiglu_absmax(
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    weight_ptr,
    fp8_scale_ptr,
    g_ptr, u_ptr, a_ptr, partial_ptr, COUNT:tl.constexpr,
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
        activation_g=tl.load(g_ptr+offs,offs<COUNT,0).to(tl.float32)
        activation_u=tl.load(u_ptr+offs,offs<COUNT,0).to(tl.float32)
        activation_silu=tl.div_rn(activation_g,1.0+libdevice.exp(-activation_g)).to(tl.bfloat16)
        activation_value=(activation_silu.to(tl.float32)*activation_u).to(tl.bfloat16)
        tl.store(a_ptr+offs,activation_value,offs<COUNT)
        tl.store(partial_ptr+part,tl.max(tl.abs(activation_value.to(tl.float32)),0))

