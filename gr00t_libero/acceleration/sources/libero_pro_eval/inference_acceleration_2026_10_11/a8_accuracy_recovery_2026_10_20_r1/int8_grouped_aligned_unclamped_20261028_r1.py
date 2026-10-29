"""Tune grouped QKV decoding using current aligned-index and BF16x2 math."""
import torch,triton,json,statistics
import triton.language as tl
from pathlib import Path
from int8_refined_divide_20261028_r1 import _divide
from int8_aligned_index_decoder_20261028_r1 import _affine_bf16x4,_int8_pair
REPORT_PATH=None
CHOICES={}
RECORDS=[]
CONFIGS=[(4,128),(8,128),(16,64),(8,64)]
@triton.jit
def _decode_group_component(
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
    PID_N,
):
    """Decode one packed index and store all d4 BF16 weight components."""
    pid_n = PID_N
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
def _decode_grouped_qkv_weights(P0,C0,S0,H0,O0,W0,P1,C1,S1,H1,O1,W1,P2,C2,S2,H2,O2,W2,
    N0:tl.constexpr,N1:tl.constexpr,N2:tl.constexpr,K:tl.constexpr,
    B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
    I0:tl.constexpr,I1:tl.constexpr,I2:tl.constexpr,
    M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,
    BN:tl.constexpr,BV:tl.constexpr):
    p=tl.program_id(0)
    T0:tl.constexpr=triton.cdiv(N0,BN)
    T1:tl.constexpr=triton.cdiv(N1,BN)
    if p<T0:
        _decode_group_component(P0,C0,S0,H0,O0,W0,N0,K,K//64,64,4,B0,I0,M0,BN,BV,p)
    elif p<T0+T1:
        _decode_group_component(P1,C1,S1,H1,O1,W1,N1,K,K//64,64,4,B1,I1,M1,BN,BV,p-T0)
    else:
        _decode_group_component(P2,C2,S2,H2,O2,W2,N2,K,K//64,64,4,B2,I2,M2,BN,BV,p-T0-T1)

def decode(modules,modes,validate):
    import int8_refined_policy_20261028_r1 as refined
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    if not all(type(m) is Packed for m in modules):return [refined.decoder.decode(m,mode) for m,mode in zip(modules,modes)]
    assert len(modules)==len(modes)==3
    k=modules[0].in_features
    assert all(m.in_features==k and m.vector_dim==4 and m.group_size==64 and m.shift.numel()==1 and m.packed_indices.data_ptr()%4==0 for m in modules)
    ns=[m.out_features for m in modules]
    key=tuple(ns+[k]+[m.index_bits for m in modules]+[m.index_window_bytes for m in modules]+list(modes))
    def baseline():return [refined.decoder.decode(m,mode) for m,mode in zip(modules,modes)]
    def run(cfg):
        bn,bv=cfg
        outputs=[torch.empty((n,k),device=modules[0].codebook.device,dtype=torch.int8) for n in ns]
        args=[]
        for m,out in zip(modules,outputs):args.extend([m.packed_indices,m.codebook,m.scales,m.shift,out,m._a8_perrow_weight_scale])
        _decode_grouped_qkv_weights[(sum(triton.cdiv(n,bn) for n in ns),triton.cdiv(k//4,bv))](*args,*ns,k,*[m.index_bits for m in modules],*[m.index_window_bytes for m in modules],*modes,bn,bv,num_warps=4,enable_fp_fusion=False)
        return outputs
    assert key in CHOICES, ('UnseenFrozenGroupedShape',key)
    cfg=CHOICES[key]
    outputs=baseline() if cfg is None else run(cfg)
    if validate:
        expected=baseline()
        assert all(torch.equal(x,y) for x,y in zip(outputs,expected)),'GroupedAlignedEveryWeightBytes'
    return outputs


def install(selection):
    import hashlib
    p=Path(selection['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==selection['sha256']
    data=json.loads(p.read_text()); assert data['all_candidate_weight_bytes_exact']
    assert not CHOICES
    for row in data['choices']:
        key=tuple(row['shape']); assert key not in CHOICES
        CHOICES[key]=None if row['selected'] is None else tuple(row['selected'])
