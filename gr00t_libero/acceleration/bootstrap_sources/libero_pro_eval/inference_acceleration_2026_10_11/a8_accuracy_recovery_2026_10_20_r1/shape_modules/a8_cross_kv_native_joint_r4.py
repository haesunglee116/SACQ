import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Fuse K weight decoding with AQ, retain native GEMMs and compact M156 outputs.

The original frozen decoder tile and quantizer kernels are reused. K and V
retain separate original Sw. FP8 weight buffers are temporary, used one at a
time; no decoded weights are cached across projections or queries.
"""
import torch
import triton

def evaluate(x, members, names, scales, debug=False):
    import a8_joint_pair_r1 as fp8
    assert len(members) == len(names) == 2 and tuple(x.shape) == (1, ROLLOUT_SHAPE.rows, 2048)
    assert x.dtype == torch.bfloat16 and x.is_cuda
    kmod, vmod = members
    assert all(((m.in_features, m.out_features, m.vector_dim, m.group_size) == (2048, 1536, 4, 64) for m in members))
    a = x.reshape(ROLLOUT_SHAPE.rows, 2048).contiguous()
    rows, k = a.shape
    n = kmod.out_features
    aq = torch.empty((rows, k), device=a.device, dtype=torch.float8_e4m3fn)
    wq = torch.empty((n, k), device=a.device, dtype=torch.float8_e4m3fn)
    parts = triton.cdiv(a.numel(), 4096)
    partials = torch.empty(parts, device=a.device, dtype=torch.float32)
    sa = torch.empty((), device=a.device, dtype=torch.float32)
    bn, bv, warps = fp8.DECODE_TILES[fp8._decode_key(kmod)]
    blocks = triton.cdiv(n, bn) * triton.cdiv(k // 4, bv)
    fp8._decode_codebook_and_absmax[blocks + parts,](kmod.packed_indices, kmod.codebook, kmod.scales, kmod.shift, wq, scales[names[0]], a, partials, a.numel(), N=n, K=k, NUM_GROUPS=k // kmod.group_size, GROUP_SIZE=kmod.group_size, VECTOR_DIM=4, INDEX_BITS=kmod.index_bits, INDEX_WINDOW_BYTES=kmod.index_window_bytes, RECIP=kmod._a8_fast_weight_div, BLOCK_N=bn, BLOCK_V=bv, num_warps=warps, enable_fp_fusion=False)
    fp8._finish_quantize[triton.cdiv(rows * k, 1024),](a, partials, aq, sa, a.numel(), rows * k, parts, triton.next_power_of_2(parts), 1024, num_warps=4)
    details = {}
    if debug:
        assert not torch.cuda.is_current_stream_capturing()
        from fused_ffup_cutlass_probe_r1b import bitsha
        details[names[0]] = bitsha(wq)
    yk = torch._scaled_mm(aq, wq.t(), sa, scales[names[0]], bias=kmod.bias, out_dtype=torch.bfloat16, use_fast_accum=False)
    if isinstance(yk, tuple):
        yk = yk[0]
    del wq
    wq = fp8.decode(vmod, scales[names[1]], True)
    if debug:
        details[names[1]] = bitsha(wq)
    yv = torch._scaled_mm(aq, wq.t(), sa, scales[names[1]], bias=vmod.bias, out_dtype=torch.bfloat16, use_fast_accum=False)
    if isinstance(yv, tuple):
        yv = yv[0]
    del wq
    outputs = [yk.reshape(1, ROLLOUT_SHAPE.rows, 1536), yv.reshape(1, ROLLOUT_SHAPE.rows, 1536)]
    assert all((y.is_contiguous() and y.untyped_storage().nbytes() == y.numel() * y.element_size() for y in outputs))
    if debug:
        return (outputs, dict(weight_FP8_sha256=details, aq=aq, sa=sa))
    return outputs
