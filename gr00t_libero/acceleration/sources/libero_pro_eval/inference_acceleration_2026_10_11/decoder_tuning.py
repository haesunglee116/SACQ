"""Tune launch geometry only; preserve the existing BF16 decoder arithmetic."""
import statistics
import torch
import triton
from codebook_linear_triton_2026_08_23 import _dequantize_packed_codebook_weight

CONFIGS = [(16,32,4), (8,64,4), (16,64,4), (16,128,4), (32,32,4), (32,64,4), (8,128,4)]
SELECTED = {}


def key(m):
    return (m.out_features, m.in_features, m.group_size, m.vector_dim,
            m.index_bits, m.index_window_bytes)


def dequantize(m, config=None):
    if m.vector_dim != 4:
        raise ValueError('Only the existing d4 decoder is supported')
    bn, bv, nw = config or SELECTED.get(key(m), CONFIGS[0])
    weight = torch.empty((m.out_features,m.in_features), dtype=torch.bfloat16, device=m.codebook.device)
    _dequantize_packed_codebook_weight[(triton.cdiv(m.out_features,bn),triton.cdiv(m.in_features//4,bv))](
        m.packed_indices,m.codebook,m.scales,m.shift,weight,
        N=m.out_features,K=m.in_features,NUM_GROUPS=m.in_features//m.group_size,
        GROUP_SIZE=m.group_size,VECTOR_DIM=4,INDEX_BITS=m.index_bits,
        INDEX_WINDOW_BYTES=m.index_window_bytes,BLOCK_N=bn,BLOCK_V=bv,num_warps=nw)
    return weight


def linear(x, m):
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise TypeError('CUDA BF16 input required')
    return torch.nn.functional.linear(x,dequantize(m),m.bias)


def sweep(policy, bi, ai, iterations, warmup):
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    layers = {n:m for n,m in policy.model.named_modules() if type(m) is Packed}
    calls = {n:0 for n in layers}
    handles = []
    def count(n):
        def hook(m, args):
            calls[n] += 1
        return hook
    for n,m in layers.items():
        handles.append(m.register_forward_pre_hook(count(n)))
    try:
        bo = policy.model.backbone(bi)
        policy.model.action_head.get_action(bo,ai)
    finally:
        for h in handles:
            h.remove()
    # Exact runtime caches AdaLN outputs and hoists cross-attention K/V.
    for block in policy.model.action_head.model.transformer_blocks:
        for n,m in layers.items():
            if m is block.norm1.linear:
                calls[n] = 0
            if m in (block.attn1.to_k,block.attn1.to_v) and m.in_features != block.attn1.to_q.in_features:
                calls[n] = min(calls[n],1)
    groups = {}
    for n,m in layers.items():
        if calls[n]:
            groups.setdefault(key(m),dict(module=m,calls=0,names=[]))
            groups[key(m)]['calls'] += calls[n]
            groups[key(m)]['names'].append(n)
    result = []
    for geometry,g in sorted(groups.items()):
        m=g['module']; reference=m.dequantize_weight(); timings=[]
        for config in CONFIGS:
            candidate=dequantize(m,config)
            if not torch.equal(candidate.view(torch.int16),reference.view(torch.int16)):
                raise RuntimeError(f'Decoder launch changed weights: {geometry}/{config}')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    candidate=dequantize(m,config)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured=dequantize(m,config)
            graph.replay();torch.cuda.synchronize()
            assert torch.equal(captured.view(torch.int16),reference.view(torch.int16))
            samples=[]
            for _ in range(iterations):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                samples.append(start.elapsed_time(end))
            timings.append(dict(config=config,median_ms=statistics.median(samples),samples_ms=samples))
            del graph,captured,candidate
        best=min(timings,key=lambda x:x['median_ms'])
        SELECTED[geometry]=tuple(best['config'])
        row=dict(geometry=geometry,calls=g['calls'],layers=g['names'],selected=best['config'],timings=timings)
        result.append(row)
        print('DECODE_TUNING',geometry,'selected',best['config'],'ms',best['median_ms'],flush=True)
        del reference
    return dict(groups=result,iterations=iterations,warmup=warmup,weights_exact=True,
                caveat='Per-geometry microbenchmark; full-policy timing and action gates decide acceptance')
