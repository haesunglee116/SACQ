"""Post-timing event-node attribution of the already optimized backbone.

A separate diagnostic graph is released afterwards. Its temporary memory and
instrumented timings do not replace deployment memory or official latency.
"""
import ctypes
import statistics
from pathlib import Path
import torch
from transformers.feature_extraction_utils import BatchFeature


def run(policy,bi,contract):
    owner=policy.model.backbone._omni_backbone_graph_owner
    original=owner['original'];stream=owner['stream'];static=owner['static']
    hf=policy.model.backbone.model.model
    modules={'vision':hf.visual,'language':hf.language_model,'embedding':hf.language_model.embed_tokens}
    old={n:m.forward for n,m in modules.items()}
    reference={k:v.clone() for k,v in policy.model.backbone(bi).items()}
    rng=torch.cuda.get_rng_state().clone()
    path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcudart.so' in line)
    lib=ctypes.CDLL(path);record=lib.cudaEventRecordWithFlags
    record.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint);record.restype=ctypes.c_int
    counts={n:0 for n in modules};pairs={};order=[];instrument=False
    def record_event(event):
        err=record(event.cuda_event,stream.cuda_stream,1)
        assert err==0,('cudaEventRecordWithFlags',err)
    for name,m in modules.items():
        def forward(*args,_name=name,_old=old[name],**kwargs):
            slot=counts[_name];counts[_name]+=1
            if instrument:
                assert slot<len(pairs[_name]),'New phase call appeared during capture'
                order.append((_name,slot,'start'));record_event(pairs[_name][slot][0])
            output=_old(*args,**kwargs)
            if instrument:
                order.append((_name,slot,'end'));record_event(pairs[_name][slot][1])
            return output
        m.forward=forward
    graph=None
    try:
        expected_counts=None
        with torch.cuda.stream(stream):
            for _ in range(contract['warmup']):
                counts.update({n:0 for n in modules})
                output=original(BatchFeature(data=dict(static)))
                if expected_counts is None:expected_counts=dict(counts)
                assert counts==expected_counts,'Phase calls changed at the fixed input'
            assert all(expected_counts.values()),expected_counts
            for name,count in expected_counts.items():
                pairs[name]=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(count)]
            total=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for events in [total,*[p for values in pairs.values() for p in values]]:
                events[0].record();events[1].record()
            torch.cuda.synchronize()
            counts.update({n:0 for n in modules});instrument=True
            graph=torch.cuda.CUDAGraph()
            # Independent diagnostic pool; the original runtime graph remains
            # intact, and its measured resident memory was recorded already.
            with torch.cuda.graph(graph,stream=stream):
                record_event(total[0])
                output=original(BatchFeature(data=dict(static)))
                record_event(total[1])
            assert counts==expected_counts
            graph.replay();total[1].synchronize()
            assert all(torch.equal(output[k],v) for k,v in reference.items()),'Observer backbone mismatch'
            rows=[]
            for _ in range(contract['num_iterations']):
                graph.replay();total[1].synchronize()
                intervals={n:[dict(start_ms=total[0].elapsed_time(a),end_ms=total[0].elapsed_time(b),duration_ms=a.elapsed_time(b))
                    for a,b in values] for n,values in pairs.items()}
                rows.append(dict(total_ms=total[0].elapsed_time(total[1]),intervals=intervals))
        stack=[];nested=False
        for name,slot,position in order:
            if position=='start':
                nested |= bool(stack);stack.append((name,slot))
            else:assert stack.pop()==(name,slot)
        assert not stack
        assert torch.equal(rng,torch.cuda.get_rng_state()),'Observer changed RNG'
        medians={n:statistics.median(sum(x['duration_ms'] for x in row['intervals'][n]) for row in rows) for n in modules}
        total_median=statistics.median(row['total_ms'] for row in rows)
        other=None if nested else statistics.median(row['total_ms']-sum(x['duration_ms'] for values in row['intervals'].values() for x in values) for row in rows)
        result=dict(status='BACKBONE_SPLIT_COMPLETE',diagnostic_only=True,median_ms={**medians,'total':total_median,'other_exclusive':other},
            samples=rows,phase_calls=expected_counts,capture_order=order,nested_intervals=nested,
            own_backbone_exact=True,rng_unchanged=True,cuda_runtime_library=path,
            interpretation='Post-timing isolated backbone graph, consistent with official separate-component timing; not interleaved serving-engine latency. Event instrumentation and diagnostic graph memory excluded from official timing/resident. Nested scopes are not additive.')
    finally:
        for name,m in modules.items():m.forward=old[name]
        instrument=False
        if graph is not None:del graph
        del reference
    return result
