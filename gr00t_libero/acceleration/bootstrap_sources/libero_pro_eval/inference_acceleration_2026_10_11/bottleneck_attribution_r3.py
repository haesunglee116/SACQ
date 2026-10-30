"""Read-only numerical/cost attribution of the existing exact A16 runtime.

Microbenchmarks are not deployment speed claims. Cache-pressure probes do not
measure physical DRAM bytes; profiler kernel-time sums may overlap across streams.
"""
from collections import defaultdict
import statistics
import torch


def region(name):
    if 'language_model' in name: return 'language'
    if '.visual.' in name: return 'vision'
    if 'action_head' in name: return 'action_head'
    return 'other'


def capture_inputs(model):
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    samples={}; handles=[]
    def hook(name):
        def collect(m,args):
            x=args[0]
            codec='sacq_codebook' if type(m) is Packed else 'bf16_linear'
            key=(region(name),codec,x.numel()//x.shape[-1],m.out_features,m.in_features,
                 m.bias is not None,getattr(m,'index_bits',None),getattr(m,'index_window_bytes',None))
            if key not in samples:
                samples[key]=dict(name=name,module=m,x_cpu=x.detach().cpu().clone(),geometry=key)
        return collect
    for name,m in model.named_modules():
        # Only actual SACQ/native linear counterparts. Other formats remain in
        # the full-model profiler, not silently attributed to the SACQ decoder.
        if type(m) not in (torch.nn.Linear,Packed): continue
        if not ('language_model.layers.' in name or 'action_head.model.transformer_blocks.' in name): continue
        if '.norm1.linear' in name: continue  # r6 uses fixed-schedule output cache
        handles.append(m.register_forward_pre_hook(hook(name)))
    return samples,handles


def category(name):
    n=name.lower()
    if '_dequantize_packed_codebook_weight' in n: return 'sacq_weight_decode'
    if '_hqq4_dequant_kernel' in n: return 'hqq_weight_decode'
    if n.startswith('_decode'): return 'dictionary_weight_decode'
    if 'gemm' in n or 'gemv' in n or 'matmul' in n or 'splitkreduce' in n: return 'gemm_and_reduction'
    if 'flash' in n or 'fmha' in n or 'attention' in n: return 'attention'
    if 'layernorm' in n or 'layer_norm' in n or 'rmsnorm' in n: return 'normalization'
    if 'memcpy' in n or 'memset' in n: return 'copy_or_fill'
    return 'other'


def profile_official(policy,obs,official,c,expected_rng):
    bb=policy.model.backbone; ah=policy.model.action_head
    old_bb=bb.forward; old_ah=ah.get_action
    def wrap(fn,label):
        def wrapped(*args,**kwargs):
            with torch.profiler.record_function(label):
                value=fn(*args,**kwargs)
                torch.cuda.synchronize()
            return value
        return wrapped
    bb.forward=wrap(old_bb,'ATTRIBUTION_BACKBONE')
    ah.get_action=wrap(old_ah,'ATTRIBUTION_ACTION_HEAD')
    try:
        official.set_seed(c['seed'])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            official.benchmark_components(policy,obs,c['num_iterations'],c['warmup'])
        assert torch.equal(torch.cuda.get_rng_state(),expected_rng), 'Profiler changed RNG stream'
    finally:
        bb.forward=old_bb;ah.get_action=old_ah
    events=prof.events()
    ranges=[(e.time_range.start,e.time_range.end,e.name) for e in events
            if e.device_type==torch.autograd.DeviceType.CPU and e.name in ('ATTRIBUTION_BACKBONE','ATTRIBUTION_ACTION_HEAD')]
    counts=defaultdict(int)
    intervals=[]
    for start,end,label in sorted(ranges):
        index=counts[label];counts[label]+=1
        intervals.append((start,end,label,index))
    expected_calls=c['warmup']+c['num_iterations']
    assert dict(counts)=={'ATTRIBUTION_BACKBONE':expected_calls,'ATTRIBUTION_ACTION_HEAD':expected_calls},dict(counts)
    grouped=defaultdict(lambda:dict(count=0,total_us=0.0))
    unmatched=0
    for e in events:
        if e.device_type!=torch.autograd.DeviceType.CUDA: continue
        # Kineto creates device-side user annotation events too; those are
        # envelopes, not kernels, and must not be double-counted.
        if e.name in ('ATTRIBUTION_BACKBONE','ATTRIBUTION_ACTION_HEAD'): continue
        match=next(((label,index) for start,end,label,index in intervals
                    if start<=e.time_range.start and e.time_range.end<=end),None)
        if match is None:
            unmatched+=1; stage='unassigned'
        else:
            label,index=match
            if index<c['warmup']:continue
            stage=label.removeprefix('ATTRIBUTION_').lower()
        key=(stage,category(e.name),e.name)
        grouped[key]['count']+=1;grouped[key]['total_us']+=e.device_time_total
    rows=[dict(stage=k[0],category=k[1],kernel=k[2],**v,
               sum_ms_per_measured_query=v['total_us']/1000/c['num_iterations'])
          for k,v in grouped.items()]
    return dict(rows=rows,component_calls=dict(counts),unassigned_events=unmatched,
                rng_exact=True,iterations=c['num_iterations'],warmup=c['warmup'],
                caveat='CUDA kernel duration sums can overlap on streams. Profiled wall time is excluded from speed claims. No hardware DRAM counters collected.')


def graph_measure(fn,c,pressure=None):
    # Record timing nodes inside the CUDA graph. Events outside graph.replay()
    # can include host enqueue gaps for tiny kernels and bias cache controls.
    import ctypes
    from pathlib import Path
    cudart_path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libcudart.so' in line)
    cudart=ctypes.CDLL(cudart_path)
    record=cudart.cudaEventRecordWithFlags
    record.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint)
    record.restype=ctypes.c_int
    # cudaEventRecordExternal=0x01 in the installed CUDA driver_types.h.
    def external_record(event):
        err=record(event.cuda_event,torch.cuda.current_stream().cuda_stream,1)
        if err:raise RuntimeError(f'cudaEventRecordWithFlags failed: {err}')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(c['warmup']):value=fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();end.record();end.synchronize()  # initialize handles
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        external_record(start)
        value=fn()
        external_record(end)
    samples=[]
    for _ in range(c['num_iterations']):
        if pressure is not None:pressure.zero_()
        graph.replay();end.synchronize()
        samples.append(start.elapsed_time(end))
    assert all(v>=0 for v in samples)
    return dict(median_ms=statistics.median(samples),samples_ms=samples,
                event_placement='external CUDA timing-event nodes inside the same graph',
                cuda_runtime_library=cudart_path),value.clone()


def microbench(samples,c,emit):
    import decoder_tuning as decoder
    l2=torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    # Diagnostic control only: touch four times the hardware-reported L2 size.
    # Excluded from event timing; not asserted to be a perfect cache flush.
    pressure=torch.empty((4*l2,),dtype=torch.uint8,device='cuda')
    rows=[]
    with torch.inference_mode():
        for key,sample in sorted(samples.items(),key=lambda item:str(item[0])):
            m=sample['module'];x=sample['x_cpu'].to('cuda'); packed=key[1]=='sacq_codebook'
            w=decoder.dequantize(m) if packed else m.weight
            ref=torch.nn.functional.linear(x,w,m.bias)
            row=dict(name=sample['name'],geometry=key,input_shape=list(x.shape),
                     dense_weight_bytes=w.numel()*w.element_size(),
                     logical_dense_gemm_flops=2*(x.numel()//x.shape[-1])*m.out_features*m.in_features,
                     measurements={})
            if packed:
                reference_w=m.dequantize_weight()
                assert torch.equal(w.view(torch.int16),reference_w.view(torch.int16))
                del reference_w
                row['stored_weight_payload_bytes']=sum(getattr(m,n).numel()*getattr(m,n).element_size()
                    for n in ('packed_indices','codebook','scales','shift'))
            else:row['stored_weight_payload_bytes']=row['dense_weight_bytes']
            cases=[('dense_gemm',lambda:torch.nn.functional.linear(x,w,m.bias),ref)]
            if packed:
                cases.extend([('decode_only',lambda:decoder.dequantize(m),w),
                              ('decode_then_gemm',lambda:decoder.linear(x,m),ref)])
            for label,fn,expected in cases:
                for mode,pbuf in [('repeated',None),('cache_pressure',pressure)]:
                    timing,output=graph_measure(fn,c,pbuf)
                    timing['output_bitwise_exact']=torch.equal(output.view(torch.int16),expected.view(torch.int16))
                    assert timing['output_bitwise_exact'],(sample['name'],label,mode)
                    row['measurements'][label+'_'+mode]=timing
                    del output
            rows.append(row);emit(rows)
            print('ATTRIBUTION_MICRO',sample['name'],key,{k:round(v['median_ms']*1000,3) for k,v in row['measurements'].items()},flush=True)
            del x,w,ref
    return dict(rows=rows,L2_cache_bytes=l2,cache_pressure_bytes=pressure.numel(),
                iterations=c['num_iterations'],warmup=c['warmup'],
                scope='One actual input per region/format/M/N/K/bias/index geometry; language and action transformer linears. Fixed-schedule AdaLN excluded. Native and SACQ inputs come from their respective unoptimized references.',
                caveats=['Dense GEMM for a SACQ layer uses its decoded quantized weight to isolate decode overhead, not the original BF16 weight.',
                         'Repeated microbenchmarks favor cache reuse. Cache-pressure results are sensitivity controls, not measured DRAM traffic or real end-to-end latency.',
                         'Separate component medians are not additive; decoder can warm the weight cache for following GEMM.',
                         'Do not sum representative-layer timings to claim full-model cost; full-model profiler covers the real executed kernels.'])
