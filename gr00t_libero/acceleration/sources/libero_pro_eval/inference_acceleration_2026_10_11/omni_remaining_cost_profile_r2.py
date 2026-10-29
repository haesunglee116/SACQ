"""Read-only numerical diagnostics of the frozen Omni inference implementation.

Deployment measurements precede all observers. Expanded weights exist only in
diagnostics. Activity duration, union duration and counterfactual latency have
different meanings and are never combined into an additive speedup budget.
"""
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


def save(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def stats(values):
    return dict(median_ms=statistics.median(values), min_ms=min(values),
                max_ms=max(values), samples_ms=values)


def telemetry():
    r = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return dict(time_ns=time.time_ns(), csv=r.stdout.strip(), exitcode=r.returncode)


def union_duration(intervals):
    total = 0.; end = None
    for a, b in sorted(intervals):
        assert b >= a
        if end is None or a > end:
            total += b-a; end=b
        elif b > end:
            total += b-end; end=b
    return total


def category(name):
    if re.search(r'dequant|decode|decompress|packed.*weight', name, re.I): return 'weight_decode'
    if re.search(r'flash|fmha|attention', name, re.I): return 'attention'
    if re.search(r'gemm|xmma|mma|matmul', name, re.I): return 'gemm'
    if re.search(r'memcpy|memset', name, re.I): return 'copy'
    return 'other'


def analyze_trace(trace, count):
    events = trace['traceEvents']
    queries = sorted([e for e in events if e.get('ph')=='X' and
                      e.get('name','').startswith('cost/query/')], key=lambda e:e['ts'])
    assert len(queries)==count, ('query count',len(queries),count)
    phases = [e for e in events if e.get('ph')=='X' and e.get('name') in
              ('cost/preprocessing','cost/backbone','cost/action')]
    external = {}
    for e in events:
        if e.get('ph')!='X' or 'External id' not in e.get('args',{}): continue
        if e.get('cat') not in ('cuda_runtime','cpu_op'): continue
        enclosing = [p for p in phases if p.get('tid')==e.get('tid') and
                     p['ts']<=e['ts']<p['ts']+p['dur']]
        if enclosing: external[str(e['args']['External id'])]=enclosing[-1]['name'].split('/')[-1]
    # GPU user annotations are enclosing ranges, not kernels.
    gpu = [e for e in events if e.get('ph')=='X' and
           e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    assert gpu, 'No actual CUDA kernels/copies in trace'
    groups=defaultdict(lambda:dict(sum_us=0.,count=0))
    names=defaultdict(lambda:dict(sum_us=0.,count=0))
    rows=[]
    for q in queries:
        ks=[e for e in gpu if q['ts']<=e['ts'] and e['ts']+e['dur']<=q['ts']+q['dur']]
        assert ks, ('empty query',q['name'])
        intervals=[(e['ts'],e['ts']+e['dur']) for e in ks]
        active=union_duration(intervals);span=max(b for a,b in intervals)-min(a for a,b in intervals)
        rows.append(dict(query=q['name'],profiled_cpu_completion_ms=q['dur']/1000,
                         gpu_active_union_ms=active/1000,gpu_first_to_last_ms=span/1000,
                         gaps_between_gpu_activities_ms=(span-active)/1000,
                         summed_gpu_activity_ms=sum(e['dur'] for e in ks)/1000))
        for e in ks:
            phase=external.get(str(e.get('args',{}).get('External id')), 'unattributed')
            key=phase+'/'+category(e['name'])
            groups[key]['sum_us']+=e['dur'];groups[key]['count']+=1
            names[(phase,e['name'])]['sum_us']+=e['dur'];names[(phase,e['name'])]['count']+=1
    return dict(status='TRACE_ANALYZED',queries=rows,
        median_ms={k:statistics.median(r[k] for r in rows) for k in rows[0] if k!='query'},
        activity_per_query={k:dict(sum_ms=v['sum_us']/count/1000,count=v['count']/count) for k,v in groups.items()},
        top_kernels=[dict(phase=p,name=n,ms_per_query=v['sum_us']/count/1000,calls_per_query=v['count']/count)
                     for (p,n),v in sorted(names.items(),key=lambda t:t[1]['sum_us'],reverse=True)[:80]],
        interpretation='Profiler changes latency. GPU union avoids overlap double counting. Gaps include scheduling, CPU dependency and profiler effects; not automatically removable. External-id phase attribution may be incomplete.')


def prepared(policy,bi,ai):
    bo=policy.model.backbone(bi)
    return policy.model.action_head.get_action(bo,ai).action_pred


def run(policy, obs, contract, arm, out):
    import benchmark_inference as official
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(diagnostic_only=True,arm=arm,seed=contract['seed'],
                iterations=contract['num_iterations'],warmup=contract['warmup'])
    with torch.inference_mode():
        inputs=official.prepare_model_inputs(policy,obs);bi,ai=policy.model.prepare_input(inputs)
        def whole():
            fresh=official.prepare_model_inputs(policy,obs)
            b,a=policy.model.prepare_input(fresh)
            return prepared(policy,b,a)
        # True continuous preprocessor -> backbone -> action, synchronized only
        # at request boundaries, rather than a sum of component arrays.
        before=telemetry(); result['telemetry_before_unprofiled']=before
        for label,fn in [('continuous_e2e',whole),('prepared_model',lambda:prepared(policy,bi,ai))]:
            official.set_seed(contract['seed']);values=[]
            for i in range(contract['warmup']+contract['num_iterations']):
                torch.cuda.synchronize();start=time.perf_counter()
                y=fn();torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                if i>=contract['warmup']:values.append(elapsed)
            result[label]=stats(values)
        result['resident_before_observers_GB']=torch.cuda.memory_allocated()/1e9
        result['telemetry_after_unprofiled']=telemetry()
        del y
        save(out/(arm+'_continuous.json'),result)
        official.set_seed(contract['seed'])
        refs=[prepared(policy,bi,ai).clone() for _ in range(contract['warmup']+contract['num_iterations'])]
        target_rng=torch.cuda.get_rng_state().clone()
        from torch.profiler import profile, ProfilerActivity, record_function
        official.set_seed(contract['seed'])
        for _ in range(contract['warmup']): whole()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for i in range(contract['num_iterations']):
                with record_function('cost/query/'+str(i)):
                    with record_function('cost/preprocessing'):
                        fresh=official.prepare_model_inputs(policy,obs)
                        b,a=policy.model.prepare_input(fresh)
                    with record_function('cost/backbone'):bo=policy.model.backbone(b)
                    with record_function('cost/action'):y=policy.model.action_head.get_action(bo,a).action_pred
                    torch.cuda.synchronize()
                assert torch.equal(y,refs[contract['warmup']+i]),'Profile action changed'
        assert torch.equal(target_rng,torch.cuda.get_rng_state()),'Profile RNG changed'
        local=Path('@LOCAL_RUNTIME@')/(arm+'_cost_trace_'+str(os.getpid())+'.json')
        prof.export_chrome_trace(str(local))
        trace=json.loads(local.read_text())
        analysis=analyze_trace(trace,contract['num_iterations'])
        dst=out/(arm+'_timeline.json.gz');tmp=dst.with_suffix('.tmp')
        with local.open('rb') as f,gzip.open(tmp,'wb') as g:
            import shutil
            shutil.copyfileobj(f,g)
        os.replace(tmp,dst);local.unlink()
        analysis.update(actions_exact=True,rng_exact=True,trace_path=str(dst))
        save(out/(arm+'_timeline_analysis.json'),analysis)
        print('CURRENT_TRACE_COMPLETE',arm,analysis['activity_per_query'],flush=True)
        del trace,prof,refs,y,bo,b,a,fresh
        from omni_backbone_split_profile_r1 import run as split
        result['backbone_split']=split(policy,bi,contract)
        save(out/(arm+'_backbone_split.json'),result['backbone_split'])
        if arm=='sacq_optimized':
            result['micro']=micro_and_counterfactual(policy,bi,ai,contract,out)
        result['status']='REMAINING_COST_PROFILE_COMPLETE'
        save(out/(arm+'_cost_complete.json'),result)
    return result


def decoder_for(n,m,contract,groups):
    from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear
    from hqq.core.quantize import HQQLinear
    if isinstance(m,HQQLinear):return 'hqq4',m.dequantize,m
    if id(m) in groups:
        g=groups[id(m)]
        return 'grouped_qkv',g.decode,g
    if type(m) is LosslessAffineLinear:return 'scalar_affine',m.reconstruct,m
    import decoder_tuning as default
    for key,mod in [('register_r2_selection','omni_decode_register_lut_r2'),
                    ('register_reference_selection','omni_decode_register_lut_r1'),
                    ('native_reference_selection','omni_decode_simd_r3')]:
        rows=json.loads(Path(contract[key]['path']).read_text())['layers']
        if n in rows:
            import importlib
            module=importlib.import_module(mod);cfg=rows[n]
            if key=='native_reference_selection':
                return mod,lambda:module.dequantize(m,tuple(cfg['config']),cfg['native'],cfg['index32']),m
            return mod,lambda:module.dequantize(m,tuple(cfg['config'])),m
    return 'default_decoder',lambda:default.dequantize(m),m


def region(name):
    if '.visual.' in name:return 'vision'
    if name.startswith('backbone.'):return 'language'
    return 'action'


def eager_call(policy,bi,ai):
    bb=policy.model.backbone._omni_backbone_graph_owner
    ah=policy.model.action_head._omni_action_graph_owner
    bo=bb['original'](BatchFeature(data=dict(bb['static'])))
    for k,v in ah['static_bo'].items():v.copy_(bo[k])
    return ah['body']().action_pred


def micro_and_counterfactual(policy,bi,ai,contract,out):
    import benchmark_inference as official
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear
    from hqq.core.quantize import HQQLinear
    from omni_graph_microtime_r1 import measure
    modules={n:m for n,m in policy.model.named_modules() if type(m) in (Packed,LosslessAffineLinear) or isinstance(m,HQQLinear)}
    groups={id(g.members[0]):g for g in policy.model._omni_action_grouped_qkv_owner}
    # Later Q/K/V slots consume views and never perform their own GEMM.
    consumed={id(m) for g in groups.values() for m in g.members[1:]}
    modules={n:m for n,m in modules.items() if id(m) not in consumed}
    decoders={n:decoder_for(n,m,contract,groups) for n,m in modules.items()}
    selections={key:json.loads(Path(contract[key]['path']).read_text())['layers']
                for key in ('register_r2_selection','register_reference_selection','native_reference_selection')}
    stack=[];calls=Counter();cases={};handles=[];original_linear=F.linear
    def pre(n):
        def hook(m,args):stack.append(n)
        return hook
    def post(n):
        def hook(m,args,output):
            if isinstance(m,HQQLinear):
                before_count=calls[n]
                w=m.dequantize();captured=collect(args[0],w,m.bias)
                assert torch.equal(captured,output),'HQQ observer changed matmul/bias path'
                assert calls[n]==before_count+1
            assert stack.pop()==n
        return hook
    for n,m in modules.items():
        handles.extend([m.register_forward_pre_hook(pre(n)),m.register_forward_hook(post(n))])
    def collect(x,w,bias=None):
        if stack:
            n=stack[-1];m=modules[n];kind,decode,owner=decoders[n]
            bits=m.index_bits if hasattr(m,'index_bits') else (m.meta['nbits'] if isinstance(m,HQQLinear) else m.bits)
            geom=(region(n),kind,tuple(x.shape),tuple(w.shape),bits,
                  repr(getattr(m,'config',None)),bias is not None)
            # Distinguish actual selected launch configs even at equal shapes.
            cfgs=[]
            for key in ('register_r2_selection','register_reference_selection','native_reference_selection'):
                cfgs.append(selections[key].get(n))
            geom=geom+(json.dumps(cfgs,sort_keys=True),)
            calls[n]+=1
            if geom not in cases:
                cases[geom]=dict(name=n,module=m,decode=decode,owner=owner,kind=kind,
                                x=x.detach().contiguous().cpu(),weight_sha=hashlib.sha256(w.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                                bias=None if bias is None else bias.clone(),members=set())
            cases[geom]['members'].add(n)
        if stack and isinstance(modules[stack[-1]],HQQLinear):
            y=torch.matmul(x,w.t())
            if bias is not None:y+=bias
            return y
        return original_linear(x,w,bias)
    official.set_seed(contract['seed'])
    refs=[prepared(policy,bi,ai).clone() for _ in range(contract['warmup']+contract['num_iterations'])]
    target_rng=torch.cuda.get_rng_state().clone()
    F.linear=collect
    official.set_seed(contract['seed'])
    try:
        for ref in refs:
            y=eager_call(policy,bi,ai)
            assert torch.equal(y,ref),'Eager post-cache observer changed action'
        assert torch.equal(target_rng,torch.cuda.get_rng_state()),'Observer RNG'
    finally:
        F.linear=original_linear
        for h in handles:h.remove()
    assert cases and not stack
    local=Path('@LOCAL_RUNTIME@/current_cost_'+str(os.getpid()));local.mkdir()
    rows=[];artifacts=[]
    for idx,(geometry,case) in enumerate(sorted(cases.items(),key=lambda t:repr(t[0]))):
        x=case['x'].cuda();w=case['decode']();bias=case['bias']
        sha=hashlib.sha256(w.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        assert sha==case['weight_sha'],'Extracted decoder is not active weight'
        dt,dw=measure(case['decode'],contract)
        assert torch.equal(dw.view(torch.int16),w.view(torch.int16))
        def gemm(weight):
            if case['kind']=='hqq4':
                y=torch.matmul(x,weight.t())
                if bias is not None:y+=bias
                return y
            return F.linear(x,weight,bias)
        gt,gy=measure(lambda:gemm(w),contract)
        st,sy=measure(lambda:gemm(case['decode']()),contract)
        assert torch.equal(gy,sy),'Diagnostic split/dense GEMM changed output'
        rows.append(dict(case=idx,region=geometry[0],kind=case['kind'],input_shape=list(x.shape),weight_shape=list(w.shape),
            layers=sorted(case['members']),calls_per_query=sum(calls[n] for n in case['members'])/len(refs),
            weight_bits_exact=True,output_exact=True,decode=dt,dense_native_gemm=gt,decode_plus_native_gemm=st,
            split_minus_dense_us=st['median_us']-gt['median_us'],
            dense_weight_bytes=w.numel()*w.element_size(),input_sha256=hashlib.sha256(case['x'].view(torch.uint8).numpy().tobytes()).hexdigest()))
        # Actual operands for an isolated hardware-counter permission probe.
        f=local/(str(idx)+'.pt');torch.save(dict(x=x.cpu(),weight=w.cpu(),bias=None if bias is None else bias.cpu()),f)
        artifacts.append((idx,f))
        save(out/'actual_geometry_micros.json',dict(status='MICROS_RUNNING',rows=rows))
        print('CURRENT_COST_MICRO',idx,geometry[0],case['kind'],dt['median_us'],gt['median_us'],st['median_us'],flush=True)
        del x,w,dw,gy,sy
    micros=dict(status='ACTUAL_GEOMETRY_MICROS_COMPLETE',rows=rows,
        call_counts=dict(calls),own25actions_rng_exact=True,
        interpretation='Hot repeated-geometry measurements. split-minus-dense also changes cache/read/write behavior; standalone decode is not an additive wall-time saving. Current runtime uses native GEMM after reconstruction; no custom fused-MMA tile to bypass.')
    save(out/'actual_geometry_micros.json',micros)
    counter=cache_counterfactual(policy,bi,ai,contract,modules,groups,calls,refs,target_rng,out)
    hardware=hardware_probe(artifacts,rows,contract,out)
    for _,f in artifacts:f.unlink()
    local.rmdir()
    del refs,ref,y
    return dict(cases=len(rows),called_projection_layers=len(calls),counterfactual=counter,hardware_counters=hardware)


def cache_counterfactual(policy,bi,ai,contract,modules,groups,calls,refs,target_rng,out):
    import benchmark_inference as official
    from omni_graph_microtime_r1 import measure
    from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear
    from hqq.core.quantize import HQQLinear
    rows=[]
    def measure_policy(fn):
        # Each comparison starts from precisely the same approved seed stream.
        stream=torch.cuda.current_stream()
        for _ in range(contract['warmup']):fn()
        torch.cuda.synchronize()
        runtime_path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                          if '/libcudart.so' in line)
        record=ctypes.CDLL(runtime_path).cudaEventRecordWithFlags
        record.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint);record.restype=ctypes.c_int
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();end.record();end.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            assert record(start.cuda_event,stream.cuda_stream,1)==0
            value=fn()
            assert record(end.cuda_event,stream.cuda_stream,1)==0
        official.set_seed(contract['seed'])
        samples=[]
        for i,ref in enumerate(refs):
            graph.replay();end.synchronize()
            if i>=contract['warmup']:samples.append(start.elapsed_time(end)*1000)
            assert torch.equal(value,ref),'Counterfactual graph action changed'
        assert torch.equal(target_rng,torch.cuda.get_rng_state()),'Counterfactual graph RNG'
        del graph,value
        return dict(median_us=statistics.median(samples),samples_us=samples,
                    exact25actions_rng=True,event_placement='inside diagnostic whole-model graph')
    for part in ('vision','language','action'):
        fn=lambda:eager_call(policy,bi,ai)
        baseline=measure_policy(fn)
        old=[];weights=[]
        try:
            for n,count in calls.items():
                if region(n)!=part:continue
                m=modules[n];kind,decode,owner=decoder_for(n,m,contract,groups)
                w=decode();weights.append(w)
                if kind=='grouped_qkv':
                    old.append((owner,'decode',owner.decode));owner.decode=lambda _w=w:_w
                elif type(m) is LosslessAffineLinear:
                    old.append((m,'reconstruct',m.reconstruct));m.reconstruct=lambda _w=w:_w
                elif isinstance(m,HQQLinear):
                    old.append((m,'dequantize',m.dequantize));m.dequantize=lambda _w=w:_w
                else:
                    # Keep the existing once-per-action cross-KV schedule.
                    attribute='_aq26_kv_stock' if hasattr(m,'_aq26_kv_stock') else 'forward'
                    old.append((m,attribute,getattr(m,attribute)))
                    setattr(m,attribute,lambda x,_w=w,_m=m:F.linear(x,_w,_m.bias))
            assert weights,part
            cached=measure_policy(fn)
            row=dict(region=part,baseline=baseline,dense_cache=cached,
                     difference_ms=(baseline['median_us']-cached['median_us'])/1000,
                     diagnostic_expanded_weight_bytes=sum(w.numel()*w.element_size() for w in weights),
                     patched_projections=len(weights),actions_exact=True,rng_exact=True)
            rows.append(row);save(out/'cache_counterfactual.json',dict(status='COUNTERFACTUAL_RUNNING',rows=rows))
            print('CURRENT_CACHE_COUNTERFACTUAL',part,row['difference_ms'],flush=True)
        finally:
            for owner,name,value in old:setattr(owner,name,value)
            del weights
    result=dict(status='CURRENT_CACHE_COUNTERFACTUAL_COMPLETE',rows=rows,
        interpretation='Independent whole-model diagnostic graphs, same arithmetic,25actions/RNG checked. Region dense cache removes decode and changes weight residency/cache pressure. Differences are conditional and not additive, not deployment changes or achievable gains under original memory bound.')
    save(out/'cache_counterfactual.json',result)
    return result


def hardware_probe(artifacts,rows,contract,out):
    # ncu is an unprivileged diagnostic subprocess. No driver/security changes.
    tool=Path('/usr/local/cuda/bin/ncu')
    if not tool.exists():
        result=dict(status='UNAVAILABLE',reason='ncu executable absent')
    else:
        largest=max(rows,key=lambda r:r['dense_native_gemm']['median_us']*r['calls_per_query'])
        artifact=dict(artifacts)[largest['case']]
        script=Path(__file__).with_name('omni_cost_ncu_operand_r1.py')
        command=[str(tool),'--profile-from-start','off','--set','basic','--launch-count','1',
                 '--csv',contract['omni_runtime']['python'],str(script),str(artifact)]
        try:r=subprocess.run(command,capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired as e:
            result=dict(status='UNAVAILABLE',reason='ncu access probe timed out',command=command)
            save(out/'hardware_counter_access.json',result)
            return result
        log=out/'ncu_permission_probe.log';log.write_text(r.stdout+'\n'+r.stderr)
        result=dict(status='COUNTERS_COMPLETE' if r.returncode==0 and 'gpu__' in r.stdout else 'UNAVAILABLE',
                    exitcode=r.returncode,actual_geometry_case=largest['case'],command=command,log=str(log),
                    reason='No elevated privileges attempted; see raw ncu output. This one-kernel probe cannot establish model-wide saturation.')
    save(out/'hardware_counter_access.json',result)
    return result
