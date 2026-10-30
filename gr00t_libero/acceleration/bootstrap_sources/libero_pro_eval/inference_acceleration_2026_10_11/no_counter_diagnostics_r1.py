"""Read-only diagnostics; no hardware counters or runtime changes.

Instrumentation is excluded from speed/memory claims. Logical bytes/FLOPs are
not physical traffic. Kernel sums are not additive wall time.
"""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import torch


def capture_inventory(model):
    rows = []; handles = []
    def hook(name):
        def record(m, args):
            if not torch.cuda.is_current_stream_capturing(): return
            if not args or not isinstance(args[0], torch.Tensor): return
            x = args[0]
            if x.ndim < 2: return
            n = getattr(m, 'out_features', None)
            k = getattr(m, 'in_features', None)
            if n is None or k is None: return
            tensors = list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False))
            payload = {key: dict(shape=list(t.shape), dtype=str(t.dtype), bytes=t.numel()*t.element_size())
                       for key,t in tensors}
            mm = x.numel() // x.shape[-1]
            rows.append(dict(name=name, type=type(m).__name__, input_shape=list(x.shape),
                M=mm,N=n,K=k, dense_flops=2*mm*n*k, input_bytes=x.numel()*x.element_size(),
                output_bytes=mm*n*x.element_size(),dense_weight_bytes=n*k*2,
                registered_payload=payload, registered_payload_bytes=sum(t['bytes'] for t in payload.values()),
                index_bits=getattr(m,'index_bits',None),group_size=getattr(m,'group_size',None)))
        return record
    for name,m in model.named_modules():
        if hasattr(m,'in_features') and hasattr(m,'out_features'):
            handles.append(m.register_forward_pre_hook(hook(name)))
    return rows,handles


def telemetry():
    commands = {
      'gpu': ['nvidia-smi','--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used','--format=csv'],
      'cpu_affinity': ['taskset','-pc',str(os.getpid())],
    }
    out={}
    for k,cmd in commands.items():
        try:
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=10)
            out[k]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as e:out[k]=dict(error=str(e))
    return out


def profile(policy,obs,official,c,expected_rng,out):
    bb=policy.model.backbone; ah=policy.model.action_head
    originals=(bb.forward,ah.get_action,official.prepare_model_inputs)
    def wrap(fn,label,sync=False):
        def call(*args,**kwargs):
            with torch.profiler.record_function(label):
                result=fn(*args,**kwargs)
                if sync:torch.cuda.synchronize()
            return result
        return call
    bb.forward=wrap(originals[0],'DIAG_BACKBONE',True)
    ah.get_action=wrap(originals[1],'DIAG_ACTION',True)
    official.prepare_model_inputs=wrap(originals[2],'DIAG_PREPROCESS')
    before=telemetry()
    try:
        official.set_seed(c['seed'])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
            official.benchmark_components(policy,obs,c['num_iterations'],c['warmup'])
        assert torch.equal(torch.cuda.get_rng_state(),expected_rng),'Profiler RNG mismatch'
    finally:
        bb.forward,ah.get_action,official.prepare_model_inputs=originals
    # Local temporary trace avoids creating a large uncompressed Lustre file.
    with tempfile.TemporaryDirectory(prefix='sacq_trace_') as tmp:
        path=Path(tmp)/'trace.json';prof.export_chrome_trace(str(path))
        trace=json.loads(path.read_text())
        with gzip.open(out.with_suffix('.trace.json.gz'),'wt',compresslevel=1) as f:json.dump(trace,f)
    cpu=[]
    for e in prof.key_averages():
        cpu.append(dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,total_cpu_ms=e.cpu_time_total/1000))
    info=dict(status='PROFILE_COMPLETE',rng_exact=True,before=before,after=telemetry(),
        top_cpu_self=sorted(cpu,key=lambda x:-x['self_cpu_ms'])[:80],
        trace=str(out.with_suffix('.trace.json.gz')),
        caveat='Instrumented timings are diagnostic only; no hardware counters collected.')
    out.write_text(json.dumps(info,indent=2))


def binary_resources(out):
    libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '.so' in line and ('codebook_' in line or 'ffn_extension' in line)})
    rows=[]
    for lib in libs:
        row=dict(path=lib,sha256=hashlib.sha256(Path(lib).read_bytes()).hexdigest())
        try:
            r=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',lib],text=True,capture_output=True,timeout=90)
            row.update(returncode=r.returncode,resource_usage=r.stdout,stderr=r.stderr)
        except Exception as e:row['error']=str(e)
        rows.append(row)
    out.write_text(json.dumps(rows,indent=2))
