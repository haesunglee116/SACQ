"""CUDA graph for deterministic, frozen DiT forward; separate from training.

Capture failures and numerical mismatches raise, never silently enable a graph.
All noise sampling stays in the caller. Static output is cloned to avoid aliasing
across subsequent denoising calls. This module does not authorize deployment.
"""
import torch
from rollout_exact_acceleration_20260920 import rng_snapshot, rng_restore, exact


class AuditedDiTGraph:
    def __init__(self, module):
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise RuntimeError('Only frozen eval DiT is supported')
        self.original = module.forward
        self.entries = {}
        self.events = []
        self.verify_each_call = True

    def key(self, kw):
        entries = []
        for k,v in sorted(kw.items()):
            if isinstance(v,torch.Tensor):
                if v.device.type != 'cuda': raise RuntimeError('CPU tensor in graph inputs')
                entries.append((k,tuple(v.shape),tuple(v.stride()),v.dtype,str(v.device)))
            elif v is None or type(v) in (str,int,float,bool):
                entries.append((k,v))
            else:raise TypeError((k,type(v)))
        return (torch.is_inference_mode_enabled(),torch.is_autocast_enabled('cuda'),
                torch.get_autocast_dtype('cuda'),tuple(entries))

    def __call__(self,*args,**kw):
        if args or kw.get('return_all_hidden_states',False) or torch.is_grad_enabled():
            raise RuntimeError('Unsupported DiT graph signature or enabled gradients')
        key=self.key(kw)
        if key not in self.entries:
            before=rng_snapshot()
            try:
                expected=self.original(**kw).clone()
                static={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in kw.items()}
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):self.original(**static)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=self.original(**static)
                graph.replay();torch.cuda.synchronize()
                if not exact(before,rng_snapshot()):
                    raise RuntimeError('DiT consumed RNG during capture/warmup')
                if not torch.equal(expected,out):
                    raise RuntimeError(f'Graph capture output mismatch: {(expected.float()-out.float()).abs().max().item()}')
                self.entries[key]=(static,graph,out)
                self.events.append(dict(status='CAPTURE_EXACT_PASS',shape_key=str(key),rng_unchanged=True))
            finally:rng_restore(before)
        static,graph,out=self.entries[key]
        for k,v in kw.items():
            if isinstance(v,torch.Tensor):static[k].copy_(v)
        graph.replay()
        result=out.clone()
        if self.verify_each_call:
            before=rng_snapshot()
            expected=self.original(**kw)
            if not exact(before,rng_snapshot()):
                rng_restore(before)
                raise RuntimeError('DiT reference consumed RNG')
            if not torch.equal(result,expected):
                raise RuntimeError('Graph replay differs on a new real input')
        return result
