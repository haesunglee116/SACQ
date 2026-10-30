"""Cache only outputs depending on the fixed denoising timestep/position.

Caller explicitly selects the original step before invoking the original body.
Not a general input-value cache: rebuild for any schedule/model/layout change.
"""
import torch


class FixedScheduleOutputs:
    def __init__(self,modules):
        self.entries=[dict(name=n,module=m,original=m.forward,outputs=[],layouts=[]) for n,m in modules]
        self.step=None

    def prepare(self,run):
        handles=[]
        def collect(e):
            def hook(m,args,value):
                if m.training or torch.is_grad_enabled() or not isinstance(value,torch.Tensor):
                    raise RuntimeError('Fixed-schedule cache requires tensor-only inference')
                e['layouts'].append([(v.shape,v.dtype,v.device) for v in args])
                e['outputs'].append(value.detach().clone())
            return hook
        try:
            for e in self.entries:
                handles.append(e['module'].register_forward_hook(collect(e)))
            run()
        finally:
            for h in handles: h.remove()
        counts={len(e['outputs']) for e in self.entries}
        if len(counts)!=1 or 0 in counts:
            raise RuntimeError('Incomplete fixed schedule')
        self.steps=counts.pop()

    def enable(self):
        for e in self.entries:
            def forward(*args,_e=e):
                if self.step is None or _e['module'].training or torch.is_grad_enabled():
                    raise RuntimeError('Explicit step and inference mode required')
                if [(v.shape,v.dtype,v.device) for v in args] != _e['layouts'][self.step]:
                    raise ValueError('Fixed-schedule input layout changed')
                return _e['outputs'][self.step]
            e['module'].forward=forward

    def select_step(self,step):
        if not 0<=step<self.steps:
            raise ValueError('Invalid schedule step')
        self.step=step

    def receipt(self):
        return dict(steps=self.steps,modules=[e['name'] for e in self.entries],
                    bytes=sum(v.numel()*v.element_size() for e in self.entries for v in e['outputs']),
                    expanded_weights_cached=False)
