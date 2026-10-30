"""Opt-in fixed-schedule AdaLN output cache; inference only.

Caller must replay the same timestep schedule, dtype and batch shape used to
prepare the cache. Rebuild after any weight/runtime/schedule change. Only the
small modulation outputs are cached, never reconstructed weights. Not thread safe.
"""
import torch
from exact_pointwise import modulate


class FixedAdaLN:
    def __init__(self, norms):
        self.norms = list(norms)
        self.originals = [n.forward for n in self.norms]
        self.outputs = [[] for _ in self.norms]
        self.step = None
        self.ready = False

    def prepare(self, run_reference):
        if self.ready:
            raise RuntimeError("Create a new cache after changing model or schedule")
        handles = []
        def hook(i):
            def collect(module, args):
                if module.training or torch.is_grad_enabled():
                    raise RuntimeError("FixedAdaLN requires eval + no_grad")
                if len(args) != 2:
                    raise RuntimeError("Expected norm(x, temb)")
                self.outputs[i].append(module.linear(module.silu(args[1])).detach().clone())
            return collect
        try:
            for i, norm in enumerate(self.norms):
                handles.append(norm.register_forward_pre_hook(hook(i)))
            with torch.no_grad():
                run_reference()
        finally:
            for h in handles:
                h.remove()
        counts = {len(v) for v in self.outputs}
        if not self.norms or len(counts) != 1 or 0 in counts:
            raise RuntimeError("Incomplete or unequal AdaLN schedule")
        self.steps = counts.pop()
        self.ready = True

    def select_step(self, step):
        if not self.ready or not 0 <= step < self.steps:
            raise RuntimeError("Unprepared or invalid timestep")
        self.step = step

    def enable(self):
        if not self.ready:
            raise RuntimeError("Prepare first")
        for i, norm in enumerate(self.norms):
            def forward(x, temb, _i=i, _n=norm):
                if _n.training or torch.is_grad_enabled() or self.step is None:
                    raise RuntimeError("Inference-only cache needs explicit timestep")
                modulation = self.outputs[_i][self.step]
                if modulation.shape[0] != x.shape[0] or modulation.device != x.device or modulation.dtype != x.dtype:
                    raise RuntimeError("Cache input layout changed")
                return modulate(_n.norm(x), modulation)
            norm.forward = forward

    def restore(self):
        for norm, forward in zip(self.norms, self.originals):
            norm.forward = forward

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for row in self.outputs for t in row)
