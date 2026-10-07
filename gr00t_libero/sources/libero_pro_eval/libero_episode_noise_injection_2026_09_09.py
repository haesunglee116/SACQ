"""Instance-local injection into GR00T's single diffusion-noise draw.

Does not monkeypatch global torch or choose any scientific seed/shape. Callers
must provide the frozen episode/call tensors before each policy invocation.
"""
import contextlib
import inspect
import types


class NoiseInjection:
    def __init__(self, head):
        self.head = head
        self.original = head.get_action_with_features
        raw = inspect.unwrap(self.original.__func__)
        self.torch = raw.__globals__["torch"]
        self.active = None
        self.draws = 0
        scope = dict(raw.__globals__)
        scope["torch"] = self
        patched = types.FunctionType(raw.__code__, scope, raw.__name__, raw.__defaults__, raw.__closure__)
        patched.__kwdefaults__ = raw.__kwdefaults__
        head.get_action_with_features = types.MethodType(self.torch.no_grad()(patched), head)

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def randn(self, *args, **kwargs):
        if self.active is None:
            raise RuntimeError("Missing explicitly selected episode/call noise")
        if self.draws:
            raise RuntimeError("Unexpected second diffusion noise draw")
        shape = kwargs.get("size", args[0] if len(args) == 1 else args)
        if tuple(shape) != tuple(self.active.shape):
            raise ValueError(("Noise shape mismatch", shape, tuple(self.active.shape)))
        if kwargs.get("dtype") != self.active.dtype:
            raise ValueError("Noise dtype mismatch: implicit cast forbidden")
        self.draws += 1
        return self.active.to(device=kwargs["device"]).clone()

    @contextlib.contextmanager
    def selected(self, noise):
        if self.active is not None:
            raise RuntimeError("Concurrent/nested policy invocation is not supported")
        self.active, self.draws = noise, 0
        try:
            yield
            if self.draws != 1:
                raise RuntimeError("Expected exactly one consumed diffusion noise draw")
        finally:
            self.active = None

    def close(self):
        if self.active is not None:
            raise RuntimeError("Cannot remove injection during inference")
        self.head.get_action_with_features = self.original
