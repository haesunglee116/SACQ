"""Opt-in rollout instrumentation and exact-gated fixed-weight caching.

No scientific defaults, seed generation, simulator changes or graph activation.
An adapter must supply its existing noise-preserving policy call. Cached weights
are runtime memory, never a compressed-resident-memory measurement.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time
import types


class Profile:
    def __init__(self, folder, label):
        self.path = Path(folder) / f'{label}_{os.getpid()}.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.last = 0.
        self.phases = {}
        self.events = []

    @contextlib.contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            d = self.phases.setdefault(name, {'seconds': 0., 'calls': 0})
            d['seconds'] += time.perf_counter() - start
            d['calls'] += 1

    def timed(self, name, fn, *a, **kw):
        with self.phase(name):
            return fn(*a, **kw)

    def save(self, force=False):
        now = time.perf_counter()
        if not force and now - self.last < 10:
            return
        self.last = now
        value = dict(time=time.time(), pid=os.getpid(), elapsed=now-self.started,
                     phases=self.phases, events=self.events,
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     phase_note='Nested phases are not additive; CUDA timing synchronized only in canary.')
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(self.path)


def rng_snapshot():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def rng_restore(state):
    import numpy as np
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def exact(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class FrozenWeightCache:
    """Cache only the audited pure decoder, leaving forward/GEMM untouched."""
    def __init__(self, model):
        import torch
        self.rows = []
        for name, m in model.named_modules():
            # Do not guess semantics of arbitrary dequantizers or HQQ modules.
            if (type(m).__name__ != 'PrecomputedPackedVectorCodebookLinear'
                    or type(m).__module__ != 'gr00t.eval.cosmos_quant_shim'):
                continue
            if getattr(m, 'runtime_backend', None) != 'dense_reconstruct':
                raise RuntimeError(f'Unsupported backend: {name}')
            if any(p.requires_grad for p in m.parameters()):
                raise RuntimeError(f'Cache requires frozen weights: {name}')
            original = m.dequantize_weight
            with torch.inference_mode():
                weight = original().detach()
                if not torch.equal(weight, original()):
                    raise RuntimeError(f'Non-deterministic decoder: {name}')
            # This cache is installed only after the immutable overlay is loaded.
            # No checkpoint mutation/training is permitted during its lifetime.
            self.rows.append((m, original, weight))

    def enable(self):
        for m, original, weight in self.rows:
            m.dequantize_weight = types.MethodType(lambda self, _w=weight: _w, m)

    def disable(self):
        for m, original, weight in self.rows:
            m.dequantize_weight = original


class PolicyAcceleration:
    """Warm up, then compare full actions with identical caller-supplied noise.

    Three real calls are a runtime canary, not new benchmark samples. Each returns
    the baseline action. Any mismatch disables caching; no altered action is sent
    to the simulator. RNG after each check equals the baseline call's RNG state.
    """
    def __init__(self, profile, cache=True):
        self.profile = profile
        self.allow_cache = cache
        self.cache = None
        self.calls = 0
        self.checks = 0
        self.baseline_seconds = 0.
        self.cached_seconds = 0.
        self.state = 'WARMUP'

    def __call__(self, model, call):
        import torch
        self.calls += 1
        if self.calls <= 3 or not self.allow_cache or self.state in ('DISABLED', 'ENABLED'):
            value = self.profile.timed('policy_' + self.state.lower(), call)
            self.profile.save()
            return value
        if self.cache is None:
            candidates = [m for m in model.modules() if type(m).__name__ == 'PrecomputedPackedVectorCodebookLinear'
                          and type(m).__module__ == 'gr00t.eval.cosmos_quant_shim']
            if torch.cuda.is_available() and candidates:
                required = sum(m.out_features*m.in_features*2 for m in candidates)
                free, _ = torch.cuda.mem_get_info()
                if free < required + 3*1024**3:
                    self.state = 'DISABLED'
                    self.profile.events.append(dict(status='CACHE_SKIPPED_MEMORY', free=free, required=required))
                    self.profile.save(True)
                    return call()
            with self.profile.phase('cache_setup'):
                self.cache = FrozenWeightCache(model)
            if not self.cache.rows:
                self.state = 'DISABLED'
                self.profile.events.append(dict(status='CACHE_NOT_APPLICABLE'))
                return call()
            self.state = 'CANARY'
        before = rng_snapshot()
        self.cache.disable()
        torch.cuda.synchronize()
        t = time.perf_counter()
        baseline = call()
        torch.cuda.synchronize()
        baseline_time = time.perf_counter() - t
        after = rng_snapshot()
        rng_restore(before)
        self.cache.enable()
        try:
            torch.cuda.synchronize()
            t = time.perf_counter()
            accelerated = call()
            torch.cuda.synchronize()
            cached_time = time.perf_counter() - t
            # Policy information may include runtime metadata: compare full action
            # dictionary only, preserving the baseline auxiliary return value.
            equal = exact(baseline[0], accelerated[0]) if isinstance(baseline, tuple) else exact(baseline, accelerated)
        except BaseException:
            self.cache.disable()
            self.state = 'DISABLED'
            raise
        finally:
            rng_restore(after)
        self.checks += 1
        self.baseline_seconds += baseline_time
        self.cached_seconds += cached_time
        self.profile.events.append(dict(status='ACTION_EXACT_PASS' if equal else 'ACTION_MISMATCH_CACHE_REJECTED',
            call=self.calls, layers=len(self.cache.rows), baseline_seconds=baseline_time,
            cached_seconds=cached_time))
        if not equal:
            self.state = 'DISABLED'
            self.cache.disable()
        elif self.checks >= 3:
            self.state = 'ENABLED' if self.cached_seconds < self.baseline_seconds else 'DISABLED'
            if self.state == 'DISABLED':
                self.cache.disable()
            self.profile.events.append(dict(status='CACHE_' + self.state,
                measured_policy_speedup=self.baseline_seconds/self.cached_seconds,
                packed_memory_claim=False))
        else:
            self.cache.disable()
        self.profile.save(True)
        return baseline
