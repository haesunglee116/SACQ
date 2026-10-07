"""Default, opt-out-only operational rollout acceleration for GR00T.

Only frozen rollout inference is supported. Caller owns the approved sample and
noise contract. New workers prove full-action/RNG equivalence before enabling
cache and graphs. No calibration training, batch, kernel or simulator changes.
"""
import atexit
import hashlib
from pathlib import Path
import time

from rollout_exact_acceleration_20260920 import (
    Profile, PolicyAcceleration, exact, rng_restore, rng_snapshot,
)
from rollout_display_heartbeat_20260920 import DisplayHeartbeat
from dit_graph_audit_20260920 import AuditedDiTGraph


class AsyncProfile(Profile):
    def __init__(self, folder, label):
        super().__init__(folder,label)
        self.writer=DisplayHeartbeat(self.path)
        atexit.register(self.close)

    def save(self, force=False):
        now=time.perf_counter()
        if not force and now-self.last<10:return
        self.last=now
        self.writer.publish(dict(time=time.time(),elapsed=now-self.started,
            phases=self.phases,events=self.events,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            display_only=True,phase_note='Nested phases are not additive'))

    def close(self):
        if self.writer.closing:return
        self.save(True)
        if not self.writer.close():
            raise RuntimeError('Display profile writer did not finish: '+str(self.writer.error))


class DefaultAcceleration:
    def __init__(self, profile):
        self.profile=profile
        self.cache=PolicyAcceleration(profile)
        self.graph=None
        self.state='WAIT_CACHE'
        self.checks=0
        self.eager_seconds=0.
        self.graph_seconds=0.

    def __call__(self, model, call):
        import torch
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Default accelerator forbids training or mutable weights')
        if self.cache.state not in ('ENABLED','DISABLED'):
            return self.cache(model,call)
        head=model.action_head.model
        if self.state=='ENABLED':
            result=self.profile.timed('policy_graph_enabled',call)
            self.profile.save()
            return result
        if self.state=='DISABLED':
            return self.profile.timed('policy_graph_disabled',call)
        if self.graph is None:
            if type(head.forward).__name__ == 'ExactDiTGraph':
                raise RuntimeError('An inherited graph must be disabled by a versioned adapter before cache installation')
            self.graph=AuditedDiTGraph(head)
            self.state='CANARY'
        before=rng_snapshot()
        head.forward=self.graph.original
        baseline=call()
        after=rng_snapshot()
        try:
            rng_restore(before);head.forward=self.graph;self.graph.verify_each_call=True
            candidate=call()
            if not exact(_action(baseline),_action(candidate)) or not exact(after,rng_snapshot()):
                raise RuntimeError('Full-action or RNG CUDA Graph canary mismatch')
            self.graph.verify_each_call=False
            timings={}
            for mode in ('eager','graph'):
                head.forward=self.graph.original if mode=='eager' else self.graph
                rng_restore(before);torch.cuda.synchronize();start=time.perf_counter()
                value=call();torch.cuda.synchronize();timings[mode]=time.perf_counter()-start
                if not exact(_action(baseline),_action(value)) or not exact(after,rng_snapshot()):
                    raise RuntimeError('Timed full-action or RNG mismatch')
        except BaseException as exc:
            head.forward=self.graph.original
            self.state='DISABLED'
            self.profile.events.append(dict(status='GRAPH_REJECTED',error=repr(exc)))
            self.profile.save(True)
            # Stop this new worker; never continue in a potentially poisoned CUDA
            # context or label a numerical mismatch as an exact fallback.
            raise
        finally:rng_restore(after)
        self.checks+=1
        self.eager_seconds+=timings['eager'];self.graph_seconds+=timings['graph']
        self.profile.events.append(dict(status='GRAPH_FULL_ACTION_EXACT_PASS',rng_equal=True,timings=timings))
        if self.checks>=3:
            self.state='ENABLED' if self.graph_seconds<self.eager_seconds else 'DISABLED'
            self.profile.events.append(dict(status='GRAPH_'+self.state,
                speedup=self.eager_seconds/self.graph_seconds))
        head.forward=self.graph if self.state=='ENABLED' else self.graph.original
        self.profile.save(True)
        return baseline

    def audit(self):
        return dict(cache_state=self.cache.state,graph_state=self.state,
            full_action_checks=self.checks,graph_events=[] if self.graph is None else self.graph.events,
            profile=str(self.profile.path),training_supported=False)


def _action(value):
    return value[0] if isinstance(value,tuple) else value
