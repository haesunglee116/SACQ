"""Prepare GPTQ search candidates using the existing BF16/576-request recipe.

Explicit job inputs only. Run one suite/bit/layer-shard per invocation; merge
verified receipts before allocation. No allocation or rollout is launched here.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from .scalar_bank import digest


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def verify(row):
    p = Path(row['path'])
    if not p.is_absolute() or digest(p) != row['sha256']:
        raise ValueError('Missing or changed input: '+str(p))
    return p


def run(path):
    path = Path(path).resolve(); job = json.loads(path.read_text())
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Use a fresh output directory; completed candidates are immutable')
    records = job['records']; targets = job['targets']
    if [r['position'] for r in records] != [i for i in range(640) if i % 10]:
        raise ValueError('Requires original ordered 576 training positions; no validation consumption')
    if not targets or len({t['layer'] for t in targets}) != len(targets):
        raise ValueError('Explicit unique target layer shard required')
    for t in targets:
        if t['bits'] not in (3, 4) or t['scope'] not in ('vlm', 'dit'):
            raise ValueError('Only allocation candidate layers at 3/4 bits')
    for label in ('model', 'processor'):
        root = Path(job[label]).resolve()
        refs = job[label+'_files']
        if not refs or not root.is_dir() or any(not Path(r['path']).resolve().is_relative_to(root) for r in refs):
            raise ValueError('Checkpoint manifest must bind the configured '+label+' directory')
    for row in records + job['model_files'] + job['processor_files']:
        verify(row)
    trainer = verify(job['trainer'])
    workspace = Path(job['workspace']).resolve(); b = workspace/'libero_pro_eval'
    sys.path[:0] = [str(b), str(b/'Isaac-GR00T-N1.7')]
    for key in list(os.environ):
        if key.startswith(('VLM_', 'DIT_', 'SACQ_', 'ACTION_HEAD_')):
            os.environ.pop(key)
    import torch
    import transformers
    from .gptq_core import GPTQAccumulator
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    for cls in (transformers.AutoProcessor, transformers.Qwen3VLProcessor, transformers.Qwen3VLForConditionalGeneration):
        old = cls.from_pretrained
        def local(cls, name, *args, _old=old, **kwargs):
            return _old(job['processor'] if name == 'nvidia/Cosmos-Reason2-2B' else name, *args, **kwargs)
        cls.from_pretrained = classmethod(local)
    spec = importlib.util.spec_from_file_location('scalar_trainer', trainer)
    ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
    from gr00t.policy.server_client import MsgSerializer
    from measure_hotspot_protection_proxy import collate_request, move_to_device
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    policy = ft.Gr00tPolicy(ft.EmbodimentTag.resolve('LIBERO_PANDA'), job['model'], device='cuda:0', strict=True)
    policy.model.requires_grad_(False).eval()
    accs = {}; calls = {}; handles = []
    for t in targets:
        name = t['layer']; m = policy.model.get_submodule(name)
        if not isinstance(m, torch.nn.Linear) or list(m.weight.shape) != t['shape'] or m.weight.dtype != torch.bfloat16:
            raise ValueError('Target does not match untouched BF16 model')
        accs[name] = GPTQAccumulator(m); calls[name] = 0
        def hook(m, args, result, name=name):
            x = args[0].detach()
            if not torch.isfinite(x).all():
                raise ValueError('Nonfinite statistics input')
            accs[name].add_batch(x); calls[name] += 1
        handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i, row in enumerate(records):
                payload = MsgSerializer.from_bytes(Path(row['path']).read_bytes())
                bb, ai = policy.model.prepare_input(collate_request(policy, payload['observation']))
                action = ft.run_action_grad(policy,
                    move_to_device(bb, torch.device('cuda:0'), torch.bfloat16),
                    move_to_device(ai, torch.device('cuda:0'), torch.bfloat16), seed=7000+row['position'])
                if not torch.isfinite(action).all():
                    raise ValueError('Nonfinite action')
                if (i+1) % 16 == 0:
                    write(out/'STATE.json', dict(requests=i+1, total=576))
    finally:
        for handle in handles:
            handle.remove()
    artifacts = []
    for t in targets:
        n = t['layer']; expected = 576*(policy.model.action_head.num_inference_timesteps if t['scope']=='dit' else 1)
        if calls[n] != expected:
            raise ValueError('Unexpected statistics hook count')
        acc = accs.pop(n)
        weight, loss, damp = acc.fasterquant(bits=t['bits'], groupsize=64, blocksize=128, percdamp=.01, symmetric=False)
        if damp != .01 or not torch.isfinite(weight).all():
            raise ValueError('GPTQ numerical failure')
        dest = out/(n+'.pt'); tmp = dest.with_suffix('.tmp')
        with tmp.open('wb') as f:
            torch.save(dict(layer=n, weight=weight.cpu(), bits=t['bits'], group_size=64,
                blocksize=128, damping=damp, act_order=False, symmetric=False,
                requests=576, statistics_model='untouched BF16', loss=loss), f)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(dest)
        artifacts.append(dict(suite=job['suite'], layer=n, bits=t['bits'], method='GPTQ',
                              path=str(dest.resolve()), sha256=digest(dest)))
        del acc, weight
    torch.cuda.synchronize()
    write(out/'COMPLETE.json', dict(schema='sacq_gptq_scalar_shard_v1', job=dict(path=str(path),sha256=digest(path)),
          artifacts=artifacts, wall_seconds=time.perf_counter()-started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def merge(receipts, output):
    if Path(output).exists():
        raise FileExistsError(output)
    rows = []; refs = []; keys = set()
    for receipt in receipts:
        p = Path(receipt).resolve(); d = json.loads(p.read_text())
        if d['schema'] != 'sacq_gptq_scalar_shard_v1':
            raise ValueError('Not a scalar preparation receipt')
        verify(d['job'])
        for row in d['artifacts']:
            verify(row); key = (row['suite'], row['layer'], row['bits'])
            if key in keys:
                raise ValueError('Duplicate candidate across shards')
            keys.add(key); rows.append(row)
        refs.append(dict(path=str(p), sha256=digest(p)))
    write(output, dict(schema='sacq_gptq_scalar_bank_v1', artifacts=rows, receipts=refs))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run'); r.add_argument('job')
    m = sub.add_parser('merge'); m.add_argument('--output', required=True); m.add_argument('receipts', nargs='+')
    a = p.parse_args()
    if a.command == 'run': run(a.job)
    else: merge(a.receipts, a.output)
