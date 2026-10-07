"""Approved fixed seed noise table, independent of rollout completion order.

Caller must supply action shape/dtype/max calls from the frozen model/rollout
contract, not defaults. This utility alone is not a scientific launch approval.
"""
from pathlib import Path
import hashlib
import json
import os


def build_table(output, *, seed, episodes, calls, shape, dtype, device):
    import torch
    output = Path(output)
    if output.exists():
        raise FileExistsError('Immutable noise destination already exists')
    if episodes <= 0 or calls <= 0 or not shape or min(shape) <= 0:
        raise ValueError('Explicit positive dimensions required')
    output.mkdir(parents=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    manifest = dict(seed=seed, episodes=episodes, calls=calls, shape=list(shape),
                    dtype=str(dtype), generator_device=str(device),
                    torch_version=torch.__version__,
                    order='episode_then_policy_call; one draw per call', files=[])
    for episode in range(episodes):
        values = torch.stack([
            torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype).cpu()
            for _ in range(calls)
        ])
        target = output / f'episode_{episode:03d}.pt'
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            torch.save(values, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        manifest['files'].append(dict(episode=episode, name=target.name,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    temporary = output / 'manifest.tmp'
    with temporary.open('w') as handle:
        json.dump(manifest, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'manifest.json')
    return manifest


def read_episode(output, episode):
    import torch
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    row = manifest['files'][episode]
    if row['episode'] != episode:
        raise ValueError('Episode mapping mismatch')
    path = output / row['name']
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Noise checksum mismatch')
    values = torch.load(path, map_location='cpu', weights_only=True)
    if list(values.shape) != [manifest['calls'], *manifest['shape']]:
        raise ValueError('Noise shape mismatch')
    return values
