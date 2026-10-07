"""Generate explicit fixed episode noise with the bundled original algorithm.

This prepares inputs only. It does not select a rollout protocol or establish
that regenerated tensors equal archived tensors from another runtime/device.
"""
import argparse
import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).parent/'sources/libero_pro_eval/fixed_episode_noise_2026_09_09.py'


def validate(spec):
    keys = {'schema', 'output', 'seed', 'episodes', 'calls', 'shape', 'dtype', 'device', 'torch_version'}
    if not isinstance(spec, dict) or set(spec) != keys or spec['schema'] != 'sacq_noise_inputs_v1':
        raise ValueError('All noise settings must be explicit; unknown fields are rejected')
    if not isinstance(spec['output'], str) or not Path(spec['output']).is_absolute():
        raise ValueError('Absolute output path required')
    for key in ('seed', 'episodes', 'calls'):
        if type(spec[key]) is not int or spec[key] < (0 if key == 'seed' else 1):
            raise ValueError('Explicit nonnegative seed and positive counts required')
    if not isinstance(spec['shape'], list) or not spec['shape'] or any(type(v) is not int or v <= 0 for v in spec['shape']):
        raise ValueError('Explicit positive shape dimensions required')
    if spec['dtype'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unsupported noise dtype')
    if not isinstance(spec['device'], str) or not spec['device']:
        raise ValueError('Explicit generator device required')
    if not isinstance(spec['torch_version'], str) or not spec['torch_version']:
        raise ValueError('Explicit torch version required')


def generate(spec):
    validate(spec)
    import torch
    if torch.__version__ != spec['torch_version']:
        raise ValueError('Noise generator torch version differs from the explicit contract')
    device = torch.device(spec['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA generators are supported')
    module_spec = importlib.util.spec_from_file_location('release_original_noise', SOURCE)
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    return original.build_table(spec['output'], seed=spec['seed'], episodes=spec['episodes'],
        calls=spec['calls'], shape=spec['shape'], dtype=getattr(torch, spec['dtype']), device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    result = generate(spec)
    print(json.dumps({'manifest': str(Path(spec['output'])/'manifest.json'),
                      'episodes': result['episodes'], 'calls': result['calls']}))


if __name__ == '__main__':
    main()
