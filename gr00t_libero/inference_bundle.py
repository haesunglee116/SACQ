"""Export explicit existing model inputs into a relocatable inference bundle.

Copies weights; allow disk space for both original and exported artifacts.
No fitting, model selection, downloads, or publication is performed.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from inference import sha


def export(spec, output):
    if spec.get('schema') != 'sacq_inference_export_v1':
        raise ValueError('Unsupported export specification')
    if spec['condition'] not in ('bf16', 'sacq', 'sacq1.5'):
        raise ValueError('Only BF16 and SACQ A16 are supported')
    for k in ('seed', 'denoising_steps'):
        if type(spec[k]) is not int or spec[k] < (1 if k == 'denoising_steps' else 0) or (k == 'seed' and spec[k] >= 2**32):
            raise ValueError('Explicit valid ' + k + ' required')
    output = Path(output).resolve()
    inputs = [Path(spec[k]).resolve() for k in ('model', 'processor')]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError('Output and input directories must be separate')
    output.mkdir(parents=True, exist_ok=False)
    cfg = {k: spec[k] for k in ('condition', 'suite', 'seed', 'denoising_steps')}
    cfg.update(schema='sacq_single_request_v1', files={})
    def add(source, target, expected=None):
        source = Path(source)
        before = sha(source)
        if expected is not None and before != expected:
            raise ValueError('Input hash mismatch: ' + str(source))
        dest = output / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if sha(dest) != before or sha(source) != before:
            raise ValueError('Input changed during export: ' + str(source))
        cfg['files'][target] = before
        return dict(path=target, sha256=before)
    try:
        for key, source in zip(('model', 'processor'), inputs):
            if not source.is_dir():
                raise ValueError('Missing local ' + key)
            cfg[key] = key
            # Materialize file symlinks (e.g. Hugging Face snapshots), not cache directories.
            for path in sorted(source.rglob('*')):
                if '.git' in path.relative_to(source).parts or '.cache' in path.relative_to(source).parts:
                    continue
                if path.is_symlink() and path.is_dir():
                    raise ValueError('Directory symlink must be materialized before export: ' + str(path))
                if path.is_file():
                    add(path, str(Path(key) / path.relative_to(source)))
            if not any(n.startswith(key + '/') for n in cfg['files']):
                raise ValueError('Empty ' + key)
        request = spec['request']
        cfg['request'] = add(request['path'], 'request.msgpack', request['sha256'])['path']
        if spec['condition'] != 'bf16':
            model = copy.deepcopy(spec['workshop_model'])
            for key in ('vlm', 'dit', 'vlm_spec', 'dit_spec'):
                ref = model[key]
                model[key] = add(ref['path'], 'quantized/' + key + Path(ref['path']).suffix, ref['sha256'])
            for i, ref in enumerate(model['gptq']):
                copied = add(ref['path'], f'quantized/background/{i:04d}.pt', ref['sha256'])
                # Retain installation fields, omit historical machine-local metadata.
                model['gptq'][i] = dict(layer=ref['layer'], bits=ref['bits'], **copied)
            cfg['workshop_model'] = {k: model[k] for k in (
                'vlm', 'dit', 'vlm_spec', 'dit_spec', 'gptq', 'expected_sacq_count', 'expected_gptq_count')}
        elif 'workshop_model' in spec:
            raise ValueError('BF16 export must not include workshop_model')
        (output / 'inference.json').write_text(json.dumps(cfg, indent=2) + '\n')
    except BaseException as error:
        (output / 'EXPORT_FAILED.json').write_text(json.dumps(dict(error=str(error))))
        raise
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--spec', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    cfg = export(json.loads(a.spec.read_text()), a.output)
    print(json.dumps(dict(status='EXPORTED', files=len(cfg['files']), bundle=str(a.output))))


if __name__ == '__main__':
    main()
