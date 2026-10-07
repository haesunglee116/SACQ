"""Prepare installation directives from an explicit calibrated allocation.

For DiT, 16 is the pinned compiler's BF16-or-codebook sentinel, not SACQ BPW.
VLM codebooks require no force entry: the installer's global selector is 2.
Templates explicitly supply any fixed-background directives outside allocation.
"""
import argparse
import csv
import json
from pathlib import Path
try:
    from .jobs import Builder, parse_force_specs
    from .pipeline import sha
except ImportError:
    from jobs import Builder, parse_force_specs
    from pipeline import sha


def prepare(spec_path, output_dir):
    spec_path = Path(spec_path).resolve()
    builder = Builder()
    spec = builder.json(dict(path=str(spec_path), sha256=sha(spec_path)))
    if spec.get('schema') != 'sacq_installation_specs_v1':
        raise ValueError('Unsupported installation specification schema')
    job = builder.json(spec['calibration_job'])
    allocation = builder.read(spec['allocation'])
    if allocation != Path(job['allocation']).resolve():
        raise ValueError('Allocation path differs from calibration job')
    with allocation.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['layer'] for r in rows}) != len(rows):
        raise ValueError('Allocation requires unique layer rows')
    selected = {r['layer']: r for r in job['artifacts']}
    background = {r['layer']: r['bits'] for r in job['gptq']}
    if len(selected) != len(job['artifacts']) or len(background) != len(job['gptq']):
        raise ValueError('Duplicate calibrated layer identity')
    if set(selected) & set(background):
        raise ValueError('Codebook and background layers overlap')
    if set(spec['templates']) != {'vlm', 'dit'}:
        raise ValueError('Both explicit installation templates required')
    values = {scope: parse_force_specs(builder.read(spec['templates'][scope]).read_text())
              for scope in ('vlm', 'dit')}
    codebook_names = set()
    for row in rows:
        name, scope, k = row['layer'], row['scope'], int(row['k'])
        if scope not in values or k <= 0:
            raise ValueError('Invalid allocation scope or codebook size')
        if k < 80000:
            artifact = selected.get(name, {})
            if artifact.get('scope') != scope or artifact.get('k') != k:
                raise ValueError('Allocation differs from calibrated codebook selection')
            bits = 16
            codebook_names.add(name)
        else:
            bits = {80000: 3, 90000: 4}.get(k)
            if bits is None or background.get(name) != bits:
                raise ValueError('Allocation differs from scalar background precision')
        other = 'dit' if scope == 'vlm' else 'vlm'
        if name in values[other]:
            raise ValueError('Template places candidate in the wrong scope')
        if scope == 'vlm' and k < 80000:
            values[scope].pop(name, None)
        else:
            values[scope][name] = bits
    if codebook_names != set(selected):
        raise ValueError('Allocation does not cover every calibrated codebook')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    refs = {}
    for scope, mapping in values.items():
        path = out/(scope+'_spec.txt')
        path.write_text(''.join(f'{name}={bits}\n' for name, bits in sorted(mapping.items())))
        refs[scope+'_spec'] = dict(path=str(path), sha256=sha(path))
    (out/'inputs.json').write_text(json.dumps(list(builder.inputs.values()), indent=2)+'\n')
    result = dict(status='INSTALLATION_SPECS_PREPARED_NOT_MODEL_VALIDATION',
                  calibration=refs, dit_codebook_sentinel=16,
                  vlm_codebook_directive='omit force entry; use installer global selector 2',
                  meaning='These are compiler selectors, not nominal precision')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.spec, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
