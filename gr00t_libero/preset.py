"""Apply recorded SACQ2 settings to explicit external job input bindings.

This prepares builder specs only; it neither launches a job nor grants approval.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRESET = ROOT/'presets/gr00t_libero_sacq2_seed17.json'


def assign(mapping, key, value):
    if key in mapping and mapping[key] != value:
        raise ValueError(f'Binding conflicts with preset: {key}')
    mapping[key] = copy.deepcopy(value)


def apply(stage, bindings, preset_path=PRESET):
    preset_path = Path(preset_path).resolve()
    preset = json.loads(preset_path.read_text())
    if preset['schema'] != 'sacq_reference_preset_v1':
        raise ValueError('Unsupported preset')
    result = copy.deepcopy(bindings)
    selection = preset['allocation']
    allocation = preset_path.parent/selection['file']
    if hashlib.sha256(allocation.read_bytes()).hexdigest() != selection['sha256']:
        raise ValueError('Preset allocation content changed')
    if stage in ('index', 'calibration'):
        assign(result, 'schema', 'sacq_worker_build_v1')
        assign(result, 'allocation', dict(path=str(allocation), sha256=selection['sha256']))
        job = result.setdefault('job', {})
        suite = job.get('suite', result.get('bank_selection', {}).get('suite'))
        if suite not in preset['suites']:
            raise ValueError('Explicit supported suite required in job or bank_selection')
        if len(result.get('gptq', [])) != selection['frozen_gptq_layers']:
            raise ValueError('Reference requires 111 explicit frozen GPTQ payload records')
        if stage == 'index':
            assign(job, 'index_settings', preset['index_settings'])
            positions = [row.get('position') for row in result.get('records', [])]
            if positions != [i for i in range(640) if i % 10 != 0]:
                raise ValueError('Index requires the ordered 576 training positions, excluding validation')
        else:
            assign(job, 'training', preset['training'])
            if len(result.get('all_records', [])) != 640:
                raise ValueError('Calibration requires 640 ordered request records')
    elif stage == 'evaluation':
        plan = result.setdefault('plan', {})
        assign(plan, 'rollout', preset['rollout'])
        assign(plan, 'language_action_nominal_bpw', selection['actual_nominal_bpw'])
        assign(plan, 'method', 'final_seed17')
        assign(plan, 'diagnostic_only', True)
        assign(plan, 'held_out', False)
    else:
        raise ValueError('Supported binding stages: index, calibration, evaluation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('index', 'calibration', 'evaluation'))
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = apply(args.stage, json.loads(args.bindings.read_text()))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(f'Prepared reference builder spec: {args.output}; no experiment launched')


if __name__ == '__main__':
    main()
