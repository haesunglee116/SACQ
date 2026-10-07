"""Audit explicitly supplied development/evaluation episode identities.

This checks declared provenance and task coverage, not the truth of a capture's
labels or scientific approval. It never creates diagnostic-overlap approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

STAGES = ('fitting', 'allocation', 'calibration_train', 'calibration_validation', 'rollout')
KEYS = ('benchmark_family', 'task', 'base_seed', 'episode_id')


def read_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Every manifest requires an explicit path and SHA-256')
    path = Path(ref['path'])
    if not path.is_absolute():
        raise ValueError('Manifest paths must be absolute')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError(f'Manifest identity mismatch: {path}')
    return json.loads(raw)


def identity(row):
    if not isinstance(row, dict) or not set(KEYS).issubset(row):
        raise ValueError('Every sample needs explicit family, task, base seed and episode ID')
    for key in KEYS[:2]:
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f'Nonempty {key} required')
    for key in KEYS[2:]:
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f'Explicit nonnegative integer {key} required')
    return tuple(row[key] for key in KEYS)


def samples(manifest):
    if manifest.get('schema') == 'sacq_prepared_capture_inputs_v1':
        rows = [sample for record in manifest['records'] for sample in record['observations']]
    elif manifest.get('schema') == 'sacq_episode_identities_v1':
        rows = manifest['samples']
    else:
        raise ValueError('Expected prepared captures or explicit episode identities')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Every stage requires a nonempty explicit sample list')
    return [identity(row) for row in rows]


def audit(spec):
    if spec.get('schema') != 'sacq_sample_audit_spec_v1' or set(spec.get('stages', {})) != set(STAGES):
        raise ValueError('An audit requires all five named stage manifests')
    registry = read_ref(spec['task_registry'])
    if registry.get('schema') != 'sacq_task_registry_v1':
        raise ValueError('Explicit task registry required')
    tasks = registry.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Task registry must not be empty')
    expected = set()
    for row in tasks:
        if set(row) != {'benchmark_family', 'task'} or not all(isinstance(v, str) and v for v in row.values()):
            raise ValueError('Registry tasks require explicit family and task strings')
        key = row['benchmark_family'], row['task']
        if key in expected:
            raise ValueError('Duplicate registry task')
        expected.add(key)
    pools, coverage = {}, {}
    for stage in STAGES:
        expanded = samples(read_ref(spec['stages'][stage]))
        pools[stage] = set(expanded)
        if stage == 'rollout' and len(expanded) != len(pools[stage]):
            raise ValueError('Rollout manifest repeats an episode identity')
        actual = {row[:2] for row in pools[stage]}
        coverage[stage] = {
            'listed_samples': len(expanded), 'unique_episodes': len(pools[stage]),
            'included_tasks': len(actual), 'expected_tasks': len(expected),
            'missing_tasks': [list(row) for row in sorted(expected-actual)],
            'unexpected_tasks': [list(row) for row in sorted(actual-expected)],
        }
    overlaps = {stage: [dict(zip(KEYS, row)) for row in sorted(pools[stage] & pools['rollout'])]
                for stage in STAGES[:-1]}
    disjoint = not any(overlaps.values())
    full_scope = all(not row['missing_tasks'] and not row['unexpected_tasks'] for row in coverage.values())
    return {
        'schema': 'sacq_sample_audit_v1',
        'status': 'PASS_DISJOINT' if disjoint and full_scope else 'FAIL',
        'inputs': {'task_registry': spec['task_registry'], 'stages': spec['stages']},
        'coverage': coverage, 'development_rollout_overlaps': overlaps,
        'calibration_train_validation_overlap_episodes': len(pools['calibration_train'] & pools['calibration_validation']),
        'meaning': 'Episode-level development/rollout separation and full coverage of the supplied registry. '
                   'Calibration train/validation overlap is reported separately, not silently treated as independence. '
                   'Registry completeness and sample labels require provenance; this is not execution authorization.',
    }


def validate_receipt(receipt):
    """Recompute from bound manifests rather than trusting a handwritten PASS."""
    if receipt.get('schema') != 'sacq_sample_audit_v1':
        raise ValueError('Unsupported public sample-audit receipt')
    inputs = receipt['inputs']
    actual = audit(dict(schema='sacq_sample_audit_spec_v1', **inputs))
    if actual != receipt or actual['status'] != 'PASS_DISJOINT':
        raise ValueError('Sample audit is failed, changed or inconsistent with its inputs')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error('Choose a new output; existing audit receipts are not overwritten')
    result = audit(json.loads(a.spec.read_text()))
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'coverage': result['coverage']}))
    return 0 if result['status'] == 'PASS_DISJOINT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
