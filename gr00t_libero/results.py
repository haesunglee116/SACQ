"""Aggregate an explicit expected episode set; never glob or infer denominators.

Input manifest: schema=sacq_episode_results_v1, label, aggregation
(episode_micro or task_macro), and samples=[{path, expected:{...}}].
Expected fields mirror the pinned rollout worker's durable episode records.
Partial reports expose completed-only rates separately from final success rates.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

IDENTITY_FIELDS = ('method', 'suite', 'task', 'base_seed', 'policy_seed',
                   'episode_id', 'n_envs', 'protocol_sha256')


def percentage(successes, count):
    return 100.0 * successes / count if count else None


def aggregate(manifest, *, allow_partial=False):
    if manifest.get('schema') != 'sacq_episode_results_v1':
        raise ValueError('Unsupported episode manifest')
    if manifest.get('aggregation') not in ('episode_micro', 'task_macro'):
        raise ValueError('Explicit aggregation rule required')
    if not isinstance(manifest.get('label'), str) or not manifest['label']:
        raise ValueError('Explicit experimental condition label required')
    samples = manifest['samples']
    if not samples:
        raise ValueError('Expected episode set must not be empty')
    seen, paths, missing, evidence = set(), set(), [], []
    groups = defaultdict(lambda: dict(expected=0, completed=0, successes=0))
    for item in samples:
        expected = item['expected']
        if set(expected) != set(IDENTITY_FIELDS):
            raise ValueError('Every expected identity field must be explicit')
        for key in ('method', 'suite', 'task', 'protocol_sha256'):
            if not isinstance(expected[key], str) or not expected[key]:
                raise ValueError(f'Expected identity requires {key}')
        for key in ('base_seed', 'policy_seed', 'episode_id', 'n_envs'):
            if type(expected[key]) is not int or expected[key] < 0:
                raise ValueError(f'Expected identity requires nonnegative integer {key}')
        if expected['n_envs'] == 0:
            raise ValueError('n_envs must be positive')
        identity = tuple(expected[k] for k in ('method', 'suite', 'task', 'base_seed', 'episode_id'))
        path = Path(item['path'])
        if not path.is_absolute():
            raise ValueError('Result paths must be absolute')
        path = path.resolve()
        if identity in seen or path in paths:
            raise ValueError('Duplicate expected episode or result path')
        seen.add(identity); paths.add(path)
        group = groups[(expected['method'], expected['base_seed'], expected['suite'], expected['task'])]
        group['expected'] += 1
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = path.read_bytes()
        record = json.loads(payload)
        if record.get('status') != 'COMPLETE':
            raise ValueError(f'Incomplete episode record: {path}')
        for key in IDENTITY_FIELDS:
            if type(record.get(key)) is not type(expected[key]) or record[key] != expected[key]:
                raise ValueError(f'Episode identity mismatch ({key}): {path}')
        if type(record.get('success')) is not bool:
            raise ValueError('Episode success must be a boolean')
        group['completed'] += 1
        group['successes'] += int(record['success'])
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()))
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} expected episodes are missing; no final aggregate produced')
    tasks = []
    for (method, seed, suite, task), counts in sorted(groups.items()):
        complete = counts['completed'] == counts['expected']
        rate = percentage(counts['successes'], counts['completed'])
        tasks.append(dict(method=method, base_seed=seed, suite=suite, task=task, **counts,
                          success_rate_pct=rate if complete else None,
                          completed_success_rate_pct=rate))
    completed = sum(row['completed'] for row in tasks)
    successes = sum(row['successes'] for row in tasks)
    if manifest['aggregation'] == 'episode_micro':
        rate = percentage(successes, completed)
    else:
        rates = [row['completed_success_rate_pct'] for row in tasks if row['completed']]
        rate = sum(rates)/len(rates) if rates else None
    return dict(schema='sacq_episode_summary_v1', label=manifest['label'],
                status='PARTIAL' if missing else 'COMPLETE', aggregation=manifest['aggregation'],
                expected=len(samples), completed=completed, successes=successes,
                success_rate_pct=rate if not missing else None,
                completed_success_rate_pct=rate, tasks=tasks, missing=missing, inputs=evidence,
                meaning='Explicit episode-set aggregation; not an independent scientific protocol approval')



def from_plans(spec):
    """Expand the pinned worker's episode set from explicit, hashed plans."""
    if spec.get('schema')!='sacq_result_manifest_build_v1':
        raise ValueError('Unsupported result manifest build schema')
    if spec.get('aggregation') not in ('episode_micro','task_macro') or not isinstance(spec.get('label'),str) or not spec['label']:
        raise ValueError('Explicit label and aggregation required')
    if not isinstance(spec.get('plans'),list) or not spec['plans']:
        raise ValueError('Explicit nonempty plan list required')
    profile=json.loads((Path(__file__).parent/'profiles/legacy_rollout.json').read_text())['rollout']
    inputs=[]
    def read(ref):
        path=Path(ref['path'])
        if not path.is_absolute():raise ValueError('Plan/task references must be absolute')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('Plan/task identity mismatch')
        inputs.append(dict(path=str(path.resolve()),sha256=ref['sha256']))
        return json.loads(data)
    samples=[];seen=set();condition=None
    for row in spec['plans']:
        plan=read(row['plan']);tasks=read(row['tasks'])
        if plan.get('method')!='final_seed17' or plan.get('rollout')!=profile:
            raise ValueError('Plan differs from supported pinned rollout')
        root=Path(plan['root'])
        if not root.is_absolute() or Path(row['tasks']['path']).resolve()!=(root/'tasks.json').resolve():
            raise ValueError('Plan task registry path mismatch')
        current=(plan['condition'],plan['language_action_nominal_bpw'])
        if condition is None:condition=current
        elif current!=condition:raise ValueError('Do not aggregate different quantization conditions')
        index=plan['worker_task_index']
        if type(index) is not int or not 0<=index<len(tasks):raise ValueError('Invalid worker task index')
        task=tasks[index]
        for key in ('suite','task'):
            value=task[key]
            if not isinstance(value,str) or not value or '/' in value or '\\' in value or value in ('.','..'):
                raise ValueError('Invalid task path component')
        if plan['selected_suites']!=[task['suite']]:raise ValueError('Plan suite differs from task')
        identity=(task['suite'],task['task'],profile['environment_seed'])
        if identity in seen:raise ValueError('Duplicate planned task episodes')
        seen.add(identity)
        for episode in range(profile['episodes_per_task']):
            path=root/'episodes'/plan['method']/task['suite']/task['task']/f'{episode:03d}.json'
            expected=dict(method=plan['method'],suite=task['suite'],task=task['task'],
                base_seed=profile['environment_seed'],policy_seed=profile['policy_seed'],
                episode_id=episode,n_envs=profile['n_envs'],protocol_sha256=row['plan']['sha256'])
            samples.append(dict(path=str(path),expected=expected))
    return dict(schema='sacq_episode_results_v1',label=spec['label'],aggregation=spec['aggregation'],
                samples=samples,plan_inputs=inputs,
                meaning='Expected samples for supplied plans only; not benchmark coverage or execution approval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path)
    mode.add_argument('--build-spec', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.build_spec:
        if args.allow_partial:parser.error('--allow-partial is only for aggregation')
        result=from_plans(json.loads(args.build_spec.read_text()))
        with args.output.open('x') as target:
            json.dump(result,target,indent=2);target.write('\n')
        print(f"Prepared {len(result['samples'])} expected episodes; no results read or rollout launched")
        return
    payload = args.manifest.read_bytes()
    result = aggregate(json.loads(payload), allow_partial=args.allow_partial)
    result['manifest_sha256'] = hashlib.sha256(payload).hexdigest()
    with args.output.open('x') as target:
        json.dump(result, target, indent=2)
        target.write('\n')
    print(f"{result['status']}: {result['completed']}/{result['expected']} episodes")


if __name__ == '__main__':
    main()
