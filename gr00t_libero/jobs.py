"""Build explicit index/calibration/evaluation jobs for the pinned GR00T–LIBERO workers.

No model imports, scheduler calls, data selection, or scientific defaults.
The generated files are preparation artifacts, never execution authorization.
"""
import argparse
import copy
import csv
import json
import re
from pathlib import Path

try:
    from . import pipeline
    from .configuration import validate_training, validate_index
    from .bank import PackedBank
except ImportError:
    import pipeline
    from configuration import validate_training, validate_index
    from bank import PackedBank


def parse_force_specs(text):
    """Parse the pinned compiler's exact-layer installation format strictly."""
    values = {}
    for entry in re.split(r'[;\r\n]+', text):
        entry = entry.strip()
        if not entry or entry.startswith('#'):
            continue
        try:
            name, value = entry.split('=', 1)
            name, bits = name.strip(), int(value.strip())
        except ValueError as exc:
            raise ValueError('Malformed installation force-bit specification') from exc
        if not name or name in values or bits not in (3, 4, 16):
            raise ValueError('Installation specs require unique layers and supported precision')
        values[name] = bits
    return values


class Builder:
    def __init__(self):
        self.inputs = {}

    def read(self, ref):
        path = Path(ref['path'])
        if not path.is_absolute():
            raise ValueError('Input paths must be absolute')
        path = path.resolve()
        digest = pipeline.sha(path)
        if digest != ref['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        self.inputs[str(path)] = dict(path=str(path), sha256=digest)
        return path

    def json(self, ref):
        return json.loads(self.read(ref).read_text())

    def records(self, rows, label):
        if not isinstance(rows, list):
            raise ValueError(f'{label} must be an explicit list')
        result = copy.deepcopy(rows)
        for row in result:
            row['path'] = str(self.read(row))
        return result

    def layers(self, rows, label):
        result = self.records(rows, label)
        names = [r['layer'] for r in result]
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate layers in {label}')
        return result

    def base(self, spec):
        if spec.get('schema') != 'sacq_worker_build_v1':
            raise ValueError('Unsupported builder schema')
        if not Path(spec['workspace']).is_absolute():
            raise ValueError('Workspace must be an explicit absolute path')
        job = copy.deepcopy(spec['job'])
        for key in ('out', 'local_qa', 'batch_cache', 'local_output'):
            if key in job and not Path(job[key]).is_absolute():
                raise ValueError(f'{key} must be an absolute path')
        derived = {'trainer', 'cached_launcher', 'allocation', 'artifacts', 'gptq', 'targets', 'records',
                   'all_records', 'sacq_count', 'gptq_count'}
        if derived.intersection(job):
            raise ValueError('Derived job fields must be supplied through builder inputs')
        for key in ('model', 'processor'):
            path = Path(job[key])
            if not path.is_absolute() or not path.is_dir():
                raise ValueError(f'{key} must be an existing absolute directory')
        # Trainer and cache launcher are pinned files, not untracked CLI paths.
        job['trainer'] = str(self.read(spec['trainer']))
        job['allocation'] = str(self.read(spec['allocation']))
        with open(job['allocation'], newline='') as source:
            rows = list(csv.DictReader(source))
        names = [r['layer'] for r in rows]
        if not rows or len(names) != len(set(names)):
            raise ValueError('Allocation must contain unique layer rows')
        # These sentinels come from the pinned fixed_model.py, not a new policy.
        for row in rows:
            k = int(row['k'])
            if k <= 0 or (k >= 80000 and k not in (80000, 90000)):
                raise ValueError('Unsupported allocation candidate encoding')
            if row['scope'] not in ('vlm', 'dit'):
                raise ValueError('Unsupported worker scope')
        selected = {r['layer']: r for r in rows if int(r['k']) < 80000}
        if ('artifacts' in spec)==('bank_manifest' in spec):
            raise ValueError('Specify exactly one of artifacts or bank_manifest')
        if 'bank_manifest' in spec:
            choice=spec.get('bank_selection',{})
            if set(choice)!={'suite','init'} or not all(isinstance(v,str) and v for v in choice.values()):
                raise ValueError('Bank selection requires explicit suite and init')
            if 'suite' in job and job['suite']!=choice['suite']:
                raise ValueError('Bank selection suite differs from job suite')
            bank=PackedBank(self.json(spec['bank_manifest']))
            artifacts=[]
            for layer,row in selected.items():
                scope,k=row['scope'],int(row['k'])
                path=bank.find_packed(scope,layer,k,suite=choice['suite'],init=choice['init'],verify=False)
                key=(scope,layer,k,choice['suite'],choice['init'])
                entry=bank.rows.get(key)
                if entry is None:
                    entry=bank.rows[(scope,layer,k,'shared',choice['init'])]
                artifacts.append(dict(layer=layer,scope=scope,k=k,path=str(path),sha256=entry['sha256']))
            job['artifacts']=self.layers(artifacts,'artifacts')
        else:
            if 'bank_selection' in spec:
                raise ValueError('bank_selection requires bank_manifest')
            job['artifacts'] = self.layers(spec['artifacts'], 'artifacts')
        job['gptq'] = self.layers(spec['gptq'], 'gptq')
        if {r['layer'] for r in job['artifacts']} != set(selected):
            raise ValueError('Selected bank must exactly cover allocation codebook layers')
        for row in job['artifacts']:
            expected = selected[row['layer']]
            if row['scope'] != expected['scope'] or int(row['k']) != int(expected['k']):
                raise ValueError('Bank candidate differs from allocation')
        background = {r['layer']: r for r in job['gptq']}
        if set(background).intersection(selected):
            raise ValueError('Codebook and frozen background layers overlap')
        for row in job['gptq']:
            if type(row.get('bits')) is not int:
                raise ValueError('Background precision must be explicit')
        for row in rows:
            if int(row['k']) >= 80000:
                expected = 3 if int(row['k']) == 80000 else 4
                if background.get(row['layer'], {}).get('bits') != expected:
                    raise ValueError('Missing or inconsistent scalar allocation background')
        return job

    def build(self, stage, spec):
        if stage == 'evaluation':
            return self.evaluation(spec)
        job = self.base(spec)
        if stage == 'index':
            validate_index(job.get('index_settings'))
            names = spec['target_layers']
            artifacts = {r['layer']: r for r in job['artifacts']}
            if not names or len(set(names)) != len(names) or not set(names).issubset(artifacts):
                raise ValueError('Targets must be explicit unique selected codebook layers')
            job['targets'] = [copy.deepcopy(artifacts[name]) for name in names]
            job['records'] = self.records(spec['records'], 'records')
            positions = [r['position'] for r in job['records']]
            if any(type(p) is not int or p < 0 for p in positions) or len(set(positions)) != len(positions):
                raise ValueError('Index request positions must be unique nonnegative integers')
        elif stage == 'calibration':
            job['cached_launcher'] = str(self.read(spec['cached_launcher']))
            job['all_records'] = self.records(spec['all_records'], 'all_records')
            if any(Path(row['path']).suffix != '.msgpack' for row in job['all_records']):
                raise ValueError('Calibration loader requires .msgpack request files')
            validate_training(job['training'], len(job['all_records']))
            corrected = {}
            reference_records = None
            reference_index_settings = None
            for shard in spec['index_shards']:
                source = self.json(shard['job'])
                complete = self.json(shard['completion'])
                if complete.get('status') != 'COMPLETE_DIAGNOSTIC' or complete.get('job_sha256') != shard['job']['sha256']:
                    raise ValueError('Index completion does not belong to the completed job')
                settings = validate_index(source.get('index_settings'))
                if reference_index_settings is None:
                    reference_index_settings = settings
                elif settings != reference_index_settings:
                    raise ValueError('Index shards used different correction settings')
                for key in ('allocation', 'artifacts', 'gptq', 'model', 'processor', 'trainer'):
                    if source.get(key) != job[key]:
                        raise ValueError(f'Index/calibration input mismatch: {key}')
                if reference_records is None:
                    reference_records = source['records']
                elif source['records'] != reference_records:
                    raise ValueError('Index shards used different request identities or order')
                selected = {r['layer']: r for r in job['artifacts']}
                if any(row != selected.get(row['layer']) for row in source['targets']):
                    raise ValueError('Index targets differ from the selected original bank')
                expected = {r['layer'] for r in source['targets']}
                outputs = self.layers(complete['outputs'], 'index outputs')
                if {r['layer'] for r in outputs} != expected:
                    raise ValueError('Index shard output coverage mismatch')
                for row in outputs:
                    if row.get('job_sha256') != shard['job']['sha256'] or row.get('indices_only_verified') is not True:
                        raise ValueError('Missing index-only completion evidence')
                    if row['layer'] in corrected:
                        raise ValueError('Overlapping index shards')
                    corrected[row['layer']] = row
            if set(corrected) != {r['layer'] for r in job['artifacts']}:
                raise ValueError('Completed index shards must cover every selected codebook layer')
            for row in job['artifacts']:
                result = corrected[row['layer']]
                row.update(path=result['path'], sha256=result['sha256'])
            job.update(sacq_count=len(job['artifacts']), gptq_count=len(job['gptq']))
        else:
            raise ValueError('Supported stages: index, calibration')
        pipeline.validate_job(stage, job, Path(spec['workspace']).resolve())
        return job

    def evaluation(self, spec):
        if spec.get('schema') != 'sacq_worker_build_v1':
            raise ValueError('Unsupported builder schema')
        plan = copy.deepcopy(spec['plan'])
        pipeline.validate_job('evaluation', plan, Path(spec['workspace']).resolve())
        profile = Path(__file__).parent/'profiles/legacy_rollout.json'
        expected = self.json(dict(path=str(profile.resolve()), sha256=pipeline.sha(profile)))['rollout']
        if plan['rollout'] != expected:
            raise ValueError('Rollout differs from the supported pinned worker contract')
        if plan['diagnostic_only'] is not True or plan['held_out'] is not False:
            raise ValueError('Pinned worker supports diagnostic-only evidence')
        for key in ('root', 'repo', 'socket_root', 'worker_source', 'sim_source',
                    'sim_python', 'cosmos', 'noise_root', 'reset_audit_root'):
            if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
                raise ValueError(f'Evaluation requires absolute {key}')
        for key in ('models', 'reset_task_contract_sha256', 'paired_initial_observation_sha256'):
            if not isinstance(plan.get(key), dict):
                raise ValueError(f'Evaluation requires explicit {key}')
        if not isinstance(plan.get('suites'), list) or not plan['suites']:
            raise ValueError('Evaluation requires explicit suites')
        if plan.get('method') != 'final_seed17' or plan.get('reset_identity_status') != 'PASS':
            raise ValueError('Pinned method and original reset PASS required')
        if type(plan.get('rollout_authorized')) is not bool:
            raise ValueError('Explicit rollout_authorized declaration required')
        if plan.get('render_reference_mode') != 'same_process_saved_rng_exact':
            raise ValueError('Pinned renderer reference mode required')
        if type(plan.get('language_action_nominal_bpw')) not in (float, int) or plan['language_action_nominal_bpw'] <= 0:
            raise ValueError('Explicit positive nominal BPW required')
        tasks_path = self.read(spec['tasks'])
        if tasks_path != (Path(plan['root'])/'tasks.json').resolve():
            raise ValueError('Task registry must be the one read by the worker')
        tasks = json.loads(tasks_path.read_text())
        index = plan['worker_task_index']
        if not 0 <= index < len(tasks):
            raise ValueError('Task index outside explicit registry')
        task = tasks[index]
        for key in ('suite', 'task', 'bddl', 'language'):
            if not isinstance(task.get(key), str) or not task[key]:
                raise ValueError(f'Task requires {key}')
        if type(task.get('task_id')) is not int:
            raise ValueError('Task requires explicit task_id')
        suite = task['suite']
        if plan.get('selected_suites') != [suite] or suite not in plan['suites']:
            raise ValueError('Selected suite differs from worker task')
        if suite not in plan['models'] or not Path(plan['models'][suite]).is_absolute():
            raise ValueError('Explicit suite checkpoint required')
        key = suite+'/'+task['task']
        if not plan['reset_task_contract_sha256'].get(key):
            raise ValueError('Selected task has no reset contract identity')
        plan['sources'] = self.records(plan['sources'], 'sources')
        source_paths = {r['path'] for r in plan['sources']}
        for key in ('worker_source', 'sim_source'):
            if str(Path(plan[key]).resolve()) not in source_paths:
                raise ValueError(f'{key} is not among hashed sources')
        if 'workshop_model' in plan:
            raise ValueError('workshop_model is derived from completed calibration inputs')
        if plan['condition'] == 'bf16':
            if 'calibration' in spec:
                raise ValueError('BF16 must not install a calibration overlay')
        elif plan['condition'] in ('sacq', 'sacq1.5'):
            calibration = spec['calibration']
            job = self.json(calibration['job'])
            complete = self.json(calibration['completion'])
            if complete.get('status') != 'COMPLETE_DIAGNOSTIC' or complete.get('job_sha256') != calibration['job']['sha256']:
                raise ValueError('Calibration completion identity mismatch')
            if complete.get('updates') != job['training']['expected_updates']:
                raise ValueError('Calibration completion budget mismatch')
            if job['suite'] != suite or job['model'] != plan['models'][suite] or job['processor'] != plan['cosmos']:
                raise ValueError('Calibration/evaluation model or suite mismatch')
            overlay = {}
            for key in ('vlm', 'dit'):
                ref = complete['outputs'][key]
                overlay[key] = dict(path=str(self.read(ref)), sha256=ref['sha256'])
            for key in ('vlm_spec', 'dit_spec'):
                ref = calibration[key]
                path = self.read(ref)
                codebooks = {r['layer'] for r in job['artifacts']}
                for name, bits in parse_force_specs(path.read_text()).items():
                    if name in codebooks and (key == 'vlm_spec' or bits != 16):
                        raise ValueError('Installation spec conflicts with calibrated codebook layer: '+name)
                overlay[key] = dict(path=str(path), sha256=ref['sha256'])
            overlay['gptq'] = self.layers(job['gptq'], 'calibration background')
            if job['sacq_count'] != len(job['artifacts']) or job['gptq_count'] != len(overlay['gptq']):
                raise ValueError('Calibration layer counts are inconsistent')
            overlay['expected_sacq_count'] = job['sacq_count']
            overlay['expected_gptq_count'] = job['gptq_count']
            plan['workshop_model'] = overlay
        else:
            raise ValueError('Installer supports bf16 and sacq (legacy alias sacq1.5)')
        return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('index', 'calibration', 'evaluation'))
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    builder = Builder()
    spec = builder.json(dict(path=str(spec_path), sha256=pipeline.sha(spec_path)))
    job = builder.build(args.stage, spec)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pipeline.atomic(out/'job.json', job)
    refs = list(builder.inputs.values())
    refs.append(dict(path=str(out/'job.json'), sha256=pipeline.sha(out/'job.json')))
    pipeline.atomic(out/'inputs.json', refs)
    print(f'Prepared {out}/job.json; no experiment launched')


if __name__ == '__main__':
    main()
