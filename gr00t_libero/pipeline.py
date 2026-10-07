"""Portable source staging and explicit, receipt-checked stage execution.

This driver does not manufacture scientific settings or missing job manifests.
It launches the supplied research workers only after explicit input validation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import posixpath
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager, ExitStack

GR00T_REVISION='4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e'
LIBERO_REVISION='8f1084e3132a39270c3a13ebe37270a43ece2a01'
ROOT=Path(__file__).resolve().parent
STAGES=('fitting','allocation','index','calibration','evaluation')
B='libero_pro_eval/'
F='gr00t_final_seed17_20260922/'
PIN=B+'quantization_analysis/workshop_fixedalloc_refit_20260919/source/'
ENTRYPOINTS={
 'fitting':['release_stage.py'],
 'allocation':['release_stage.py'],
 'index':[F+'index_worker.py'],
 'calibration':[F+'calibration_worker.py'],
 'evaluation':[F+'rollout_worker.py'],
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def inside(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes workspace')
    return p

def export_runtime(checkout, destination, revision=None):
    """Export the pinned commit, ignoring mutable/untracked checkout contents."""
    checkout=Path(checkout).resolve()
    revision = GR00T_REVISION if revision is None else revision
    commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse',revision+'^{commit}'],text=True,stderr=subprocess.PIPE).strip()
    if commit!=revision:
        raise ValueError('Unexpected runtime revision')
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['git','-C',str(checkout),'archive','--format=tar',commit],stdout=stream,check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream,mode='r:') as archive:
            members=archive.getmembers()
            by_name={member.name:member for member in members}
            resolved={}
            for member in members:
                inside(destination,member.name)
                if member.issym():
                    target=member
                    seen=set()
                    while target.issym():
                        if target.name in seen or target.linkname.startswith('/'):
                            raise ValueError('Runtime archive has an unsafe symbolic link')
                        seen.add(target.name)
                        name=posixpath.normpath(posixpath.join(posixpath.dirname(target.name),target.linkname))
                        inside(destination,name)
                        if name not in by_name:
                            raise ValueError('Runtime archive symbolic link target missing')
                        target=by_name[name]
                    if not target.isfile():
                        raise ValueError('Runtime archive links must resolve to regular files')
                    resolved[member.name]=target
                elif not (member.isdir() or member.isfile()):
                    raise ValueError('Runtime archive contains an unsupported entry')
            destination.mkdir(parents=True)
            for member in members:
                target=inside(destination,member.name)
                if member.isdir():
                    target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(resolved.get(member.name,member)) as source,target.open('wb') as output:
                        shutil.copyfileobj(source,output)
                    target.chmod(resolved.get(member.name,member).mode & 0o777)
    return commit


def prepare(workspace, gr00t_checkout=None, libero_checkout=None):
    workspace=Path(workspace).resolve()
    if workspace.exists():raise ValueError('Choose a NEW workspace; existing trees are never overwritten')
    if libero_checkout is not None and gr00t_checkout is None:
        raise ValueError('LIBERO preparation requires the pinned GR00T checkout too')
    rows=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for r in rows:
        p=inside(ROOT/'sources',r['file'])
        if sha(p)!=r['export_sha256']:raise ValueError('Export source changed: '+r['file'])
    if any(c in str(workspace) for c in ('"', "'", '\\', '\n')):
        raise ValueError('Workspace path contains unsupported quoting characters')
    workspace.mkdir(parents=True)
    runtime_files=[]
    if gr00t_checkout is not None:
        try:
            export_runtime(gr00t_checkout,workspace/B/'Isaac-GR00T-N1.7')
            if libero_checkout is not None:
                external=workspace/B/'Isaac-GR00T-N1.7/external_dependencies/LIBERO'
                if external.is_dir() and not any(external.iterdir()):
                    external.rmdir()
                if external.exists():
                    raise ValueError('GR00T runtime unexpectedly contains a nonempty LIBERO tree')
                export_runtime(libero_checkout,external,LIBERO_REVISION)
        except BaseException:
            shutil.rmtree(workspace)
            raise
        runtime_files=[str(p.relative_to(workspace)) for p in (workspace/B/'Isaac-GR00T-N1.7').rglob('*') if p.is_file()]
    staged=[]
    for r in rows:
        dest=inside(workspace,r['file']);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text((ROOT/'sources'/r['file']).read_text().replace('@WORKSPACE@',str(workspace)))
        staged.append({'file':r['file'],'sha256':sha(dest)})
    shutil.copyfile(ROOT/'stage.py',workspace/'release_stage.py')
    staged.append({'file':'release_stage.py','sha256':sha(workspace/'release_stage.py')})
    shutil.copyfile(ROOT/'configuration.py',workspace/F/'release_configuration.py')
    staged.append({'file':F+'release_configuration.py','sha256':sha(workspace/F/'release_configuration.py')})
    shutil.copyfile(ROOT/'bank.py',workspace/'release_bank.py')
    staged.append({'file':'release_bank.py','sha256':sha(workspace/'release_bank.py')})
    shutil.copyfile(ROOT/'cache_identity.py',workspace/F/'release_cache_identity.py')
    staged.append({'file':F+'release_cache_identity.py','sha256':sha(workspace/F/'release_cache_identity.py')})
    overlaid={row['file'] for row in staged}
    staged.extend({'file':name,'sha256':sha(workspace/name)} for name in sorted(runtime_files) if name not in overlaid)
    atomic(workspace/'SOURCE_RECEIPT.json',{'sources':staged,
           'gr00t_revision':GR00T_REVISION if gr00t_checkout is not None else None,
           'libero_revision':LIBERO_REVISION if libero_checkout is not None else None})
    return {'status':'SOURCES_STAGED_NOT_EXPERIMENT_READY','files':len(staged),'workspace':str(workspace)}

def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('schema')!='sacq_gr00t_libero_execution_v1':raise ValueError('Wrong configuration schema')
    for key in ('workspace','python','stages','protocol_manifest'):
        if not cfg.get(key):raise ValueError('Required configuration: '+key)
    if set(cfg['stages'])!=set(STAGES):raise ValueError('All five stages must be specified')
    return cfg

def resolve_input(cfg, stage, item):
    if 'from_stage' not in item:
        return {'path':str(Path(item['path']).resolve()),'sha256':item['sha256']}
    previous=item['from_stage']
    if previous not in STAGES[:STAGES.index(stage)]:
        raise ValueError('Input reference must name an earlier stage')
    record=Path(cfg['workspace'])/'pipeline_receipts'/(previous+'.json')
    receipt=json.loads(record.read_text())
    if receipt.get('status')!='COMPLETE':raise ValueError('Referenced stage is not complete')
    matches=[r for r in receipt['outputs'] if r['path']==str(Path(item['path']).resolve())]
    if len(matches)!=1:raise ValueError('Referenced output is not in completion receipt')
    return matches[0]

def stage_protocol(cfg,stage):
    """Allow late-bound output references without rewriting earlier contracts."""
    base_ref=cfg['protocol_manifest']
    selected=cfg['stages'][stage].get('protocol_manifest',base_ref)
    for ref in (base_ref,selected):
        if sha(ref['path'])!=ref['sha256']:
            raise ValueError('Protocol identity mismatch')
    if selected!=base_ref:
        base=json.loads(Path(base_ref['path']).read_text())
        extension=json.loads(Path(selected['path']).read_text())
        if {k:v for k,v in base.items() if k!='inputs'}!={k:v for k,v in extension.items() if k!='inputs'}:
            raise ValueError('Stage protocol may only append input references; settings and approval must not change')
        original=base.get('inputs',[]); updated=extension.get('inputs',[])
        if not isinstance(original,list) or not isinstance(updated,list) or updated[:len(original)]!=original:
            raise ValueError('Stage protocol must preserve original input references')
        for ref in updated[len(original):]:
            if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not Path(ref['path']).is_absolute():
                raise ValueError('Additional protocol input requires an absolute path and hash')
            if sha(ref['path'])!=ref['sha256']:
                raise ValueError('Additional protocol input identity mismatch')
    return selected

@contextmanager
def evaluation_execution(root, protocol_sha256):
    """Own a real worker execution interval; never manufacture scientific approval."""
    import fcntl
    path=root/'EXECUTION_RECEIPT.json'
    with (root/'public_execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('execution_status')=='RUNNING':
                raise ValueError('Existing RUNNING execution must be resolved before relaunch')
            if previous.get('schema')!='sacq_public_evaluation_execution_v1' or previous.get('protocol_sha256')!=protocol_sha256:
                raise ValueError('Refusing to replace an external or different-protocol execution receipt')
        record=dict(schema='sacq_public_evaluation_execution_v1',execution_status='RUNNING',
                    protocol_sha256=protocol_sha256,launcher_pid=os.getpid(),
                    meaning='Public driver execution interval; not protocol approval or success-rate evidence')
        atomic(path,record)
        try:
            yield
        except BaseException as exc:
            atomic(path,dict(record,execution_status='FAILED',error_type=type(exc).__name__))
            raise
        else:
            atomic(path,dict(record,execution_status='FINISHED'))

def fingerprint(cfg,stage):
    payload={k:cfg[k] for k in ('workspace','python','protocol_manifest')}
    payload['stage_protocol_manifest']=stage_protocol(cfg,stage)
    if stage=='evaluation':payload['manage_evaluation_receipts']=cfg.get('manage_evaluation_receipts',False)
    payload['stage']=cfg['stages'][stage]
    payload['sources']=sha(Path(cfg['workspace'])/'SOURCE_RECEIPT.json')
    payload['inputs']=[resolve_input(cfg,stage,r) for r in cfg['stages'][stage]['inputs']]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_job(stage, job, workspace):
    """Check worker input structure without inventing scientific settings.

    This is a structural check, not a replacement for the worker's scientific
    contract, artifact checks, or model execution.
    """
    if not isinstance(job, dict):
        raise ValueError('Job configuration must be an object')
    if stage in ('fitting', 'allocation'):
        if not isinstance(job.get('workspace'), str) or not Path(job['workspace']).is_absolute():
            raise ValueError('Stage workspace must be explicit and absolute')
        if Path(job['workspace']).resolve() != workspace:
            raise ValueError('Stage workspace differs from verified workspace')
        allowed = ('metrics', 'vlm', 'dit') if stage == 'fitting' else ('allocation',)
        if job.get('operation') not in allowed:
            raise ValueError('Operation does not match pipeline stage')
        required = {'processor': str}
        required.update({'checkpoint': str} if stage == 'fitting' else
                        {'models': dict, 'allocation_requests': str, 'allocation_work': str})
        if stage == 'allocation':
            if ('bank' in job) == ('bank_manifest' in job):
                raise ValueError('Specify exactly one of bank or bank_manifest')
            required['bank_manifest' if 'bank_manifest' in job else 'bank'] = dict if 'bank_manifest' in job else str
    elif stage == 'index':
        required = dict(out=str, processor=str, trainer=str, model=str,
                        allocation=str, artifacts=list, gptq=list, targets=list, records=list, index_settings=dict)
    elif stage == 'calibration':
        required = dict(out=str, processor=str, trainer=str, model=str,
                        allocation=str, artifacts=list, gptq=list, all_records=list,
                        sacq_count=int, gptq_count=int, local_qa=str, cached_launcher=str,
                        batch_cache=str, local_output=str, suite=str, tag=str, training=dict)
    else:
        required = dict(root=str, worker_task_index=int, rollout=dict, sources=list,
                        repo=str, worker_source=str, socket_root=str, condition=str,
                        diagnostic_only=bool, held_out=bool)
    for key, expected in required.items():
        value = job.get(key)
        if type(value) is not expected or (expected in (str, list, dict) and not value):
            raise ValueError(f'{stage} job requires nonempty {key}: {expected.__name__}')
    if stage == 'allocation' and 'bank_manifest' in job:
        ref=job['bank_manifest']
        if not all(isinstance(ref.get(k),str) and ref[k] for k in ('path','sha256')):
            raise ValueError('bank_manifest requires path and sha256')
    for key in ('artifacts', 'gptq', 'targets'):
        for item in job.get(key, []):
            if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k] for k in ('layer', 'path')):
                raise ValueError(f'{stage}.{key} requires layer/path records')
    for key in ('records', 'all_records'):
        for item in job.get(key, []):
            if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']:
                raise ValueError(f'{stage}.{key} requires request path records')
            if stage == 'index' and type(item.get('position')) is not int:
                raise ValueError('index.records requires explicit integer position')
    if stage == 'calibration':
        for count, records in (('sacq_count', 'artifacts'), ('gptq_count', 'gptq')):
            if job[count] != len(job[records]):
                raise ValueError(f'{count} differs from supplied layer records')

def check(cfg,stage):
    workspace=Path(cfg['workspace']).resolve()
    receipt=json.loads((workspace/'SOURCE_RECEIPT.json').read_text())
    for r in receipt['sources']:
        if sha(inside(workspace,r['file']))!=r['sha256']:raise ValueError('Staged source drift: '+r['file'])
    if not Path(cfg['python']).is_file():raise ValueError('Explicit model Python executable missing')
    protocol=stage_protocol(cfg,stage)
    if sha(protocol['path'])!=protocol['sha256']:raise ValueError('Protocol identity mismatch')
    specification=json.loads(Path(protocol['path']).read_text())
    if specification.get('benchmark')!='gr00t_libero':raise ValueError('Wrong benchmark scope')
    if specification.get('scientific_settings_complete') is not True:
        raise ValueError('Scientific settings are not resolved; do not infer them')
    if specification.get('sample_audit_status') not in ('PASS_DISJOINT','APPROVED_DIAGNOSTIC_ONLY'):
        raise ValueError('Missing development/evaluation sample audit')
    if specification['sample_audit_status']=='APPROVED_DIAGNOSTIC_ONLY' and (
        specification.get('diagnostic_only') is not True or specification.get('held_out') is not False):
        raise ValueError('Diagnostic evidence must not claim held-out status')
    audit=specification.get('sample_audit_receipt')
    if not audit or sha(audit['path'])!=audit['sha256']:
        raise ValueError('Missing or changed sample audit receipt')
    audit_data=json.loads(Path(audit['path']).read_text())
    if audit_data.get('schema')=='sacq_sample_audit_v1':
        module_spec=importlib.util.spec_from_file_location('release_sample_audit',ROOT/'sample_audit.py')
        module=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.validate_receipt(audit_data)
        if specification['sample_audit_status']!=audit_data['status']:
            raise ValueError('Protocol/sample audit status mismatch')
    row=cfg['stages'][stage]
    if not row.get('commands') or not row.get('inputs') or not row.get('outputs'):
        raise ValueError('Each stage needs commands, hashed inputs and expected outputs')
    inputs=[resolve_input(cfg,stage,r) for r in row['inputs']]
    for r in inputs:
        if sha(r['path'])!=r['sha256']:raise ValueError('Input identity mismatch: '+r['path'])
    for command in row['commands']:
        if command['entrypoint'] not in ENTRYPOINTS[stage]:raise ValueError('Unapproved stage entrypoint')
        if not isinstance(command.get('argv'),list):raise ValueError('Explicit argv list required')
        launcher=command.get('launcher',[])
        if launcher and (launcher[:2]!=['-m','torch.distributed.run'] or stage!='calibration'):
            raise ValueError('Only explicit calibration torchrun launcher supported')
        if any(not isinstance(x,str) for x in command['argv']+launcher):raise ValueError('Arguments must be strings')
        if any(key in command.get('env',{}) for key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD')):
            raise ValueError('Runtime override outside validated source layout')
        argv=command['argv']
        if stage in ('fitting','allocation'):
            config_flags=[x for x in argv if x=='--stage-config' or x.startswith('--stage-config=')]
            if config_flags!=['--stage-config'] or argv.index('--stage-config')+1>=len(argv):
                raise ValueError('Exactly one explicit --stage-config is required')
            config_file=argv[argv.index('--stage-config')+1]
        else:
            if len(argv)!=(2 if stage=='evaluation' else 1):
                raise ValueError('Expected explicit job JSON and, for evaluation, task index')
            config_file=argv[0]
        config_path=Path(config_file)
        if not config_path.is_absolute():
            raise ValueError('Job/stage configuration must use an absolute path')
        if not any(r['path']==str(config_path.resolve()) for r in inputs):
            raise ValueError('Job/stage configuration must be a hashed stage input')
        job=json.loads(config_path.read_text())
        validate_job(stage,job,workspace)
        if stage=='allocation' and 'bank_manifest' in job:
            ref=job['bank_manifest']
            if not Path(ref['path']).is_absolute() or sha(ref['path'])!=ref['sha256']:
                raise ValueError('Bank manifest identity mismatch')
            if not any(r['path']==str(Path(ref['path']).resolve()) and r['sha256']==ref['sha256'] for r in inputs):
                raise ValueError('Bank manifest must be a hashed stage input')
        if stage=='evaluation':
            root=Path(job['root'])
            protocol_path=root/'protocol.json'
            managed=cfg.get('manage_evaluation_receipts',False)
            if type(managed) is not bool:raise ValueError('manage_evaluation_receipts must be boolean')
            if managed:
                if sha(protocol_path)!=protocol['sha256']:
                    raise ValueError('Managed evaluation root must contain the checked stage protocol')
            else:
                authorization=json.loads((root/'EXECUTION_RECEIPT.json').read_text())
                if authorization.get('execution_status')!='RUNNING':
                    raise ValueError('Evaluation requires its original authorized execution receipt')
                if authorization.get('protocol_sha256')!=sha(protocol_path):
                    raise ValueError('Evaluation authorization protocol mismatch')
            evaluation_protocol=json.loads(protocol_path.read_text())
            plan_ref={'path':str(config_path.resolve()),'sha256':sha(config_path)}
            if plan_ref not in evaluation_protocol.get('inputs',[]):
                raise ValueError('Evaluation plan is absent from authorized protocol inputs')
            if argv[1]!=str(job['worker_task_index']):
                raise ValueError('Evaluation task argument differs from plan')
    return specification

def run(config_path,stage):
    import fcntl
    cfg=load_config(config_path);workspace=Path(cfg['workspace']).resolve()
    records=workspace/'pipeline_receipts';records.mkdir(exist_ok=True)
    with (records/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec=check(cfg,stage)
        signature=fingerprint(cfg,stage)
        for previous in STAGES[:STAGES.index(stage)]:
            p=records/(previous+'.json')
            receipt=json.loads(p.read_text())
            check(cfg,previous)
            if receipt.get('status')!='COMPLETE' or receipt.get('stage_sha256')!=fingerprint(cfg,previous):
                raise ValueError('Prerequisite not complete under this configuration: '+previous)
            for r in receipt['outputs']:
                if sha(r['path'])!=r['sha256']:raise ValueError('Prerequisite output drift')
        row=cfg['stages'][stage]
        env=dict(os.environ)
        env['PYTHONPATH']=os.pathsep.join(str(workspace/p) for p in (
          B.rstrip('/'),B+'Isaac-GR00T-N1.7',F.rstrip('/'),
          'simplerenv_n17_feasibility_2026_08_27','libero_transfer_quantstats_20260922'))
        atomic(records/(stage+'.json'),{'status':'RUNNING','stage_sha256':signature})
        i=None
        result=None
        try:
            with ExitStack() as executions:
                if stage=='evaluation' and cfg.get('manage_evaluation_receipts',False):
                    roots=sorted({Path(json.loads(Path(c['argv'][0]).read_text())['root']).resolve() for c in row['commands']})
                    for root in roots:
                        executions.enter_context(evaluation_execution(root,stage_protocol(cfg,stage)['sha256']))
                for i,c in enumerate(row['commands']):
                    result=None
                    command=[cfg['python'],*c.get('launcher',[]),str(inside(workspace,c['entrypoint'])),*c['argv']]
                    with (records/f'{stage}.{i}.log').open('a') as log:
                        result=subprocess.run(command,cwd=workspace,env=dict(env,**c.get('env',{})),stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError('Stage failed; inspect log. No downstream stage started.')
                outputs=[{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in row['outputs']]
                atomic(records/(stage+'.json'),{'status':'COMPLETE','stage_sha256':signature,'outputs':outputs,
                       'sample_audit_status':spec['sample_audit_status'],'meaning':'commands exited successfully and declared outputs exist; not a success-rate audit'})
        except (Exception, KeyboardInterrupt) as exc:
            failure={'status':'FAILED', 'stage_sha256':signature,
                     'command_index':i, 'error_type':type(exc).__name__, 'error':str(exc)}
            if result is not None:
                failure['returncode']=result.returncode
            atomic(records/(stage+'.json'),failure)
            raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--workspace',required=True);p.add_argument('--gr00t-checkout');p.add_argument('--libero-checkout')
    p=sub.add_parser('catalog')
    for name in ('check','run'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--stage',choices=STAGES,required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(args.workspace,args.gr00t_checkout,args.libero_checkout),indent=2))
    elif args.command=='catalog':print(json.dumps(ENTRYPOINTS,indent=2))
    elif args.command=='check':check(load_config(args.config),args.stage);print('INPUT_PREFLIGHT_PASS_NOT_MODEL_EXECUTION')
    else:run(args.config,args.stage)

if __name__=='__main__':main()
