"""Synthetic orchestration tests only; no scientific jobs or audit claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

MODULE=Path(__file__).resolve().parents[1]/'gr00t_libero'/'pipeline.py'
spec=importlib.util.spec_from_file_location('release_pipeline',MODULE)
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class PipelineTests(unittest.TestCase):
    def prepare_managed_evaluation(self):
        extension=json.loads((self.root/'protocol.json').read_text())
        extension['inputs']=[self.ref('evaluation.json')]
        self.write('evaluation_root/protocol.json',extension)
        self.cfg['manage_evaluation_receipts']=True
        self.cfg['stages']['evaluation']['protocol_manifest']=self.ref('evaluation_root/protocol.json')

    def test_managed_rollout_records_actual_execution_interval(self):
        for stage in p.STAGES[:-1]:self.execute(stage)
        self.prepare_managed_evaluation()
        self.write('execution.json',self.cfg)
        receipt=self.root/'evaluation_root/EXECUTION_RECEIPT.json'
        self.assertFalse(receipt.exists())
        def worker(*args,**kwargs):
            self.assertEqual(json.loads(receipt.read_text())['execution_status'],'RUNNING')
            (self.root/'evaluation.out').write_text('fixture')
            return SimpleNamespace(returncode=0)
        with patch.object(p.subprocess,'run',worker):p.run(self.root/'execution.json','evaluation')
        self.assertEqual(json.loads(receipt.read_text())['execution_status'],'FINISHED')

    def test_managed_rollout_failure_does_not_leave_running(self):
        for stage in p.STAGES[:-1]:self.execute(stage)
        self.prepare_managed_evaluation();self.write('execution.json',self.cfg)
        with patch.object(p.subprocess,'run',return_value=SimpleNamespace(returncode=9)):
            with self.assertRaises(RuntimeError):p.run(self.root/'execution.json','evaluation')
        receipt=json.loads((self.root/'evaluation_root/EXECUTION_RECEIPT.json').read_text())
        self.assertEqual(receipt['execution_status'],'FAILED')

    def test_execution_lease_rejects_duplicate_and_unresolved_running(self):
        root=self.root/'evaluation_root';root.mkdir()
        with p.evaluation_execution(root,'fixture'):
            with self.assertRaises(BlockingIOError):
                with p.evaluation_execution(root,'fixture'):pass
        self.write('evaluation_root/EXECUTION_RECEIPT.json',{'execution_status':'RUNNING'})
        with self.assertRaisesRegex(ValueError,'Existing RUNNING'):
            with p.evaluation_execution(root,'fixture'):pass

    def test_managed_rollout_requires_checked_protocol(self):
        self.prepare_managed_evaluation()
        changed=json.loads((self.root/'evaluation_root/protocol.json').read_text())
        changed['held_out']=True
        self.write('evaluation_root/protocol.json',changed)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):p.check(self.cfg,'evaluation')

    def test_all_stages_accept_late_evaluation_plan_without_rewriting_prior_receipts(self):
        original=json.loads((self.root/'protocol.json').read_text())
        for stage in p.STAGES[:-1]:self.execute(stage)
        saved=(self.root/'pipeline_receipts/fitting.json').read_bytes()
        self.authorize_evaluation()
        extension=json.loads((self.root/'protocol.json').read_text())
        self.write('evaluation-protocol.json',extension)
        self.write('protocol.json',original)
        self.cfg['protocol_manifest']=self.ref('protocol.json')
        self.cfg['stages']['evaluation']['protocol_manifest']=self.ref('evaluation-protocol.json')
        self.execute('evaluation')
        self.assertEqual((self.root/'pipeline_receipts/fitting.json').read_bytes(),saved)
        result=json.loads((self.root/'pipeline_receipts/evaluation.json').read_text())
        self.assertEqual(result['status'],'COMPLETE')

    def test_late_stage_protocol_inputs_preserve_previous_fingerprint(self):
        before=p.fingerprint(self.cfg,'fitting')
        self.write('generated-plan.json',{'generated':True})
        extension=json.loads((self.root/'protocol.json').read_text())
        extension['inputs']=[self.ref('generated-plan.json')]
        self.write('evaluation-protocol.json',extension)
        self.cfg['stages']['evaluation']['protocol_manifest']=self.ref('evaluation-protocol.json')
        self.assertEqual(p.stage_protocol(self.cfg,'evaluation'),self.ref('evaluation-protocol.json'))
        self.assertEqual(p.fingerprint(self.cfg,'fitting'),before)
        self.write('generated-plan.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'Additional protocol input identity'):
            p.stage_protocol(self.cfg,'evaluation')

    def test_stage_protocol_cannot_change_settings_or_drop_inputs(self):
        base=json.loads((self.root/'protocol.json').read_text())
        base['inputs']=[self.ref('audit.json')]
        self.write('protocol.json',base);self.cfg['protocol_manifest']=self.ref('protocol.json')
        for update in ({'held_out':True},{'inputs':[]}):
            extension=dict(base,**update);self.write('extension.json',extension)
            self.cfg['stages']['evaluation']['protocol_manifest']=self.ref('extension.json')
            with self.assertRaises(ValueError):p.stage_protocol(self.cfg,'evaluation')

    def test_public_audit_recomputed_before_execution(self):
        spec=importlib.util.spec_from_file_location('public_audit',MODULE.with_name('sample_audit.py'))
        audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
        self.write('registry.json',dict(schema='sacq_task_registry_v1',
            tasks=[dict(benchmark_family='fixture',task='task')]))
        stages={}
        for name in audit.STAGES:
            self.write(name+'_samples.json',dict(schema='sacq_episode_identities_v1',samples=[
                dict(benchmark_family='fixture',task='task',base_seed=17 if name=='rollout' else 7,episode_id=0)]))
            stages[name]=self.ref(name+'_samples.json')
        receipt=audit.audit(dict(schema='sacq_sample_audit_spec_v1',
            task_registry=self.ref('registry.json'),stages=stages))
        self.write('audit.json',receipt)
        protocol=json.loads((self.root/'protocol.json').read_text())
        protocol.update(sample_audit_status='PASS_DISJOINT',sample_audit_receipt=self.ref('audit.json'))
        self.write('protocol.json',protocol); self.cfg['protocol_manifest']=self.ref('protocol.json')
        p.check(self.cfg,'fitting')
        self.write('fitting_samples.json',{})
        with self.assertRaisesRegex(ValueError,'Manifest identity mismatch'):
            p.check(self.cfg,'fitting')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        p.atomic(self.root/'SOURCE_RECEIPT.json',{'sources':[]})
        self.write('audit.json',{'synthetic_test_only':True})
        protocol={'benchmark':'gr00t_libero','scientific_settings_complete':True,
                  'sample_audit_status':'APPROVED_DIAGNOSTIC_ONLY','diagnostic_only':True,
                  'held_out':False,'sample_audit_receipt':self.ref('audit.json')}
        self.write('protocol.json',protocol)
        self.cfg={'schema':'sacq_gr00t_libero_execution_v1','workspace':str(self.root),
                  'python':sys.executable,'protocol_manifest':self.ref('protocol.json'),'stages':{}}
        for stage in p.STAGES:
            self.write(stage+'.json',self.job(stage))
            argv=[str(self.root/(stage+'.json'))]
            if stage in ('fitting','allocation'):argv=['--stage-config',*argv]
            if stage=='evaluation':argv+=['0']
            self.cfg['stages'][stage]={'commands':[{'entrypoint':p.ENTRYPOINTS[stage][0],'argv':argv}],
                                      'inputs':[self.ref(stage+'.json')],
                                      'outputs':[str(self.root/(stage+'.out'))]}

    def job(self, stage):
        common = dict(out='output', processor='processor', trainer='trainer',
                      model='model', allocation='allocation',
                      artifacts=[dict(layer='layer', path='artifact')],
                      gptq=[dict(layer='background', path='background')])
        if stage == 'fitting':
            return dict(workspace=str(self.root), operation='vlm', processor='processor', checkpoint='model')
        if stage == 'allocation':
            return dict(workspace=str(self.root), operation='allocation', processor='processor',
                        models={'suite': 'model'}, bank='bank', allocation_requests='requests', allocation_work='work')
        if stage == 'index':
            return dict(common, index_settings={'synthetic':True}, targets=[dict(layer='layer', path='target')],
                        records=[dict(path='request', position=0)])
        if stage == 'calibration':
            return dict(common, all_records=[dict(path='request')], sacq_count=1, gptq_count=1,
                        local_qa='qa', cached_launcher='launcher', batch_cache='cache',
                        local_output='local_output', suite='suite', tag='tag', training={'synthetic': True})
        return dict(root=str(self.root/'evaluation_root'), worker_task_index=0,
                    rollout={'synthetic': True}, sources=['synthetic'], repo='repo',
                    worker_source='worker', socket_root='sockets', condition='condition',
                    diagnostic_only=True, held_out=False)

    def replace_job(self, stage, job):
        self.write(stage+'.json', job)
        self.cfg['stages'][stage]['inputs']=[self.ref(stage+'.json')]

    def authorize_evaluation(self, register=True):
        self.write('evaluation_root/protocol.json',
                   {'inputs': [self.ref('evaluation.json')] if register else []})
        self.write('evaluation_root/EXECUTION_RECEIPT.json',
                   {'execution_status': 'RUNNING',
                    'protocol_sha256': self.ref('evaluation_root/protocol.json')['sha256']})

    def test_valid_worker_schemas(self):
        for stage in ('fitting', 'allocation', 'index', 'calibration'):
            with self.subTest(stage=stage): p.check(self.cfg, stage)
        self.authorize_evaluation()
        p.check(self.cfg, 'evaluation')

    def test_bank_manifest_is_registered_and_pinned(self):
        self.write('bank.json', {'schema': 'sacq_packed_bank_v1', 'artifacts': []})
        job=self.job('allocation')
        job.pop('bank')
        job['bank_manifest']=self.ref('bank.json')
        self.replace_job('allocation',job)
        with self.assertRaisesRegex(ValueError,'hashed stage input'):
            p.check(self.cfg,'allocation')
        self.cfg['stages']['allocation']['inputs'].append(self.ref('bank.json'))
        p.check(self.cfg,'allocation')
        job['bank']='legacy'
        self.replace_job('allocation',job)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            p.check(self.cfg,'allocation')

    def test_empty_jobs_rejected(self):
        for stage in ('index', 'calibration'):
            self.replace_job(stage, {})
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'requires nonempty'):
                p.check(self.cfg, stage)

    def test_duplicate_stage_config_rejected(self):
        self.cfg['stages']['fitting']['commands'][0]['argv'].append('--stage-config=/other.json')
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            p.check(self.cfg, 'fitting')

    def test_workspace_mismatch_rejected(self):
        job=self.job('fitting'); job['workspace']=str(self.root/'other')
        self.replace_job('fitting', job)
        with self.assertRaisesRegex(ValueError, 'differs from verified'):
            p.check(self.cfg, 'fitting')

    def test_index_position_required(self):
        job=self.job('index'); del job['records'][0]['position']
        self.replace_job('index', job)
        with self.assertRaisesRegex(ValueError, 'integer position'):
            p.check(self.cfg, 'index')

    def test_calibration_counts_match_records(self):
        job=self.job('calibration'); job['sacq_count']=2
        self.replace_job('calibration', job)
        with self.assertRaisesRegex(ValueError, 'differs from supplied'):
            p.check(self.cfg, 'calibration')

    def test_evaluation_requires_registered_plan(self):
        self.authorize_evaluation(register=False)
        with self.assertRaisesRegex(ValueError, 'absent from authorized'):
            p.check(self.cfg, 'evaluation')

    def test_evaluation_task_argument_matches_plan(self):
        self.authorize_evaluation()
        self.cfg['stages']['evaluation']['commands'][0]['argv'][1]='1'
        with self.assertRaisesRegex(ValueError, 'task argument'):
            p.check(self.cfg, 'evaluation')

    def write(self,name,value):p.atomic(self.root/name,value)
    def ref(self,name):return {'path':str(self.root/name),'sha256':p.sha(self.root/name)}
    def execute(self,stage):
        self.write('execution.json',self.cfg)
        def mock_run(*args,**kwargs):
            (self.root/(stage+'.out')).write_text('synthetic output')
            return SimpleNamespace(returncode=0)
        with patch.object(p.subprocess,'run',mock_run):p.run(self.root/'execution.json',stage)

    def test_missing_output_marks_failed(self):
        self.write('execution.json', self.cfg)
        with patch.object(p.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            with self.assertRaises(FileNotFoundError):
                p.run(self.root/'execution.json', 'fitting')
        receipt=json.loads((self.root/'pipeline_receipts/fitting.json').read_text())
        self.assertEqual(receipt['status'], 'FAILED')
        self.assertEqual(receipt['error_type'], 'FileNotFoundError')

    def test_spawn_failure_marks_failed(self):
        self.write('execution.json', self.cfg)
        with patch.object(p.subprocess, 'run', side_effect=OSError('synthetic spawn failure')):
            with self.assertRaises(OSError):
                p.run(self.root/'execution.json', 'fitting')
        receipt=json.loads((self.root/'pipeline_receipts/fitting.json').read_text())
        self.assertEqual(receipt['status'], 'FAILED')
        self.assertNotIn('returncode', receipt)

    def test_nonzero_exit_marks_failed(self):
        self.write('execution.json', self.cfg)
        with patch.object(p.subprocess, 'run', return_value=SimpleNamespace(returncode=2)):
            with self.assertRaises(RuntimeError):
                p.run(self.root/'execution.json', 'fitting')
        receipt=json.loads((self.root/'pipeline_receipts/fitting.json').read_text())
        self.assertEqual(receipt['status'], 'FAILED')
        self.assertEqual(receipt['returncode'], 2)

    def test_missing_job_rejected(self):
        self.cfg['stages']['index']['commands'][0]['argv']=[str(self.root/'missing.json')]
        with self.assertRaises(ValueError):p.check(self.cfg,'index')

    def test_wrong_operation_rejected(self):
        self.write('allocation.json',{'operation':'vlm'})
        self.cfg['stages']['allocation']['inputs']=[self.ref('allocation.json')]
        with self.assertRaises(ValueError):p.check(self.cfg,'allocation')

    def test_changed_stage_config_rejected(self):
        self.write('fitting.json',{'operation':'dit'})
        with self.assertRaises(ValueError):p.check(self.cfg,'fitting')

    def test_unhashed_stage_config_rejected(self):
        self.cfg['stages']['fitting']['inputs']=[self.ref('audit.json')]
        with self.assertRaises(ValueError):p.check(self.cfg,'fitting')

    def test_forward_output_reference_and_future_changes(self):
        self.execute('fitting')
        self.cfg['stages']['allocation']['inputs'].append(
            {'from_stage':'fitting','path':str(self.root/'fitting.out')})
        self.execute('allocation')
        receipt=json.loads((self.root/'pipeline_receipts'/'allocation.json').read_text())
        self.assertEqual(receipt['status'],'COMPLETE')

    def test_upstream_configuration_drift_rejected(self):
        self.execute('fitting')
        self.cfg['stages']['fitting']['commands'][0]['argv']+=['--changed']
        with self.assertRaises(ValueError):self.execute('allocation')

    def test_upstream_output_drift_rejected(self):
        self.execute('fitting')
        (self.root/'fitting.out').write_text('changed')
        with self.assertRaises(ValueError):self.execute('allocation')

if __name__=='__main__':unittest.main()
