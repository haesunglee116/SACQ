"""Synthetic aggregation checks; no experiment results are read."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from gr00t_libero.results import aggregate


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.manifest=dict(schema='sacq_episode_results_v1',label='synthetic',
                           aggregation='episode_micro',samples=[])
        for index,(task,episode,success) in enumerate([('a',0,True),('b',0,False),('b',1,False)]):
            expected=dict(method='fixture',suite='suite',task=task,base_seed=7,policy_seed=123,
                          episode_id=episode,n_envs=1,protocol_sha256='synthetic-hash')
            path=self.root/f'{index}.json'
            path.write_text(json.dumps(dict(expected,status='COMPLETE',success=success)))
            self.manifest['samples'].append(dict(path=str(path),expected=expected))

    def test_micro_and_macro_are_explicit(self):
        self.assertAlmostEqual(aggregate(self.manifest)['success_rate_pct'],100/3)
        self.manifest['aggregation']='task_macro'
        self.assertEqual(aggregate(self.manifest)['success_rate_pct'],50)

    def test_missing_episode_blocks_final_result(self):
        (self.root/'2.json').unlink()
        with self.assertRaisesRegex(ValueError,'expected episodes are missing'):
            aggregate(self.manifest)
        partial=aggregate(self.manifest,allow_partial=True)
        self.assertEqual(partial['status'],'PARTIAL')
        self.assertIsNone(partial['success_rate_pct'])
        self.assertEqual(partial['completed_success_rate_pct'],50)
        self.assertEqual(partial['expected'],3)
        self.assertEqual(partial['completed'],2)

    def test_duplicate_sample_rejected(self):
        self.manifest['samples'].append(copy.deepcopy(self.manifest['samples'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            aggregate(self.manifest)

    def test_different_seed_rejected(self):
        path=self.root/'0.json';record=json.loads(path.read_text());record['base_seed']=17
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'base_seed'):
            aggregate(self.manifest)

    def test_different_checkpoint_plan_rejected(self):
        path=self.root/'0.json';record=json.loads(path.read_text());record['protocol_sha256']='other'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'protocol_sha256'):
            aggregate(self.manifest)

    def test_extra_unlisted_file_not_counted(self):
        (self.root/'extra.json').write_text('{}')
        self.assertEqual(aggregate(self.manifest)['completed'],3)

    def test_no_implicit_aggregation_rule(self):
        del self.manifest['aggregation']
        with self.assertRaisesRegex(ValueError,'aggregation rule'):
            aggregate(self.manifest)

    def test_boolean_success_required(self):
        path=self.root/'0.json';record=json.loads(path.read_text());record['success']='false'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'boolean'):
            aggregate(self.manifest)


if __name__=='__main__':
    unittest.main()

class PlanExpansionTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        from gr00t_libero import results
        self.module=results
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.profile=json.loads((Path(results.__file__).parent/'profiles/legacy_rollout.json').read_text())['rollout']
        self.tasks=self.write('tasks.json',[dict(suite='suite',task='task')])
        self.plan=dict(method='final_seed17',rollout=self.profile,root=str(self.root),
                       condition='sacq',language_action_nominal_bpw=2.0,worker_task_index=0,selected_suites=['suite'])
        self.ref=self.write('plan.json',self.plan)
        self.spec=dict(schema='sacq_result_manifest_build_v1',label='synthetic',aggregation='episode_micro',
                       plans=[dict(plan=self.ref,tasks=self.tasks)])

    def write(self,name,value):
        import hashlib
        path=self.root/name;path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_pinned_episode_expansion_and_consumption(self):
        manifest=self.module.from_plans(self.spec)
        self.assertEqual(len(manifest['samples']),50)
        self.assertEqual([r['expected']['episode_id'] for r in manifest['samples']],list(range(50)))
        for sample in manifest['samples']:
            path=Path(sample['path']);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(sample['expected'],status='COMPLETE',success=True)))
        result=aggregate(manifest)
        self.assertEqual((result['completed'],result['successes']),(50,50))
        self.assertEqual(result['success_rate_pct'],100)

    def test_changed_plan_or_duplicate_task_rejected(self):
        self.spec['plans'].append(copy.deepcopy(self.spec['plans'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.module.from_plans(self.spec)
        self.spec['plans'].pop()
        (self.root/'plan.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.module.from_plans(self.spec)

    def test_mixed_conditions_rejected(self):
        other=dict(self.plan,condition='bf16')
        self.spec['plans'].append(dict(plan=self.write('other.json',other),tasks=self.tasks))
        with self.assertRaisesRegex(ValueError,'different quantization'):self.module.from_plans(self.spec)

    def test_unsupported_protocol_rejected(self):
        self.plan['rollout']=dict(self.profile,environment_seed=0)
        self.spec['plans'][0]['plan']=self.write('plan.json',self.plan)
        with self.assertRaisesRegex(ValueError,'pinned rollout'):self.module.from_plans(self.spec)
