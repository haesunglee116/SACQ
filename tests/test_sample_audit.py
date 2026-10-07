"""Episode separation must survive policy-call changes and receipt tampering."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'gr00t_libero/sample_audit.py'
spec = importlib.util.spec_from_file_location('sample_audit', path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

class SampleAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.row = dict(benchmark_family='fixture', task='a', base_seed=7, episode_id=0)
        self.spec = dict(schema='sacq_sample_audit_spec_v1',
            task_registry=self.write('registry', dict(schema='sacq_task_registry_v1',
                tasks=[dict(benchmark_family='fixture', task='a')])), stages={})
        for stage in a.STAGES:
            self.replace(stage, [dict(self.row, base_seed=17 if stage=='rollout' else 7)])

    def write(self, name, data):
        path = self.root / (name+'.json')
        path.write_text(json.dumps(data))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def replace(self, stage, rows):
        self.spec['stages'][stage] = self.write(stage, dict(schema='sacq_episode_identities_v1', samples=rows))

    def test_pass_recomputed_and_train_validation_overlap_reported(self):
        result = a.audit(self.spec)
        self.assertEqual(result['status'], 'PASS_DISJOINT')
        self.assertEqual(result['calibration_train_validation_overlap_episodes'], 1)
        a.validate_receipt(result)

    def test_policy_call_difference_does_not_hide_overlap(self):
        self.replace('rollout', [dict(self.row, policy_call=99)])
        result = a.audit(self.spec)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(len(result['development_rollout_overlaps']['fitting']), 1)

    def test_validation_only_overlap_fails(self):
        self.replace('calibration_validation', [dict(self.row, base_seed=17)])
        self.assertEqual(a.audit(self.spec)['status'], 'FAIL')

    def test_missing_and_unexpected_tasks_fail(self):
        self.replace('allocation', [dict(self.row, task='b')])
        result = a.audit(self.spec)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['coverage']['allocation']['missing_tasks'], [['fixture','a']])
        self.assertEqual(result['coverage']['allocation']['unexpected_tasks'], [['fixture','b']])

    def test_rollout_duplicates_rejected(self):
        self.replace('rollout', [self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'repeats'): a.audit(self.spec)

    def test_missing_provenance_rejected(self):
        row = dict(self.row); del row['episode_id']
        self.replace('fitting', [row])
        with self.assertRaises(ValueError): a.audit(self.spec)

    def test_changed_manifest_and_forged_receipt_rejected(self):
        result = a.audit(self.spec)
        forged = copy.deepcopy(result)
        forged['coverage']['fitting']['listed_samples'] = 900
        with self.assertRaises(ValueError): a.validate_receipt(forged)
        Path(self.spec['stages']['fitting']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): a.validate_receipt(result)

    def test_prepared_capture_repeated_calls_collapse_to_episode(self):
        self.spec['stages']['fitting'] = self.write('captures', dict(
            schema='sacq_prepared_capture_inputs_v1', records=[dict(observations=[
                dict(self.row, policy_call=0), dict(self.row, policy_call=1)])]))
        result = a.audit(self.spec)
        self.assertEqual(result['coverage']['fitting']['listed_samples'], 2)
        self.assertEqual(result['coverage']['fitting']['unique_episodes'], 1)

    def test_cli_failed_audit_is_nonzero_and_not_overwritten(self):
        self.replace('rollout', [self.row])
        path = self.root/'spec.json'; path.write_text(json.dumps(self.spec))
        output = self.root/'receipt.json'
        argv = ['--spec', str(path), '--output', str(output)]
        self.assertEqual(a.main(argv), 1)
        self.assertEqual(json.loads(output.read_text())['status'], 'FAIL')
        with self.assertRaises(SystemExit): a.main(argv)

if __name__ == '__main__': unittest.main()
