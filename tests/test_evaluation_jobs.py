"""Evaluation plan preparation tests; no model, simulator, or authorization made."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from gr00t_libero import jobs


class EvaluationJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rollout = json.loads((Path(jobs.__file__).parent/'profiles/legacy_rollout.json').read_text())['rollout']
        worker = self.file('worker.py', '# worker')
        sim = self.file('sim.py', '# simulator')
        tasks = self.file('tasks.json', json.dumps([
            dict(suite='suite', task='task', task_id=0, bddl='/bddl', language='instruction')]))
        plan = dict(root=str(self.root), worker_task_index=0, rollout=self.rollout,
            sources=[worker, sim], repo='/repo', worker_source=worker['path'],
            sim_source=sim['path'], sim_python='/python', socket_root='/sockets',
            cosmos='/processor', models={'suite': '/model'}, noise_root='/noise',
            reset_audit_root='/reset', reset_task_contract_sha256={'suite/task': 'fixture'},
            paired_initial_observation_sha256={}, suites=['suite'], selected_suites=['suite'],
            method='final_seed17', reset_identity_status='PASS', rollout_authorized=False,
            render_reference_mode='same_process_saved_rng_exact', condition='bf16',
            diagnostic_only=True, held_out=False, language_action_nominal_bpw=16)
        self.spec = dict(schema='sacq_worker_build_v1', workspace=str(self.root), tasks=tasks, plan=plan)

    def file(self, name, text):
        path = self.root/name
        path.write_text(text)
        return dict(path=str(path), sha256=jobs.pipeline.sha(path))

    def calibrated(self):
        spec = copy.deepcopy(self.spec)
        spec['plan'].update(condition='sacq1.5', language_action_nominal_bpw=1.5)
        job = dict(suite='suite', model='/model', processor='/processor',
            training={'expected_updates': 1}, sacq_count=1, gptq_count=1, artifacts=[{'layer':'b'}],
            gptq=[dict(self.file('bg.pt', 'background'), layer='a', bits=4)])
        ref = self.file('calibration.json', json.dumps(job))
        outputs = dict(vlm=self.file('vlm.pt', 'vlm overlay'), dit=self.file('dit.pt', 'dit overlay'))
        completion = self.file('TRAINING_COMPLETE.json', json.dumps(dict(
            status='COMPLETE_DIAGNOSTIC', job_sha256=ref['sha256'], updates=1, outputs=outputs)))
        spec['calibration'] = dict(job=ref, completion=completion,
            vlm_spec=self.file('vlm.txt', 'a=16'), dit_spec=self.file('dit.txt', 'b=16'))
        return spec

    def test_bf16_preparation_preserves_authorization_and_samples(self):
        result = jobs.Builder().build('evaluation', self.spec)
        self.assertEqual(result, self.spec['plan'])
        self.assertFalse(result['rollout_authorized'])
        self.assertFalse((self.root/'EXECUTION_RECEIPT.json').exists())

    def test_calibrated_overlays_linked_from_completion(self):
        result = jobs.Builder().build('evaluation', self.calibrated())
        self.assertEqual(result['workshop_model']['vlm']['path'], str(self.root/'vlm.pt'))
        self.assertEqual(result['workshop_model']['dit']['path'], str(self.root/'dit.pt'))
        self.assertEqual(result['workshop_model']['gptq'][0]['bits'], 4)
        self.assertFalse(result['rollout_authorized'])

    def test_sacq_condition_supports_explicit_bit_label(self):
        spec = self.calibrated()
        spec['plan'].update(condition='sacq', language_action_nominal_bpw=2.0)
        result = jobs.Builder().build('evaluation', spec)
        self.assertEqual(result['language_action_nominal_bpw'], 2.0)
        self.assertEqual(result['workshop_model']['expected_sacq_count'], 1)
        self.assertEqual(result['workshop_model']['expected_gptq_count'], 1)

    def test_different_seed_not_silently_accepted(self):
        self.spec['plan']['rollout']['environment_seed'] = 7
        with self.assertRaisesRegex(ValueError, 'supported pinned'):
            jobs.Builder().build('evaluation', self.spec)

    def test_out_of_range_task_rejected(self):
        self.spec['plan']['worker_task_index'] = 1
        with self.assertRaisesRegex(ValueError, 'outside explicit'):
            jobs.Builder().build('evaluation', self.spec)

    def test_unhashed_worker_rejected(self):
        self.spec['plan']['sources'] = self.spec['plan']['sources'][1:]
        with self.assertRaisesRegex(ValueError, 'hashed sources'):
            jobs.Builder().build('evaluation', self.spec)

    def test_calibration_model_mismatch_rejected(self):
        spec = self.calibrated()
        spec['plan']['models']['suite'] = '/other'
        with self.assertRaisesRegex(ValueError, 'model or suite mismatch'):
            jobs.Builder().build('evaluation', spec)

    def test_wrong_completion_rejected(self):
        spec = self.calibrated()
        ref = spec['calibration']['completion']
        data = json.loads(Path(ref['path']).read_text())
        data['job_sha256'] = 'wrong'
        spec['calibration']['completion'] = self.file('wrong.json', json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'completion identity'):
            jobs.Builder().build('evaluation', spec)

    def test_changed_overlay_rejected(self):
        spec = self.calibrated()
        (self.root/'vlm.pt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            jobs.Builder().build('evaluation', spec)

    def test_stale_scalar_spec_cannot_override_calibrated_codebook(self):
        spec = self.calibrated()
        spec['calibration']['dit_spec'] = self.file('stale.txt', 'b=3\n')
        with self.assertRaisesRegex(ValueError, 'conflicts with calibrated codebook'):
            jobs.Builder().build('evaluation', spec)

    def test_vlm_codebook_force16_would_disable_overlay(self):
        spec = self.calibrated()
        spec['calibration']['vlm_spec'] = self.file('wrong_vlm.txt', 'b=16')
        with self.assertRaisesRegex(ValueError, 'conflicts with calibrated codebook'):
            jobs.Builder().build('evaluation', spec)

    def test_malformed_or_duplicate_installation_entries_rejected(self):
        for text in ('b=16;b=16', 'b=oops', 'not an assignment'):
            with self.subTest(text=text):
                spec = self.calibrated()
                spec['calibration']['dit_spec'] = self.file('invalid.txt', text)
                with self.assertRaises(ValueError):
                    jobs.Builder().build('evaluation', spec)

    def test_empty_vlm_specs_and_explicit_codebook_sentinel_are_supported(self):
        spec = self.calibrated()
        spec['calibration']['vlm_spec'] = self.file('empty.txt', '\n')
        spec['calibration']['dit_spec'] = self.file('current.txt', '# generated\nb=16\n')
        result = jobs.Builder().build('evaluation', spec)
        self.assertEqual(result['workshop_model']['expected_sacq_count'], 1)

    def test_diagnostic_must_not_be_promoted_to_held_out(self):
        self.spec['plan'].update(diagnostic_only=False, held_out=True)
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            jobs.Builder().build('evaluation', self.spec)


if __name__ == '__main__':
    unittest.main()
