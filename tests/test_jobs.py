"""Synthetic file-linkage tests; no model or scientific data is loaded."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from gr00t_libero import jobs


class JobBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('model', 'processor'):
            (self.root/name).mkdir()
        self.spec = dict(schema='sacq_worker_build_v1', workspace=str(self.root),
            job=dict(index_settings=json.loads((Path(jobs.__file__).parent/'profiles/legacy_index.json').read_text())['index_settings'], out=str(self.root/'index'), model=str(self.root/'model'),
                     processor=str(self.root/'processor')),
            trainer=self.file('trainer.py', '# fixture'),
            allocation=self.file('allocation.csv', 'layer,scope,k\na,vlm,16\nb,dit,8\nc,dit,80000\n'),
            artifacts=[dict(self.file('a.pt', 'original a'), layer='a', scope='vlm', k=16),
                       dict(self.file('b.pt', 'original b'), layer='b', scope='dit', k=8)],
            gptq=[dict(self.file('c.pt', 'scalar'), layer='c', bits=3)],
            target_layers=['a', 'b'],
            records=[dict(self.file('request.msgpack', 'request'), position=0)])

    def file(self, name, content):
        p = self.root/name
        p.write_text(content)
        return dict(path=str(p), sha256=jobs.pipeline.sha(p))

    def calibration(self):
        index = jobs.Builder().build('index', self.spec)
        source = self.file('index.json', json.dumps(index))
        outputs = []
        for row in index['targets']:
            outputs.append(dict(self.file('corrected_'+row['layer']+'.pt', 'corrected'),
                layer=row['layer'], job_sha256=source['sha256'], indices_only_verified=True))
        complete = self.file('complete.json', json.dumps(dict(status='COMPLETE_DIAGNOSTIC',
            job_sha256=source['sha256'], outputs=outputs)))
        spec = copy.deepcopy(self.spec)
        spec['job'].update(out=str(self.root/'calibration'), local_qa=str(self.root/'qa'),
            batch_cache=str(self.root/'cache'), local_output=str(self.root/'local_output'),
            suite='explicit_fixture_suite', tag='fixture',
            training=dict(limit=2, epochs=1, global_batch=1, codebook_lr=1e-4, scale_lr=1e-5,
                          val_every=2, steps_per_invocation=0, fp32_codebook_grad=True,
                          resume=False, expected_updates=1))
        spec.update(cached_launcher=self.file('cache.py', '# fixture'),
            all_records=[spec['records'][0], self.file('request2.msgpack', 'second request')], index_shards=[dict(job=source, completion=complete)])
        return spec

    def with_bank(self, spec):
        spec=copy.deepcopy(spec)
        artifacts=spec.pop('artifacts')
        suite=spec['job'].get('suite','explicit_fixture_suite')
        rows=[dict(r,suite='shared' if r['scope']=='vlm' else suite,init='aware') for r in artifacts]
        # Unselected candidates do not need to be opened by this job builder.
        rows.append(dict(layer='unselected',scope='vlm',k=16,suite='shared',init='aware',
                         path=str(self.root/'unselected.pt'),sha256='0'*64))
        spec['bank_manifest']=self.file('bank.json',json.dumps(dict(schema='sacq_packed_bank_v1',artifacts=rows)))
        spec['bank_selection']=dict(suite=suite,init='aware')
        return spec

    def test_calibration_rejects_files_the_loader_would_skip(self):
        spec = self.calibration()
        spec['all_records'][1] = self.file('unsupported.bin', 'capture')
        with self.assertRaisesRegex(ValueError, r'requires .msgpack'):
            jobs.Builder().build('calibration', spec)

    def test_bank_manifest_selects_same_artifacts_as_manual_input(self):
        expected=jobs.Builder().build('index',self.spec)
        spec=self.with_bank(self.spec)
        builder=jobs.Builder()
        actual=builder.build('index',spec)
        self.assertEqual(actual,expected)
        self.assertIn(spec['bank_manifest']['path'],builder.inputs)
        self.assertNotIn(str(self.root/'unselected.pt'),builder.inputs)

    def test_manifest_selection_can_feed_completed_index_calibration(self):
        spec=self.calibration()
        expected=jobs.Builder().build('calibration',spec)
        actual=jobs.Builder().build('calibration',self.with_bank(spec))
        self.assertEqual(actual,expected)

    def test_bank_selection_requires_explicit_unambiguous_input(self):
        spec=self.with_bank(self.spec)
        spec['artifacts']=self.spec['artifacts']
        with self.assertRaisesRegex(ValueError,'exactly one'):jobs.Builder().build('index',spec)
        spec.pop('artifacts');spec.pop('bank_selection')
        with self.assertRaisesRegex(ValueError,'explicit suite'):jobs.Builder().build('index',spec)

    def test_wrong_bank_suite_does_not_fall_back_to_another_suite(self):
        spec=self.with_bank(self.spec)
        spec['bank_selection']['suite']='another_suite'
        with self.assertRaises(FileNotFoundError):jobs.Builder().build('index',spec)
        spec['job']['suite']='conflicting_suite'
        with self.assertRaisesRegex(ValueError,'differs from job'):jobs.Builder().build('index',spec)

    def test_index_selects_exact_candidates_and_preserves_positions(self):
        builder = jobs.Builder()
        job = builder.build('index', self.spec)
        self.assertEqual(job['targets'], self.spec['artifacts'])
        self.assertEqual(job['records'], self.spec['records'])
        self.assertEqual(job['allocation'], self.spec['allocation']['path'])
        self.assertIn(self.spec['trainer']['path'], builder.inputs)

    def test_calibration_uses_corrected_bank_and_derived_counts(self):
        spec = self.calibration()
        job = jobs.Builder().build('calibration', spec)
        self.assertEqual(job['sacq_count'], 2)
        self.assertEqual(job['gptq_count'], 1)
        self.assertTrue(all(Path(r['path']).name.startswith('corrected_') for r in job['artifacts']))
        self.assertEqual([r['k'] for r in job['artifacts']], [16, 8])
        self.assertEqual(job['all_records'], spec['all_records'])

    def test_changed_artifact_rejected(self):
        Path(self.spec['artifacts'][0]['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            jobs.Builder().build('index', self.spec)

    def test_wrong_candidate_rejected(self):
        self.spec['artifacts'][0]['k'] = 8
        with self.assertRaisesRegex(ValueError, 'candidate differs'):
            jobs.Builder().build('index', self.spec)

    def test_missing_allocation_layer_rejected(self):
        self.spec['artifacts'].pop()
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            jobs.Builder().build('index', self.spec)

    def test_duplicate_targets_rejected(self):
        self.spec['target_layers'] = ['a', 'a']
        with self.assertRaisesRegex(ValueError, 'unique selected'):
            jobs.Builder().build('index', self.spec)

    def test_duplicate_request_positions_rejected(self):
        self.spec['records'] *= 2
        with self.assertRaisesRegex(ValueError, 'unique nonnegative'):
            jobs.Builder().build('index', self.spec)

    def test_background_precision_mismatch_rejected(self):
        self.spec['gptq'][0]['bits'] = 4
        with self.assertRaisesRegex(ValueError, 'scalar allocation'):
            jobs.Builder().build('index', self.spec)

    def test_index_source_mismatch_rejected(self):
        spec = self.calibration()
        spec['job']['model'] = str(self.root/'processor')
        with self.assertRaisesRegex(ValueError, 'input mismatch: model'):
            jobs.Builder().build('calibration', spec)

    def test_incomplete_shards_rejected(self):
        spec = self.calibration()
        spec['index_shards'] = []
        with self.assertRaisesRegex(ValueError, 'cover every'):
            jobs.Builder().build('calibration', spec)

    def test_overlapping_shards_rejected(self):
        spec = self.calibration()
        spec['index_shards'] *= 2
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            jobs.Builder().build('calibration', spec)

    def test_wrong_completion_identity_rejected(self):
        spec = self.calibration()
        ref = spec['index_shards'][0]['completion']
        data = json.loads(Path(ref['path']).read_text())
        data['job_sha256'] = 'wrong'
        spec['index_shards'][0]['completion'] = self.file('wrong.json', json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            jobs.Builder().build('calibration', spec)

    def test_changed_corrected_bank_rejected(self):
        spec = self.calibration()
        (self.root/'corrected_a.pt').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            jobs.Builder().build('calibration', spec)

    def test_cli_writes_hashed_inputs_without_overwrite(self):
        spec = self.file('spec.json', json.dumps(self.spec))
        out = self.root/'prepared'
        command = [sys.executable, str(Path(jobs.__file__).resolve()), 'index',
                   '--spec', spec['path'], '--output-dir', str(out)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        inputs = json.loads((out/'inputs.json').read_text())
        for ref in inputs:
            self.assertEqual(jobs.pipeline.sha(ref['path']), ref['sha256'])
        self.assertIn(spec, inputs)
        original = (out/'job.json').read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual((out/'job.json').read_bytes(), original)

    def test_index_only_evidence_required(self):
        spec = self.calibration()
        ref = spec['index_shards'][0]['completion']
        data = json.loads(Path(ref['path']).read_text())
        data['outputs'][0]['indices_only_verified'] = False
        spec['index_shards'][0]['completion'] = self.file('not_index_only.json', json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'index-only'):
            jobs.Builder().build('calibration', spec)

    def test_mixed_request_sets_across_shards_rejected(self):
        spec = self.calibration()
        original = spec['index_shards'][0]
        job = json.loads(Path(original['job']['path']).read_text())
        job['records'][0]['position'] = 17
        job_ref = self.file('different_records_job.json', json.dumps(job))
        complete = json.loads(Path(original['completion']['path']).read_text())
        complete['job_sha256'] = job_ref['sha256']
        for row in complete['outputs']:
            row['job_sha256'] = job_ref['sha256']
        receipt = self.file('different_records_complete.json', json.dumps(complete))
        spec['index_shards'].append(dict(job=job_ref, completion=receipt))
        with self.assertRaisesRegex(ValueError, 'different request identities'):
            jobs.Builder().build('calibration', spec)

    def split_index_shards(self, spec, second_settings=None):
        original = spec['index_shards'][0]
        source = json.loads(Path(original['job']['path']).read_text())
        completion = json.loads(Path(original['completion']['path']).read_text())
        shards = []
        for i, target in enumerate(source['targets']):
            job = copy.deepcopy(source)
            job['targets'] = [target]
            if i == 1 and second_settings is not None:
                job['index_settings'] = second_settings
            job_ref = self.file(f'shard_{i}.json', json.dumps(job))
            output = copy.deepcopy(next(r for r in completion['outputs'] if r['layer'] == target['layer']))
            output['job_sha256'] = job_ref['sha256']
            receipt = self.file(f'shard_{i}_complete.json', json.dumps(dict(
                status='COMPLETE_DIAGNOSTIC', job_sha256=job_ref['sha256'], outputs=[output])))
            shards.append(dict(job=job_ref, completion=receipt))
        spec['index_shards'] = shards
        return spec

    def test_consistent_disjoint_index_shards_join(self):
        spec = self.calibration()
        expected = jobs.Builder().build('calibration', spec)
        actual = jobs.Builder().build('calibration', self.split_index_shards(spec))
        self.assertEqual(actual, expected)

    def test_mixed_index_numerical_settings_rejected(self):
        for key, value in [('damping', 0.01), ('noise_seed_base', 19),
                           ('block_columns', 256), ('allow_tf32_matmul', True)]:
            with self.subTest(key=key):
                spec = self.calibration()
                settings = copy.deepcopy(self.spec['job']['index_settings'])
                # Guarantee a different value even if the source profile changes.
                settings[key] = not settings[key] if type(value) is bool else value
                self.assertNotEqual(settings, self.spec['job']['index_settings'])
                spec = self.split_index_shards(spec, settings)
                with self.assertRaisesRegex(ValueError, 'different correction settings'):
                    jobs.Builder().build('calibration', spec)

    def test_missing_index_settings_in_completed_shard_rejected(self):
        spec = self.split_index_shards(self.calibration(), {})
        with self.assertRaisesRegex(ValueError, 'Index settings must explicitly'):
            jobs.Builder().build('calibration', spec)

    def test_relative_workspace_rejected(self):
        self.spec['workspace'] = 'relative'
        with self.assertRaisesRegex(ValueError, 'absolute path'):
            jobs.Builder().build('index', self.spec)

    def test_no_hidden_job_override(self):
        self.spec['job']['artifacts'] = []
        with self.assertRaisesRegex(ValueError, 'Derived job fields'):
            jobs.Builder().build('index', self.spec)


if __name__ == '__main__':
    unittest.main()
