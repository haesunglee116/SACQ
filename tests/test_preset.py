import copy
import json
import unittest
from gr00t_libero.preset import apply, PRESET


class PresetTests(unittest.TestCase):
    def bindings(self):
        return {'job': {'suite': 'libero_spatial'},
                'gptq': [{'layer': str(i)} for i in range(111)],
                'records': [{'position': i} for i in range(640) if i % 10 != 0]}

    def test_index_excludes_validation_and_preserves_bindings(self):
        binding = self.bindings(); original = copy.deepcopy(binding)
        result = apply('index', binding)
        self.assertEqual(result['job']['index_settings']['damping'], 0.1)
        self.assertEqual(binding, original)
        result['records'][0]['position'] = 0
        with self.assertRaisesRegex(ValueError, 'ordered 576'):
            apply('index', result)

    def test_conflicting_training_is_not_overwritten(self):
        binding = self.bindings(); binding['all_records'] = [{}]*640
        result = apply('calibration', binding)
        self.assertEqual(result['job']['training']['expected_updates'], 1152)
        result['job']['training']['codebook_lr'] = 1e-3
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            apply('calibration', result)

    def test_wrong_background_count_rejected(self):
        binding = self.bindings(); binding['gptq'].pop()
        with self.assertRaisesRegex(ValueError, '111'):
            apply('index', binding)

    def test_rollout_never_promotes_historical_data(self):
        result = apply('evaluation', {'plan': {}})
        self.assertEqual(result['plan']['rollout']['environment_seed'], 17)
        self.assertFalse(result['plan']['held_out'])
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            apply('evaluation', {'plan': {'held_out': True}})

    def test_allocation_is_actual_recorded_331_codebooks(self):
        import csv
        preset = json.loads(PRESET.read_text())
        with (PRESET.parent/preset['allocation']['file']).open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 338)
        self.assertEqual(sum(int(x['k']) < 80000 for x in rows), 331)


class UpstreamPresetTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suites = json.loads(PRESET.read_text())['suites']

    def bindings(self):
        return dict(workspace=str(self.root/'workspace'), processor=str(self.root/'processor'),
                    models={s: str(self.root/s) for s in self.suites}, output_root=str(self.root/'results'))

    def captures(self, name, names):
        folder = self.root/name
        folder.mkdir()
        for name in names:
            (folder/(name+'.msgpack')).write_bytes(b'fixture')
        return str(folder)

    def test_fitting_full_candidate_coverage_and_order(self):
        from gr00t_libero.preset_stages import prepare
        b = self.bindings()
        b['vlm_requests'] = self.captures('vlm', map(str, range(64)))
        b['dit_requests'] = self.captures('dit', map(str, range(64)))
        row = prepare('fitting', b, self.root/'plan')
        self.assertEqual(len(row['commands']), 44)
        self.assertEqual(len(row['outputs']), 8128+8)
        self.assertEqual(len(set(row['outputs'])), len(row['outputs']))
        groups = json.loads((self.root/'plan/inventory-spec.json').read_text())['groups']
        self.assertEqual(sum(len(g['layers']) for g in groups), 8128)
        for c in row['commands'][:4]:
            self.assertIn('--calib-calls', c['argv'])
        self.assertFalse(json.loads((self.root/'plan/preset-receipt.json').read_text())['executed'])

    def test_allocation_keeps_original_search_and_intermediate_selection(self):
        from gr00t_libero.preset_stages import prepare
        b = self.bindings()
        b['allocation_requests'] = self.captures('alloc', [f'{s}_{i:02}' for s in self.suites for i in range(10)])
        bank = self.root/'bank.json'; bank.write_text('{}')
        b['bank_manifest'] = str(bank)
        row = prepare('allocation', b, self.root/'plan')
        argv = row['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--target-nominal-bits')+1], '1.25')
        self.assertEqual(argv[argv.index('--n-gpus')+1], '8')
        self.assertTrue(any(p.endswith('allocation_ckpt2.00.csv') for p in row['outputs']))
        self.assertEqual(len(row['inputs']), 42)

    def test_incomplete_capture_set_rejected_before_writing(self):
        from gr00t_libero.preset_stages import prepare
        b = self.bindings()
        b['vlm_requests'] = self.captures('vlm', ['one'])
        b['dit_requests'] = self.captures('dit', ['one'])
        with self.assertRaisesRegex(ValueError, '64'):
            prepare('fitting', b, self.root/'plan')
        self.assertFalse((self.root/'plan').exists())

    def test_hyperparameter_override_rejected(self):
        from gr00t_libero.preset_stages import prepare
        b = self.bindings(); b['min_bit'] = 2
        with self.assertRaisesRegex(ValueError, 'path-only'):
            prepare('allocation', b, self.root/'plan')
