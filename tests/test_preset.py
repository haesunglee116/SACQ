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
