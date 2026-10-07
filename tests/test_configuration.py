"""Configuration and trainer-hook checks without importing the VLA runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from gr00t_libero.configuration import calibration_argv, validate_training

ROOT = Path(__file__).resolve().parents[1]/'gr00t_libero'
TRAINER = ROOT/'sources/libero_pro_eval/quantization_analysis/workshop_metricv2_calibration_20260919/source/train_centroid_ft_2026_08_05.py'


def stable_ast(value):
    # Python 3.12 adds empty type_params fields; ignore only those empty fields.
    if isinstance(value, ast.AST):
        return [type(value).__name__, {key: stable_ast(item) for key, item in ast.iter_fields(value)
                                     if not (key == 'type_params' and not item)}]
    if isinstance(value, list):
        return [stable_ast(item) for item in value]
    return value


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.training = json.loads((ROOT/'profiles/legacy_calibration.json').read_text())['training']
        self.job = dict(training=self.training, trainer='/trainer.py', suite='explicit_suite',
                        tag='explicit_tag', allocation='/allocation.csv', all_records=[{}]*640)

    def test_legacy_arguments_and_budget(self):
        split = validate_training(self.training, 640)
        self.assertEqual(split, dict(train_records=576, validation_records=64, epoch_budget=1152))
        self.assertEqual(calibration_argv(self.job), [
            '/trainer.py', '--suite', 'explicit_suite', '--limit', '640', '--epochs', '12',
            '--global-batch', '6', '--out-tag', 'explicit_tag', '--alloc', '/allocation.csv',
            '--lr', '0.0001', '--lr-scale', '1e-05', '--val-every', '10', '--steps', '0',
            '--fp32-codebook-grad', '--resume'])

    def test_explicit_settings_drive_arguments(self):
        self.training.update(codebook_lr=2e-4, scale_lr=3e-5, resume=False, fp32_codebook_grad=False)
        args = calibration_argv(self.job)
        self.assertEqual(args[args.index('--lr')+1], '0.0002')
        self.assertEqual(args[args.index('--lr-scale')+1], '3e-05')
        self.assertNotIn('--resume', args)
        self.assertNotIn('--fp32-codebook-grad', args)

    def test_missing_or_unknown_setting_rejected(self):
        for variant in ({k:v for k,v in self.training.items() if k != 'codebook_lr'},
                        dict(self.training, lr=1e-4)):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_training(variant, 640)

    def test_invalid_learning_rate_rejected(self):
        for value in (True, -1., 0., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training(dict(self.training, codebook_lr=value), 640)

    def test_mismatched_record_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'limit differs'):
            validate_training(self.training, 639)

    def test_empty_or_indivisible_training_split_rejected(self):
        for fields in (dict(val_every=1), dict(global_batch=7)):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, 'divisible'):
                validate_training(dict(self.training, **fields), 640)

    def test_impossible_completion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            validate_training(dict(self.training, expected_updates=1153), 640)

    def test_unreachable_completion_without_stop_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Without an early stop'):
            validate_training(dict(self.training, expected_updates=1000), 640)

    def test_background_hook_precedes_resume_and_optimizer(self):
        tree = ast.parse(TRAINER.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        hook = next(n for n in main.body if isinstance(n, ast.If) and
                    ast.unparse(n.test) == 'install_background is not None')
        step_base = next(n for n in main.body if isinstance(n, ast.Assign) and
                         any(isinstance(t, ast.Name) and t.id == 'step_base' for t in n.targets))
        self.assertLess(main.body.index(hook), main.body.index(step_base))
        calls = []
        namespace = dict(install_background=lambda *args: calls.append(args),
                         policy=object(), trainables=object(), device=object())
        exec(compile(ast.Module(body=[hook], type_ignores=[]), '<hook-test>', 'exec'), namespace)
        self.assertEqual(calls, [(namespace['policy'], namespace['trainables'], namespace['device'])])
        namespace['install_background'] = None
        exec(compile(ast.Module(body=[hook], type_ignores=[]), '<hook-test>', 'exec'), namespace)
        self.assertEqual(len(calls), 1)

    def test_trainer_math_matches_submission_ast(self):
        tree = ast.parse(TRAINER.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertEqual([a.arg for a in main.args.kwonlyargs], ['install_background', 'output_dir'])
        main.args.kwonlyargs = []
        main.args.kw_defaults = []
        output = next(n for n in main.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'out_dir' for t in n.targets))
        self.assertIsInstance(output.value, ast.IfExp)
        output.value = output.value.orelse
        main.body = [n for n in main.body if not (isinstance(n, ast.If) and
                     ast.unparse(n.test) == 'install_background is not None')]
        digest = hashlib.sha256(json.dumps(stable_ast(tree), sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, '2116ed8e00ecf2a4b106fa10fca4414e230b979a324df6b676ab91345086b966')


if __name__ == '__main__':
    unittest.main()

class IndexConfigurationTests(unittest.TestCase):
    def settings(self):
        return json.loads((ROOT/'profiles/legacy_index.json').read_text())['index_settings']

    def test_reference_settings_and_explicit_overrides(self):
        from gr00t_libero.configuration import validate_index
        settings=self.settings()
        self.assertEqual(validate_index(settings)['block_columns'],128)
        settings.update(damping=.05,block_columns=64,noise_seed_base=9000,allow_tf32_matmul=False)
        self.assertEqual(validate_index(settings),settings)

    def test_missing_invalid_or_unknown_settings_rejected(self):
        from gr00t_libero.configuration import validate_index
        for settings in ({},dict(self.settings(),damping=float('nan')),dict(self.settings(),block_columns=4),
                         dict(self.settings(),noise_seed_base=True),dict(self.settings(),allow_tf32_cudnn=1),
                         dict(self.settings(),unknown=1)):
            with self.subTest(settings=settings),self.assertRaises(ValueError):validate_index(settings)

    def test_worker_reads_explicit_settings_at_numerical_call_sites(self):
        worker=ROOT/'sources/gr00t_final_seed17_20260922/index_worker.py'
        tree=ast.parse(worker.read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        correction=next(n for n in calls if ast.unparse(n.func)=='core.compensate_packed')
        values={k.arg:ast.unparse(k.value) for k in correction.keywords}
        self.assertEqual(values['damping'],"settings['damping']")
        self.assertEqual(values['block_columns'],"settings['block_columns']")
        action=next(n for n in calls if ast.unparse(n.func)=='ft.run_action_grad')
        self.assertEqual(next(ast.unparse(k.value) for k in action.keywords if k.arg=='seed'),"settings['noise_seed_base'] + row['position']")
        moment=next(n for n in calls if ast.unparse(n.func)=='stats.FullMoment')
        self.assertEqual(ast.unparse(moment.args[2]),"settings['moment_chunk_rows']")
