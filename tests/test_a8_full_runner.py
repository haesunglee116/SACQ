import json
from pathlib import Path
import tempfile
import unittest

from gr00t_libero.acceleration.prepare import ROOT, digest, prepare, verify_prepared
from gr00t_libero.acceleration.run import prepare_run


class FullA8PackagingTests(unittest.TestCase):
    def test_unified_source_inventory_has_no_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = prepare(root / 'workspace', root / 'runtime', include_bootstrap=True)
            self.assertEqual(len(receipt['sources']), len({r['file'] for r in receipt['sources']}))
            self.assertEqual(verify_prepared(root / 'workspace'), receipt)
            self.assertTrue(receipt['include_bootstrap'])

    def fixture(self, root):
        # Synthetic packaging fixture only; never executes a model or CUDA.
        bundle = root / 'bundle'; bundle.mkdir(); (bundle / 'templates').mkdir()
        preset = json.loads((ROOT / 'presets/final_w1p5a8.json').read_text())
        bc = dict(perrow_method=preset['projection'], accuracy_recovery={'method': preset['accuracy_recovery_method']},
                  seed=42, warmup=5, num_iterations=20, suite='libero_10', environment={})
        values = {'bootstrap.json': bc, 'fixture.json': {},
                  'experiment.json': {'bootstrap': {'path': '@CONFIG@/bootstrap.json', 'sha256': 'RENDERED'}}}
        for name, value in values.items():
            (bundle / 'templates' / name).write_text(json.dumps(value))
        cfg = dict(schema='sacq_a8_fixed_input_bundle_v1', source_contract_sha256=preset['original_contract_sha256'],
                   templates=list(values), seed=42, warmup=5, measured=20, suite='libero_10',
                   files={str(p.relative_to(bundle)): digest(p.read_bytes()) for p in (bundle / 'templates').iterdir()})
        (bundle / 'inference.json').write_text(json.dumps(cfg))
        binary = root / 'dummy.so'; binary.write_bytes(b'unit-test-only')
        sage = json.loads((ROOT / 'presets/sage_base_build.json').read_text())
        ref = {'path': str(binary), 'sha256': digest(binary.read_bytes())}
        receipt = root / 'sage.json'; receipt.write_text(json.dumps(dict(source_commit=sage['source_commit'], files=[ref], compiled_module=ref)))
        python = root / 'venv/bin/python'; python.parent.mkdir(parents=True); python.symlink_to('/usr/bin/python3')
        runtime = dict(python=str(python), local_runtime=str(root / 'local'), gr00t=str(root), cuda_home=str(root),
                       sage_current_receipt=str(receipt), sage_original_receipt=str(receipt), sage_fp32_receipt=str(receipt))
        return bundle, runtime

    def test_templates_hashes_and_venv_path_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); bundle, runtime = self.fixture(root)
            command, env, cwd = prepare_run(bundle, root / 'work', runtime, root / 'out', expected_manifest={'inference': json.loads((bundle / 'inference.json').read_text())})
            self.assertEqual(command[0], runtime['python'])
            self.assertNotEqual(command[0], str(Path(runtime['python']).resolve()))
            cfg = json.loads((root / 'out/config/experiment.json').read_text())
            ref = cfg['bootstrap']
            self.assertEqual(ref['sha256'], digest(Path(ref['path']).read_bytes()))
            self.assertNotIn('LD_PRELOAD', env)
            with self.assertRaises(FileExistsError):
                prepare_run(bundle, root / 'work', runtime, root / 'out', expected_manifest={'inference': json.loads((bundle / 'inference.json').read_text())})

    def test_input_tamper_fails_before_workspace_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); bundle, runtime = self.fixture(root)
            (bundle / 'templates/bootstrap.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prepare_run(bundle, root / 'work', runtime, root / 'out', expected_manifest={'inference': json.loads((bundle / 'inference.json').read_text())})
            self.assertFalse((root / 'work').exists())


if __name__ == '__main__':
    unittest.main()
