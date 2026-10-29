import ast
import json
from pathlib import Path
import tempfile
import unittest

from gr00t_libero.acceleration import prepare


class AccelerationReleaseTests(unittest.TestCase):
    def test_manifest_and_staging_roundtrip(self):
        rows, preset = prepare.audit()
        self.assertTrue(all(r['pinned_in_final_contract'] for r in rows))
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / 'staged'
            receipt = prepare.prepare(workspace, Path(folder) / 'runtime')
            self.assertEqual(prepare.verify_prepared(workspace), receipt)
            self.assertFalse(receipt['gpu_validated'])
            self.assertEqual(set(receipt['selections']), set(preset['selections']))
            with self.assertRaises(FileExistsError):
                prepare.prepare(workspace, Path(folder) / 'runtime')

    def test_tamper_is_rejected_even_with_modified_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / 'staged'
            receipt = prepare.prepare(workspace, Path(folder) / 'runtime')
            row = receipt['sources'][0]
            path = workspace / row['file']
            path.write_text(path.read_text() + '\n# changed\n')
            with self.assertRaises(ValueError):
                prepare.verify_prepared(workspace)
            row['sha256'] = prepare.digest(path.read_bytes())
            (workspace / 'ACCELERATION_RECEIPT.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                prepare.verify_prepared(workspace)

    def test_frozen_choice_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / 'staged'
            receipt = prepare.prepare(workspace, Path(folder) / 'runtime')
            item = next(iter(receipt['selections'].values()))
            Path(item['path']).write_text('[]')
            with self.assertRaises(ValueError):
                prepare.verify_prepared(workspace)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                prepare.contained(Path(folder), '../outside')
            with self.assertRaises(ValueError):
                prepare.contained(Path(folder), '/absolute')
            with self.assertRaises(ValueError):
                prepare.prepare(Path(folder) / "quote'bad", Path(folder) / 'runtime')

    def test_install_sequence_matches_final_worker_ast(self):
        # The new boundary checks may change; the selected kernel-install sequence may not.
        root = prepare.ROOT
        source = (root / 'reference/final_worker.py').read_text()
        start = source.index('  import int8_ordered_epilogue')
        stop = source.index('  bundle=bridge.build')
        expected = ast.parse('def f():\n' + source[start:stop]).body[0].body
        tree = ast.parse((root / 'install.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure')
        index = next(i for i, n in enumerate(function.body)
                     if isinstance(n, ast.Import) and n.names[0].name == 'int8_ordered_epilogue_20261028_r1')
        actual = function.body[index:-1]
        self.assertEqual([ast.dump(n) for n in expected], [ast.dump(n) for n in actual])

    def test_reporting_denominators_and_separate_evaluations(self):
        root = prepare.ROOT.parents[1] / 'experiments/gr00t_libero'
        latency = json.loads((root / 'latency_final.json').read_text())
        success = json.loads((root / 'success_w1p5a8.json').read_text())
        self.assertEqual(sum(s['episodes'] for s in success['suites'].values()), success['episodes'])
        self.assertEqual(sum(s['successes'] for s in success['suites'].values()), success['successes'])
        self.assertAlmostEqual(success['success_percent'], 100 * success['successes'] / success['episodes'])
        for row in latency['rows']:
            control = row['matched_optimized_bf16_ms']
            if control is not None:
                self.assertAlmostEqual(control / row['wall_ms'], row['vs_optimized_bf16'])
        self.assertFalse(latency['new_success_rollout'])
        self.assertFalse(latency['clock_normalized'])
        self.assertTrue(success['diagnostic_only'])


if __name__ == '__main__':
    unittest.main()
