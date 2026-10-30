import json
from pathlib import Path
import tempfile
import unittest

from gr00t_libero.acceleration.bootstrap import audit_bootstrap, prepare_bootstrap
from gr00t_libero.acceleration.build_sage import PRESET, patch_source
from gr00t_libero.acceleration.prepare import ROOT, digest


class BootstrapPackagingTests(unittest.TestCase):
    def test_exported_hashes_and_relocation(self):
        rows = audit_bootstrap()
        self.assertGreater(len(rows), 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'source'
            receipt = prepare_bootstrap(root, Path(folder) / 'runtime')
            for row in receipt['sources']:
                data = (root / row['file']).read_bytes()
                self.assertEqual(digest(data), row['sha256'])
                self.assertNotIn(b'@WORKSPACE@', data)
                self.assertNotIn(b'@LOCAL_RUNTIME@', data)
            self.assertFalse(receipt['gpu_validated'])
            self.assertFalse(receipt['standalone_ready'])
            bridge = next(root / r['file'] for r in receipt['sources']
                          if r['file'].endswith('rollout_bridge_perrow_int8_r5.py'))
            self.assertTrue((bridge.parent / 'bootstrap_policy_perrow_int8_r5.py').is_file())
            self.assertTrue((bridge.parent / 'shape_modules').is_dir())
            with self.assertRaises(FileExistsError):
                prepare_bootstrap(root, Path(folder) / 'runtime')

    def test_sage_unknown_source_fails_before_patching(self):
        recipe = json.loads(PRESET.read_text())
        with self.assertRaisesRegex(ValueError, 'Unexpected original'):
            patch_source('unverified source', recipe)
        self.assertEqual(recipe['cuda_arch'], '8.9')

    def test_no_machine_paths_in_export(self):
        for row in audit_bootstrap():
            source = (ROOT / 'bootstrap_sources' / row['file']).read_text()
            self.assertNotRegex(source, r'/(?:mnt/[^/]+/users|home)/[^/]+/', row['file'])


if __name__ == '__main__':
    unittest.main()
