"""Installer failure boundaries; no network, package installs or GPU jobs."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gr00t_libero.environments import install_policy as installer


class PolicyInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = self.root / 'sacq.whl'
        self.wheel.write_bytes(b'test fixture')
        self.args = ['--env', str(self.root/'env'), '--artifacts', str(self.root/'artifacts'),
                     '--sacq-wheel', str(self.wheel)]

    def test_dry_run_has_no_side_effects_and_preserves_paths_with_spaces(self):
        self.args[1] = str(self.root/'env with spaces')
        with patch.object(installer.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(installer.main(self.args + ['--dry-run']), 0)
        run.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [self.wheel])
        self.assertIn(installer.FLASH_SHA256, out.getvalue())
        self.assertIn('env with spaces', out.getvalue())

    def test_existing_directory_is_not_overwritten(self):
        (self.root/'env').mkdir()
        sentinel = self.root/'env/existing'
        sentinel.write_text('preserve')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            installer.main(self.args)
        self.assertEqual(sentinel.read_text(), 'preserve')

    def test_bad_download_cannot_install_dependencies(self):
        import json
        def download(cmd, **kwargs):
            self.assertEqual(cmd[0], 'curl')
            Path(cmd[cmd.index('--output')+1]).write_bytes(b'corrupt')
        with patch.object(installer.subprocess, 'run', side_effect=download) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                installer.main(self.args)
        self.assertEqual(run.call_count, 1)
        receipt = json.loads((self.root/'artifacts/install_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'FAILED')
        self.assertFalse((self.root/'env').exists())

    def test_foundation_excludes_build_dependencies_and_uses_local_wheel(self):
        url, wheel, content = installer.foundation(installer.REQUIREMENTS, self.root)
        self.assertTrue(url.startswith('https://github.com/'))
        self.assertIn('flash-attn @ '+wheel.as_uri(), content)
        self.assertNotIn('deepspeed==', content)
        self.assertNotIn('fast-hadamard-transform @', content)


if __name__ == '__main__':
    unittest.main()
