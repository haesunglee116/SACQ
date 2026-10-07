"""Operational logger changes must not silently patch unknown simulator code."""
from pathlib import Path
import tempfile
import unittest
from gr00t_libero.environments.configure_simulator import isolate_log


class SimulatorConfigurationTests(unittest.TestCase):
    def test_known_logger_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'logger.py'
            path.write_bytes(b'import logging\r\nfh = logging.FileHandler("/tmp/robosuite.log")\r\n')
            result = isolate_log(path)
            self.assertEqual(result['status'], 'CONFIGURED')
            self.assertNotEqual(result['before_sha256'], result['sha256'])
            self.assertIn(b"__import__('os').getuid()", path.read_bytes())
            self.assertIn(b'\r\n', path.read_bytes())
            self.assertEqual(isolate_log(path)['status'], 'ALREADY_CONFIGURED')

    def test_unknown_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'logger.py'
            original = b'print("different logger")\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                isolate_log(path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
