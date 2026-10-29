import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('dataset_verifier', Path(__file__).resolve().parents[1] / 'tools/verify_calibration_data.py')
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class DatasetVerifierTests(unittest.TestCase):
    def test_valid_and_corrupt_same_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'request').write_bytes(b'abc')
            manifest = {'suites': {'suite': {'records': [{'file': 'request', 'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]}}}
            self.assertEqual(verifier.verify(root, manifest)['status'], 'PASS')
            (root / 'request').write_bytes(b'abd')
            self.assertEqual(verifier.verify(root, manifest)['status'], 'FAIL')

    def test_missing_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['missing', '../outside', '/outside']:
                manifest = {'suites': {'suite': {'records': [{'file': name, 'size': 3, 'sha256': 'unused'}]}}}
                self.assertEqual(verifier.verify(directory, manifest)['status'], 'FAIL')

    def test_unknown_suite(self):
        with self.assertRaises(ValueError):
            verifier.verify('.', {'suites': {}}, ['missing'])
