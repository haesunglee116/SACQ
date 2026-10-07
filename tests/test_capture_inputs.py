"""Request-order and explicit-provenance boundaries, using synthetic payloads."""
import copy
from pathlib import Path
import tempfile
import unittest
from gr00t_libero.captures import prepare


class CaptureInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for i in (4, 1):
            path = self.root/f'{i}.msgpack'; path.write_bytes(bytes([i]))
            rows.append({'path': str(path), 'request_id': f'r{i}', 'position': i,
                         'observations': [{'benchmark_family': 'synthetic', 'task': 'fixture',
                                           'base_seed': 17, 'episode_id': i, 'policy_call': 0}]})
        self.spec = {'schema': 'sacq_capture_inputs_v1', 'requests': rows}

    def test_order_and_explicit_positions_preserved(self):
        result = prepare(self.spec)
        self.assertEqual([r['position'] for r in result['records']], [4, 1])
        self.assertTrue(all(len(r['sha256']) == 64 for r in result['records']))
        self.assertIn('NOT_SCIENTIFIC_APPROVAL', result['status'])

    def test_missing_episode_provenance_is_not_inferred(self):
        del self.spec['requests'][0]['observations'][0]['base_seed']
        with self.assertRaisesRegex(ValueError, 'Every observation'):
            prepare(self.spec)

    def test_duplicate_ids_positions_and_paths_rejected(self):
        for key in ('request_id', 'position', 'path'):
            spec = copy.deepcopy(self.spec)
            spec['requests'][1][key] = spec['requests'][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare(spec)

    def test_expected_hash_mismatch_rejected(self):
        self.spec['requests'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            prepare(self.spec)

    def test_many_observations_per_request_are_preserved(self):
        observations = self.spec['requests'][0]['observations']
        observations.append(dict(observations[0], episode_id=8))
        self.assertEqual(len(prepare(self.spec)['records'][0]['observations']), 2)


if __name__ == '__main__':
    unittest.main()
