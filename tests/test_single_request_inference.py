"""Portable input handoff checks; intentionally no simulated GPU PASS."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gr00t_libero'))
from inference import preflight, sha
from inference_bundle import export


class InferenceBundleTests(unittest.TestCase):
    def fixture(self, root):
        spec = dict(schema='sacq_inference_export_v1', condition='bf16', suite='libero_spatial',
                    seed=123, denoising_steps=4)
        for key in ('model', 'processor'):
            p = root / key; p.mkdir(); (p / 'config.json').write_text('{}')
            spec[key] = str(p)
        req = root / 'request.msgpack'; req.write_bytes(b'fixture only')
        spec['request'] = dict(path=str(req), sha256=sha(req))
        workspace = root / 'workspace'
        source = workspace / 'libero_pro_eval/Isaac-GR00T-N1.7/gr00t/policy/gr00t_policy.py'
        source.parent.mkdir(parents=True); source.write_text('# fixture')
        (workspace / 'SOURCE_RECEIPT.json').write_text(json.dumps(dict(sources=[dict(
            file=str(source.relative_to(workspace)), sha256=sha(source))])))
        return spec, workspace

    def test_relocation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec, workspace = self.fixture(root)
            export(spec, root / 'bundle')
            with self.assertRaises(FileExistsError): export(spec, root / 'bundle')
            (root / 'bundle').rename(root / 'moved')
            cfg, plan = preflight(root / 'moved', workspace)
            self.assertEqual(plan['condition'], 'bf16')
            self.assertNotIn(str(root), json.dumps(cfg))

    def test_tampered_and_unlisted_model_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec, workspace = self.fixture(root)
            bundle = root / 'bundle'; export(spec, bundle)
            extra = bundle / 'model/extra.safetensors'; extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'inventory'): preflight(bundle, workspace)
            extra.unlink(); (bundle / 'request.msgpack').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): preflight(bundle, workspace)

    def test_sacq_binding_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec, workspace = self.fixture(root)
            ref = spec['request']
            spec['condition'] = 'sacq'
            spec['workshop_model'] = dict(vlm=ref, dit=ref, vlm_spec=ref, dit_spec=ref,
                gptq=[dict(layer='example', bits=4, **ref)], expected_sacq_count=1, expected_gptq_count=1)
            bundle = root / 'bundle'; export(spec, bundle)
            cfg, plan = preflight(bundle, workspace)
            self.assertTrue(Path(plan['workshop_model']['vlm']['path']).is_absolute())
            cfg['workshop_model']['vlm']['path'] = '../request.msgpack'
            (bundle / 'inference.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): preflight(bundle, workspace)

    def test_missing_settings_and_nested_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec, _ = self.fixture(root)
            with self.assertRaises(ValueError): export(spec, root / 'model/bundle')
            spec.pop('seed')
            with self.assertRaises(KeyError): export(spec, root / 'bundle')


if __name__ == '__main__': unittest.main()
