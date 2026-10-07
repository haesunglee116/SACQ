import copy
import json
from pathlib import Path
import tempfile
import unittest
from gr00t_libero.installation_specs import prepare
from gr00t_libero.pipeline import sha


class InstallationSpecsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        allocation = self.file('allocation.csv', 'layer,scope,k\na,vlm,512\nb,dit,256\nc,dit,80000\n')
        job = dict(allocation=allocation['path'],
            artifacts=[dict(layer='a', scope='vlm', k=512), dict(layer='b', scope='dit', k=256)],
            gptq=[dict(layer='c', bits=3)])
        self.spec = dict(schema='sacq_installation_specs_v1', calibration_job=self.file('job.json', json.dumps(job)),
            allocation=allocation, templates=dict(vlm=self.file('vlm.txt', ''),
            dit=self.file('dit.txt', 'b=3;c=4;fixed=4')))

    def file(self, name, text):
        p = self.root/name; p.write_text(text)
        return dict(path=str(p), sha256=sha(p))

    def run_spec(self, spec=None):
        ref = self.file('spec.json', json.dumps(self.spec if spec is None else spec))
        return prepare(ref['path'], self.root/'output')

    def test_real_selection_replaces_stale_specs_and_preserves_background(self):
        result = self.run_spec()
        self.assertEqual((self.root/'output/vlm_spec.txt').read_text(), '')
        self.assertEqual((self.root/'output/dit_spec.txt').read_text(), 'b=16\nc=3\nfixed=4\n')
        for ref in result['calibration'].values():
            self.assertEqual(sha(ref['path']), ref['sha256'])
        with self.assertRaises(FileExistsError): self.run_spec()

    def test_different_allocation_content_or_identity_rejected(self):
        Path(self.spec['allocation']['path']).write_text('layer,scope,k\na,vlm,256\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.run_spec()
        self.spec['allocation']['sha256'] = sha(self.spec['allocation']['path'])
        with self.assertRaisesRegex(ValueError, 'differs from calibrated'): self.run_spec()
        self.assertFalse((self.root/'output').exists())

    def test_wrong_scope_template_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['templates']['vlm'] = self.file('wrong.txt', 'b=16')
        with self.assertRaisesRegex(ValueError, 'wrong scope'): self.run_spec(spec)


if __name__ == '__main__': unittest.main()
