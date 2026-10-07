"""Synthetic noise fixtures; no scientific episode or seed selection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

path=Path(__file__).resolve().parents[1]/'gr00t_libero/noise_inputs.py'
spec=importlib.util.spec_from_file_location('noise_inputs',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NoiseInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.spec=dict(schema='sacq_noise_inputs_v1',output=str(Path(self.tmp.name)/'noise'),
            seed=13,episodes=2,calls=3,shape=[1,2,4],dtype='float32',device='cpu',torch_version=torch.__version__)

    def test_exact_draw_order_and_global_rng_unchanged(self):
        before=torch.get_rng_state().clone()
        manifest=m.generate(self.spec)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        generator=torch.Generator(device='cpu').manual_seed(13)
        for episode in range(2):
            expected=torch.stack([torch.randn((1,2,4),generator=generator) for _ in range(3)])
            actual=torch.load(Path(self.spec['output'])/manifest['files'][episode]['name'],weights_only=True)
            self.assertTrue(torch.equal(expected,actual))
        with self.assertRaises(FileExistsError):m.generate(self.spec)

    def test_runtime_mismatch_fails_before_output(self):
        self.spec['torch_version']='not-the-installed-version'
        with self.assertRaisesRegex(ValueError,'version'):m.generate(self.spec)
        self.assertFalse(Path(self.spec['output']).exists())

    def test_missing_or_invalid_settings_fail_before_output(self):
        for key in self.spec:
            incomplete=dict(self.spec);del incomplete[key]
            with self.assertRaises(ValueError):m.validate(incomplete)
        for key,value in [('seed',True),('episodes',0),('shape',[0]),('dtype','int8'),('output','relative')]:
            invalid=dict(self.spec);invalid[key]=value
            with self.assertRaises(ValueError):m.validate(invalid)
        self.assertFalse(Path(self.spec['output']).exists())

if __name__=='__main__':unittest.main()
