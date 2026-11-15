"""CPU checks for the BF16 statistics contract without importing the model stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'gr00t_libero/sources/libero_pro_eval/quantization_analysis/workshop_fixedalloc_refit_20260919/source/dit_metrics.py'

class Weight:
    def __init__(self, value, dtype='bf16'):
        self.value, self.dtype = value, dtype
    def detach(self): return self
    def cpu(self): return self

class Linear:
    def __init__(self, weight): self.weight = weight

class StatisticsBackgroundTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        nodes = [n for n in tree.body if
                 isinstance(n, ast.FunctionDef) and n.name == 'verify_bf16_weights'
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VLM_ENV' for t in n.targets)]
        self.ns = {'torch': SimpleNamespace(nn=SimpleNamespace(Linear=Linear),
                   bfloat16='bf16', equal=lambda a,b: a.value == b.value)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'), self.ns)
    def test_bf16_environment_has_no_overlay_or_forced_bits(self):
        env = self.ns['VLM_ENV']
        for prefix in ('VLM','DIT'):
            self.assertEqual(env[prefix+'_BITS'],'16')
            self.assertEqual(env[prefix+'_FORCE_BITS_SPECS_FILE'],'')
            self.assertEqual(env[prefix+'_CODEBOOK_OVERLAY_PATH'],'')
    def test_unchanged_weights_pass(self):
        model=SimpleNamespace(named_modules=lambda:[('layer',Linear(Weight(7)))])
        self.ns['verify_bf16_weights'](model,{'layer':Weight(7)})
    def test_changed_precision_or_values_fail(self):
        for w in (Weight(8),Weight(7,'fp16')):
            model=SimpleNamespace(named_modules=lambda:[('layer',Linear(w))])
            with self.assertRaises(RuntimeError):
                self.ns['verify_bf16_weights'](model,{'layer':Weight(7)})
