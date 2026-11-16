import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scalar_bank', ROOT/'gr00t_libero/scalar_bank.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class ScalarBankTests(unittest.TestCase):
    def setup_bank(self, root, **changes):
        artifact = root/'candidate.pt'; artifact.write_bytes(b'fixture')
        row = dict(suite='s',layer='l',bits=3,method='GPTQ',path=str(artifact),sha256=m.digest(artifact))
        row.update(changes)
        path = root/'bank.json'; path.write_text(json.dumps(dict(schema='sacq_gptq_scalar_bank_v1',artifacts=[row])))
        return m.ScalarBank(dict(path=str(path),sha256=m.digest(path))), row

    def test_downstream_requires_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,row = self.setup_bank(Path(tmp))
            allocation=[dict(layer='l',k='80000')]
            bank.check_selected('s',allocation,[row])
            with self.assertRaises(ValueError):
                bank.check_selected('s',allocation,[dict(row,sha256='0'*64)])

    def test_tamper_missing_and_cross_suite_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,row = self.setup_bank(Path(tmp))
            bank.entry('s','l',3)
            with self.assertRaises(KeyError): bank.entry('other','l',3)
            with self.assertRaises(KeyError): bank.entry('s','l',4)
            Path(row['path']).write_bytes(b'mutated artifact')
            with self.assertRaises(ValueError): bank.entry('s','l',3)

    def test_hqq_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): self.setup_bank(Path(tmp),method='HQQ')

    def test_tensor_load_identity_and_precision(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); artifact = root/'candidate.pt'
            weight = torch.arange(12, dtype=torch.bfloat16).reshape(3,4)
            torch.save(dict(layer='l', bits=3, weight=weight), artifact)
            row = dict(suite='s', layer='l', bits=3, method='GPTQ', path=str(artifact), sha256=m.digest(artifact))
            path = root/'bank.json'; path.write_text(json.dumps(dict(schema='sacq_gptq_scalar_bank_v1', artifacts=[row])))
            bank = m.ScalarBank(dict(path=str(path), sha256=m.digest(path)))
            self.assertTrue(torch.equal(bank.weight('s','l',3,(3,4)), weight))
            with self.assertRaises(ValueError): bank.weight('s','l',3,(4,3))

    def test_runtime_has_no_hqq_candidate_fallback(self):
        import ast
        path=ROOT/'gr00t_libero/sources/libero_pro_eval/coord_descent_final_multisuite_2026_08_24.py'
        tree=ast.parse(path.read_text()); replica=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SuiteReplica')
        method=next(n for n in replica.body if isinstance(n,ast.FunctionDef) and n.name=='weight')
        source=ast.unparse(method)
        self.assertNotIn('quantize_weight_hqq',source)
        self.assertIn('SCALAR_BANK.weight',source)

if __name__=='__main__': unittest.main()
