"""A generated bank must not require old experiment directory names."""
import hashlib
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from gr00t_libero.bank import PackedBank
from gr00t_libero.stage import manifest_preflight, manifest_packed_path

class AllocationBankHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.a=SimpleNamespace(LADDER=[90000,8,4],HQQ_SENTINEL_K={90000:4},
            BIT_OF_K={90000:4,8:.75,4:.5},SUITES=('suite-a','suite-b'))
        self.rows=[]
        for scope,suite,layer in [('vlm','shared','language'),('dit','suite-a','action'),('dit','suite-b','action')]:
            path=root/(scope+'_'+suite+'.pt')
            torch.save({'packed':{'codebook':torch.zeros(8,4)}},path)
            self.rows.append(dict(scope=scope,suite=suite,layer=layer,k=8,init='aware',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def bank(self):return PackedBank(dict(schema='sacq_packed_bank_v1',artifacts=self.rows))

    def test_arbitrary_directory_with_exact_manifest_is_accepted(self):
        self.assertEqual(manifest_preflight(self.a,self.bank(),['language'],['action'],.75),3)

    def test_missing_suite_or_rate_is_not_filled_by_fallback(self):
        with self.assertRaises(FileNotFoundError):manifest_preflight(self.a,self.bank(),['language'],['action'],.5)
        self.rows.pop()
        with self.assertRaises(FileNotFoundError):manifest_preflight(self.a,self.bank(),['language'],['action'],.75)

    def test_changed_bytes_and_wrong_codebook_rejected(self):
        path=Path(self.rows[0]['path']);path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):manifest_preflight(self.a,self.bank(),['language'],['action'],.75)
        torch.save({'packed':{'codebook':torch.zeros(4,4)}},path)
        self.rows[0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'codebook size'):manifest_preflight(self.a,self.bank(),['language'],['action'],.75)

    def decoder_namespace(self):
        source=Path(__file__).parents[1]/'gr00t_libero/sources/libero_pro_eval/coord_descent_final_multisuite_2026_08_24.py'
        tree=ast.parse(source.read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('exact_packed_path','packed_weight_gpu')]
        self.assertEqual(len(functions),2)
        namespace=dict(torch=torch,Path=Path,KB={8:'075'})
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
        return namespace

    def test_runtime_decoder_accepts_manifest_path_with_identical_bf16_result(self):
        row=self.rows[0];path=Path(row['path'])
        packed=dict(codebook=torch.arange(32,dtype=torch.float32).reshape(8,4),
            indices=torch.arange(16).remainder(8).reshape(1,16),scales=torch.tensor([[.5]]),
            shift=.25,out_features=1,in_features=64,group_size=64)
        torch.save({'packed':packed},path);row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        canonical=path.parent/'vlm_packed_u075';canonical.mkdir();legacy=canonical/'language.pt';legacy.write_bytes(path.read_bytes())
        ns=self.decoder_namespace();ns['find_packed']=lambda *args,**kw:legacy
        expected=ns['packed_weight_gpu']('vlm','suite-a','language',8,torch.device('cpu'))
        bank=self.bank()
        ns['exact_packed_path']=lambda kind,suite,name,k:manifest_packed_path(bank,kind,suite,name,k)
        actual=ns['packed_weight_gpu']('vlm','suite-a','language',8,torch.device('cpu'))
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(actual.dtype,torch.bfloat16)
        path.write_bytes(b'tampered after first decode')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            ns['packed_weight_gpu']('vlm','suite-a','language',8,torch.device('cpu'))

    def test_legacy_runtime_still_rejects_fallback_directory(self):
        ns=self.decoder_namespace();ns['find_packed']=lambda *args,**kw:Path(self.rows[0]['path'])
        with self.assertRaisesRegex(AssertionError,'forbids fallback'):
            ns['packed_weight_gpu']('vlm','suite-a','language',8,torch.device('cpu'))

if __name__=='__main__':unittest.main()
