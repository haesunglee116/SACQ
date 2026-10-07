"""Synthetic candidate lookup tests; no model or experimental data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('release_bank',Path(__file__).resolve().parents[1]/'gr00t_libero/bank.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'candidate.pt'
        self.path.write_bytes(b'synthetic')
        self.row=dict(scope='vlm',layer='layer',k=16,suite='shared',init='aware',path=str(self.path),sha256=b.digest(self.path))

    def bank(self,*rows):
        return b.PackedBank(dict(schema='sacq_packed_bank_v1',artifacts=list(rows or [self.row])))

    def test_explicit_shared_lookup(self):
        bank=self.bank()
        for suite in ('libero_goal','libero_spatial'):
            self.assertEqual(bank.find_packed('vlm','layer',16,suite=suite,verify=False),self.path)
        with self.assertRaises(FileNotFoundError):
            bank.find_packed('vlm','layer',8,verify=False)

    def test_suite_specific_has_no_fallback(self):
        bank=self.bank(dict(self.row,scope='dit',suite='libero_goal'))
        self.assertEqual(bank.find_packed('dit','layer',16,suite='libero_goal',verify=False),self.path)
        with self.assertRaises(FileNotFoundError):
            bank.find_packed('dit','layer',16,suite='libero_spatial',verify=False)

    def test_duplicate_and_ambiguous_candidates_rejected(self):
        for rows in ((self.row,self.row),(self.row,dict(self.row,suite='libero_goal'))):
            with self.assertRaises(ValueError): self.bank(*rows)
        with self.assertRaises(ValueError): self.bank(dict(self.row,scope='dit'))

    def test_checksums_cached_until_file_changes(self):
        bank=self.bank()
        with patch.object(b,'digest',wraps=b.digest) as digest:
            bank.find_packed('vlm','layer',16,verify=False)
            bank.find_packed('vlm','layer',16,verify=False)
            self.assertEqual(digest.call_count,1)
            self.path.write_bytes(b'changed file contents')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                bank.find_packed('vlm','layer',16,verify=False)

    def test_manifest_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'manifest identity mismatch'):
            b.PackedBank.from_ref(dict(path=str(self.path),sha256='0'*64))

    def test_payload_k_checked(self):
        import torch
        torch.save({'packed':{'codebook':torch.zeros(16,4)}},self.path)
        row=dict(self.row,sha256=b.digest(self.path))
        self.assertEqual(self.bank(row).find_packed('vlm','layer',16),self.path)
        with self.assertRaisesRegex(ValueError,'size differs'):
            self.bank(dict(row,k=8)).find_packed('vlm','layer',8)

class InventoryBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.group=dict(scope='vlm',suite='shared',init='aware',k=16,directory=str(self.root),layers=['layer.a','layer.b'])
        self.spec=dict(schema='sacq_fitted_inventory_build_v1',groups=[self.group])
        for name in self.group['layers']:(self.root/(name+'.pt')).write_bytes(name.encode())

    def test_expands_explicit_list_ignoring_unrequested_files(self):
        (self.root/'unrequested.pt').write_bytes(b'not selected')
        result=b.build_inventory(self.spec)
        self.assertEqual([r['layer'] for r in result['artifacts']],self.group['layers'])
        for row in result['artifacts']:self.assertEqual(row['sha256'],b.digest(row['path']))

    def test_missing_candidate_is_not_silently_omitted(self):
        (self.root/'layer.b.pt').unlink()
        with self.assertRaises(FileNotFoundError):b.build_inventory(self.spec)

    def test_duplicate_scope_or_path_layer_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            b.build_inventory(dict(self.spec,groups=[self.group,self.group]))
        self.group['layers']=['../other']
        with self.assertRaisesRegex(ValueError,'module name'):b.build_inventory(self.spec)

    def test_dit_output_name_and_payload_size(self):
        import torch
        self.group.update(scope='dit',suite='libero_goal',layers=['layer.a'])
        torch.save({'packed':{'codebook':torch.zeros(16,4)}},self.root/'layer.a__ck.pt')
        self.assertTrue(b.build_inventory(self.spec,check_payloads=True)['artifacts'][0]['path'].endswith('__ck.pt'))
        self.group['k']=32
        with self.assertRaisesRegex(ValueError,'size differs'):b.build_inventory(self.spec,check_payloads=True)

    def test_cli_writes_consumer_manifest_and_refuses_overwrite(self):
        import json,subprocess,sys
        source=self.root/'spec.json';source.write_text(json.dumps(self.spec))
        output=self.root/'output'
        result=subprocess.run([sys.executable,str(Path(b.__file__)),'--build-spec',str(source),'--output-dir',str(output)],capture_output=True,text=True,check=True)
        receipt=json.loads(result.stdout)
        bank=b.PackedBank.from_ref(receipt['bank_manifest'])
        self.assertEqual(len(bank.rows),2)
        refs=json.loads((output/'inputs.json').read_text())
        for row in refs:self.assertEqual(b.digest(row['path']),row['sha256'])
        original=(output/'bank.json').read_bytes()
        with self.assertRaises(FileExistsError):b.write_inventory(source,output)
        self.assertEqual((output/'bank.json').read_bytes(),original)
