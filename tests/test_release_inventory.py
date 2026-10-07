"""Release inventory consistency; no environment installation or model import."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_check',ROOT/'tools/release_check.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'reference_sources').mkdir()
        (self.root/'reference_sources/a.py').write_text('')
        (self.root/'gr00t_libero/sources').mkdir(parents=True)
        (self.root/'gr00t_libero/UNRESOLVED_IMPORTS.json').write_text('{}')
        self.rows=[{'file':'reference_sources/a.py','bundled_reference_imports_not_installed':[]}]

    def errors(self):
        (self.root/'REFERENCE_DEPENDENCIES.json').write_text(json.dumps({'snapshots':self.rows}))
        return m.dependency_inventory_errors(self.root)

    def test_current_inventory_matches_sources(self):
        self.assertEqual(m.dependency_inventory_errors(ROOT),[])

    def test_deleted_and_unlisted_files(self):
        self.assertEqual(self.errors(),[])
        (self.root/'reference_sources/a.py').unlink()
        (self.root/'reference_sources/b.py').write_text('')
        errors=self.errors()
        self.assertTrue(any('absent reference' in e for e in errors))
        self.assertTrue(any('missing from dependency' in e for e in errors))

    def test_duplicate_and_missing_bundled_dependency(self):
        self.rows[0]['bundled_reference_imports_not_installed']=['absent']
        self.rows.append(dict(self.rows[0]))
        errors=self.errors()
        self.assertTrue(any('duplicate' in e for e in errors))
        self.assertTrue(any('dependency absent' in e for e in errors))

    def test_unresolved_source_path_must_exist_inside_bundle(self):
        (self.root/'gr00t_libero/UNRESOLVED_IMPORTS.json').write_text(json.dumps({'dependency':['missing.py','../../reference_sources/a.py']}))
        self.assertEqual(len(self.errors()),2)
