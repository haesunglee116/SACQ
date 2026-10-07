"""Pinned checkout materialization tests with a synthetic local Git repository."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runtime_pipeline',ROOT/'gr00t_libero/pipeline.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class RuntimePrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.repo=self.root/'repo';self.repo.mkdir()
        self.git('init','-q')
        (self.repo/'base.py').write_text('original\n')
        (self.repo/'pyproject.toml').write_text('# synthetic upstream\n')
        self.git('add','.')
        self.git('-c','user.name=Synthetic Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
        self.revision=self.git('rev-parse','HEAD').strip()

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],text=True,stderr=subprocess.PIPE)

    def test_pinned_tree_ignores_worktree_changes_and_applies_overlays(self):
        (self.repo/'base.py').write_text('modified\n')
        (self.repo/'untracked.py').write_text('not exported\n')
        workspace=self.root/'workspace'
        with patch.object(p,'GR00T_REVISION',self.revision):
            p.prepare(workspace,self.repo)
        runtime=workspace/p.B/'Isaac-GR00T-N1.7'
        self.assertEqual((runtime/'base.py').read_text(),'original\n')
        self.assertFalse((runtime/'untracked.py').exists())
        self.assertNotEqual((runtime/'pyproject.toml').read_text(),'# synthetic upstream\n')
        receipt=json.loads((workspace/'SOURCE_RECEIPT.json').read_text())
        self.assertEqual(receipt['gr00t_revision'],self.revision)
        self.assertEqual(len(receipt['sources']),len({r['file'] for r in receipt['sources']}))
        for row in receipt['sources']:
            self.assertEqual(p.sha(workspace/row['file']),row['sha256'])

    def test_missing_commit_leaves_no_partial_workspace(self):
        workspace=self.root/'workspace'
        with patch.object(p,'GR00T_REVISION','0'*40), self.assertRaises(subprocess.CalledProcessError):
            p.prepare(workspace,self.repo)
        self.assertFalse(workspace.exists())

    def test_existing_destination_is_preserved(self):
        workspace=self.root/'workspace';workspace.mkdir()
        sentinel=workspace/'keep';sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError,'NEW workspace'):
            p.prepare(workspace,self.repo)
        self.assertEqual(sentinel.read_text(),'keep')

    def test_libero_commit_and_overlays_are_materialized_and_receipted(self):
        workspace=self.root/'with_libero'
        with patch.object(p,'GR00T_REVISION',self.revision), patch.object(p,'LIBERO_REVISION',self.revision):
            p.prepare(workspace,self.repo,self.repo)
        external=workspace/p.B/'Isaac-GR00T-N1.7/external_dependencies/LIBERO'
        self.assertEqual((external/'base.py').read_text(),'original\n')
        self.assertTrue((external/'libero/libero/envs/utils.py').is_file())
        receipt=json.loads((workspace/'SOURCE_RECEIPT.json').read_text())
        self.assertEqual(receipt['libero_revision'],self.revision)
        for row in receipt['sources']:
            self.assertEqual(p.sha(workspace/row['file']),row['sha256'])

    def test_missing_libero_commit_cleans_new_workspace(self):
        workspace=self.root/'missing_libero'
        with patch.object(p,'GR00T_REVISION',self.revision), patch.object(p,'LIBERO_REVISION','0'*40):
            with self.assertRaises(subprocess.CalledProcessError):
                p.prepare(workspace,self.repo,self.repo)
        self.assertFalse(workspace.exists())

    def test_libero_without_gr00t_fails_before_writing(self):
        workspace=self.root/'incomplete'
        with self.assertRaisesRegex(ValueError,'requires'):
            p.prepare(workspace,libero_checkout=self.repo)
        self.assertFalse(workspace.exists())

    def test_internal_file_link_materialized(self):
        (self.repo/'alias.py').symlink_to('base.py')
        self.git('add','alias.py')
        self.git('-c','user.name=Synthetic Test','-c','user.email=test@example.invalid','commit','-qm','link')
        with patch.object(p,'GR00T_REVISION',self.git('rev-parse','HEAD').strip()):
            destination=self.root/'export'
            p.export_runtime(self.repo,destination)
        self.assertFalse((destination/'alias.py').is_symlink())
        self.assertEqual((destination/'alias.py').read_text(),'original\n')

    def test_external_link_rejected_before_extraction(self):
        (self.repo/'alias.py').symlink_to('../outside.py')
        self.git('add','alias.py')
        self.git('-c','user.name=Synthetic Test','-c','user.email=test@example.invalid','commit','-qm','external link')
        destination=self.root/'export'
        with patch.object(p,'GR00T_REVISION',self.git('rev-parse','HEAD').strip()), self.assertRaises(ValueError):
            p.export_runtime(self.repo,destination)
        self.assertFalse(destination.exists())
