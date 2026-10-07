import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('cache_identity',Path(__file__).resolve().parents[1]/'gr00t_libero/cache_identity.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.job={k:str(self.root/k) for k in ('out','local_output','local_qa','batch_cache')}

    def test_resume_same_job_and_reject_changed_job(self):
        first=c.calibration_paths(self.job,'a'*64)
        (Path(self.job['local_output'])/'refs.pt').write_bytes(b'synthetic')
        self.assertEqual(c.calibration_paths(self.job,'a'*64),first)
        with self.assertRaisesRegex(ValueError,'different job'):
            c.calibration_paths(self.job,'b'*64)
        self.assertEqual((Path(self.job['local_output'])/'refs.pt').read_bytes(),b'synthetic')

    def test_separate_job_caches_under_shared_parent(self):
        first=c.calibration_paths(self.job,'a'*64)
        second=dict(self.job,out=str(self.root/'new_out'),local_output=str(self.root/'new_local'))
        other=c.calibration_paths(second,'b'*64)
        self.assertNotEqual(first,other)
        self.assertEqual(first[0].parent,other[0].parent)

    def test_unowned_existing_outputs_not_adopted(self):
        output=Path(self.job['local_output']);output.mkdir()
        (output/'refs.pt').write_bytes(b'synthetic')
        with self.assertRaisesRegex(ValueError,'no job identity'):
            c.calibration_paths(self.job,'a'*64)
        self.assertFalse((output/'RELEASE_JOB_IDENTITY.json').exists())

    def test_same_local_and_durable_directory(self):
        self.job['local_output']=self.job['out']
        c.calibration_paths(self.job,'a'*64)
        c.calibration_paths(self.job,'a'*64)
