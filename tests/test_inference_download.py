import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gr00t_libero'))
from download_inference import materialize
from inference import sha

class DownloadTests(unittest.TestCase):
    def fixture(self, root, member='request.msgpack'):
        archive=root/'example.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            info=tarfile.TarInfo(member);info.size=4;t.addfile(info,io.BytesIO(b'demo'))
        import hashlib
        digest=hashlib.sha256(b'demo').hexdigest()
        model=root/'model';model.mkdir();(model/'config.json').write_bytes(b'{}')
        cfg=dict(schema='sacq_single_request_v1',condition='sacq',workshop_model={},files={'request.msgpack':digest,'model/config.json':sha(model/'config.json')})
        manifest=dict(schema='sacq_inference_download_v1',inference=cfg,
            archive=dict(url=archive.as_uri(),sha256=sha(archive)),archive_files={'request.msgpack':digest},
            upstream=[dict(kind='model',target='model/config.json',repo_id='synthetic',revision='synthetic',filename='config.json')])
        return manifest,model
    def test_materialize_resume_and_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,model=self.fixture(root);out=root/'out'
            materialize(m,out,local_model=model)
            materialize(m,out,local_model=model)
            self.assertEqual((out/'request.msgpack').read_bytes(),b'demo')
            different=copy.deepcopy(m);different['inference']['condition']='bf16'
            with self.assertRaises(ValueError):materialize(different,out,local_model=model)
    def test_reject_unexpected_archive_member_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,model=self.fixture(root,'../escape')
            with self.assertRaisesRegex(ValueError,'Unexpected archive'):materialize(m,root/'out',local_model=model)
            self.assertFalse((root/'escape').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,model=self.fixture(root);(model/'config.json').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'Upstream file differs'):materialize(m,root/'out',local_model=model)
            self.assertFalse((root/'out/inference.json').exists())
    def test_reject_unowned_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,model=self.fixture(root);out=root/'out';out.mkdir();(out/'keep').write_text('keep')
            with self.assertRaises(ValueError):materialize(m,out,local_model=model)
            self.assertEqual((out/'keep').read_text(),'keep')
if __name__=='__main__':unittest.main()
