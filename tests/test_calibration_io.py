"""Exercise durable-save plumbing with bytes, without importing the trainer."""
import ast
import os
from pathlib import Path
import shutil
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'gr00t_libero/sources/gr00t_final_seed17_20260922/calibration_worker.py'


class CalibrationIOTests(unittest.TestCase):
    def saver(self, out):
        tree=ast.parse(SOURCE.read_text())
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='durable_save')
        def original_save(payload,path):
            Path(path).write_bytes(payload)
        namespace=dict(original_save=original_save,Path=Path,shutil=shutil,os=os,j={'out':str(out)})
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<save-test>','exec'),namespace)
        return namespace['durable_save']

    def test_same_source_and_destination_does_not_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            self.saver(Path(tmp))(b'complete checkpoint',path)
            self.assertEqual(path.read_bytes(),b'complete checkpoint')

    def test_distinct_destination_receives_complete_atomic_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'durable';out.mkdir()
            path=root/'checkpoint.pt'
            self.saver(out)(b'complete checkpoint',path)
            self.assertEqual(path.read_bytes(),(out/path.name).read_bytes())
            self.assertFalse((out/'checkpoint.pt.tmp').exists())


if __name__=='__main__':
    unittest.main()
