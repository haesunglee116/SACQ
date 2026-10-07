"""Probe wrapper and stdlib subprocess tests; no VLA environment is started."""
from pathlib import Path
import os
import signal
import time
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from gr00t_libero.runtime_check import probe, _run_isolated

class RuntimeCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'SOURCE_RECEIPT.json').write_text('{}')

    def test_success_is_import_only_with_offline_environment(self):
        with patch('gr00t_libero.runtime_check._run_isolated',return_value=subprocess.CompletedProcess([],0,'done','')) as run:
            report=probe(self.root,sys.executable,10)
        self.assertEqual(report['status'],'IMPORTS_PASSED')
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
        self.assertEqual(run.call_args.kwargs['timeout'],10)

    def test_failure_is_not_hidden(self):
        with patch('gr00t_libero.runtime_check._run_isolated',return_value=subprocess.CompletedProcess([],1,'','missing module')):
            report=probe(self.root,sys.executable,10)
        self.assertEqual(report['status'],'IMPORT_FAILED')
        self.assertEqual(report['stderr'],'missing module')

    def test_timeout_is_not_a_pass_or_proven_import_failure(self):
        error=subprocess.TimeoutExpired('python',10,output=b'partial',stderr=b'')
        with patch('gr00t_libero.runtime_check._run_isolated',side_effect=error):
            report=probe(self.root,sys.executable,10)
        self.assertEqual(report['status'],'TIMEOUT_UNVERIFIED')
        self.assertEqual(report['stdout'],'partial')

    def test_real_subprocess_reports_last_started_import(self):
        with patch('gr00t_libero.runtime_check.MODULES', ('json', 'sacq_deliberately_missing_module')):
            report = probe(self.root, sys.executable, 10)
        import json
        events = [json.loads(line) for line in report['stdout'].splitlines()]
        self.assertEqual(report['status'], 'IMPORT_FAILED')
        self.assertEqual(events[0], {'event': 'python_started'})
        self.assertEqual(events[2]['event'], 'import_completed')
        self.assertEqual(events[2]['module'], 'json')
        self.assertEqual(events[-1], {'event': 'import_started', 'module': 'sacq_deliberately_missing_module'})

    def test_progress_file_matches_report_and_cannot_be_overwritten(self):
        log = self.root / 'progress.jsonl'
        with patch('gr00t_libero.runtime_check.MODULES', ('json',)):
            report = probe(self.root, sys.executable, 10, log)
            self.assertEqual(report['status'], 'IMPORTS_PASSED')
            self.assertEqual(log.read_text(), report['stdout'])
            self.assertIn('import_completed', log.read_text())
            with self.assertRaises(FileExistsError):
                probe(self.root, sys.executable, 10, log)
        self.assertEqual(log.read_text(), report['stdout'])

    def test_timeout_preserves_flushed_progress_file(self):
        modules = self.root / 'libero_pro_eval'
        modules.mkdir()
        (modules / 'slow_probe_fixture.py').write_text('import time; time.sleep(10)\n')
        log = self.root / 'timeout.jsonl'
        with patch('gr00t_libero.runtime_check.MODULES', ('slow_probe_fixture',)):
            report = probe(self.root, sys.executable, 0.5, log)
        self.assertEqual(report['status'], 'TIMEOUT_UNVERIFIED')
        self.assertEqual(log.read_text(), report['stdout'])
        self.assertIn('import_started', report['stdout'])
        self.assertIn('slow_probe_fixture', report['stdout'])
        self.assertNotIn('import_completed', report['stdout'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process-state regression')
    def test_timeout_stops_child_processes_in_probe_group(self):
        code = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(child.pid,flush=True); time.sleep(60)"
        )
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            _run_isolated([sys.executable, '-u', '-c', code], timeout=1,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pid = int(caught.exception.stdout.strip())
        try:
            for _ in range(100):
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                except (FileNotFoundError, ProcessLookupError):
                    break
                state = next(line for line in status.splitlines() if line.startswith('State:'))
                if '(zombie)' in state or '(dead)' in state:
                    break
                time.sleep(0.01)
            else:
                self.fail('Probe descendant remained running after timeout')
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
