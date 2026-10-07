"""Offline import-only probe of a prepared GR00T environment, not model execution."""
from contextlib import nullcontext
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

MODULES=('torch','transformers','gr00t.policy.gr00t_policy','index_worker','calibration_worker','fixed_model')


def _run_isolated(command, *, timeout, **kwargs):
    """On POSIX, timeout cleanup includes descendants in the probe's group."""
    with subprocess.Popen(command, start_new_session=(os.name == 'posix'), **kwargs) as process:
        def stop():
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from error
        except BaseException:
            stop()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def probe(workspace, python, timeout_seconds, progress_path=None):
    workspace=Path(workspace).resolve();python=Path(python).absolute()
    if not workspace.is_dir() or not (workspace/'SOURCE_RECEIPT.json').is_file():
        raise ValueError('Use a prepared source workspace')
    if not python.is_file() or timeout_seconds<=0:
        raise ValueError('Explicit runtime Python and positive timeout required')
    paths=[workspace/'libero_pro_eval/Isaac-GR00T-N1.7',workspace/'libero_pro_eval',
           workspace/'gr00t_final_seed17_20260922',workspace/'libero_transfer_quantstats_20260922',
           workspace/'simplerenv_n17_feasibility_2026_08_27']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',
             PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,paths)))
    code=('import importlib,json,time\n'
          "print(json.dumps({'event':'python_started'}),flush=True)\n"
          f'for name in {MODULES!r}:\n'
          " print(json.dumps({'event':'import_started','module':name}),flush=True)\n"
          ' start=time.monotonic();importlib.import_module(name)\n'
          " print(json.dumps({'event':'import_completed','module':name,'seconds':time.monotonic()-start}),flush=True)\n")
    # A new file is required so a diagnostic never truncates existing evidence.
    stream = nullcontext(None) if progress_path is None else Path(progress_path).open('x+', encoding='utf-8')
    with stream as progress:
        try:
            result=_run_isolated([str(python),'-u','-c',code],cwd=workspace,env=env,text=True,
                                  stdout=progress if progress is not None else subprocess.PIPE,
                                  stderr=subprocess.PIPE,timeout=timeout_seconds)
            report=dict(status='IMPORTS_PASSED' if result.returncode==0 else 'IMPORT_FAILED',
                        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired as error:
            def text(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
            report=dict(status='TIMEOUT_UNVERIFIED',stdout=text(error.stdout),stderr=text(error.stderr))
        if progress is not None:
            progress.seek(0)
            report['stdout']=progress.read()
    report.update(modules=list(MODULES),timeout_seconds=timeout_seconds,
                  scope='Imports only; no model construction, data loading, forward pass or simulator rollout requested')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--timeout-seconds',required=True,type=int)
    parser.add_argument('--progress-file', help='New JSONL file with flushed module progress events')
    args=parser.parse_args()
    result=probe(args.workspace,args.python,args.timeout_seconds,args.progress_file)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='IMPORTS_PASSED' else 1


if __name__=='__main__':raise SystemExit(main())
