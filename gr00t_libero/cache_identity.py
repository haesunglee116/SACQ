"""Bind mutable calibration outputs to one explicit job; isolate derived caches."""
import fcntl
import json
import os
from pathlib import Path


def bind_output(path, job_sha256):
    directory=Path(path).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'RELEASE_JOB_IDENTITY.json'
    with (directory/'.release_identity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.exists():
            if json.loads(marker.read_text())!={'job_sha256':job_sha256}:
                raise ValueError('Calibration output belongs to a different job; use a new directory')
        else:
            if any(p.name!='.release_identity.lock' for p in directory.iterdir()):
                raise ValueError('Existing calibration output has no job identity; use a new directory')
            temporary=marker.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('w') as output:
                json.dump({'job_sha256':job_sha256},output)
                output.flush();os.fsync(output.fileno())
            os.replace(temporary,marker)
    return directory


def calibration_paths(job, job_sha256):
    if len(job_sha256)!=64 or any(c not in '0123456789abcdef' for c in job_sha256):
        raise ValueError('Expected SHA256 job identity')
    for path in sorted({str(Path(job[k]).resolve()) for k in ('out','local_output')}):
        bind_output(path,job_sha256)
    qa=Path(job['local_qa'])/'release_jobs'/job_sha256
    batches=Path(job['batch_cache'])/'release_jobs'/job_sha256
    qa.mkdir(parents=True,exist_ok=True)
    batches.mkdir(parents=True,exist_ok=True)
    return qa,batches
