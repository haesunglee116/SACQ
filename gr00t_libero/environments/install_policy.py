"""Install the tested policy dependencies into a NEW Linux x86_64 environment.

No model downloads, scientific settings, or jobs are selected by this installer.
Use --dry-run to inspect the commands without writing files or using the network.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from urllib.parse import urlsplit

REQUIREMENTS = Path(__file__).with_name('policy-linux-py310-20261006.txt')
FLASH_SHA256 = 'f09928f2c7e4dfbe6efca5a997862898495263c87afe662c18fb1936eb192eab'
HADAMARD_COMMIT = '1cc807efbd6cc001df359822d60bf6052dd66859'
HADAMARD_URL = 'https://github.com/Dao-AILab/fast-hadamard-transform.git'


def foundation(requirements, artifact_dir):
    lines = requirements.read_text().splitlines()
    flashes = [line.split(' @ ', 1)[1] for line in lines if line.startswith('flash-attn @ ')]
    if len(flashes) != 1 or not flashes[0].startswith('https://github.com/Dao-AILab/flash-attention/'):
        raise ValueError('Expected exactly one official FlashAttention wheel')
    url = flashes[0]
    wheel = artifact_dir / Path(urlsplit(url).path).name
    out = []
    for line in lines:
        if line.startswith(('deepspeed==', 'fast-hadamard-transform @ ')):
            continue
        out.append('flash-attn @ ' + wheel.as_uri() if line.startswith('flash-attn @ ') else line)
    return url, wheel, '\n'.join(out) + '\n'


def commands(env_dir, artifact_dir, python, uv, foundation_file, sacq_wheel):
    interpreter = str(env_dir / 'bin/python')
    pip = [uv, 'pip', 'install', '--python', interpreter, '--link-mode=copy']
    source = str(artifact_dir / 'fast-hadamard-transform')
    return [
        [uv, 'venv', '--python', python, str(env_dir)],
        [interpreter, '-c', 'import sys; assert sys.version_info[:3] == (3, 10, 19), sys.version'],
        pip + ['--extra-index-url', 'https://download.pytorch.org/whl/cu128',
               '--index-strategy', 'unsafe-best-match', '-r', str(foundation_file)],
        pip + ['--no-deps', '--reinstall-package', 'torchcodec', '--index-url',
               'https://pypi.org/simple', 'torchcodec===0.4.0'],
        pip + ['--no-build-isolation', 'deepspeed==0.17.6'],
        ['git', 'clone', HADAMARD_URL, source],
        ['git', '-C', source, 'checkout', '--detach', HADAMARD_COMMIT],
        pip + ['--no-deps', '--no-build-isolation', source],
        pip + ['--no-deps', str(sacq_wheel)],
        [uv, 'pip', 'check', '--python', interpreter],
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', required=True, type=Path)
    p.add_argument('--artifacts', required=True, type=Path)
    p.add_argument('--sacq-wheel', required=True, type=Path)
    p.add_argument('--python', default='3.10.19', help='Python 3.10 interpreter or uv selector')
    p.add_argument('--uv', default='uv')
    p.add_argument('--build-jobs', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    env_dir, artifacts, sacq_wheel = a.env.resolve(), a.artifacts.resolve(), a.sacq_wheel.resolve()
    if env_dir.exists() or artifacts.exists():
        p.error('--env and --artifacts must both be new paths; existing files are never overwritten')
    if env_dir == artifacts or env_dir in artifacts.parents or artifacts in env_dir.parents:
        p.error('Environment and artifact directories must be separate, non-nested paths')
    if a.build_jobs < 1:
        p.error('--build-jobs must be positive')
    if not a.dry_run and (sys.platform != 'linux' or platform.machine() != 'x86_64'):
        p.error('This tested dependency set supports Linux x86_64 only')
    if not sacq_wheel.is_file() or sacq_wheel.suffix != '.whl':
        p.error('--sacq-wheel must point to a built wheel')
    url, flash_wheel, content = foundation(REQUIREMENTS, artifacts)
    foundation_file = artifacts / 'foundation.txt'
    steps = commands(env_dir, artifacts, a.python, a.uv, foundation_file, sacq_wheel)
    download = ['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                '--max-time', '1800', '--output', str(flash_wheel) + '.partial', url]
    if a.dry_run:
        print(shlex.join(download))
        print(f'# Verify wheel SHA-256: {FLASH_SHA256}; write foundation.txt with local wheel URI')
        for cmd in steps:
            print(shlex.join(cmd))
        return 0
    artifacts.mkdir(parents=True)
    receipt = {'status': 'RUNNING', 'completed_commands': [],
               'requirements_sha256': hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
               'sacq_wheel_sha256': hashlib.sha256(sacq_wheel.read_bytes()).hexdigest(),
               'scope': 'Dependency installation only, not model or benchmark validation'}
    def save():
        temp = artifacts / 'install_receipt.json.tmp'
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(artifacts / 'install_receipt.json')
    build_env = dict(os.environ, UV_HTTP_TIMEOUT='120', MAX_JOBS=str(a.build_jobs),
                     FAST_HADAMARD_TRANSFORM_FORCE_BUILD='TRUE')
    def run(cmd):
        print(shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=build_env)
        receipt['completed_commands'].append(cmd)
        save()
    save()
    try:
        run(download)
        partial = Path(str(flash_wheel) + '.partial')
        if hashlib.sha256(partial.read_bytes()).hexdigest() != FLASH_SHA256:
            raise ValueError('FlashAttention wheel checksum mismatch; refusing installation')
        partial.replace(flash_wheel)
        foundation_file.write_text(content)
        for cmd in steps:
            run(cmd)
        inventory = subprocess.check_output([a.uv, 'pip', 'freeze', '--python', str(env_dir / 'bin/python')], text=True)
        (artifacts / 'installed.txt').write_text(inventory)
        receipt['status'] = 'DEPENDENCIES_INSTALLED'
    except BaseException as exc:
        receipt.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
