"""Build pinned original and current-stream SageAttention packages, without GPU execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .prepare import ROOT, contained, digest


def prepare(source, output, patched):
    source, output = Path(source).resolve(), Path(output).resolve()
    recipe = json.loads((ROOT / 'presets/sage_base_build.json').read_text())
    if output.exists():
        raise FileExistsError('Use a new source output directory')
    for row in recipe['files']:
        if digest(contained(source, row['file']).read_bytes()) != row['sha256']:
            raise ValueError('Original Sage source mismatch: ' + row['file'])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.sage-base-', dir=output.parent))
    try:
        for row in recipe['files']:
            dest = contained(tmp, row['file'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained(source, row['file']), dest)
        if patched:
            subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                            str(ROOT / 'sage_current_stream.patch')], cwd=tmp, check=True)
            for row in recipe['patches']:
                if digest((tmp / row['relative_path']).read_bytes()) != row['patched_sha256']:
                    raise ValueError('Stream patch differs from recorded source')
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    return recipe


def build(source, output, cuda_home, max_jobs):
    output, cuda_home = Path(output).resolve(), Path(cuda_home).resolve()
    if output.exists():
        raise FileExistsError('Use a new build root')
    if max_jobs < 1 or not (cuda_home / 'bin/nvcc').is_file():
        raise ValueError('Positive build concurrency and CUDA toolkit required')
    version = subprocess.check_output([str(cuda_home / 'bin/nvcc'), '--version'], text=True)
    if 'release 12.8' not in version:
        raise ValueError('The recorded Sage compiler is CUDA 12.8')
    import torch
    if torch.__version__ != '2.11.0+cu129' or torch.cuda.is_initialized():
        raise RuntimeError('Build in a fresh CPU-only process with PyTorch 2.11.0+cu129')
    output.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home),
               TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS=str(max_jobs))
    receipts = {}
    for name, patched in [('original', False), ('current_stream', True)]:
        directory = output / name
        recipe = prepare(source, directory / 'source', patched)
        library = directory / 'lib'
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--build-lib', str(library),
                        '--build-temp', str(directory / 'build')], cwd=directory / 'source',
                       env=env, check=True)
        for file in (directory / 'source/sageattention').rglob('*.py'):
            dest = library / 'sageattention' / file.relative_to(directory / 'source/sageattention')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
        files = [{'path': str(p), 'sha256': digest(p.read_bytes())}
                 for p in sorted(library.rglob('*')) if p.is_file()]
        receipt = dict(status='CPU_BUILD_COMPLETE_NO_GPU_NUMERICS', root=str(directory),
                       pythonpath=str(library), source_commit=recipe['source_commit'], files=files,
                       nvcc=version, GPU_execution=False)
        target = directory / 'BUILD_RECEIPT.json'
        target.write_text(json.dumps(receipt, indent=2) + '\n')
        receipts[name] = str(target)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cuda-home', required=True, type=Path)
    parser.add_argument('--max-jobs', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.cuda_home, args.max_jobs), indent=2))


if __name__ == '__main__':
    main()
