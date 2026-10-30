"""Reproduce the recorded Sage FP32 current-stream extension from pinned source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .prepare import ROOT, contained, digest

PRESET = ROOT / 'presets/sage_fp32_build.json'
TARGET = 'csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu'


def patch_source(source, recipe):
    if digest(source.encode()) != recipe['original_source_sha256']:
        raise ValueError('Unexpected original Sage CUDA source')
    anchor = 'kernel_func<<<grid, block, smem_max>>>'
    if source.count(anchor) != 1:
        raise ValueError('Expected one launch site')
    symbol = recipe['symbol']
    patched = source.replace('#include "attn_cuda_sm89.h"',
        '#include <c10/cuda/CUDAStream.h>\n#include "attn_cuda_sm89.h"', 1)
    patched = patched.replace(anchor,
        'kernel_func<<<grid, block, smem_max, c10::cuda::getCurrentCUDAStream()>>>')
    patched = patched.replace('torch::Tensor ' + symbol + '(',
                              'torch::Tensor a8_recovery_' + symbol + '(', 1)
    start = patched.index('torch::Tensor a8_recovery_' + symbol + '(')
    signature = patched[start:patched.index('{', start)].strip() + ';'
    binding = ('#include <torch/extension.h>\n' + signature +
               '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("' + symbol +
               '", &a8_recovery_' + symbol + ');}\n')
    if digest(patched.encode()) != recipe['patched_source_sha256']:
        raise ValueError('Patched source differs from recorded implementation')
    if digest(binding.encode()) != recipe['binding_sha256']:
        raise ValueError('Binding differs from recorded implementation')
    return patched, binding


def prepare_source(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    recipe = json.loads(PRESET.read_text())
    if (source / '.git').exists():
        revision = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != recipe['source_commit']:
            raise ValueError('Checkout does not match the pinned Sage commit')
    # Verify every header and CUDA unit before compilation, including dirty worktrees.
    for row in recipe['source_manifest']:
        expected = recipe['original_source_sha256'] if row['file'] == TARGET else row['sha256']
        if digest(contained(source, row['file']).read_bytes()) != expected:
            raise ValueError('Sage source mismatch: ' + row['file'])
    if digest((source / 'LICENSE').read_bytes()) != recipe['license_sha256']:
        raise ValueError('Unexpected Sage license')
    patched, binding = patch_source((source / TARGET).read_text(), recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.sage-source-', dir=output.parent))
    try:
        # Copy only pinned inputs, never arbitrary build products from the checkout.
        for row in recipe['source_manifest']:
            dest = contained(temporary, row['file'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contained(source, row['file']), dest)
        shutil.copyfile(source / 'LICENSE', temporary / 'SAGE_LICENSE')
        (temporary / TARGET).write_text(patched)
        (temporary / 'binding.cpp').write_text(binding)
        (temporary / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return recipe


def build(source, output, cuda_home, max_jobs=2):
    if max_jobs < 1:
        raise ValueError('max_jobs must be positive')
    cuda_home = Path(cuda_home).resolve()
    if not (cuda_home / 'bin/nvcc').is_file():
        raise ValueError('CUDA toolkit with nvcc required')
    if shutil.which('ninja') is None:
        raise RuntimeError('Ninja must be installed and on PATH')
    recipe = prepare_source(source, output)
    output = Path(output).resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home),
                      TORCH_CUDA_ARCH_LIST=recipe['cuda_arch'], MAX_JOBS=str(max_jobs))
    import torch
    if torch.__version__ != recipe['torch_version']:
        raise RuntimeError('Use the recorded PyTorch build: ' + recipe['torch_version'])
    if torch.cuda.is_initialized():
        raise RuntimeError('Build in a fresh CPU-only process')
    from torch.utils.cpp_extension import load
    directory = output / 'build'
    directory.mkdir()
    module = load(name=recipe['module_name'],
                  sources=[str(output / 'binding.cpp'), str(output / TARGET)],
                  extra_cflags=recipe['compiler_flags_from_pinned_setup']['CXX_FLAGS'],
                  extra_cuda_cflags=recipe['compiler_flags_from_pinned_setup']['NVCC_FLAGS'],
                  build_directory=str(directory), with_cuda=True, verbose=True)
    def ref(path):
        path = Path(path).resolve()
        return {'path': str(path), 'sha256': digest(path.read_bytes())}
    receipt = dict(status='CPU_BUILD_COMPLETE_NO_GPU_NUMERICS',
                   source_commit=recipe['source_commit'], torch_version=torch.__version__,
                   compiled_module=ref(module.__file__), patched_source=ref(output / TARGET),
                   binding=ref(output / 'binding.cpp'), kernel_math_unchanged=True,
                   GPU_numerical_and_graph_PASS=False,
                   compiler_flags_from_pinned_setup=recipe['compiler_flags_from_pinned_setup'])
    (output / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--cuda-home', type=Path)
    parser.add_argument('--max-jobs', type=int, default=2)
    args = parser.parse_args()
    if args.prepare_only:
        prepare_source(args.source, args.output)
        print('PINNED_SOURCE_PREPARED_NOT_COMPILED')
    else:
        if args.cuda_home is None:
            parser.error('--cuda-home is required for compilation')
        receipt = build(args.source, args.output, args.cuda_home, args.max_jobs)
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
