"""Run the final fixed-input W1.5A8 model, retaining original numerical gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .prepare import ROOT, contained, prepare, verify_prepared


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_run(bundle, workspace, runtime, output, *, expected_manifest=None):
    if not __debug__:
        raise RuntimeError('Python -O disables required numerical gates')
    bundle, workspace, output = [Path(x).resolve() for x in (bundle, workspace, output)]
    cfg = json.loads((bundle / 'inference.json').read_text())
    if cfg.get('schema') != 'sacq_a8_fixed_input_bundle_v1':
        raise ValueError('Expected the fixed-input A8 bundle')
    if expected_manifest is None:
        expected_manifest = json.loads((ROOT / 'presets/inference_libero10_w1p5a8.json').read_text())
    if {k: v for k, v in cfg.items() if k != 'upstream'} != expected_manifest['inference']:
        raise ValueError('Bundle specification differs from the published fixed-input release')
    preset = json.loads((ROOT / 'presets/final_w1p5a8.json').read_text())
    if cfg['source_contract_sha256'] != preset['original_contract_sha256']:
        raise ValueError('Unexpected baseline contract')
    for name, checksum in cfg['files'].items():
        if sha(contained(bundle, name)) != checksum:
            raise ValueError('Input checksum mismatch: ' + name)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    required = ['python', 'local_runtime', 'gr00t', 'cuda_home',
                'sage_current_receipt', 'sage_original_receipt', 'sage_fp32_receipt']
    if any(k not in runtime for k in required):
        raise ValueError('Missing runtime bindings: ' + ', '.join(k for k in required if k not in runtime))
    # Resolving a venv interpreter symlink discards its site-packages.
    runtime = {k: str(Path(v).absolute() if k == 'python' else Path(v).resolve())
               for k, v in runtime.items()}
    for key in ['sage_current_receipt', 'sage_original_receipt', 'sage_fp32_receipt']:
        receipt = json.loads(Path(runtime[key]).read_text())
        if receipt['source_commit'] != json.loads((ROOT / 'presets/sage_base_build.json').read_text())['source_commit']:
            raise ValueError('Wrong Sage source revision')
        refs = receipt.get('files', []) if key != 'sage_fp32_receipt' else [receipt['compiled_module']]
        if not refs:
            raise ValueError('Sage receipt lacks file identities')
        for ref in refs:
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Sage runtime file changed: ' + ref['path'])
    Path(runtime['local_runtime']).mkdir(parents=True, exist_ok=True)
    if not workspace.exists():
        prepare(workspace, runtime['local_runtime'], include_bootstrap=True)
    staged = verify_prepared(workspace)
    if not staged.get('include_bootstrap') or staged['local_runtime'] != runtime['local_runtime']:
        raise ValueError('Workspace was not prepared with this complete runtime binding')
    output.mkdir(parents=True)
    config = output / 'config'
    config.mkdir()
    bindings = {'BUNDLE': str(bundle), 'WORKSPACE': str(workspace), 'OUTPUT': str(output),
                'CONFIG': str(config), 'GR00T': runtime['gr00t'], 'PYTHON': runtime['python'],
                'SAGE_CURRENT_RECEIPT': runtime['sage_current_receipt'],
                'SAGE_ORIGINAL_RECEIPT': runtime['sage_original_receipt'],
                'SAGE_FP32_RECEIPT': runtime['sage_fp32_receipt']}
    active, done = set(), set()
    def render(value):
        if isinstance(value, list):
            return [render(x) for x in value]
        if isinstance(value, dict):
            result = {k: render(v) for k, v in value.items()}
            if 'path' in result and 'sha256' in result:
                path = result['path']
                if path.startswith(str(config) + '/'):
                    emit(str(Path(path).relative_to(config)))
                if not path.startswith('omitted_provenance:'):
                    result['sha256'] = sha(path)
            return result
        if isinstance(value, str):
            for key, path in bindings.items():
                value = value.replace('@' + key + '@', path)
            if '@' in value and any('@' + key in value for key in bindings):
                raise ValueError('Unresolved path binding')
        return value
    def emit(name):
        if name in done:
            return
        if name in active or name not in cfg['templates']:
            raise ValueError('Invalid template dependency: ' + name)
        active.add(name)
        value = render(json.loads(contained(bundle / 'templates', name).read_text()))
        contained(config, name).write_text(json.dumps(value, indent=2) + '\n')
        active.remove(name); done.add(name)
    for name in cfg['templates']:
        emit(name)
    bc = json.loads((config / 'bootstrap.json').read_text())
    if bc['perrow_method'] != preset['projection'] or bc['accuracy_recovery']['method'] != preset['accuracy_recovery_method']:
        raise ValueError('Quantization recipe differs from the final selection')
    if (bc['seed'], bc['warmup'], bc['num_iterations'], bc['suite']) != (cfg['seed'], cfg['warmup'], cfg['measured'], cfg['suite']):
        raise ValueError('Timing/sample settings differ from the bundle')
    (output / 'bootstrap').mkdir()
    (output / 'orders/forward').mkdir(parents=True)
    shutil.copyfile(config / 'fixture.json', output / 'fixture.json')
    worker = workspace / 'libero_pro_eval/inference_acceleration_2026_10_11/a8_accuracy_recovery_2026_10_20_r1/final_worker.py'
    shutil.copyfile(ROOT / 'reference/final_worker.py', worker)
    env = dict(os.environ)
    # Scientific quantization/noise configuration is owned by the pinned bootstrap.
    for key in list(env):
        if key.startswith(('VLM_', 'DIT_', 'ACTION_HEAD_', 'GR00T_', 'OMNI_')) or key in ('CUBLAS_WORKSPACE_CONFIG', 'LD_PRELOAD'):
            env.pop(key)
    for key, value in bc['environment'].items():
        if key in ('PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'TRITON_CACHE_DIR'):
            continue
        if 'omitted_provenance:' not in str(value):
            env[key] = str(value)
    env.update(CUDA_HOME=runtime['cuda_home'], OMNI_BENCH_CHILD='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=runtime['gr00t'] + ':' + runtime['gr00t'] + '/scripts/deployment',
               PATH=str(Path(runtime['python']).parent) + ':' + runtime['cuda_home'] + '/bin:' + os.environ.get('PATH', ''),
               TRITON_CACHE_DIR=str(Path(runtime['local_runtime']) / 'triton_cache'))
    command = [runtime['python'], str(worker), '--protocol', str(config / 'experiment.json'),
               '--order', 'forward', '--condition', 'int8_fused']
    receipt = dict(status='PREPARED_NOT_EXECUTED', command=command,
                   source_contract_sha256=cfg['source_contract_sha256'],
                   worker_sha256=sha(worker), bundle_manifest_sha256=sha(bundle / 'inference.json'),
                   runtime=runtime, scientific_settings_changed=False)
    (output / 'RUN_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return command, env, runtime['gr00t']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    if not args.prepare_only:
        spec = json.loads((ROOT / 'presets/external_runtime.json').read_text())
        expected = spec['versions']
        files = {r['file']: r['sha256'] for r in spec['vllm_omni']['files']}
        script = ('import json,hashlib,importlib.metadata as m; '
                  'print(json.dumps({"versions":{k:m.version(k) for k in ' + repr(list(expected)) +
                  '},"files":{k:hashlib.sha256(m.distribution("vllm-omni").locate_file(k).read_bytes()).hexdigest() for k in ' + repr(list(files)) + '}}))')
        actual = json.loads(subprocess.check_output([runtime['python'], '-c', script], text=True))
        if actual != {'versions': expected, 'files': files}:
            raise RuntimeError('Runtime versions or GR00T-port source hashes differ from the recorded environment')
    command, env, cwd = prepare_run(args.bundle, args.workspace, runtime, args.output)
    if args.prepare_only:
        print('PREPARED_NOT_EXECUTED ' + str(args.output))
        return
    receipt_path = args.output / 'RUN_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    try:
        subprocess.run(command, env=env, cwd=cwd, check=True)
        result_path = args.output / 'orders/forward/int8_fused.json'
        result = json.loads(result_path.read_text())
        if not (result['all50_timed_raw_actions_exact'] and result['all4_RNG_exact'] and
                result['whole_model_reference_vs_original_INT8_R1_bytes_exact']):
            raise RuntimeError('Full-model validation did not pass')
        receipt.update(status='COMPLETE', result_sha256=sha(result_path))
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error))
        raise
    finally:
        temporary = receipt_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)
    print(json.dumps({k: result[k] for k in ('status', 'prepared_wall_median_ms',
                                          'GPU_event_median_ms', 'projection_INT8_GEMMs',
                                          'Sage_attention_calls')}, indent=2))


if __name__ == '__main__':
    main()
