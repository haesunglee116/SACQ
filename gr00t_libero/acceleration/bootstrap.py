"""Stage the original INT8 bootstrap dependency sources for integration work.

This does not resolve checkpoint contracts, install external packages or run a model.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .prepare import ROOT, audit, contained, digest


def audit_bootstrap():
    rows = json.loads((ROOT / 'BOOTSTRAP_MANIFEST.json').read_text())
    names = [row['file'] for row in rows]
    actual = {str(p.relative_to(ROOT / 'bootstrap_sources'))
              for p in (ROOT / 'bootstrap_sources').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if len(set(names)) != len(names) or set(names) != actual:
        raise ValueError('Bootstrap inventory mismatch')
    for row in rows:
        if digest(contained(ROOT / 'bootstrap_sources', row['file']).read_bytes()) != row['export_sha256']:
            raise ValueError('Bootstrap hash mismatch: ' + row['file'])
    return rows


def prepare_bootstrap(workspace, local_runtime):
    workspace, local_runtime = Path(workspace).resolve(), Path(local_runtime).resolve()
    if workspace.exists():
        raise FileExistsError('Use a new workspace; existing sources are never overwritten')
    for value in (str(workspace), str(local_runtime)):
        if any(c in value for c in "\n\r'\"\\@"):
            raise ValueError('Unsupported source-rendering path')
    rows = audit_bootstrap()
    audit()  # Includes the exact bridge identity.
    workspace.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.sacq-bootstrap-', dir=workspace.parent))
    try:
        receipt = dict(schema='sacq_bootstrap_source_stage_v1', sources=[],
                       gpu_validated=False, standalone_ready=False)
        for row in rows:
            data = contained(ROOT / 'bootstrap_sources', row['file']).read_text()
            data = data.replace('@WORKSPACE@', str(workspace)).replace('@LOCAL_RUNTIME@', str(local_runtime))
            if '@USER_HOME@' in data:
                raise ValueError('Unbound home path')
            dest = contained(temporary, row['file'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(data)
            receipt['sources'].append(dict(file=row['file'], sha256=digest(dest.read_bytes())))
        bridge_name = 'rollout_bridge_perrow_int8_r5.py'
        bridge = temporary / 'libero_pro_eval/inference_acceleration_2026_10_11/a8_accuracy_recovery_2026_10_20_r1' / bridge_name
        shutil.copyfile(ROOT / 'reference' / bridge_name, bridge)
        receipt['sources'].append(dict(file=str(bridge.relative_to(temporary)), sha256=digest(bridge.read_bytes())))
        (temporary / 'BOOTSTRAP_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.rename(workspace)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--local-runtime', type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps({'status': 'SOURCE_HASHES_PASS', 'sources': len(audit_bootstrap())}))
    else:
        if args.workspace is None or args.local_runtime is None:
            parser.error('--workspace and --local-runtime are required')
        receipt = prepare_bootstrap(args.workspace, args.local_runtime)
        print(json.dumps({'status': 'SOURCE_PREPARED_NOT_GPU_VALIDATED', 'sources': len(receipt['sources'])}))


if __name__ == '__main__':
    main()
