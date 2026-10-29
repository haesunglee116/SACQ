"""Verify and stage the final kernel overlay without importing GPU dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contained(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes source root: {name}')
    return path


def audit(root=ROOT):
    root = Path(root)
    rows = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    names = [row['file'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source entries')
    actual = {str(p.relative_to(root / 'sources')) for p in (root / 'sources').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != set(names):
        raise ValueError('Source inventory does not match exported files')
    for row in rows:
        path = contained(root / 'sources', row['file'])
        if path.is_symlink() or digest(path.read_bytes()) != row['export_sha256']:
            raise ValueError(f"Source hash mismatch: {row['file']}")
    preset = json.loads((root / 'presets/final_w1p5a8.json').read_text())
    for item in preset['selections'].values():
        if digest(contained(root, item['file']).read_bytes()) != item['sha256']:
            raise ValueError(f"Selection hash mismatch: {item['file']}")
    provenance = json.loads((root / 'reference/PROVENANCE.json').read_text())
    if digest((root / 'reference/final_worker.py').read_bytes()) != provenance['export_sha256']:
        raise ValueError('Reference worker hash mismatch')
    bridge = provenance['bridge']
    if digest(contained(root / 'reference', bridge['file']).read_bytes()) != bridge['export_sha256']:
        raise ValueError('Reference bridge hash mismatch')
    return rows, preset


def prepare(workspace, local_runtime, root=ROOT):
    """Create an isolated source overlay. This neither installs a model nor runs a job."""
    root = Path(root).resolve()
    workspace, local_runtime = Path(workspace).resolve(), Path(local_runtime).resolve()
    if workspace.exists():
        raise FileExistsError('Use a new workspace; existing sources are never overwritten')
    for value in [str(workspace), str(local_runtime)]:
        if any(c in value for c in "\n\r'\"\\@"):
            raise ValueError('Unsupported characters in source-rendering path')
    rows, preset = audit(root)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.sacq-accel-', dir=workspace.parent))
    try:
        receipt = {'schema': 'sacq_acceleration_stage_v1', 'workspace': str(workspace),
                   'local_runtime': str(local_runtime), 'sources': [], 'selections': {},
                   'original_contract_sha256': preset['original_contract_sha256'],
                   'gpu_validated': False}
        for row in rows:
            data = contained(root / 'sources', row['file']).read_text()
            data = data.replace('@WORKSPACE@', str(workspace)).replace('@LOCAL_RUNTIME@', str(local_runtime))
            if '@USER_HOME@' in data:
                raise ValueError('Unbound home path in source')
            dest = contained(temporary, row['file'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(data)
            receipt['sources'].append({'file': row['file'], 'sha256': digest(dest.read_bytes())})
        for key, item in preset['selections'].items():
            dest = temporary / 'selections' / Path(item['file']).name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(contained(root, item['file']), dest)
            receipt['selections'][key] = {'path': str(workspace / 'selections' / dest.name),
                                           'sha256': item['sha256']}
        (temporary / 'ACCELERATION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.rename(workspace)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return receipt


def verify_prepared(workspace):
    workspace = Path(workspace).resolve()
    receipt = json.loads((workspace / 'ACCELERATION_RECEIPT.json').read_text())
    rows, preset = audit()
    if receipt['workspace'] != str(workspace):
        raise ValueError('Workspace moved since preparation; prepare a new copy')
    if receipt['original_contract_sha256'] != preset['original_contract_sha256']:
        raise ValueError('Unexpected recipe')
    # Recompute expected rendered contents, not just hashes from a mutable receipt.
    expected = {}
    for row in rows:
        data = contained(ROOT / 'sources', row['file']).read_text()
        data = data.replace('@WORKSPACE@', str(workspace)).replace('@LOCAL_RUNTIME@', receipt['local_runtime'])
        expected[row['file']] = digest(data.encode())
    if len(receipt['sources']) != len(expected) or {r['file']: r['sha256'] for r in receipt['sources']} != expected:
        raise ValueError('Prepared source receipt differs from canonical export')
    for name, checksum in expected.items():
        if digest(contained(workspace, name).read_bytes()) != checksum:
            raise ValueError(f'Prepared source changed: {name}')
    if set(receipt['selections']) != set(preset['selections']):
        raise ValueError('Selection inventory changed')
    for key, item in receipt['selections'].items():
        expected_hash = preset['selections'][key]['sha256']
        expected_path = workspace / 'selections' / Path(preset['selections'][key]['file']).name
        if Path(item['path']) != expected_path or item['sha256'] != expected_hash or digest(expected_path.read_bytes()) != expected_hash:
            raise ValueError(f'Selection changed: {key}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--local-runtime', type=Path)
    args = parser.parse_args()
    if args.check:
        rows, preset = audit()
        print(json.dumps({'status': 'PASS', 'sources': len(rows), 'selections': len(preset['selections'])}))
    else:
        if args.workspace is None or args.local_runtime is None:
            parser.error('--workspace and --local-runtime are required')
        result = prepare(args.workspace, args.local_runtime)
        print(json.dumps({'status': 'SOURCE_PREPARED_NOT_GPU_VALIDATED', 'workspace': result['workspace']}))


if __name__ == '__main__':
    main()
