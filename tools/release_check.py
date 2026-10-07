"""Source-distribution checks; no credentials, network access, or model execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

IGNORED = {'.git', '.venv', '__pycache__', 'build', 'dist', '.pytest_cache'}
ARTIFACT_SUFFIXES = {'.pt', '.pth', '.safetensors', '.msgpack', '.pkl', '.npz', '.npy'}
PATTERNS = {
    'private workspace path': re.compile(r'/(?:mnt/lustre|home/haeseong|root/\.ssh)/'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'cloud access key': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
}


def source_files(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(p in IGNORED or p.endswith('.egg-info') for p in relative.parts):
            continue
        if path.is_symlink():
            yield path
        elif path.is_file():
            yield path



def dependency_inventory_errors(root):
    """Check inventory paths and coverage; this is not runtime import validation."""
    errors=[]
    reference=root/'reference_sources'
    rows=json.loads((root/'REFERENCE_DEPENDENCIES.json').read_text())['snapshots']
    declared=[row['file'] for row in rows]
    actual={str(p.relative_to(root)) for p in reference.rglob('*.py')}
    if len(declared)!=len(set(declared)):
        errors.append('Reference dependency inventory contains duplicate files')
    for name in sorted(set(declared)-actual):
        errors.append(f'{name}: dependency inventory refers to an absent reference source')
    for name in sorted(actual-set(declared)):
        errors.append(f'{name}: reference source missing from dependency inventory')
    names={p.stem for p in reference.rglob('*.py')}
    for row in rows:
        for module in row.get('bundled_reference_imports_not_installed',[]):
            if module not in names:
                errors.append(f"{row['file']}: bundled reference dependency absent: {module}")
    sources=(root/'gr00t_libero/sources').resolve()
    unresolved=json.loads((root/'gr00t_libero/UNRESOLVED_IMPORTS.json').read_text())
    for module,files in unresolved.items():
        for name in files:
            path=(sources/name).resolve()
            if not path.is_relative_to(sources) or not path.is_file():
                errors.append(f'{name}: unresolved-import inventory source absent for {module}')
    return errors


def audit(root):
    errors = []
    count = 0
    for path in source_files(root):
        name = str(path.relative_to(root))
        count += 1
        if path.is_symlink():
            errors.append(f'{name}: symlinks are not release source')
            continue
        if path.suffix in ARTIFACT_SUFFIXES or path.name == '.env' or path.name.startswith('.env.'):
            errors.append(f'{name}: local artifact or environment file')
            continue
        data = path.read_bytes()
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            errors.append(f'{name}: binary file requires explicit release review')
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f'{name}: {label}')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                errors.append(f'{name}: syntax error at line {exc.lineno}')
    manifest = root/'gr00t_libero/SOURCE_MANIFEST.json'
    base = root/'gr00t_libero/sources'
    for row in json.loads(manifest.read_text()):
        path = (base/row['file']).resolve()
        if not path.is_relative_to(base.resolve()):
            errors.append('Source manifest path escapes source directory')
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['export_sha256']:
            errors.append(f"{row['file']}: exported source hash mismatch")
    errors.extend(dependency_inventory_errors(root))
    return dict(files_checked=count, errors=errors,
                publication_license='present, review required' if (root/'LICENSE').is_file() else 'pending owner decision')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--publication', action='store_true', help='Also require the owner-selected top-level license')
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.publication and not (args.root/'LICENSE').is_file():
        result['errors'].append('Top-level license remains undecided')
    print(json.dumps(result, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    sys.exit(main())
