"""Verify published calibration request files without loading model or pickle data."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(data_root, manifest, suites=None):
    root = Path(data_root).resolve()
    selected = list(manifest['suites']) if not suites else suites
    unknown = set(selected) - set(manifest['suites'])
    if unknown:
        raise ValueError(f'Unknown suites: {sorted(unknown)}')
    errors, count, total = [], 0, 0
    for suite in selected:
        for record in manifest['suites'][suite]['records']:
            name = record['file']
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root):
                errors.append(f'{name}: path outside data root')
                continue
            if not path.is_file():
                errors.append(f'{name}: missing file')
                continue
            if path.stat().st_size != record['size']:
                errors.append(f'{name}: size mismatch')
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                errors.append(f'{name}: SHA-256 mismatch')
                continue
            count += 1
            total += record['size']
    return {'status': 'FAIL' if errors else 'PASS', 'verified_files': count,
            'verified_bytes': total, 'suites': selected, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'data/libero_calibration/manifest.json')
    parser.add_argument('--suite', action='append', help='Verify only this suite; repeat for multiple suites')
    args = parser.parse_args()
    result = verify(args.data_root, json.loads(args.manifest.read_text()), args.suite)
    print(json.dumps(result, indent=2))
    return result['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
