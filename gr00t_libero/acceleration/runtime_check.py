"""Report whether installed distributions match the recorded A8 environment."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from .prepare import ROOT


def check_versions(resolve=version):
    spec = json.loads((ROOT / 'presets/external_runtime.json').read_text())
    checks = {}
    for name, expected in spec['versions'].items():
        try:
            actual = resolve(name)
        except PackageNotFoundError:
            actual = None
        checks[name] = {'expected': expected, 'actual': actual, 'match': actual == expected}
    return {'status': 'VERSIONS_MATCH' if all(r['match'] for r in checks.values()) else 'RUNTIME_MISMATCH',
            'packages': checks, 'gpu_validated': False, 'model_ready': False,
            'note': 'Version matching does not verify patched sources, CUDA extensions or model outputs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check_versions()
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as f:
            f.write(text)
    print(text, end='')
    raise SystemExit(0 if report['status'] == 'VERSIONS_MATCH' else 1)


if __name__ == '__main__':
    main()
