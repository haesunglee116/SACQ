"""Set an explicit LIBERO asset config and isolate robosuite's shared log path.

Run with the simulator interpreter. Does not select tasks, seeds or episodes.
"""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path


def isolate_log(path):
    data = path.read_bytes()
    old = b'fh = logging.FileHandler("/tmp/robosuite.log")'
    new = b'fh = logging.FileHandler(f"/tmp/robosuite_{__import__(\'os\').getuid()}.log")'
    before = hashlib.sha256(data).hexdigest()
    if data.count(new) == 1 and old not in data:
        return {'status': 'ALREADY_CONFIGURED', 'sha256': before}
    if data.count(old) != 1:
        raise ValueError('Unknown robosuite logger; refusing an unrecognized source modification')
    updated = data.replace(old, new)
    compile(updated, str(path), 'exec')
    path.write_bytes(updated)
    return {'status': 'CONFIGURED', 'before_sha256': before,
            'sha256': hashlib.sha256(updated).hexdigest()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero', required=True, type=Path, help='Explicit LIBERO source root')
    p.add_argument('--config-dir', required=True, type=Path, help='New configuration directory')
    a = p.parse_args(argv)
    root = a.libero.resolve() / 'libero/libero'
    config = a.config_dir.resolve()
    if config.exists():
        p.error('--config-dir must be new; existing configuration is never overwritten')
    locations = {key: root / rel for key, rel in {
        'benchmark_root': '.', 'assets': 'assets', 'bddl_files': 'bddl_files',
        'init_states': 'init_files', 'datasets': '../datasets'}.items()}
    for key, path in locations.items():
        if key != 'datasets' and not path.is_dir():
            p.error(f'Missing LIBERO {key}: {path}')
    installed = distribution('robosuite')
    if installed.version != '1.4.0':
        p.error('This logger adapter supports robosuite 1.4.0 only')
    report = isolate_log(Path(installed.locate_file('robosuite/utils/log_utils.py')))
    config.mkdir(parents=True)
    # JSON strings are valid YAML scalars, including paths with spaces/quotes.
    (config/'config.yaml').write_text(''.join(f'{key}: {json.dumps(str(value))}\n'
                                           for key, value in locations.items()))
    (config/'configuration_receipt.json').write_text(json.dumps({
        'logger': report, 'LIBERO_CONFIG_PATH': str(config),
        'scope': 'Path configuration only, not simulator or benchmark validation'}, indent=2)+'\n')
    print(f'Set LIBERO_CONFIG_PATH={config} in the simulator process environment.')


if __name__ == '__main__':
    main()
