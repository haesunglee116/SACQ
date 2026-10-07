"""Prepare ordered request references from explicit capture provenance.

Does not capture episodes, choose samples/splits, deserialize model payloads,
or issue a scientific authorization receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(spec):
    if spec.get('schema') != 'sacq_capture_inputs_v1':
        raise ValueError('Expected sacq_capture_inputs_v1')
    requests = spec.get('requests')
    if not isinstance(requests, list) or not requests:
        raise ValueError('An explicit nonempty ordered request list is required')
    seen_ids, seen_positions, seen_paths = set(), set(), set()
    records = []
    for row in requests:
        request_id, position = row.get('request_id'), row.get('position')
        if not isinstance(request_id, str) or not request_id or request_id in seen_ids:
            raise ValueError('Request IDs must be explicit, nonempty and unique')
        if type(position) is not int or position < 0 or position in seen_positions:
            raise ValueError('Original request positions must be explicit unique nonnegative integers')
        if not isinstance(row.get('path'), str) or not Path(row['path']).is_absolute():
            raise ValueError('Request paths must be explicit and absolute')
        path = Path(row['path']).resolve()
        if path.suffix != '.msgpack' or not path.is_file() or path in seen_paths:
            raise ValueError('Requests must be distinct existing .msgpack files')
        identities = row.get('observations')
        if not isinstance(identities, list) or not identities:
            raise ValueError('Declare episode provenance for every observation in the request batch')
        normalized = []
        required = {'benchmark_family', 'task', 'base_seed', 'episode_id', 'policy_call'}
        for identity in identities:
            if not isinstance(identity, dict) or set(identity) != required:
                raise ValueError('Every observation requires family, task, base_seed, episode_id and policy_call')
            if not all(isinstance(identity[k], str) and identity[k] for k in ('benchmark_family', 'task')):
                raise ValueError('Family and task must be nonempty strings')
            if not all(type(identity[k]) is int and identity[k] >= 0 for k in ('base_seed', 'episode_id', 'policy_call')):
                raise ValueError('Seeds, episode IDs and policy calls must be explicit nonnegative integers')
            normalized.append(dict(identity))
        sha = digest(path)
        if 'sha256' in row and row['sha256'] != sha:
            raise ValueError('Input capture hash differs from the supplied identity')
        records.append(dict(path=str(path), sha256=sha, request_id=request_id,
                            position=position, observations=normalized))
        seen_ids.add(request_id); seen_positions.add(position); seen_paths.add(path)
    return {
        'schema': 'sacq_prepared_capture_inputs_v1',
        'status': 'INPUT_REFERENCES_PREPARED_NOT_SCIENTIFIC_APPROVAL',
        'records': records,
        'order': 'Supplied order retained; no sorting, resampling or split assignment',
        'limitations': 'Episode provenance and payload request IDs are caller declarations. '
                       'This tool does not deserialize observations or prove episode-disjointness.',
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error('Output must be new; existing manifests are never overwritten')
    result = prepare(json.loads(a.spec.read_text()))
    # Exclusive creation also rejects a competing writer after the preflight.
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'requests': len(result['records'])}))


if __name__ == '__main__':
    main()
