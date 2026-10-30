"""Download the pinned representative BF16/SACQ A16 inputs, then verify them.

Requires huggingface_hub; Cosmos access uses your own Hugging Face authorization.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from inference import sha, resolve


def materialize(manifest, output, *, local_model=None, local_processor=None, condition='sacq'):
    if manifest.get('schema') != 'sacq_inference_download_v1':
        raise ValueError('Unsupported download manifest')
    if condition not in ('bf16', 'sacq'):
        raise ValueError('Unsupported inference condition')
    output = Path(output).resolve()
    cfg = copy.deepcopy(manifest['inference'])
    if condition == 'bf16':
        cfg['condition'] = 'bf16'; cfg.pop('workshop_model', None)
    output.mkdir(parents=True, exist_ok=True)
    owner = output / 'DOWNLOAD_SPEC.json'
    identity = dict(manifest=manifest, condition=condition)
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise ValueError('Output belongs to a different download specification')
    elif any(output.iterdir()):
        raise ValueError('Use a new directory or resume this downloader\'s own output')
    else:
        owner.write_text(json.dumps(identity, indent=2) + '\n')
    archive = manifest['archive']
    tarpath = output / 'download.tar.gz'
    def correct(path, digest):
        return path.is_file() and sha(path) == digest
    wanted = manifest['archive_files']
    if not all(correct(resolve(output, name), digest) for name, digest in wanted.items()):
        if not correct(tarpath, archive['sha256']):
            partial = output / 'download.tar.gz.partial'
            print('Downloading SACQ artifacts and example observation', flush=True)
            req = urllib.request.Request(archive['url'], headers={'User-Agent': 'SACQ-inference-download'})
            with urllib.request.urlopen(req, timeout=120) as source, partial.open('wb') as dest:
                shutil.copyfileobj(source, dest, length=8 << 20)
            if sha(partial) != archive['sha256']:
                raise ValueError('Archive checksum mismatch')
            partial.replace(tarpath)
        seen = set()
        with tarfile.open(tarpath, 'r:gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in wanted or member.name in seen:
                    raise ValueError('Unexpected archive member: ' + member.name)
                seen.add(member.name)
                target = resolve(output, member.name); target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.partial')
                with tar.extractfile(member) as source, partial.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                if sha(partial) != wanted[member.name]:
                    raise ValueError('Archive member checksum mismatch: ' + member.name)
                partial.replace(target)
        if seen != set(wanted):
            raise ValueError('Missing archive members')
    local = {'model': local_model, 'processor': local_processor}
    for row in manifest['upstream']:
        target = resolve(output, row['target']); digest = cfg['files'][row['target']]
        if correct(target, digest):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        directory = local[row['kind']]
        if directory:
            source = Path(directory) / Path(row['target']).relative_to(row['kind'])
        else:
            from huggingface_hub import hf_hub_download
            print('Fetching ' + row['repo_id'] + '/' + row['filename'], flush=True)
            source = Path(hf_hub_download(repo_id=row['repo_id'], revision=row['revision'], filename=row['filename']))
        if not correct(source, digest):
            raise ValueError('Upstream file differs from pinned model: ' + row['target'])
        partial = target.with_name(target.name + '.partial')
        shutil.copyfile(source, partial)
        if sha(partial) != digest:
            raise ValueError('Copy checksum mismatch: ' + row['target'])
        partial.replace(target)
    for name, digest in cfg['files'].items():
        if not correct(resolve(output, name), digest):
            raise ValueError('Incomplete bundle: ' + name)
    config = output / 'inference.json'
    temp = output / 'inference.json.partial'
    temp.write_text(json.dumps(cfg, indent=2) + '\n'); temp.replace(config)
    print('BUNDLE_READY ' + str(output), flush=True)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--manifest', type=Path, default=Path(__file__).parent/'presets/inference_spatial_w1p5a16.json')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--condition', choices=('bf16', 'sacq'), default='sacq')
    p.add_argument('--local-model', type=Path, help='Reuse an exact existing checkpoint; hashes must match')
    p.add_argument('--local-processor', type=Path, help='Reuse an exact existing Cosmos snapshot; hashes must match')
    a = p.parse_args()
    materialize(json.loads(a.manifest.read_text()), a.output, local_model=a.local_model,
                local_processor=a.local_processor, condition=a.condition)


if __name__ == '__main__':
    main()
