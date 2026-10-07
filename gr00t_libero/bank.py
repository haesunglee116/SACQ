"""Explicit fitted-bank lookup, independent of historical directory names."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


class PackedBank:
    def __init__(self, manifest):
        if manifest.get('schema')!='sacq_packed_bank_v1':
            raise ValueError('Unsupported fitted bank schema')
        rows=manifest.get('artifacts')
        if not isinstance(rows,list) or not rows:
            raise ValueError('Explicit nonempty artifact inventory required')
        self.rows={}
        self.verified={}
        for item in rows:
            row=dict(item)
            if row.get('scope') not in ('vlm','dit') or row.get('init') not in ('aware','vanilla'):
                raise ValueError('Explicit scope and initialization required')
            if type(row.get('k')) is not int or row['k']<=0:
                raise ValueError('Explicit positive integer codebook size required')
            for key in ('layer','suite','path','sha256'):
                if not isinstance(row.get(key),str) or not row[key]:
                    raise ValueError(f'Artifact requires {key}')
            if row['suite']=='shared' and row['scope']!='vlm':
                raise ValueError('Only VLM artifacts may be explicitly shared')
            if not Path(row['path']).is_absolute():
                raise ValueError('Artifact paths must be absolute')
            row['path']=str(Path(row['path']).resolve())
            if len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):
                raise ValueError('Artifact requires a SHA256 hex digest')
            identity=(row['scope'],row['layer'],row['k'],row['suite'],row['init'])
            if identity in self.rows:
                raise ValueError('Duplicate fitted candidate identity')
            self.rows[identity]=row
        for scope,layer,k,suite,init in self.rows:
            if suite!='shared' and (scope,layer,k,'shared',init) in self.rows:
                raise ValueError('Ambiguous shared and suite-specific fitted candidate')

    @classmethod
    def from_ref(cls, reference):
        path=Path(reference['path'])
        if not path.is_absolute():
            raise ValueError('Bank manifest path must be absolute')
        if digest(path)!=reference['sha256']:
            raise ValueError('Bank manifest identity mismatch')
        return cls(json.loads(path.read_text()))

    def find_packed(self, kind, name, k, suite='libero_spatial', init='aware', verify=True, exact=False):
        """Match the historical registry API, without directory fallback.

        `verify` retains the original optional codebook-header check. Checksums
        are checked even when it is false, once per unchanged file in a process.
        `exact` is retained for caller compatibility; all manifest lookups are exact.
        """
        key=(kind,name,k,suite,init)
        row=self.rows.get(key)
        if row is None and kind=='vlm':
            row=self.rows.get((kind,name,k,'shared',init))
        if row is None:
            raise FileNotFoundError(f'No explicitly registered fitted candidate: {key}')
        path=Path(row['path']);stat=path.stat()
        stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        previous=self.verified.get(str(path))
        if previous!=(stamp,row['sha256']):
            if digest(path)!=row['sha256']:
                raise ValueError(f'Fitted artifact identity mismatch: {path}')
            self.verified[str(path)]=(stamp,row['sha256'])
        if verify:
            import torch
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            packed=payload.get('packed') if isinstance(payload,dict) else None
            if not isinstance(packed,dict) or 'codebook' not in packed or packed['codebook'].shape[0]!=k:
                raise ValueError(f'Fitted codebook size differs from manifest: {path}')
        return path


def build_inventory(spec, *, check_payloads=False):
    """Expand declared fitter output groups, never discover scientific scope."""
    if spec.get('schema')!='sacq_fitted_inventory_build_v1':
        raise ValueError('Unsupported fitted inventory build schema')
    groups=spec.get('groups')
    if not isinstance(groups,list) or not groups:
        raise ValueError('Explicit nonempty fitting groups required')
    rows=[]
    for group in groups:
        layers=group.get('layers')
        if not isinstance(layers,list) or not layers:
            raise ValueError('Each fitting group requires an explicit layer list')
        directory=Path(group['directory'])
        if not directory.is_absolute():
            raise ValueError('Fitting output directory must be absolute')
        for layer in layers:
            if not isinstance(layer,str) or not layer or '/' in layer or '\\' in layer or layer in ('.','..'):
                raise ValueError('Layer must be an exact module name, not a file path')
            suffix='.pt' if group['scope']=='vlm' else '__ck.pt'
            path=(directory/(layer+suffix)).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Fitted candidate path escapes output directory')
            rows.append(dict(scope=group['scope'],suite=group['suite'],init=group['init'],
                             k=group['k'],layer=layer,path=str(path),sha256='0'*64))
    # Validate identities before reading or deserializing candidate files.
    PackedBank(dict(schema='sacq_packed_bank_v1',artifacts=rows))
    for row in rows:
        row['sha256']=digest(row['path'])
    result=dict(schema='sacq_packed_bank_v1',artifacts=rows)
    bank=PackedBank(result)
    if check_payloads:
        for scope,layer,k,suite,init in bank.rows:
            bank.find_packed(scope,layer,k,suite=suite,init=init,verify=True)
    return result


def write_inventory(spec_path, output_dir, *, check_payloads=False):
    spec_path=Path(spec_path).resolve()
    source=spec_path.read_bytes()
    result=build_inventory(json.loads(source),check_payloads=check_payloads)
    out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=out/'bank.json'
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    inputs=[dict(path=str(spec_path),sha256=hashlib.sha256(source).hexdigest()),
            dict(path=str(manifest),sha256=digest(manifest))]
    inputs.extend(dict(path=r['path'],sha256=r['sha256']) for r in result['artifacts'])
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    return dict(status='INVENTORY_PREPARED_NOT_SCIENTIFIC_APPROVAL',artifacts=len(result['artifacts']),
                bank_manifest=inputs[1],payload_headers_checked=check_payloads)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest',type=Path)
    mode.add_argument('--build-spec',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-payloads',action='store_true',help='Deserialize trusted packed files and check K')
    args=parser.parse_args()
    if args.build_spec:
        if not args.output_dir:parser.error('--build-spec requires --output-dir')
        print(json.dumps(write_inventory(args.build_spec,args.output_dir,check_payloads=args.check_payloads),indent=2))
        return
    if args.output_dir:parser.error('--output-dir is only valid with --build-spec')
    path=args.manifest.resolve()
    bank=PackedBank.from_ref(dict(path=str(path),sha256=digest(path)))
    for scope,layer,k,suite,init in bank.rows:
        bank.find_packed(scope,layer,k,suite=suite,init=init,verify=args.check_payloads,exact=True)
    print(json.dumps(dict(status='CHECKSUMS_VERIFIED',artifacts=len(bank.rows),payload_headers_checked=args.check_payloads)))


if __name__=='__main__':
    main()
