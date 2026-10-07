"""Validate a locally built source archive by unpacking and staging its sources.

This executes the archive's own release tools; use only trusted local builds.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='sacq-source-check-') as temporary:
        root=Path(temporary)
        with tarfile.open(args.archive) as archive:
            for member in archive.getmembers():
                target=(root/member.name).resolve()
                if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected source-archive entry')
            archive.extractall(root,filter='data')
        entries=list(root.iterdir())
        if len(entries)!=1 or not entries[0].is_dir():
            raise ValueError('Expected one source archive root')
        source=entries[0]
        subprocess.run([sys.executable,str(source/'tools/release_check.py'),
                        '--root',str(source)],check=True,cwd=source)
        staged=root/'staged'
        result=subprocess.run([sys.executable,str(source/'gr00t_libero/pipeline.py'),
                               'prepare','--workspace',str(staged)],check=True,cwd=source,
                              capture_output=True,text=True)
        receipt=json.loads((staged/'SOURCE_RECEIPT.json').read_text())
        print(json.dumps(dict(archive=args.archive.name,status='PASS',
                              staged_sources=len(receipt['sources']))))


if __name__=='__main__':
    main()
