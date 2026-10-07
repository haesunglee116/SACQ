import os, json, hashlib
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def save(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2)); t.replace(p)

