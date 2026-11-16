"""Verified suite-specific GPTQ scalar candidates, shared by search and refinement."""
import hashlib
import json
from pathlib import Path

BITS = {80000: 3, 90000: 4}  # Historical CSV encoding; not a quantizer name.

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

class ScalarBank:
    def __init__(self, reference):
        p = Path(reference['path'])
        if not p.is_absolute() or digest(p) != reference['sha256']:
            raise ValueError('GPTQ bank manifest hash mismatch')
        self.reference = dict(reference)
        data = json.loads(p.read_text())
        if data.get('schema') != 'sacq_gptq_scalar_bank_v1':
            raise ValueError('Explicit GPTQ scalar bank required')
        self.rows = {}
        for row in data['artifacts']:
            key = (row['suite'], row['layer'], row['bits'])
            if key in self.rows or row['bits'] not in (3, 4):
                raise ValueError('Duplicate or unsupported scalar candidate')
            if row.get('method') != 'GPTQ' or not Path(row['path']).is_absolute():
                raise ValueError('Scalar candidate must be an explicit GPTQ artifact')
            self.rows[key] = row
        if not self.rows:
            raise ValueError('Empty GPTQ bank')
        self.checked = {}

    def entry(self, suite, layer, bits):
        row = self.rows[(suite, layer, bits)]
        p = Path(row['path']); stat = p.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.checked.get(str(p)) != identity:
            if digest(p) != row['sha256']:
                raise ValueError('GPTQ candidate bytes changed')
            self.checked[str(p)] = identity
        return row

    def weight(self, suite, layer, bits, shape):
        import torch
        row = self.entry(suite, layer, bits)
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
        w = value['weight']
        if (value['layer'] != layer or value['bits'] != bits or
                tuple(w.shape) != tuple(shape) or not torch.isfinite(w).all()):
            raise ValueError('GPTQ candidate tensor identity mismatch')
        if w.dtype != torch.bfloat16:
            raise ValueError('Candidate must retain original GPTQ BF16 output')
        return w.contiguous()

    def preflight(self, suites, names):
        for suite in suites:
            for name in names:
                for bits in (3, 4):
                    self.entry(suite, name, bits)

    def check_selected(self, suite, allocation, background):
        byname = {r['layer']: r for r in background}
        for row in allocation:
            k = int(row['k'])
            if k not in BITS:
                continue
            expected = self.entry(suite, row['layer'], BITS[k])
            actual = byname.get(row['layer'], {})
            if actual.get('sha256') != expected['sha256'] or actual.get('bits') != BITS[k]:
                raise ValueError('Downstream scalar must reuse the exact allocation GPTQ artifact')
