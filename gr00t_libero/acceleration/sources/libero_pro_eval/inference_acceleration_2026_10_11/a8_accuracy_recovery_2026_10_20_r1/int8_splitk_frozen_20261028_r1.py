"""Replay exact-gated INT32 split-K selections with frozen baseline fallback."""
import json,hashlib
from pathlib import Path
from int8_splitk_exact_20261028_r1 import compute

def enable(selection):
    raw=Path(selection['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==selection['sha256']
    rows=json.loads(raw);choices={}
    for row in rows:
        assert row['all_candidate_bytes_exact']
        cfg=None if row['selected'] is None else tuple(row['selected'])
        assert cfg is None or cfg in [(2,32,64,128),(4,32,64,128),(2,64,64,128),(4,64,64,128)]
        key=tuple(row['shape']);assert key not in choices;choices[key]=cfg
    import int8_torch_epilogue_runtime_r2 as runtime
    previous=runtime.linear
    def linear(a,w,sa,sw,bias,fma,actual_rows=None):
        m=a.shape[0] if actual_rows is None else actual_rows
        if m>160 or w.shape[1]<4096 or w.shape[0]>2048:return previous(a,w,sa,sw,bias,fma,actual_rows=actual_rows)
        key=(m,w.shape[0],w.shape[1],bias is not None);assert key in choices
        cfg=choices[key]
        if cfg is None:return previous(a,w,sa,sw,bias,fma,actual_rows=actual_rows)
        assert fma
        return compute(a,w,sa,sw,bias,m,cfg)
    runtime.linear=linear
