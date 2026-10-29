"""Immutable warp choices for the original exact per-row activation kernel."""
import hashlib
import json
from pathlib import Path
import torch
import triton
import perrow_int8_padded_producer_r1 as original

def enable(selection_ref):
    raw=Path(selection_ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==selection_ref['sha256']
    choices={}
    for row in json.loads(raw):
        assert row['all_candidate_bytes_and_scales_exact']
        key=tuple(row['shape']);assert key not in choices
        warps=row['selected_warps'];assert warps in (2,4,8,16)
        choices[key]=warps
    def activation(x,mode):
        assert x.ndim==2 and x.dtype==torch.bfloat16 and x.is_contiguous() and mode==0
        key=tuple(x.shape);assert key in choices,('UnknownFrozenActivationShape',key)
        m,k=key;mp=max(32,triton.cdiv(m,16)*16)
        q=torch.empty((mp,k),device=x.device,dtype=torch.int8)
        s=torch.empty((mp,1),device=x.device,dtype=torch.float32)
        original._activation[(mp,)](x,q,s,m,k,triton.next_power_of_2(k),mode,num_warps=choices[key],enable_fp_fusion=False)
        return q,s
    original.activation=activation
