"""Replay immutable exact-gated refined-decode launch geometry."""
import hashlib
import json
from pathlib import Path
import torch
import triton
import int8_aligned_index_decoder_20261028_r1 as decoder

def install(selection_ref):
    raw=Path(selection_ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==selection_ref['sha256']
    allowed={(4,64,4),(4,128,4),(8,64,4),(8,128,4),(16,32,4),(16,64,4),(16,128,4),(32,32,4),(32,64,4),(32,128,8)}
    choices={}
    for row in json.loads(raw):
        assert row['all_candidate_weight_bytes_exact']
        key=tuple(row['shape']);assert key not in choices
        cfg=None if row['chosen'] is None else tuple(row['chosen'])
        assert cfg is None or cfg in allowed
        choices[key]=cfg
    original=decoder.decode
    def decode(m,mode):
        assert mode in (0,6)
        from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
        if mode==0 or type(m) is not Packed:return original(m,mode)
        key=(m.out_features,m.in_features,m.group_size,m.index_bits,m.index_window_bytes)
        assert key in choices,('UnknownFrozenDecodeShape',key)
        cfg=choices[key]
        if cfg is None:return original(m,mode)
        n,k=m.out_features,m.in_features;sw=m._a8_perrow_weight_scale
        out=torch.empty((n,k),device=sw.device,dtype=torch.int8)
        bn,bv,warps=cfg
        decoder._decode_codebook_perrow_int8[(triton.cdiv(n,bn),triton.cdiv(k//4,bv))](m.packed_indices,m.codebook,m.scales,m.shift,out,sw,n,k,k//m.group_size,m.group_size,4,m.index_bits,m.index_window_bytes,mode,bn,bv,num_warps=warps,enable_fp_fusion=False)
        return out
    decoder.decode=decode
