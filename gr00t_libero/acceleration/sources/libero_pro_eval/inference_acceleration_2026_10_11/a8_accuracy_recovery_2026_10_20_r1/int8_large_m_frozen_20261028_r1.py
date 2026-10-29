"""Replay immutable, byte-gated launch choices; no tuning during model setup."""
import hashlib
import json
from pathlib import Path
import torch
import triton
from int8_small_m_gemm_20261028_r1 import _int8_scaled_gemm


def enable(selection_ref):
    import int8_torch_epilogue_runtime_r2 as runtime
    previous = runtime.linear
    raw = Path(selection_ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == selection_ref['sha256']
    records = json.loads(raw)
    choices = {}
    from int8_large_m_tuned_20261028_r1 import CONFIGS
    allowed = set(CONFIGS)
    for row in records:
        assert row['all_candidate_bytes_exact']
        key = tuple(row['shape'])
        assert key not in choices
        cfg = None if row['selected'] is None else tuple(row['selected'])
        assert cfg is None or cfg in allowed
        choices[key] = cfg

    def linear(a, w, sa, sw, bias, fma, actual_rows=None):
        m = a.shape[0] if actual_rows is None else actual_rows
        if m <= 160:
            return previous(a, w, sa, sw, bias, fma, actual_rows=actual_rows)
        n, k = w.shape
        key = (m, n, k, bias is not None)
        assert key in choices, ('UnknownFrozenINT8Shape', key)
        cfg = choices[key]
        if cfg is None:
            return previous(a, w, sa, sw, bias, fma, actual_rows=m)
        assert fma and a.dtype == w.dtype == torch.int8
        assert a.is_contiguous() and w.is_contiguous()
        assert k * 127 * 127 <= torch.iinfo(torch.int32).max
        bm, bn, bk, warps, stages = cfg
        y = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        _int8_scaled_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a, w, sa, sw, bias if bias is not None else y, y,
            m, n, k, bias is not None, bm, bn, bk,
            num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        return y

    runtime.linear = linear
