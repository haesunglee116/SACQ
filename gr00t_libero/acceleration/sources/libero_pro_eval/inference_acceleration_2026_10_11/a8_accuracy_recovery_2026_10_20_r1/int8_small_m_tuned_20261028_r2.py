"""Shape-specific exact INT8 GEMM selection; baseline retained when faster."""
import json
import statistics
from pathlib import Path
import torch
import triton
from int8_small_m_gemm_20261028_r1 import _int8_scaled_gemm

# Includes every previous candidate, plus pipeline and warp variants.
_BASE = [(32,64,128),(64,64,128),(32,128,128),(64,128,128),(32,64,64),(32,128,64)]
CONFIGS = [(*tile,4,3) for tile in _BASE]
CONFIGS += [(bm,bn,bk,warps,stages) for bm,bn,bk in [(64,64,128),(32,64,64),(64,128,128)]
            for warps,stages in [(4,2),(4,4),(8,3),(8,4)]]
CONFIGS += [(16,64,128,4,3),(16,128,128,4,3),(64,64,256,4,3),(64,128,64,4,3)]


def enable(report_path):
    import int8_torch_epilogue_runtime_r2 as runtime
    previous = runtime.linear
    selected, records = {}, []

    def run(a, w, sa, sw, bias, m, cfg):
        n, k = w.shape
        bm, bn, bk, warps, stages = cfg
        y = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        _int8_scaled_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a, w, sa, sw, bias if bias is not None else y, y,
            m, n, k, bias is not None, bm, bn, bk,
            num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        return y

    def timing(fn):
        for _ in range(5):
            y = fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        values = []
        for _ in range(20):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return statistics.median(values)

    def linear(a, w, sa, sw, bias, fma, actual_rows=None):
        m = a.shape[0] if actual_rows is None else actual_rows
        if m > 160:
            return previous(a, w, sa, sw, bias, fma, actual_rows=actual_rows)
        n, k = w.shape
        assert fma and a.dtype == w.dtype == torch.int8
        assert a.is_contiguous() and w.is_contiguous()
        assert k * 127 * 127 <= torch.iinfo(torch.int32).max
        key = (m, n, k, bias is not None)
        if key not in selected:
            assert not torch.cuda.is_current_stream_capturing()
            reference = previous(a, w, sa, sw, bias, fma, actual_rows=m)
            timings = []
            for cfg in CONFIGS:
                y = run(a, w, sa, sw, bias, m, cfg)
                assert torch.equal(y.view(torch.int16), reference.view(torch.int16)), ('TunedINT8Output', key, cfg)
                del y
                timings.append((timing(lambda: run(a, w, sa, sw, bias, m, cfg)), cfg))
            baseline = timing(lambda: previous(a, w, sa, sw, bias, fma, actual_rows=m))
            best_ms, cfg = min(timings)
            selected[key] = cfg if best_ms < baseline else None
            records.append(dict(shape=key, selected=selected[key], baseline_ms=baseline,
                                candidates=[dict(median_ms=t, tile=c) for t, c in timings],
                                all_candidate_bytes_exact=True))
            p = Path(report_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix('.tmp')
            tmp.write_text(json.dumps(records, indent=2))
            tmp.replace(p)
            print('INT8_GEMM_TILE', records[-1], flush=True)
            del reference
        cfg = selected[key]
        if cfg is None:
            return previous(a, w, sa, sw, bias, fma, actual_rows=m)
        return run(a, w, sa, sw, bias, m, cfg)

    runtime.linear = linear
