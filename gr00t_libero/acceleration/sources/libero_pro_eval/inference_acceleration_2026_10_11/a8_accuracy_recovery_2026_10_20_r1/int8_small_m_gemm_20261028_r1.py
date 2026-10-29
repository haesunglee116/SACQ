"""Small-M INT8 GEMM with native-order scaling and fused BF16 epilogue."""
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_scaled_gemm(A, W, SA, SW, BIAS, Y,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(triton.cdiv(K, BK)):
        kk = block * BK + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + rn[None, :] * K + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    sw = tl.load(SW + rn, rn < N, 1.0)
    sa = tl.load(SA + rm, rm < M, 1.0)
    scaled = acc.to(tl.float32) * sw[None, :]
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, rn < N, 0).to(tl.float32)
        value = tl.fma(sa[:, None], scaled, bias[None, :])
    else:
        value = sa[:, None] * scaled
    tl.store(Y + rm[:, None] * N + rn[None, :], value.to(tl.bfloat16),
             (rm[:, None] < M) & (rn[None, :] < N))


def enable():
    import int8_torch_epilogue_runtime_r2 as runtime
    previous = runtime.linear

    def linear(a, w, sa, sw, bias, fma, actual_rows=None):
        m = a.shape[0] if actual_rows is None else actual_rows
        if m > 160:
            return previous(a, w, sa, sw, bias, fma, actual_rows=actual_rows)
        assert fma and a.dtype == w.dtype == torch.int8
        assert a.is_contiguous() and w.is_contiguous()
        n, k = w.shape
        assert k * 127 * 127 <= torch.iinfo(torch.int32).max
        y = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        _int8_scaled_gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            a, w, sa, sw, bias if bias is not None else y, y,
            m, n, k, bias is not None, 32, 64, 128,
            num_warps=4, num_stages=3, enable_fp_fusion=False)
        return y

    runtime.linear = linear
