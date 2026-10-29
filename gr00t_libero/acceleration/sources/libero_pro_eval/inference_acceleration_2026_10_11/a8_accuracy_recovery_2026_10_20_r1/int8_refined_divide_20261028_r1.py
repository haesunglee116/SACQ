"""Fixed-weight candidate only; every encoded weight is byte-gated."""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from perrow_int8_exact_producers_r1 import _divide as original_divide

@triton.jit
def _divide(x, s, MODE:tl.constexpr):
    if MODE == 6:
        reciprocal = libdevice.rcp_rn(s)
        quotient = x * reciprocal
        residual = tl.fma(-quotient, s, x)
        return tl.fma(residual, reciprocal, quotient)
    else:
        return original_divide(x, s, MODE)
