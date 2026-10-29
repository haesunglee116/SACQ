"""Diagnostic A8 adapters; existing compressed payloads remain immutable.

Dynamic per-tensor absmax RTN. No dense weight cache and no silent BF16
fallback for a requested real 8-bit GEMM. The fake paths are error probes,
not speed implementations. Callers must supply an explicitly approved scope.
"""
from dataclasses import dataclass
import types

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Quantizer:
    format: str

    def __post_init__(self):
        if self.format not in ("fp8_e4m3fn", "int8_symmetric"):
            raise ValueError(self.format)

    def quantize(self, x):
        # Compute in FP32 so absmax and scaling do not overflow in BF16.
        z = x.float()
        limit = 448.0 if self.format == "fp8_e4m3fn" else 127.0
        maximum = z.abs().amax()
        # Zero input must use a positive scale; do not invent a clipping epsilon.
        scale = torch.where(maximum > 0, maximum / limit, torch.ones_like(maximum))
        normalized = (z / scale).clamp(-limit, limit)
        if self.format == "fp8_e4m3fn":
            q = normalized.to(torch.float8_e4m3fn)
        else:
            q = normalized.round().to(torch.int8)
        return q, scale

    def fake(self, x):
        q, scale = self.quantize(x)
        return (q.float() * scale).to(x.dtype)


def real_linear(x, weight, bias, quantizer):
    """No retained FP8/INT8 weights. Padding is numerical zero, not a sample.

    cuBLAS low-precision alignment constraints are handled explicitly. Padded
    tokens are cropped before returning and never reach subsequent layers.
    FP8 uses FP32 accumulation; INT8 uses INT32 then FP32 rescaling.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("This experiment requires BF16 source operands")
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("Matching CUDA operands required")
    k = weight.shape[1]
    n = weight.shape[0]
    if x.shape[-1] != k:
        raise ValueError("Linear shape mismatch")
    a = x.reshape(-1, k).contiguous()
    m = a.shape[0]
    aq, sa = quantizer.quantize(a)
    wq, sw = quantizer.quantize(weight.contiguous())
    # INT8 GEMM on CUDA imposes a stricter M alignment than FP8. The INT8
    # path pads M to at least32; FP8 keeps16 alignment. All padding is zero.
    mp, np, kp = ((v + 15) // 16 * 16 for v in (m, n, k))
    if quantizer.format == "int8_symmetric":
        # PyTorch 2.7 _int_mm requires M > 16, including timestep/AdaLN M=1.
        mp = max(32, mp)
    if (mp, kp) != (m, k):
        padded = torch.zeros((mp, kp), dtype=aq.dtype, device=aq.device)
        padded[:m, :k].copy_(aq)
        aq = padded
    if (np, kp) != (n, k):
        padded = torch.zeros((np, kp), dtype=wq.dtype, device=wq.device)
        padded[:n, :k].copy_(wq)
        wq = padded
    # A is row-major, B is column-major for _scaled_mm. Transposing contiguous
    # NxK weights yields the required strides without another weight copy.
    if quantizer.format == "fp8_e4m3fn":
        y = torch._scaled_mm(aq, wq.t(), sa, sw,
                             out_dtype=torch.float32, use_fast_accum=False)
        # Older supported PyTorch builds returned (output, amax).
        if isinstance(y, tuple):
            y = y[0]
    else:
        if kp * 127 * 127 > torch.iinfo(torch.int32).max:
            raise OverflowError("INT8 accumulation may overflow INT32")
        y = torch._int_mm(aq, wq.t()).float() * (sa * sw)
    y = y[:m, :n]
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], n)


def evaluate_linear(x, weight, bias, format, mode):
    quantizer = Quantizer(format)
    if mode == "fake_activation":
        return F.linear(quantizer.fake(x), weight, bias)
    if mode == "fake_operands":
        return F.linear(quantizer.fake(x), quantizer.fake(weight), bias)
    if mode == "real_gemm":
        return real_linear(x, weight, bias, quantizer)
    raise ValueError(mode)


def install(model, names, format, mode):
    """Replace exactly the requested codebook Linear inputs/operations.

    Patch both ordinary forward and the optimized decoder dispatch before
    inference caches/graphs are constructed. Restore before any new installer;
    never mutate a graph after capture.
    """
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    import decoder_tuning as dec

    if len(names) != len(set(names)) or not names:
        raise ValueError("Explicit nonempty unique scope required")
    layers = {name: model.get_submodule(name) for name in names}
    if any(type(m) is not Packed for m in layers.values()):
        raise TypeError("Initial A8 contract covers codebook Linear only")
    quantizer = Quantizer(format)
    if mode not in ("fake_activation", "fake_operands", "real_gemm"):
        raise ValueError(mode)
    originals = {name: m.forward for name, m in layers.items()}
    selected = {id(m) for m in layers.values()}
    old_dispatch = dec.linear

    def linear(x, m):
        if id(m) not in selected:
            return old_dispatch(x, m)
        # Exact existing BF16 decoder: A8 does not change SACQ reconstruction.
        weight = dec.dequantize(m)
        return evaluate_linear(x, weight, m.bias, quantizer.format, mode)

    for m in layers.values():
        def forward(self, x):
            return linear(x, self)
        m.forward = types.MethodType(forward, m)
    dec.linear = linear

    def restore():
        dec.linear = old_dispatch
        for name, m in layers.items():
            m.forward = originals[name]

    receipt = dict(format=format, mode=mode, names=sorted(names),
                   activation_scale="dynamic_per_tensor_absmax",
                   weight_scale="per_tensor_absmax" if mode != "fake_activation" else None,
                   scale_dtype="float32", zero_input_scale=1.0,
                   extra_clipping=False, dense_weight_cache=False,
                   real_gemm=mode == "real_gemm",
                   accumulation="fp32" if format == "fp8_e4m3fn" else "int32_then_fp32",
                   memory_limit_enforced=False)
    return receipt, restore
