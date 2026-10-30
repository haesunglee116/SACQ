#!/usr/bin/env python3
"""One-pass HQQ 4-bit dequantisation, replacing HQQ's three-pass PyTorch path.

`Quantizer.dequantize` for a 4-bit tensor does:

    W_r = unpack_4bit_u8(W_q, dtype=compute_dtype)   # allocates and writes [2P, G]
    W_r = ((W_r - zero) * scale).reshape(shape)      # two more full passes

Each step reads and writes the whole dequantised weight, so a 730M-parameter
vision tower moves roughly 8.8 GB per action; measurement put the cost at
10.14 ms, 82% of the VLM stage's gap against BF16. Doing it in one pass moves
1.83 GB instead.

Layout, taken from the deployed meta (nbits=4, group_size=64, axis=1,
packing='4bit_u8'): the quantiser reshapes the weight to [-1, group_size] and
stores one scale and one zero per row, then packs two rows into one uint8 byte --
row r in the high nibble, row r+P in the low nibble. This kernel reproduces the
same indexing.

Arithmetic is kept in bfloat16 with an explicit round after each operation, so
the result is bit-identical to the PyTorch path rather than merely close; the
installer verifies that on every patched module and refuses the substitution for
any module whose layout does not match exactly.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hqq4_dequant_kernel(wq_ptr, scale_ptr, zero_ptr, out_ptr,
                         P, G, packed_total, BLOCK: tl.constexpr):
    """One pass over the packed bytes, two outputs per byte.

    HQQ's 4-bit layout puts row r in a byte's high nibble and row r+P in its low
    nibble, so a byte carries one element of each half of the output. Iterating
    over the *packed* tensor therefore reads every byte once and writes both
    halves, instead of iterating over the output and reading each byte twice.
    The two stores are contiguous within their halves, so both coalesce.

    Scale and zero are per row, shared by all G columns, so within a program the
    address repeats every G lanes and the loads collapse to a few transactions.

    Arithmetic stays in bfloat16 with an explicit round after each step, matching
    `((W_r - zero) * scale)` exactly as the PyTorch path computes it.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < packed_total
    row = offs // G

    byte = tl.load(wq_ptr + offs, mask=mask, other=0).to(tl.uint32)
    hi = ((byte >> 4) & 0x0F).to(tl.bfloat16)
    lo = (byte & 0x0F).to(tl.bfloat16)

    z_hi = tl.load(zero_ptr + row, mask=mask, other=0.0).to(tl.bfloat16)
    s_hi = tl.load(scale_ptr + row, mask=mask, other=0.0).to(tl.bfloat16)
    z_lo = tl.load(zero_ptr + row + P, mask=mask, other=0.0).to(tl.bfloat16)
    s_lo = tl.load(scale_ptr + row + P, mask=mask, other=0.0).to(tl.bfloat16)

    out_hi = ((hi - z_hi).to(tl.bfloat16) * s_hi).to(tl.bfloat16)
    out_lo = ((lo - z_lo).to(tl.bfloat16) * s_lo).to(tl.bfloat16)

    tl.store(out_ptr + offs, out_hi, mask=mask)
    tl.store(out_ptr + packed_total + offs, out_lo, mask=mask)


def _fused_dequantize(module) -> torch.Tensor:
    meta = module.meta
    wq = module.W_q
    if meta.get("view_as_float"):
        wq = wq.view(meta["unpack_view_dtype"])
    P, G = wq.shape
    packed_total = P * G
    out = torch.empty(2 * packed_total, dtype=torch.bfloat16, device=wq.device)
    scale = meta.get("_aq26_scale")
    if scale is None:
        scale = meta["scale"].reshape(-1).contiguous()
        zero = meta["zero"].reshape(-1).contiguous()
        meta["_aq26_scale"], meta["_aq26_zero"] = scale, zero
    else:
        zero = meta["_aq26_zero"]
    BLOCK = 2048
    _hqq4_dequant_kernel[(triton.cdiv(packed_total, BLOCK),)](
        wq, scale, zero, out, P, G, packed_total, BLOCK=BLOCK, num_warps=8)
    return out.view(2 * P, G).reshape(meta["shape"])


def install(model, verbose: bool = True) -> dict:
    """Patch every 4-bit HQQ linear whose layout matches, after verifying it."""
    from hqq.core.quantize import HQQLinear

    state = {"patched": 0, "skipped": 0, "mismatch": [], "max_rel": 0.0}
    for name, m in model.named_modules():
        if not isinstance(m, HQQLinear):
            continue
        meta = m.meta
        ok = (meta.get("nbits") == 4 and meta.get("packing") == "4bit_u8"
              and meta.get("axis") == 1
              and meta.get("compute_dtype", torch.bfloat16) == torch.bfloat16)
        if not ok:
            state["skipped"] += 1
            continue
        try:
            reference = m.dequantize()
            candidate = _fused_dequantize(m)
            if candidate.shape != reference.shape:
                raise ValueError(f"shape {candidate.shape} vs {reference.shape}")
            if not torch.equal(candidate, reference):
                d = (candidate.float() - reference.float())
                rel = float(d.norm() / reference.float().norm().clamp_min(1e-12))
                state["max_rel"] = max(state["max_rel"], rel)
                state["mismatch"].append((name, rel))
                state["skipped"] += 1
                continue
        except Exception as exc:
            state["mismatch"].append((name, f"{type(exc).__name__}: {exc}"))
            state["skipped"] += 1
            continue
        m.dequantize = (lambda mod: (lambda: _fused_dequantize(mod)))(m)
        state["patched"] += 1
    if verbose:
        print(f"[hqqf] fused dequant installed on {state['patched']} modules, "
              f"{state['skipped']} skipped, {len(state['mismatch'])} mismatched",
              flush=True)
        for entry in state["mismatch"][:3]:
            print(f"[hqqf]   mismatch {entry}", flush=True)
    return state
