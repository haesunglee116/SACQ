"""Experimental packed-codebook GEMM for GR00T.

The stock deployment shim reconstructs a complete BF16 weight and then calls
``F.linear``.  These opt-in backends keep the canonical packed artifact and
decode only the weight tile currently consumed by the GEMM.  The validated v2
and v3 entry points are lazily dispatched by the shim; the default runtime is
still the stock reconstruction path.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def packed_vector_codebook_linear(x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    """Launch vector-granular packed GEMM using buffers from the stock shim."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("Triton packed-vector backend requires CUDA BF16 input")
    if x.shape[-1] != module.in_features:
        raise ValueError(f"expected input width {module.in_features}, got {x.shape[-1]}")
    if module.group_size // module.vector_dim != 16:
        raise ValueError("vector kernel currently requires 16 vectors per scale group")
    x2 = x.reshape(-1, module.in_features).contiguous()
    m = x2.shape[0]
    y = torch.empty((m, module.out_features), dtype=x.dtype, device=x.device)
    block_m = 16 if m <= 16 else (32 if m <= 64 else 64)
    block_n, block_v, warps = 32, 16, 4
    grid = (triton.cdiv(m, block_m), triton.cdiv(module.out_features, block_n))
    bias_ptr = module.bias if module.bias is not None else y
    _packed_vector_codebook_gemm[grid](
        x2,
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        bias_ptr,
        y,
        M=m,
        N=module.out_features,
        K=module.in_features,
        NUM_GROUPS=module.in_features // module.group_size,
        GROUP_SIZE=module.group_size,
        VECTOR_DIM=module.vector_dim,
        INDEX_BITS=module.index_bits,
        INDEX_WINDOW_BYTES=module.index_window_bytes,
        HAS_BIAS=module.bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_V=block_v,
        num_warps=warps,
    )
    return y.reshape(*x.shape[:-1], module.out_features)


def packed_interleaved_codebook_linear(
    x: torch.Tensor, module: torch.nn.Module
) -> torch.Tensor:
    """Launch the v3 interleaved packed GEMM using stock-shim buffers."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("Triton interleaved backend requires CUDA BF16 input")
    if module.vector_dim != 4:
        raise ValueError("interleaved backend currently requires vector_dim=4")
    x2 = x.reshape(-1, module.in_features).contiguous()
    m = x2.shape[0]
    y = torch.empty((m, module.out_features), dtype=x.dtype, device=x.device)
    if m <= 16:
        block_m, block_n = 16, 32
        block_v = 16 if module.out_features > 2 * module.in_features else 32
    else:
        block_m, block_n = 64, 32
        block_v = 16 if module.out_features > 2 * module.in_features else 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(module.out_features, block_n))
    bias_ptr = module.bias if module.bias is not None else y
    _packed_interleaved_codebook_gemm[grid](
        x2,
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        bias_ptr,
        y,
        M=m,
        N=module.out_features,
        K=module.in_features,
        NUM_GROUPS=module.in_features // module.group_size,
        GROUP_SIZE=module.group_size,
        VECTOR_DIM=module.vector_dim,
        INDEX_BITS=module.index_bits,
        INDEX_WINDOW_BYTES=module.index_window_bytes,
        HAS_BIAS=module.bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_V=block_v,
        num_warps=4,
    )
    return y.reshape(*x.shape[:-1], module.out_features)


def packed_interleaved_splitk_codebook_linear(
    x: torch.Tensor,
    module: torch.nn.Module,
    *,
    split_k: int = 2,
    block_m: int | None = None,
    block_n: int = 32,
    block_v: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """V3 with parallel contiguous K partitions and an FP32 final reduction.

    This retains canonical bit-packed storage. ``partials`` is activation-sized
    temporary workspace, not a persistent unpacked-index or dense-weight cache.
    The experiment targets small-M contraction layers whose N-grid alone does
    not expose enough blocks to occupy the A6000.
    """
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("Triton split-K backend requires CUDA BF16 input")
    if module.vector_dim != 4:
        raise ValueError("split-K interleaved backend currently requires vector_dim=4")
    if split_k not in (2, 4, 8):
        raise ValueError(f"split_k must be 2, 4, or 8, got {split_k}")
    if x.shape[-1] != module.in_features:
        raise ValueError(f"expected input width {module.in_features}, got {x.shape[-1]}")
    x2 = x.reshape(-1, module.in_features).contiguous()
    m = x2.shape[0]
    if block_m is None:
        block_m = 16 if m <= 16 else 64
    partials = torch.empty(
        (split_k, m, module.out_features), dtype=torch.float32, device=x.device
    )
    y = torch.empty((m, module.out_features), dtype=x.dtype, device=x.device)
    grid = (
        triton.cdiv(m, block_m),
        triton.cdiv(module.out_features, block_n),
        split_k,
    )
    _packed_interleaved_codebook_gemm_splitk[grid](
        x2,
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        partials,
        M=m,
        N=module.out_features,
        K=module.in_features,
        NUM_GROUPS=module.in_features // module.group_size,
        GROUP_SIZE=module.group_size,
        VECTOR_DIM=module.vector_dim,
        INDEX_BITS=module.index_bits,
        INDEX_WINDOW_BYTES=module.index_window_bytes,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    finalize_grid = (triton.cdiv(m, block_m), triton.cdiv(module.out_features, block_n))
    bias_ptr = module.bias if module.bias is not None else y
    _finalize_interleaved_splitk[finalize_grid](
        partials,
        bias_ptr,
        y,
        M=m,
        N=module.out_features,
        SPLIT_K=split_k,
        HAS_BIAS=module.bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y.reshape(*x.shape[:-1], module.out_features)


def packed_interleaved_splitk_dispatch(
    x: torch.Tensor, module: torch.nn.Module
) -> torch.Tensor:
    """V4 hybrid: split only under-occupied small-M contraction GEMMs."""
    m = x.numel() // x.shape[-1]
    if m <= 64 and module.in_features >= 4 * module.out_features:
        # M=1 gets little benefit from four partitions (and 6-bit can regress),
        # while the observed DiT M=41 contraction consistently prefers four.
        split_k = 2 if m <= 16 else 4
        return packed_interleaved_splitk_codebook_linear(
            x, module, split_k=split_k, block_m=16 if m <= 16 else 64
        )
    return packed_interleaved_codebook_linear(x, module)


def dequantize_packed_codebook_weight(module: torch.nn.Module) -> torch.Tensor:
    """Reconstruct the exact BF16 weight with one vector-granular GPU kernel."""
    if module.vector_dim != 4:
        raise ValueError("fused dequant currently requires vector_dim=4")
    weight = torch.empty(
        (module.out_features, module.in_features),
        dtype=torch.bfloat16,
        device=module.codebook.device,
    )
    fused_config = os.environ.get("CODEBOOK_FUSED_DEQUANT_CONFIG", "").strip()
    if fused_config:
        values = tuple(
            int(value) for value in fused_config.replace("_", ",").split(",")
        )
        if len(values) != 3:
            raise ValueError(
                "CODEBOOK_FUSED_DEQUANT_CONFIG must be BLOCK_N,BLOCK_V,WARPS"
            )
        block_n, block_v, warps = values
    else:
        # BN16/BV32 was consistently best across 4/5/6-bit 1536x6144 and
        # 6144x1536 VLM-size matrices on A6000 after contiguous d4 stores.
        block_n, block_v, warps = 16, 32, 4
    grid = (
        triton.cdiv(module.out_features, block_n),
        triton.cdiv(module.in_features // module.vector_dim, block_v),
    )
    _dequantize_packed_codebook_weight[grid](
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        weight,
        N=module.out_features,
        K=module.in_features,
        NUM_GROUPS=module.in_features // module.group_size,
        GROUP_SIZE=module.group_size,
        VECTOR_DIM=module.vector_dim,
        INDEX_BITS=module.index_bits,
        INDEX_WINDOW_BYTES=module.index_window_bytes,
        BLOCK_N=block_n,
        BLOCK_V=block_v,
        num_warps=warps,
    )
    return weight


def packed_fused_dequant_codebook_linear(
    x: torch.Tensor, module: torch.nn.Module
) -> torch.Tensor:
    """Fused packed reconstruction followed by the stock cuBLAS linear."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("fused packed dequant backend requires CUDA BF16 input")
    weight = dequantize_packed_codebook_weight(module)
    bias = None if module.bias is None else module.bias.to(dtype=x.dtype)
    return F.linear(x, weight, bias)


@triton.jit
def _packed_codebook_gemm(
    x_ptr,
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    index_mask: tl.constexpr = (1 << INDEX_BITS) - 1

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        group_ids = offs_k // GROUP_SIZE
        in_group = offs_k % GROUP_SIZE
        vector_ids = in_group // VECTOR_DIM
        components = in_group % VECTOR_DIM
        linear_index_ids = (
            (offs_n[:, None] * NUM_GROUPS + group_ids[None, :])
            * vectors_per_group
            + vector_ids[None, :]
        )
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8

        words = tl.load(packed_ptr + byte_ids, mask=valid_nk, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            upper = tl.load(packed_ptr + byte_ids + 1, mask=valid_nk, other=0).to(tl.int32)
            words = words | (upper << 8)
        if INDEX_WINDOW_BYTES >= 3:
            upper2 = tl.load(packed_ptr + byte_ids + 2, mask=valid_nk, other=0).to(tl.int32)
            words = words | (upper2 << 16)
        if INDEX_WINDOW_BYTES >= 4:
            upper3 = tl.load(packed_ptr + byte_ids + 3, mask=valid_nk, other=0).to(tl.int32)
            words = words | (upper3 << 24)
        indices = (words >> bit_offsets) & index_mask

        q = tl.load(
            codebook_ptr + indices * VECTOR_DIM + components[None, :],
            mask=valid_nk,
            other=0.0,
        )
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nk,
            other=0.0,
        )
        # Match the shim's in-place BF16 multiply and add rounding before GEMM.
        weight = (q * scale).to(tl.bfloat16)
        shift = tl.load(shift_ptr)
        weight = (weight + shift).to(tl.bfloat16)

        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight))

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _compact_index_codebook_gemm(
    x_ptr,
    indices_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        group_ids = offs_k // GROUP_SIZE
        in_group = offs_k % GROUP_SIZE
        vector_ids = in_group // VECTOR_DIM
        components = in_group % VECTOR_DIM
        linear_index_ids = (
            (offs_n[:, None] * NUM_GROUPS + group_ids[None, :])
            * vectors_per_group
            + vector_ids[None, :]
        )
        indices = tl.load(indices_ptr + linear_index_ids, mask=valid_nk, other=0).to(tl.int32)
        q = tl.load(
            codebook_ptr + indices * VECTOR_DIM + components[None, :],
            mask=valid_nk,
            other=0.0,
        )
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nk,
            other=0.0,
        )
        weight = (q * scale).to(tl.bfloat16)
        weight = (weight + tl.load(shift_ptr)).to(tl.bfloat16)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight))

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _packed_vector_codebook_gemm(
    x_ptr,
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Vector-granular variant: decode one index for all VECTOR_DIM lanes."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    total_vectors: tl.constexpr = K // VECTOR_DIM
    index_mask: tl.constexpr = (1 << INDEX_BITS) - 1

    for v0 in range(0, total_vectors, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
        linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8
        words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            words = words | (
                tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
            )
        if INDEX_WINDOW_BYTES >= 3:
            words = words | (
                tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
            )
        if INDEX_WINDOW_BYTES >= 4:
            words = words | (
                tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
            )
        indices = (words >> bit_offsets) & index_mask
        group_ids = offs_v // vectors_per_group
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nv,
            other=0.0,
        )
        shift = tl.load(shift_ptr)
        for component in range(0, VECTOR_DIM):
            q = tl.load(
                codebook_ptr + indices * VECTOR_DIM + component,
                mask=valid_nv,
                other=0.0,
            )
            weight = (q * scale).to(tl.bfloat16)
            weight = (weight + shift).to(tl.bfloat16)
            offs_k = offs_v * VECTOR_DIM + component
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(weight))

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _packed_interleaved_codebook_gemm(
    x_ptr,
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Decode once/vector, interleave d4 components, issue one tensor-core dot."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    total_vectors: tl.constexpr = K // VECTOR_DIM
    index_mask: tl.constexpr = (1 << INDEX_BITS) - 1

    for v0 in range(0, total_vectors, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
        linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8
        words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            words = words | (
                tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
            )
        if INDEX_WINDOW_BYTES >= 3:
            words = words | (
                tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
            )
        if INDEX_WINDOW_BYTES >= 4:
            words = words | (
                tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
            )
        indices = (words >> bit_offsets) & index_mask
        group_ids = offs_v // vectors_per_group
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nv,
            other=0.0,
        )
        shift = tl.load(shift_ptr)
        q0 = tl.load(codebook_ptr + indices * VECTOR_DIM, mask=valid_nv, other=0.0)
        q1 = tl.load(codebook_ptr + indices * VECTOR_DIM + 1, mask=valid_nv, other=0.0)
        q2 = tl.load(codebook_ptr + indices * VECTOR_DIM + 2, mask=valid_nv, other=0.0)
        q3 = tl.load(codebook_ptr + indices * VECTOR_DIM + 3, mask=valid_nv, other=0.0)
        w0 = ((q0 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w1 = ((q1 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w2 = ((q2 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w3 = ((q3 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        # Interleaving (0,2) with (1,3) yields c0,c1,c2,c3 for each vector.
        w02 = tl.interleave(w0, w2)
        w13 = tl.interleave(w1, w3)
        weight = tl.interleave(w02, w13)
        offs_k = v0 * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(weight))

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _packed_interleaved_codebook_gemm_splitk(
    x_ptr,
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    partials_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Contiguous split-K form of the v3 packed interleaved GEMM."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    total_vectors: tl.constexpr = K // VECTOR_DIM
    total_v_blocks: tl.constexpr = (total_vectors + BLOCK_V - 1) // BLOCK_V
    blocks_per_split: tl.constexpr = (total_v_blocks + SPLIT_K - 1) // SPLIT_K
    index_mask: tl.constexpr = (1 << INDEX_BITS) - 1

    for block_offset in range(0, blocks_per_split):
        v0 = (pid_k * blocks_per_split + block_offset) * BLOCK_V
        offs_v = v0 + tl.arange(0, BLOCK_V)
        valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
        linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8
        words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            words = words | (
                tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
            )
        if INDEX_WINDOW_BYTES >= 3:
            words = words | (
                tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
            )
        if INDEX_WINDOW_BYTES >= 4:
            words = words | (
                tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
            )
        indices = (words >> bit_offsets) & index_mask
        group_ids = offs_v // vectors_per_group
        scale = tl.load(
            scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
            mask=valid_nv,
            other=0.0,
        )
        shift = tl.load(shift_ptr)
        q0 = tl.load(codebook_ptr + indices * VECTOR_DIM, mask=valid_nv, other=0.0)
        q1 = tl.load(codebook_ptr + indices * VECTOR_DIM + 1, mask=valid_nv, other=0.0)
        q2 = tl.load(codebook_ptr + indices * VECTOR_DIM + 2, mask=valid_nv, other=0.0)
        q3 = tl.load(codebook_ptr + indices * VECTOR_DIM + 3, mask=valid_nv, other=0.0)
        w0 = ((q0 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w1 = ((q1 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w2 = ((q2 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        w3 = ((q3 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
        weight = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
        offs_k = v0 * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
        x_values = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x_values, tl.trans(weight))

    partial_offsets = (
        pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    )
    tl.store(
        partials_ptr + partial_offsets,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _finalize_interleaved_splitk(
    partials_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offsets = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        acc += tl.load(
            partials_ptr + split_id * M * N + offsets,
            mask=valid,
            other=0.0,
        )
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(y_ptr + offsets, acc, mask=valid)


@triton.jit
def _dequantize_packed_codebook_weight(
    packed_ptr,
    codebook_ptr,
    scales_ptr,
    shift_ptr,
    weight_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Decode one packed index and store all d4 BF16 weight components."""
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    total_vectors: tl.constexpr = K // VECTOR_DIM
    vectors_per_group: tl.constexpr = GROUP_SIZE // VECTOR_DIM
    valid_nv = (offs_n[:, None] < N) & (offs_v[None, :] < total_vectors)
    linear_index_ids = offs_n[:, None] * total_vectors + offs_v[None, :]
    bit_positions = linear_index_ids * INDEX_BITS
    byte_ids = bit_positions // 8
    bit_offsets = bit_positions % 8
    words = tl.load(packed_ptr + byte_ids, mask=valid_nv, other=0).to(tl.int32)
    if INDEX_WINDOW_BYTES >= 2:
        words = words | (
            tl.load(packed_ptr + byte_ids + 1, mask=valid_nv, other=0).to(tl.int32) << 8
        )
    if INDEX_WINDOW_BYTES >= 3:
        words = words | (
            tl.load(packed_ptr + byte_ids + 2, mask=valid_nv, other=0).to(tl.int32) << 16
        )
    if INDEX_WINDOW_BYTES >= 4:
        words = words | (
            tl.load(packed_ptr + byte_ids + 3, mask=valid_nv, other=0).to(tl.int32) << 24
        )
    indices = (words >> bit_offsets) & ((1 << INDEX_BITS) - 1)
    group_ids = offs_v // vectors_per_group
    scale = tl.load(
        scales_ptr + offs_n[:, None] * NUM_GROUPS + group_ids[None, :],
        mask=valid_nv,
        other=0.0,
    )
    shift = tl.load(shift_ptr)
    q0 = tl.load(codebook_ptr + indices * VECTOR_DIM, mask=valid_nv, other=0.0)
    q1 = tl.load(codebook_ptr + indices * VECTOR_DIM + 1, mask=valid_nv, other=0.0)
    q2 = tl.load(codebook_ptr + indices * VECTOR_DIM + 2, mask=valid_nv, other=0.0)
    q3 = tl.load(codebook_ptr + indices * VECTOR_DIM + 3, mask=valid_nv, other=0.0)
    w0 = ((q0 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
    w1 = ((q1 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
    w2 = ((q2 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
    w3 = ((q3 * scale).to(tl.bfloat16) + shift).to(tl.bfloat16)
    values = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    offs_k = pid_v * BLOCK_V * VECTOR_DIM + tl.arange(0, BLOCK_V * VECTOR_DIM)
    valid_nk = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        values,
        mask=valid_nk,
    )


@triton.jit
def _build_activation_codebook_lut(
    x_ptr,
    codebook_ptr,
    lut_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    TOTAL_VECTORS: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    VECTOR_DIM: tl.constexpr,
    BLOCK_MV: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Build L[m,v,c] = dot(x[m,v,:], codebook[c,:])."""
    pid_mv = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_mv = pid_mv * BLOCK_MV + tl.arange(0, BLOCK_MV)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    valid_mv = offs_mv < M * TOTAL_VECTORS
    valid_c = offs_c < CODEBOOK_SIZE
    offs_m = offs_mv // TOTAL_VECTORS
    offs_v = offs_mv % TOTAL_VECTORS
    acc = tl.zeros((BLOCK_MV, BLOCK_C), dtype=tl.float32)
    for component in range(0, VECTOR_DIM):
        x = tl.load(
            x_ptr + offs_m * K + offs_v * VECTOR_DIM + component,
            mask=valid_mv,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            codebook_ptr + offs_c * VECTOR_DIM + component,
            mask=valid_c,
            other=0.0,
        ).to(tl.float32)
        acc += x[:, None] * q[None, :]
    tl.store(
        lut_ptr + offs_mv[:, None] * CODEBOOK_SIZE + offs_c[None, :],
        acc.to(tl.bfloat16),
        mask=valid_mv[:, None] & valid_c[None, :],
    )


@triton.jit
def _packed_lut_codebook_linear(
    lut_ptr,
    x_sum_ptr,
    packed_ptr,
    scales_ptr,
    shift_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TOTAL_VECTORS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    VECTORS_PER_GROUP: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    INDEX_BITS: tl.constexpr,
    INDEX_WINDOW_BYTES: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Gather activation LUT entries using packed indices and accumulate."""
    pid_m = tl.program_id(0)
    offs_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_m = offs_m < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    index_mask: tl.constexpr = (1 << INDEX_BITS) - 1

    for v0 in range(0, TOTAL_VECTORS, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        valid_v = offs_v < TOTAL_VECTORS
        linear_index_ids = offs_n * TOTAL_VECTORS + offs_v
        bit_positions = linear_index_ids * INDEX_BITS
        byte_ids = bit_positions // 8
        bit_offsets = bit_positions % 8
        words = tl.load(packed_ptr + byte_ids, mask=valid_v, other=0).to(tl.int32)
        if INDEX_WINDOW_BYTES >= 2:
            words = words | (
                tl.load(packed_ptr + byte_ids + 1, mask=valid_v, other=0).to(tl.int32) << 8
            )
        if INDEX_WINDOW_BYTES >= 3:
            words = words | (
                tl.load(packed_ptr + byte_ids + 2, mask=valid_v, other=0).to(tl.int32) << 16
            )
        if INDEX_WINDOW_BYTES >= 4:
            words = words | (
                tl.load(packed_ptr + byte_ids + 3, mask=valid_v, other=0).to(tl.int32) << 24
            )
        indices = (words >> bit_offsets) & index_mask
        lut_offsets = (
            (offs_m[:, None] * TOTAL_VECTORS + offs_v[None, :]) * CODEBOOK_SIZE
            + indices[None, :]
        )
        valid_mv = valid_m[:, None] & valid_v[None, :]
        values = tl.load(lut_ptr + lut_offsets, mask=valid_mv, other=0.0).to(tl.float32)
        group_ids = offs_v // VECTORS_PER_GROUP
        scales = tl.load(
            scales_ptr + offs_n * NUM_GROUPS + group_ids,
            mask=valid_v,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(values * scales[None, :], axis=1)

    # The stock reconstruction adds the same scalar shift to every weight.
    # Pulling this term outside the reduction avoids repeating it per output.
    if HAS_SHIFT:
        acc += tl.load(shift_ptr).to(tl.float32) * tl.load(
            x_sum_ptr + offs_m, mask=valid_m, other=0.0
        )
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)
    tl.store(y_ptr + offs_m * N + offs_n, acc, mask=valid_m)


def _unpack_compact(payload: dict[str, Any]) -> torch.Tensor:
    """Decode canonical packed indices into the smallest addressable dtype."""
    shape = tuple(int(x) for x in payload["index_shape"])
    bits = int(payload["index_bits"])
    count = math.prod(shape)
    source = torch.as_tensor(payload["packed_indices"], dtype=torch.uint8).reshape(-1)
    max_offset = 8 - math.gcd(bits, 8)
    window_bytes = (bits + max_offset + 7) // 8
    source = F.pad(source, (0, window_bytes - 1))
    positions = torch.arange(count, dtype=torch.int64) * bits
    byte_ids = positions // 8
    offsets = positions % 8
    words = source[byte_ids].to(torch.int64)
    for byte_offset in range(1, window_bytes):
        words.bitwise_or_(source[byte_ids + byte_offset].to(torch.int64) << (8 * byte_offset))
    words.bitwise_right_shift_(offsets)
    words.bitwise_and_((1 << bits) - 1)
    codebook_size = int(torch.as_tensor(payload["codebook"]).shape[0])
    dtype = torch.uint8 if codebook_size <= 256 else torch.int16
    return words.reshape(shape).to(dtype=dtype).contiguous()


class PackedCodebookTritonLinear(torch.nn.Module):
    """Direct packed-codebook linear; BF16 CUDA inference prototype."""

    def __init__(self, payload: dict[str, Any], bias: torch.Tensor | None, device: str):
        super().__init__()
        self.in_features = int(payload["in_features"])
        self.out_features = int(payload["out_features"])
        self.group_size = int(payload["group_size"])
        self.vector_dim = int(payload["vector_dim"])
        self.index_bits = int(payload["index_bits"])
        self.index_shape = tuple(int(x) for x in payload["index_shape"])
        if self.group_size % self.vector_dim:
            raise ValueError("group_size must be divisible by vector_dim")
        expected_shape = (
            self.out_features,
            self.in_features // self.group_size,
            self.group_size // self.vector_dim,
        )
        if self.index_shape != expected_shape:
            raise ValueError(f"unsupported index shape {self.index_shape}; expected {expected_shape}")
        device_obj = torch.device(device)
        packed = torch.as_tensor(payload["packed_indices"], dtype=torch.uint8)
        max_offset = 8 - math.gcd(self.index_bits, 8)
        self.index_window_bytes = (self.index_bits + max_offset + 7) // 8
        guarded = torch.zeros(
            packed.numel() + self.index_window_bytes - 1,
            dtype=torch.uint8,
            device=device_obj,
        )
        guarded[: packed.numel()].copy_(packed)
        self.register_buffer("packed_indices", guarded)
        self.register_buffer(
            "codebook", torch.as_tensor(payload["codebook"], dtype=torch.bfloat16, device=device_obj)
        )
        self.register_buffer(
            "scales", torch.as_tensor(payload["scales"], dtype=torch.bfloat16, device=device_obj)
        )
        shift_source = torch.as_tensor(payload.get("shift", 0.0))
        self.shift_is_zero = not bool(torch.any(shift_source != 0).item())
        self.register_buffer(
            "shift", shift_source.to(dtype=torch.bfloat16, device=device_obj)
        )
        if self.shift.numel() != 1:
            raise ValueError(f"prototype requires scalar shift, got {tuple(self.shift.shape)}")
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(device=device_obj, dtype=torch.bfloat16))
        config = os.environ.get("CODEBOOK_TRITON_CONFIG", "").strip()
        self.forced_config = None
        if config:
            # Underscores are convenient with Slurm's comma-delimited
            # ``--export`` syntax; accept both spellings.
            values = tuple(int(value) for value in config.replace("_", ",").split(","))
            if len(values) != 4:
                raise ValueError("CODEBOOK_TRITON_CONFIG must be BLOCK_M,BLOCK_N,BLOCK_K,WARPS")
            self.forced_config = values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("PackedCodebookTritonLinear currently requires CUDA BF16 input")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        x2 = x.reshape(-1, self.in_features).contiguous()
        m = x2.shape[0]
        y = torch.empty((m, self.out_features), dtype=x.dtype, device=x.device)
        if self.forced_config is not None:
            block_m, block_n, block_k, warps = self.forced_config
        elif m <= 16:
            block_m, block_n, block_k, warps = 16, 32, 32, 4
        elif m <= 64:
            block_m, block_n, block_k, warps = 32, 32, 32, 4
        else:
            block_m, block_n, block_k, warps = 64, 32, 32, 4
        grid = (triton.cdiv(m, block_m), triton.cdiv(self.out_features, block_n))
        bias_ptr = self.bias if self.bias is not None else y
        _packed_codebook_gemm[grid](
            x2,
            self.packed_indices,
            self.codebook,
            self.scales,
            self.shift,
            bias_ptr,
            y,
            M=m,
            N=self.out_features,
            K=self.in_features,
            NUM_GROUPS=self.in_features // self.group_size,
            GROUP_SIZE=self.group_size,
            VECTOR_DIM=self.vector_dim,
            INDEX_BITS=self.index_bits,
            INDEX_WINDOW_BYTES=self.index_window_bytes,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


class CompactIndexCodebookTritonLinear(PackedCodebookTritonLinear):
    """Direct GEMM with uint8/int16 runtime indices and no dense weight.

    This is a deliberate speed/memory point between canonical bit packing and
    int32 pre-unpacking.  K<=256 layers use one byte per 4-weight vector; the
    uncommon larger codebooks use two bytes.
    """

    def __init__(self, payload: dict[str, Any], bias: torch.Tensor | None, device: str):
        compact = _unpack_compact(payload)
        super().__init__(payload, bias=bias, device=device)
        del self.packed_indices
        self.register_buffer("indices", compact.to(device=torch.device(device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("CompactIndexCodebookTritonLinear currently requires CUDA BF16 input")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        x2 = x.reshape(-1, self.in_features).contiguous()
        m = x2.shape[0]
        y = torch.empty((m, self.out_features), dtype=x.dtype, device=x.device)
        if self.forced_config is not None:
            block_m, block_n, block_k, warps = self.forced_config
        elif m <= 16:
            block_m, block_n, block_k, warps = 16, 32, 32, 4
        elif m <= 64:
            block_m, block_n, block_k, warps = 32, 32, 32, 4
        else:
            block_m, block_n, block_k, warps = 64, 32, 32, 4
        grid = (triton.cdiv(m, block_m), triton.cdiv(self.out_features, block_n))
        bias_ptr = self.bias if self.bias is not None else y
        _compact_index_codebook_gemm[grid](
            x2,
            self.indices,
            self.codebook,
            self.scales,
            self.shift,
            bias_ptr,
            y,
            M=m,
            N=self.out_features,
            K=self.in_features,
            NUM_GROUPS=self.in_features // self.group_size,
            GROUP_SIZE=self.group_size,
            VECTOR_DIM=self.vector_dim,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


class PackedVectorCodebookTritonLinear(PackedCodebookTritonLinear):
    """Packed direct GEMM that decodes once per vector instead of per scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("PackedVectorCodebookTritonLinear currently requires CUDA BF16 input")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        x2 = x.reshape(-1, self.in_features).contiguous()
        m = x2.shape[0]
        y = torch.empty((m, self.out_features), dtype=x.dtype, device=x.device)
        # BLOCK_V=16 matches the artifact's 64/4 vectors per scale group.
        # Keep this invariant in the prototype so each vector tile shares a
        # single group scale without crossing a group boundary.
        if self.group_size // self.vector_dim != 16:
            raise ValueError("vector kernel currently requires 16 vectors per scale group")
        vector_config = os.environ.get("CODEBOOK_VECTOR_CONFIG", "").strip()
        if vector_config:
            values = tuple(
                int(value) for value in vector_config.replace("_", ",").split(",")
            )
            if len(values) != 4:
                raise ValueError("CODEBOOK_VECTOR_CONFIG must be BLOCK_M,BLOCK_N,BLOCK_V,WARPS")
            block_m, block_n, block_v, warps = values
        else:
            block_m = 16 if m <= 16 else (32 if m <= 64 else 64)
            block_n, block_v, warps = 32, 16, 4
        grid = (triton.cdiv(m, block_m), triton.cdiv(self.out_features, block_n))
        bias_ptr = self.bias if self.bias is not None else y
        _packed_vector_codebook_gemm[grid](
            x2,
            self.packed_indices,
            self.codebook,
            self.scales,
            self.shift,
            bias_ptr,
            y,
            M=m,
            N=self.out_features,
            K=self.in_features,
            NUM_GROUPS=self.in_features // self.group_size,
            GROUP_SIZE=self.group_size,
            VECTOR_DIM=self.vector_dim,
            INDEX_BITS=self.index_bits,
            INDEX_WINDOW_BYTES=self.index_window_bytes,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_V=block_v,
            num_warps=warps,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


class PackedInterleavedCodebookTritonLinear(PackedCodebookTritonLinear):
    """v3: vector-granular decode with one interleaved tensor-core dot."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("PackedInterleavedCodebookTritonLinear requires CUDA BF16 input")
        if self.vector_dim != 4:
            raise ValueError("interleaved kernel currently requires vector_dim=4")
        x2 = x.reshape(-1, self.in_features).contiguous()
        m = x2.shape[0]
        y = torch.empty((m, self.out_features), dtype=x.dtype, device=x.device)
        config = os.environ.get("CODEBOOK_INTERLEAVED_CONFIG", "").strip()
        if config:
            values = tuple(int(value) for value in config.replace("_", ",").split(","))
            if len(values) != 4:
                raise ValueError(
                    "CODEBOOK_INTERLEAVED_CONFIG must be BLOCK_M,BLOCK_N,BLOCK_V,WARPS"
                )
            block_m, block_n, block_v, warps = values
        else:
            block_m = 16 if m <= 16 else 32
            block_n, block_v, warps = 32, 16, 4
        grid = (triton.cdiv(m, block_m), triton.cdiv(self.out_features, block_n))
        bias_ptr = self.bias if self.bias is not None else y
        _packed_interleaved_codebook_gemm[grid](
            x2,
            self.packed_indices,
            self.codebook,
            self.scales,
            self.shift,
            bias_ptr,
            y,
            M=m,
            N=self.out_features,
            K=self.in_features,
            NUM_GROUPS=self.in_features // self.group_size,
            GROUP_SIZE=self.group_size,
            VECTOR_DIM=self.vector_dim,
            INDEX_BITS=self.index_bits,
            INDEX_WINDOW_BYTES=self.index_window_bytes,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_V=block_v,
            num_warps=warps,
        )
        return y.reshape(*x.shape[:-1], self.out_features)


class PackedLUTCodebookTritonLinear(PackedCodebookTritonLinear):
    """Low-K LUT-GEMM while retaining canonical bit-packed indices.

    The temporary LUT is activation-dependent and is released after each
    forward. No expanded indices or dense weights are stored persistently.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("PackedLUTCodebookTritonLinear currently requires CUDA BF16 input")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        codebook_size = int(self.codebook.shape[0])
        if codebook_size not in (16, 32, 64):
            raise ValueError(f"LUT prototype supports K=16/32/64, got {codebook_size}")
        vectors_per_group = self.group_size // self.vector_dim
        if vectors_per_group != 16:
            raise ValueError("LUT prototype currently requires 16 vectors per scale group")

        x2 = x.reshape(-1, self.in_features).contiguous()
        m = int(x2.shape[0])
        total_vectors = self.in_features // self.vector_dim
        lut = torch.empty(
            (m, total_vectors, codebook_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        block_mv = 64
        block_c = 16 if codebook_size == 16 else 32
        lut_grid = (
            triton.cdiv(m * total_vectors, block_mv),
            triton.cdiv(codebook_size, block_c),
        )
        _build_activation_codebook_lut[lut_grid](
            x2,
            self.codebook,
            lut,
            M=m,
            K=self.in_features,
            TOTAL_VECTORS=total_vectors,
            CODEBOOK_SIZE=codebook_size,
            VECTOR_DIM=self.vector_dim,
            BLOCK_MV=block_mv,
            BLOCK_C=block_c,
            num_warps=4,
        )
        y = torch.empty((m, self.out_features), dtype=x.dtype, device=x.device)
        x_sum = x2.float().sum(dim=1) if not self.shift_is_zero else y
        block_m, block_v = (16 if m <= 16 else 32), 16
        output_grid = (triton.cdiv(m, block_m), self.out_features)
        bias_ptr = self.bias if self.bias is not None else y
        _packed_lut_codebook_linear[output_grid](
            lut,
            x_sum,
            self.packed_indices,
            self.scales,
            self.shift,
            bias_ptr,
            y,
            M=m,
            N=self.out_features,
            TOTAL_VECTORS=total_vectors,
            NUM_GROUPS=self.in_features // self.group_size,
            VECTORS_PER_GROUP=vectors_per_group,
            CODEBOOK_SIZE=codebook_size,
            INDEX_BITS=self.index_bits,
            INDEX_WINDOW_BYTES=self.index_window_bytes,
            HAS_SHIFT=not self.shift_is_zero,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_V=block_v,
            num_warps=4,
        )
        return y.reshape(*x.shape[:-1], self.out_features)
