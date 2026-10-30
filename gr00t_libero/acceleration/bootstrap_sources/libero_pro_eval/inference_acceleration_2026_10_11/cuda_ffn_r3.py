"""Lazy loader for the v2 CuTe SM80 packed-codebook GEMM (shared-memory codebook).

Identical to cuda_codebook_cutlass_2026_08_23.py except that it compiles the
kernel_v2 sources into a separately named extension and build directory, so the
validated v1 build is never overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


SOURCE_DIR = Path(__file__).with_name("kernel_ffn_r3")
_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        cutlass_root = Path(
            os.environ.get("CUTLASS_ROOT", "@LOCAL_RUNTIME@/cutlass")
        )
        if not (cutlass_root / "include/cute/tensor.hpp").is_file():
            raise RuntimeError(
                f"CUTLASS headers not found at {cutlass_root}; "
                "set CUTLASS_ROOT to an official NVIDIA CUTLASS checkout"
            )
        build_root = Path(
            os.environ.get(
                "GR00T_CODEBOOK_FFN_R3_BUILD_DIR",
                "@LOCAL_RUNTIME@/codebook_ffn_extension_r3",
            )
        )
        build_root.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(
            name="gr00t_codebook_ffn_r3",
            sources=[
                str(SOURCE_DIR / "codebook_cutlass_binding.cpp"),
                str(SOURCE_DIR / "codebook_cutlass_kernel.cu"),
            ],
            extra_include_paths=[str(cutlass_root / "include")],
            build_directory=str(build_root),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "--ptxas-options=-v",
                os.environ.get(
                    "GR00T_CODEBOOK_CUTLASS_GENCODE",
                    "-gencode=arch=compute_86,code=sm_86"),
            ],
            verbose=os.environ.get("GR00T_CODEBOOK_CUTLASS_VERBOSE", "0") == "1",
        )
    return _EXTENSION


def packed_cutlass_codebook_linear(
    x: torch.Tensor, module: torch.nn.Module
) -> torch.Tensor:
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("CuTe backend requires CUDA BF16 input")
    if module.vector_dim != 4 or module.group_size != 64:
        raise ValueError("CuTe v1 requires vector_dim=4 and group_size=64")
    x2 = x.reshape(-1, module.in_features).contiguous()
    bias = (
        module.bias
        if module.bias is not None
        else torch.empty(0, device=x.device, dtype=torch.bfloat16)
    )
    extension = load_extension()
    entrypoint = (
        extension.codebook_gemm_scalarstore
        if os.environ.get("GR00T_CODEBOOK_CUTLASS_STORE_MODE", "vector")
        == "scalar"
        else extension.codebook_gemm
    )
    output = entrypoint(
        x2,
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        bias,
        module.index_bits,
        module.index_window_bytes,
    )
    return output.reshape(*x.shape[:-1], module.out_features)


def packed_cutlass_scalarstore_codebook_linear(
    x: torch.Tensor, module: torch.nn.Module
) -> torch.Tensor:
    """Pre-vectorization scalar shared-store control."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise TypeError("CuTe backend requires CUDA BF16 input")
    if module.vector_dim != 4 or module.group_size != 64:
        raise ValueError("CuTe v1 requires vector_dim=4 and group_size=64")
    x2 = x.reshape(-1, module.in_features).contiguous()
    bias = (
        module.bias
        if module.bias is not None
        else torch.empty(0, device=x.device, dtype=torch.bfloat16)
    )
    output = load_extension().codebook_gemm_scalarstore(
        x2,
        module.packed_indices,
        module.codebook,
        module.scales,
        module.shift,
        bias,
        module.index_bits,
        module.index_window_bytes,
    )
    return output.reshape(*x.shape[:-1], module.out_features)
