#!/usr/bin/env python3
"""Shared fast-path helpers for proxy scripts.

1) local_ckpt(): use the node-local /tmp checkpoint copy the eval jobs
   maintain (rule: weights load from node-local disk, not lustre); falls back
   to the lustre path if the node copy is absent/incomplete.
2) bf16 reference cache: the bf16 reference outputs are deterministic given
   (request set, limit, seed base), so compute them once and reuse --
   removes a full policy load + 64 evals (~10 min) from every proxy job."""
from __future__ import annotations

import json
from pathlib import Path

import torch

PROJ = Path("@WORKSPACE@/libero_pro_eval")
CACHE = PROJ / "quantization_analysis/bf16_refs_cache"


def local_ckpt(suite: str = "libero_spatial") -> str:
    lustre = PROJ / f"models-gr00t-n17-libero/{suite}"
    local = Path(f"@WORKSPACE@/scratch/models-gr00t-n17-libero/{suite}")
    try:
        shared_index = lustre / "model.safetensors.index.json"
        local_index = local / "model.safetensors.index.json"
        # Existence alone is not a sufficient cache key: a stale node-local
        # checkpoint can have the same filenames and silently change every
        # calibration result.  The small index must match byte-for-byte, and
        # every referenced shard must match the immutable source size+mtime.
        if local_index.read_bytes() != shared_index.read_bytes():
            raise RuntimeError("node-local checkpoint index differs from shared source")
        idx = json.loads(shared_index.read_text())["weight_map"]
        files = set(idx.values())
        fresh = all(
            (local / f).exists()
            and (local / f).stat().st_size == (lustre / f).stat().st_size
            and (local / f).stat().st_mtime_ns == (lustre / f).stat().st_mtime_ns
            for f in files
        )
        if fresh:
            print(f"[fast] node-local checkpoint: {local}", flush=True)
            return str(local)
        print(f"[fast] stale node-local checkpoint; using lustre: {lustre}", flush=True)
    except Exception as exc:
        print(f"[fast] node-local checkpoint rejected ({exc}); using lustre: {lustre}", flush=True)
    print(f"[fast] node-local checkpoint unavailable; using lustre: {lustre}", flush=True)
    return str(lustre)


def refs_cache_path(set_name: str, limit: int = 32, seed_base: int = 7000) -> Path:
    return CACHE / f"refs_{set_name}_limit{limit}_seed{seed_base}.pt"


def load_refs(set_name: str, limit: int = 32, seed_base: int = 7000) -> dict | None:
    p = refs_cache_path(set_name, limit, seed_base)
    if p.exists():
        print(f"[fast] bf16 refs cache hit: {p.name}", flush=True)
        return torch.load(p, map_location="cpu")
    return None


def save_refs(refs: dict, set_name: str, limit: int = 32, seed_base: int = 7000) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(refs, refs_cache_path(set_name, limit, seed_base))
