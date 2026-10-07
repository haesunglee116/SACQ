#!/usr/bin/env python3
"""Run centroid calibration with an exact, node-local prepared-batch cache.

The calibration trainer historically preprocesses every request independently
on every DDP rank.  That is numerically harmless but can turn 640 msgpack
reads/preprocess calls into a severe shared-filesystem bottleneck.  This
launcher monkey-patches only the imported ``load_batches`` symbol:

* rank 0 executes the original loader once;
* the result is atomically serialized to node-local storage;
* the serialized round trip is checked leaf-by-leaf for exact equality; and
* all ranks load the same prepared CPU objects.

No optimizer, request ordering, RNG, quantizer, or model code is changed.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import train_centroid_ft_2026_08_05 as trainer


CACHE_SCHEMA = "centroid_prepared_batches_exact_v1"
ORIGINAL_LOAD_BATCHES = trainer.load_batches


def _request_manifest(request_dir: Path, limit: int) -> tuple[list[str], str]:
    files = sorted(request_dir.glob("*.msgpack"))[:limit]
    rows = []
    for path in files:
        resolved = path.resolve()
        stat = resolved.stat()
        rows.append(f"{resolved}\t{stat.st_size}\t{stat.st_mtime_ns}")
    signature = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return rows, signature


def _assert_exact(left: Any, right: Any, where: str = "root") -> None:
    if type(left) is not type(right):
        raise RuntimeError(
            f"prepared-batch cache type mismatch at {where}: "
            f"{type(left)!r} != {type(right)!r}"
        )
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise RuntimeError(f"prepared-batch tensor mismatch at {where}")
        return
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise RuntimeError(f"prepared-batch ndarray mismatch at {where}")
        return
    if isinstance(left, Mapping):
        if list(left.keys()) != list(right.keys()):
            raise RuntimeError(f"prepared-batch mapping keys mismatch at {where}")
        for key in left:
            _assert_exact(left[key], right[key], f"{where}[{key!r}]")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        if len(left) != len(right):
            raise RuntimeError(f"prepared-batch sequence length mismatch at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_exact(a, b, f"{where}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"prepared-batch value mismatch at {where}: {left!r} != {right!r}")


def _cache_aware_load_batches(policy: Any, request_dir: Path, limit: int) -> list[tuple[str, Any, Any]]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    rows, signature = _request_manifest(request_dir, limit)
    cache_root = Path(
        os.environ.get(
            "CENTROID_BATCH_CACHE_ROOT",
            "@WORKSPACE@/scratch/centroid_prepared_batches_exact_v1",
        )
    )
    cache_path = cache_root / f"{signature}.pt"

    if rank == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
        payload = None
        if cache_path.exists():
            candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                candidate.get("schema") == CACHE_SCHEMA
                and candidate.get("request_manifest") == rows
                and len(candidate.get("batches", ())) == len(rows)
            ):
                payload = candidate
                print(
                    f"[batch-cache] exact cache hit path={cache_path} "
                    f"requests={len(rows)}",
                    flush=True,
                )
        if payload is None:
            original = ORIGINAL_LOAD_BATCHES(policy, request_dir, limit)
            payload = {
                "schema": CACHE_SCHEMA,
                "request_manifest": rows,
                "request_signature": signature,
                "batches": original,
            }
            temporary = cache_path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(payload, temporary)
            reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
            _assert_exact(payload, reloaded)
            os.replace(temporary, cache_path)
            print(
                f"[batch-cache] built and exhaustive exact round-trip PASS "
                f"path={cache_path} requests={len(rows)}",
                flush=True,
            )

    if dist.is_initialized():
        dist.barrier()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != CACHE_SCHEMA
        or payload.get("request_manifest") != rows
        or payload.get("request_signature") != signature
        or len(payload.get("batches", ())) != len(rows)
    ):
        raise RuntimeError(f"invalid prepared-batch cache: {cache_path}")
    batches = payload["batches"]
    request_ids = [item[0] for item in batches]
    if len(set(request_ids)) != len(request_ids):
        raise RuntimeError("prepared-batch cache contains duplicate request IDs")
    print(
        f"[batch-cache] rank={rank} validated cache requests={len(batches)} "
        f"signature={signature[:16]}",
        flush=True,
    )
    return batches


def main(*, install_background=None, output_dir=None) -> None:
    trainer.load_batches = _cache_aware_load_batches
    trainer.main(install_background=install_background, output_dir=output_dir)


if __name__ == "__main__":
    main()

