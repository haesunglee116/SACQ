#!/usr/bin/env python3
"""Fractional-rate uniform codebook for the whole DiT core (structure-aware).

Codebooks give bit rates HQQ cannot: with vector_dim=4 the index costs
log2(k)/4 bits per weight, i.e. 0.25-bit granularity
    k=256 -> 2.00 | 512 -> 2.25 | 1024 -> 2.50 | 2048 -> 2.75 | 4096 -> 3.00
Every DiT ladder point so far reached its average bit by *mixing* 2-bit
codebooks with 3-bit HQQ protection; a uniform fractional rate has never been
tried. This fits all 226 core modules at a single k with the role-appropriate
structure-aware objective (qk / vo / ffn / adaln metrics), no HQQ anywhere.

Usage: --suite libero_spatial --k 1024 --shard-index 0 --num-shards 4
Saves: fits_k{K}_<suite>/<name>__ck.pt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path

import torch

PROJ = Path("/data/workspace/libero_pro_eval")
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "Isaac-GR00T-N1.7"))

from measure_gr00t_qk_diag_codebook_2026_06_27 import quantize_vector_codebook  # noqa: E402
import fs2_fitfarm_2026_07_23 as farm  # noqa: E402

QA = PROJ / "quantization_analysis"
FS2 = QA / "gr00t_dit_fs2_2026-07-23"
MODULE_CSV = QA / "gr00t_dit_scalar_vs_vector_2026-07-17/dit_modules.csv"
SEED = 20260601

roles = {r["module_name"]: r["module_type"] for r in csv.DictReader(open(MODULE_CSV))}
for i in range(32):
    roles[f"action_head.model.transformer_blocks.{i}.norm1.linear"] = "adaln"
core = [n for n in sorted(roles) if "vl_self_attention" not in n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--modules-file", default="")
    ap.add_argument("--metrics-dir", default="")
    ap.add_argument("--fits-dir", default="")
    ap.add_argument("--alloc-csv", default="")
    ap.add_argument("--packed", action="store_true")
    ap.add_argument(
        "--packed-only", action="store_true",
        help="Save only packed state (omit legacy dense reconstruction). Requires --packed.",
    )
    ap.add_argument("--vanilla", action="store_true")
    ap.add_argument(
        "--ff-up-activation-curvature",
        choices=("none", "diag_mean"),
        default="none",
        help=(
            "none은 2026-08-03 legacy metric, diag_mean은 FF-up output "
            "importance에 저장된 E[sigma'(p)^2]를 곱한다."
        ),
    )
    ap.add_argument(
        "--proj-out-structure-aware",
        choices=("none", "diag"),
        default="none",
        help="Use captured output-head input covariance and diagonal downstream sensitivity.",
    )
    args = ap.parse_args()
    if args.packed_only and not args.packed:
        ap.error("--packed-only requires --packed")
    device = torch.device(args.device)
    farm.CKPT = PROJ / "models-gr00t-n17-libero" / args.suite
    # Vanilla Euclidean fitting never reads any structure-aware metric.  Do
    # not require or deserialize a suite metric file in that branch.  Besides
    # avoiding irrelevant I/O, this removes the stale historical
    # ``libero_spatial -> metrics/`` alias (the canonical directory is now
    # metrics_libero_spatial) without changing a single vanilla assignment.
    if args.vanilla:
        M = {}
    else:
        metrics_dir = args.metrics_dir or f"metrics_{args.suite}"
        M = torch.load(FS2 / metrics_dir / "metrics.pt", map_location="cpu")
    fits = Path(args.fits_dir) if args.fits_dir else FS2 / f"fits_k{args.k}_{args.suite}"
    fits.mkdir(parents=True, exist_ok=True)
    CB = dict(group_size=64, vector_dim=4, codebook_size=args.k, sample_size=200000, iters=12)
    kmap = {}
    if args.alloc_csv:
        kmap = {r["layer"]: int(r["k"]) for r in csv.DictReader(open(args.alloc_csv))}

    pool = core
    if args.modules_file:
        want = set(open(args.modules_file).read().split())
        pool = [n for n in core if n in want]
    if not args.vanilla and args.ff_up_activation_curvature == "diag_mean":
        for name in pool:
            if roles[name] != "ff_up":
                continue
            key = f"ffn::{name}"
            if key not in M or "activation_derivative_sq" not in M[key]:
                raise RuntimeError(
                    f"down-aware DiT fitting requires activation_derivative_sq for {name}"
                )
            derivative_sq = M[key]["activation_derivative_sq"]
            if derivative_sq.ndim != 1 or derivative_sq.numel() == 0:
                raise RuntimeError(f"invalid activation_derivative_sq shape for {name}")
            if not torch.isfinite(derivative_sq).all():
                raise RuntimeError(f"non-finite activation_derivative_sq for {name}")
    if not args.vanilla and args.proj_out_structure_aware == "diag":
        for name in ("action_head.model.proj_out_1", "action_head.model.proj_out_2"):
            key = f"proj::{name}"
            if name in pool and (key not in M or "G" not in M[key]):
                raise RuntimeError(f"proj_out structure-aware metric missing for {name}")
        key = "proj::action_head.model.proj_out_1"
        if "action_head.model.proj_out_1" in pool and "norm_out_sq" not in M.get(key, {}):
            raise RuntimeError("proj_out_1 metric missing norm_out_sq")
    tasks = [n for i, n in enumerate(pool) if i % args.num_shards == args.shard_index]
    print(f"[ck] {args.suite} k={args.k} ({(args.k.bit_length()-1)/4:.2f} bit/weight) "
          f"shard {args.shard_index}/{args.num_shards}: {len(tasks)}/{len(core)}", flush=True)

    def met(pfx, key):
        return M[pfx][key].to(device, torch.float32)

    done = skip = 0
    for name in tasks:
        outp = fits / f"{name}__ck.pt"
        if outp.exists():
            try:
                existing = torch.load(outp, map_location="cpu")
                packed_existing = existing["packed"]
                expected_k = kmap.get(name, args.k)
                if (
                    set(existing) == {"packed"}
                    and packed_existing.get("format") == "packed_vector_codebook_v1"
                    and int(packed_existing["codebook"].shape[0]) == int(expected_k)
                ):
                    skip += 1
                    continue
            except Exception:
                pass
            raise RuntimeError(f"invalid existing packed artifact blocks resume: {outp}")
        W = farm.load_w(name, device)
        if kmap:
            CB["codebook_size"] = kmap[name]
        CB["return_packed"] = args.packed
        seed = SEED + zlib.crc32(name.encode()) % 9973
        role = roles[name] if not args.vanilla else "__vanilla__"
        try:
            if "to_q" in role or "to_k" in role:
                pfx = name[: name.index(".attn") + 6]
                im, oi = (met(pfx, "Gq"), met(pfx, "q_imp")) if "to_q" in role else (met(pfx, "Gkv"), met(pfx, "k_imp"))
                Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full",
                                              scalar_importance=None, input_metric=im, output_importance=oi)
            elif "to_v" in role or "to_out" in role:
                pfx = name[: name.index(".attn") + 6]
                if "to_v" in role:
                    im, oi = met(pfx, "Vmet"), met(pfx, "v_imp")
                else:
                    im, oi = met(pfx, "Omet"), torch.ones(W.shape[0], device=device)
                Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full",
                                              scalar_importance=None, input_metric=im, output_importance=oi)
            elif "ff_" in role:
                G = met(f"ffn::{name}", "G")
                if "ff_up" in role:
                    oi = farm.load_w(name.replace("ff.net.0.proj", "ff.net.2"), device).square().sum(dim=0).clamp_min(1e-12)
                    if args.ff_up_activation_curvature == "diag_mean":
                        derivative_sq = met(f"ffn::{name}", "activation_derivative_sq")
                        if derivative_sq.shape != oi.shape:
                            # Do not let the legacy metric-missing fallback silently
                            # disable an explicitly requested scientific correction.
                            raise RuntimeError(
                                f"FF-up derivative/down-column shape mismatch for {name}: "
                                f"{tuple(derivative_sq.shape)} vs {tuple(oi.shape)}"
                            )
                        oi = oi * derivative_sq
                else:
                    oi = torch.ones(W.shape[0], device=device)
                Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full",
                                              scalar_importance=None, input_metric=G, output_importance=oi)
            elif role == "adaln":
                G = met(f"ada::{name}", "G")
                lnsq = M[f"ada::{name}"]["lnsq"].to(device, torch.float32)
                oi = torch.cat([lnsq, torch.ones_like(lnsq)]).clamp_min(1e-12)
                Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full",
                                              scalar_importance=None, input_metric=G, output_importance=oi)
            elif role == "proj_out" and args.proj_out_structure_aware == "diag":
                G = met(f"proj::{name}", "G")
                if name.endswith("proj_out_1"):
                    downstream = farm.load_w(
                        "action_head.model.proj_out_2", device
                    ).square().sum(dim=0).clamp_min(1e-12)
                    norm_sq = met(f"proj::{name}", "norm_out_sq")
                    if downstream.shape != norm_sq.shape:
                        raise RuntimeError(
                            f"proj_out_1 downstream/norm shape mismatch: "
                            f"{tuple(downstream.shape)} vs {tuple(norm_sq.shape)}"
                        )
                    # proj_out_1 emits [shift, scale].  Shift affects every whole
                    # normalized hidden channel equally; scale is modulated by
                    # that channel's observed squared normalized activation.
                    oi = torch.cat([downstream, downstream * norm_sq]).clamp_min(1e-12)
                else:
                    # proj_out_2 directly emits the denoising velocity.
                    oi = torch.ones(W.shape[0], device=device)
                Wq = quantize_vector_codebook(
                    W, **CB, seed=seed, metric="qk_full",
                    scalar_importance=None, input_metric=G, output_importance=oi,
                )
            else:
                Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="euclidean", scalar_importance=None)
        except (KeyError, ValueError) as e:
            if (
                ("ff_up" in role and args.ff_up_activation_curvature == "diag_mean")
                or (role == "proj_out" and args.proj_out_structure_aware == "diag")
            ):
                raise RuntimeError(
                    f"explicit structure-aware correction failed for {name}"
                ) from e
            print(f"[ck] {name}: metric 없음({e}) -> euclidean 폴백", flush=True)
            Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="euclidean", scalar_importance=None)
        packed = None
        if args.packed:
            Wq, packed = Wq
        rel = ((Wq.to(torch.float32) - W).square().sum() / W.square().sum()).sqrt().item()
        if packed is not None:
            payload = {"packed": packed}
            if not args.packed_only:
                payload["dense"] = Wq.to("cpu", torch.bfloat16)
            temporary = outp.with_name(f".{outp.name}.tmp.{os.getpid()}")
            torch.save(payload, temporary)
            os.replace(temporary, outp)
        else:
            temporary = outp.with_name(f".{outp.name}.tmp.{os.getpid()}")
            torch.save(Wq.to("cpu", torch.bfloat16), temporary)
            os.replace(temporary, outp)
        done += 1
        if done % 10 == 0:
            print(f"[ck] {done}/{len(tasks)} 최근 {name.split('.model.')[-1]:45} rel={rel:.4f}", flush=True)
    print(f"[ck] done={done} skipped={skip}", flush=True)


if __name__ == "__main__":
    main()

