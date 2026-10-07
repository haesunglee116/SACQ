#!/usr/bin/env python3
"""Fit farm: quantize every (module, mode) pair once and cache to disk.
Reads weights straight from the checkpoint safetensors -- no policy load.
Modes: vanilla (no metrics needed) | qk | vo | ffn | adaln (need metrics.pt)."""
from __future__ import annotations
import argparse, csv, json, sys, zlib
from pathlib import Path
import torch
from safetensors import safe_open

PROJ = Path("@WORKSPACE@/libero_pro_eval")
sys.path.insert(0, str(PROJ)); sys.path.insert(0, str(PROJ / "Isaac-GR00T-N1.7"))
from measure_gr00t_qk_diag_codebook_2026_06_27 import quantize_vector_codebook

QA = PROJ / "quantization_analysis"
CKPT = PROJ / "models-gr00t-n17-libero/libero_spatial"
MODULE_CSV = QA / "gr00t_dit_scalar_vs_vector_2026-07-17/dit_modules.csv"
CB = dict(group_size=64, vector_dim=4, codebook_size=256, sample_size=200000, iters=12)
SEED = 20260601

idx = None
_open = {}
def load_w(name, device):
    global idx
    if idx is None:
        idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    key = name + ".weight"
    f = idx[key]
    if f not in _open:
        _open[f] = safe_open(str(CKPT / f), framework="pt", device="cpu")
    return _open[f].get_tensor(key).to(device, torch.float32)

def main():
    global CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", required=True)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--metrics", default="")
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--ff-up-activation-curvature",
        choices=("none", "diag_mean"),
        default="none",
    )
    args = ap.parse_args()
    CKPT = Path(args.ckpt)
    print(f"[fit] checkpoint: {CKPT}", flush=True)
    fits = Path(args.fits_dir); fits.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    roles = {r["module_name"]: r["module_type"] for r in csv.DictReader(open(MODULE_CSV))}
    for i in range(32):
        roles[f"action_head.model.transformer_blocks.{i}.norm1.linear"] = "adaln"
    core = [n for n in sorted(roles) if "vl_self_attention" not in n]

    tasks = []
    for mode in args.modes:
        if mode == "vanilla":
            tasks += [(n, "vanilla") for n in core]
        elif mode == "qk":
            tasks += [(n, "qk") for n in core if "to_q" in roles[n] or "to_k" in roles[n]]
        elif mode == "vo":
            tasks += [(n, "vo") for n in core if "to_v" in roles[n] or "to_out" in roles[n]]
        elif mode == "ffn":
            tasks += [(n, "ffn") for n in core if "ff_" in roles[n]]
        elif mode == "adaln":
            tasks += [(n, "adaln") for n in core if roles[n] == "adaln"]
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    print(f"[fit] shard {args.shard_index}/{args.num_shards}: {len(tasks)} tasks", flush=True)

    M = torch.load(args.metrics, map_location="cpu") if args.metrics else {}
    if args.ff_up_activation_curvature == "diag_mean":
        if not args.metrics:
            raise RuntimeError("down-aware FF-up fitting requires --metrics")
        for name, role in roles.items():
            if role != "ff_up":
                continue
            key = f"ffn::{name}"
            if key not in M or "activation_derivative_sq" not in M[key]:
                raise RuntimeError(
                    f"down-aware FF-up fitting requires activation_derivative_sq for {name}"
                )
    def met(pfx, key):
        return M[pfx][key].to(device, torch.float32)

    done = skip = 0
    for i, (name, mode) in enumerate(tasks, 1):
        outp = fits / f"{name.replace('/', '_')}__{mode}.pt"
        if outp.exists():
            skip += 1; continue
        W = load_w(name, device)
        seed = SEED + zlib.crc32(name.encode()) % 9973
        role = roles[name]
        if mode == "vanilla":
            Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="euclidean", scalar_importance=None)
        elif mode in ("qk", "vo"):
            pfx = name[: name.index(".attn1") + 6]
            if "to_q" in role:   im, oi = met(pfx, "Gq"), met(pfx, "q_imp")
            elif "to_k" in role: im, oi = met(pfx, "Gkv"), met(pfx, "k_imp")
            elif "to_v" in role: im, oi = met(pfx, "Vmet"), met(pfx, "v_imp")
            else:                im, oi = met(pfx, "Omet"), torch.ones(W.shape[0], device=device)
            Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full", scalar_importance=None,
                                          input_metric=im, output_importance=oi)
        elif mode == "ffn":
            G = met(f"ffn::{name}", "G")
            if "ff_up" in role:
                oi = load_w(name.replace("ff.net.0.proj", "ff.net.2"), device).square().sum(dim=0).clamp_min(1e-12)
                if args.ff_up_activation_curvature == "diag_mean":
                    derivative_sq = met(f"ffn::{name}", "activation_derivative_sq")
                    if derivative_sq.shape != oi.shape:
                        raise ValueError(
                            f"FF-up derivative/down-column shape mismatch for {name}: "
                            f"{tuple(derivative_sq.shape)} vs {tuple(oi.shape)}"
                        )
                    oi = oi * derivative_sq
            else:
                oi = torch.ones(W.shape[0], device=device)
            Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full", scalar_importance=None,
                                          input_metric=G, output_importance=oi)
        else:  # adaln
            G = met(f"ada::{name}", "G")
            lnsq = M[f"ada::{name}"]["lnsq"].to(device, torch.float32)
            oi = torch.cat([lnsq, torch.ones_like(lnsq)]).clamp_min(1e-12)
            Wq = quantize_vector_codebook(W, **CB, seed=seed, metric="qk_full", scalar_importance=None,
                                          input_metric=G, output_importance=oi)
        torch.save(Wq.to("cpu", torch.bfloat16), outp)
        done += 1
        if i % 20 == 0:
            print(f"[fit] {i}/{len(tasks)}", flush=True)
    print(f"[fit] done={done} skipped={skip}", flush=True)

if __name__ == "__main__":
    main()
