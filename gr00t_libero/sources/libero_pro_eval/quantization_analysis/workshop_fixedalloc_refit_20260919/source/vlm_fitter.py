#!/usr/bin/env python3
"""Original full7 structure-aware VLM codebook builder + optional rotation.

Same pipeline that produced the deployed 2.0729 codebooks (Q/K full-block,
attention-aware V/O, SwiGLU gate/up, down ZTZ; calibration captured on the
train replay set), with one addition: --rotate applies a block-64 random-sign
Hadamard R to BOTH the weight (W' = W R) and the quadratic input metric
(G' = R^T G R) before fitting, and folds R back into the dense reconstruction.
For quadratic objectives this preserves the objective exactly. gate/up use an
elementwise importance (not quadratic), which does not compose exactly with
coordinate mixing -- those two roles are always fitted UNROTATED.
Saves dense reconstructions for the 101 deployed-2bit layers only
(the 3/4-bit exception layers are untouched), for the ruler override slot."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from measure_gr00t_qk_diag_action_proxy_2026_06_27 import (
    collect_qk_calibration,
    input_metric_from_cache,
    load_batches,
)
from measure_gr00t_qk_diag_codebook_2026_06_27 import (
    layer_module_name,
    project_qk,
    quantize_vector_codebook,
    repeat_kv,
)
from measure_gr00t_qkfull_swiglu_combo_proxy_2026_07_11 import build_policy
from measure_gr00t_swiglu_codebook_proxy_2026_07_11 import (
    collect_mlp_inputs,
    mlp_name,
    proj_name,
    silu_derivative,
)
from proxy_fast_2026_07_24 import local_ckpt

PROJ = Path("@WORKSPACE@/libero_pro_eval")
QA = PROJ / "quantization_analysis"
PACKED = QA / "gr00t_full7_2p1_packed_2026-07-13/full7_2p1_packed_refined.pt"
DEFAULT_REQUESTS = PROJ / "request_replay_runs/calib_4suite_2026-07-02/train/requests"
ROT_SEED = 20260601


def apply_swiglu_down_diag_curvature(
    gate_importance: torch.Tensor,
    up_importance: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate diagonal gate/up sensitivity through ``down_proj``.

    For y = W_down (SiLU(g) * u), a perturbation of intermediate row r is
    weighted by diag(W_down^T W_down)_r = ||W_down[:, r]||_2^2.  The full
    W_down^T W_down would couple different intermediate rows; this helper
    deliberately retains its diagonal so the existing per-weight ``qk_diag``
    fitter remains applicable.
    """
    if gate_importance.shape != up_importance.shape:
        raise ValueError(
            "gate/up importance shape mismatch: "
            f"{tuple(gate_importance.shape)} vs {tuple(up_importance.shape)}"
        )
    if down_weight.ndim != 2:
        raise ValueError(f"down_weight must be a matrix, got {tuple(down_weight.shape)}")
    if down_weight.shape[1] != gate_importance.shape[0]:
        raise ValueError(
            "down_proj input width must match gate/up output rows: "
            f"{down_weight.shape[1]} vs {gate_importance.shape[0]}"
        )
    down_column_energy = down_weight.square().sum(dim=0)
    if not torch.isfinite(down_column_energy).all():
        raise ValueError("non-finite down_proj diagonal curvature")
    row_weight = down_column_energy.unsqueeze(1)
    return gate_importance * row_weight, up_importance * row_weight


def hadamard(n: int, device) -> torch.Tensor:
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def make_rot(device) -> torch.Tensor:
    H = hadamard(64, device)
    gen = torch.Generator(device="cuda").manual_seed(ROT_SEED)
    signs = (torch.randint(0, 2, (64,), generator=gen, device=device, dtype=torch.int64) * 2 - 1).float()
    return H * signs.unsqueeze(0)  # orthogonal 64x64


def rot_w(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    # R: (64,64) shared or (nb,64,64) per-group
    o, i = W.shape
    Wg = W.reshape(o, i // 64, 64)
    if R.dim() == 2:
        return (Wg @ R).reshape(o, i)
    return torch.einsum("ogk,gkl->ogl", Wg, R).reshape(o, i)


def unrot_w(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    o, i = W.shape
    Wg = W.reshape(o, i // 64, 64)
    if R.dim() == 2:
        return (Wg @ R.t()).reshape(o, i)
    return torch.einsum("ogk,glk->ogl", Wg, R).reshape(o, i)


def rot_metric(G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    n = G.shape[0]
    nb = n // 64
    Gb = G.reshape(nb, 64, nb, 64)
    if R.dim() == 2:
        Gb = torch.einsum("ki,akbl->aibl", R, Gb)
        Gb = torch.einsum("akbl,lj->akbj", Gb, R)
        return Gb.reshape(n, n)
    Gb = torch.einsum("aki,akbl->aibl", R, Gb)   # R_a^T on left blocks
    Gb = torch.einsum("aibl,blj->aibj", Gb, R)   # R_b on right blocks
    return Gb.reshape(n, n)


def eig_rot_for(G: torch.Tensor, in_f: int) -> torch.Tensor:
    """Per-64-group rotation into the eigenbasis of G's diagonal blocks.
    Diagonalizes the metric the fitter actually sees, so its 4x4-slice
    approximation becomes lossless (a diagonal matrix loses nothing when
    sliced) -- the aware weighting survives rotation exactly."""
    nb = in_f // 64
    Gd = G.reshape(nb, 64, nb, 64)
    Gd = torch.stack([Gd[a, :, a, :] for a in range(nb)])
    Gd = 0.5 * (Gd + Gd.transpose(1, 2))
    _vals, vecs = torch.linalg.eigh(Gd.double())
    return vecs.to(G.dtype)  # (nb, 64, 64), columns = eigenvectors


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-dir", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--rot-mode", default="hadamard", choices=["hadamard", "eig"])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260713)
    # Codebook size sets the rate: with vector_dim=4 the index costs log2(k)/4
    # bits per weight, so k=256/512/1024/2048 -> 2.00/2.25/2.50/2.75 bit.
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--alloc-csv", type=Path, default=None)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--packed-only", action="store_true",
        help="Save only packed state (omit legacy dense reconstruction). Requires --packed.",
    )
    parser.add_argument("--only-layers", type=Path, default=None,
                        help="이 파일에 적힌 레이어만 피팅")
    parser.add_argument("--add-layers", type=Path, default=None,
                        help="기본 101층에 더해 이 파일의 레이어도 피팅 (핫스팟 11층 등)")
    parser.add_argument("--vanilla", action="store_true",
                        help="structure-aware 목적함수 끄기 (순수 euclidean k-means)")
    parser.add_argument(
        "--swiglu-down-curvature",
        choices=("none", "diag"),
        default="none",
        help=(
            "gate/up metric에 down_proj 곡률을 전파하는 방식. "
            "none은 2026-07-25 legacy metric, diag는 "
            "diag(W_down^T W_down)를 행별로 곱한다."
        ),
    )
    parser.add_argument(
        "--skip-existing-packed",
        action="store_true",
        help="Validate and skip completed packed-only layer artifacts when resuming.",
    )
    args = parser.parse_args()
    if args.packed_only and not args.packed:
        parser.error("--packed-only requires --packed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in args.only_layers.read_text().split()}) if args.only_layers else list(range(16))
    size = args.codebook_size
    kmap = {}
    if args.alloc_csv:
        import csv as _csv
        kmap = {r["layer"]: int(r["k"]) for r in _csv.DictReader(open(args.alloc_csv))}
        print(f"[bg] per-layer k from {args.alloc_csv.name}: {len(kmap)} layers", flush=True)
    print(f"[bg] codebook_size={size} -> {(size.bit_length()-1)/4:.2f} bit/weight packed={args.packed}", flush=True)
    print(f"[bg] swiglu_down_curvature={args.swiglu_down_curvature}", flush=True)
    keep = set(Path(args.only_layers).read_text().split()) if args.only_layers else set(torch.load(PACKED, map_location="cpu")["weights"].keys())
    if args.add_layers:
        # keep 은 예전 2비트 오버레이(101층) 목록에서 온다. 7월에 3/4bit 로 따로 뺀
        # 핫스팟 11층이 여기에 없어서, 배분 실험이 그 층을 건드리지 못한 채 상속됐다.
        # 112층 전부를 배분 대상으로 삼으려면 이 옵션으로 합집합을 만든다.
        extra = set(open(args.add_layers).read().split())
        keep |= extra
        print(f"[bg] add-layers: +{len(extra)}층 -> 총 {len(keep)}층", flush=True)
    if args.only_layers:
        keep &= set(open(args.only_layers).read().split())
        print(f"[bg] only-layers: {len(keep)}개만 피팅", flush=True)
    R = make_rot(device) if args.rotate else None

    def fit(weight, *, metric, scalar, imetric, oimp, seed, name):
        # ``--only-layers`` historically filtered only serialization, so a
        # sharded selected-layer run still executed k-means for every one of
        # the 112 background layers.  Fits are independent and receive a
        # deterministic per-layer seed; skipping an artifact that will not be
        # saved therefore cannot change any selected artifact value.
        if name not in keep:
            print(
                f"[bg]     {name.split('language_model.')[-1]:<40} skipped=True",
                flush=True,
            )
            return
        use_rot = args.rotate and imetric is not None  # quadratic objectives only
        Ruse = R
        if use_rot and args.rot_mode == "eig":
            Ruse = eig_rot_for(imetric.to(weight.device, torch.float32), weight.shape[1])
        W = rot_w(weight, Ruse) if use_rot else weight
        G = rot_metric(imetric, Ruse) if (use_rot and imetric is not None) else imetric
        size_l = kmap.get(name, size)
        output_path = args.out_dir / f"{name}.pt"
        if args.skip_existing_packed and output_path.is_file():
            try:
                existing = torch.load(output_path, map_location="cpu")
                packed_existing = existing["packed"]
                if (
                    set(existing) == {"packed"}
                    and packed_existing.get("format") == "packed_vector_codebook_v1"
                    and int(packed_existing["codebook"].shape[0]) == int(size_l)
                ):
                    print(
                        f"[bg]     {name.split('language_model.')[-1]:<40} resume_skip=True",
                        flush=True,
                    )
                    return
            except Exception:
                pass
            raise RuntimeError(f"invalid existing packed artifact blocks resume: {output_path}")
        if args.vanilla:
            metric, scalar, G, oimp = "euclidean", None, None, None
        out = quantize_vector_codebook(
            W, group_size=64, vector_dim=4, codebook_size=size_l,
            sample_size=args.sample_size, iters=args.iters, seed=seed,
            metric=metric, scalar_importance=scalar, input_metric=G,
            output_importance=oimp, return_packed=args.packed,
        )
        dense, packed = out if args.packed else (out, None)
        if use_rot:
            dense = unrot_w(dense, Ruse)
        rel = float((dense - weight).norm() / weight.norm())
        if name in keep:
            if packed is not None:
                payload = {"packed": packed}
                if not args.packed_only:
                    payload["dense"] = dense.to("cpu", torch.bfloat16)
                temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
                torch.save(payload, temporary)
                os.replace(temporary, output_path)
            else:
                temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
                torch.save(dense.to("cpu", torch.bfloat16), temporary)
                os.replace(temporary, output_path)
        print(f"[bg] {'rot ' if use_rot else '    '}{name.split('language_model.')[-1]:<40} rel={rel:.4f} saved={name in keep}", flush=True)

    policy = build_policy(local_ckpt(), args.device)
    batches = load_batches(policy, args.request_dir, args.limit)
    qk_caches = collect_qk_calibration(policy, batches, layers, device, max_cache_tokens_per_layer=args.max_tokens)
    mlp_caches = collect_mlp_inputs(policy, batches, layers, device, max_tokens_per_layer=args.max_tokens)
    audit = {}
    for layer in layers:
        q = qk_caches[layer]
        m = mlp_caches[layer]
        q_kept = sum(int(x.shape[0] * x.shape[1]) for x in q.x_chunks)
        m_kept = sum(int(x.shape[0]) for x in m.chunks)
        if q_kept != q.token_count or m_kept != m.token_count or len(q.x_chunks) != len(batches):
            raise RuntimeError(f"request coverage truncated at layer {layer}: q={q_kept}/{q.token_count} mlp={m_kept}/{m.token_count}")
        audit[str(layer)] = {"requests": len(batches), "q_tokens": q_kept, "mlp_tokens": m_kept}
    import json, os
    ap = args.out_dir / ("coverage_" + str(os.environ.get("SLURM_JOB_ID", os.getpid())) + ".json")
    ap.write_text(json.dumps({"status":"PASS","layers":audit,"request_ids":[x[0] for x in batches]}, indent=2))
    modules = dict(policy.model.named_modules())

    for layer in layers:
        cache = qk_caches[layer]
        attn = modules[layer_module_name(layer)]
        input_metric = input_metric_from_cache(cache, device)
        k_imp = (cache.krep_sq / max(cache.token_count, 1)).to(device, torch.float32).reshape(-1)
        q_imp = (cache.q_sq / max(cache.token_count, 1)).to(device, torch.float32)
        kv_heads = attn.k_proj.out_features // attn.head_dim
        q_grouped = q_imp.reshape(kv_heads, attn.num_key_value_groups, attn.head_dim).sum(dim=1).reshape(-1)
        for suffix, opposite, offset in (("q_proj", k_imp, 0), ("k_proj", q_grouped, 1009)):
            name = f"{layer_module_name(layer)}.{suffix}"
            weight = getattr(attn, suffix).weight.detach().to(device, torch.float32)
            fit(weight, metric="qk_full", scalar=None, imetric=input_metric,
                oimp=opposite, seed=args.seed + layer * 17 + offset, name=name)

        mlp = modules[mlp_name(layer)]
        h = mlp_caches[layer].tensor(device)
        wg = mlp.gate_proj.weight.detach().to(device, torch.float32)
        wu = mlp.up_proj.weight.detach().to(device, torch.float32)
        wd = mlp.down_proj.weight.detach().to(device, torch.float32)
        g = F.linear(h, wg); u = F.linear(h, wu)
        h_sq = h.square()
        gate_importance = (silu_derivative(g) * u).square().t().matmul(h_sq) / h.shape[0]
        up_importance = F.silu(g).square().t().matmul(h_sq) / h.shape[0]
        if args.swiglu_down_curvature == "diag":
            gate_importance, up_importance = apply_swiglu_down_diag_curvature(
                gate_importance, up_importance, wd
            )
        z = F.silu(g) * u
        ztz = z.t().matmul(z) / z.shape[0]
        ztz = 0.5 * (ztz + ztz.t())
        for suffix, weight, metric, scalar, full_metric, offset in (
            ("gate_proj", wg, "qk_diag", gate_importance, None, 0),
            ("up_proj", wu, "qk_diag", up_importance, None, 1009),
            ("down_proj", wd, "qk_full", None, ztz, 2017),
        ):
            name = proj_name(layer, suffix)
            fit(weight, metric=metric, scalar=scalar, imetric=full_metric,
                oimp=(torch.ones(weight.shape[0], device=device) if full_metric is not None else None),
                seed=args.seed + layer * 53 + offset, name=name)
        del h, wg, wu, wd, g, u, h_sq, gate_importance, up_importance, z, ztz
        gc.collect(); torch.cuda.empty_cache()

    for layer in layers:
        cache = qk_caches[layer]
        attn = modules[layer_module_name(layer)]
        v_metric = o_metric = None; tokens = 0
        for x_cpu, cos_cpu, sin_cpu, mask_cpu in zip(cache.x_chunks, cache.cos_chunks, cache.sin_chunks, cache.mask_chunks):
            x = x_cpu.to(device, torch.float32)
            cos = cos_cpu.to(device, torch.float32); sin = sin_cpu.to(device, torch.float32)
            q, _k, krep = project_qk(attn, x, cos, sin, attn.q_proj.weight, attn.k_proj.weight)
            shape = (*x.shape[:-1], -1, attn.head_dim)
            v = F.linear(x, attn.v_proj.weight.detach().to(torch.float32)).view(shape).transpose(1, 2)
            vrep = repeat_kv(v, attn.num_key_value_groups)
            logits = torch.matmul(q.float(), krep.float().transpose(2, 3)) * float(attn.scaling)
            if mask_cpu is not None and mask_cpu.ndim == 4:
                mask = mask_cpu.to(device)
                logits = logits + mask[..., : logits.shape[-2], : logits.shape[-1]].to(logits.dtype)
            probs = torch.softmax(logits, dim=-1)
            context = torch.matmul(probs, vrep.float()).transpose(1, 2).contiguous().reshape(*x.shape[:-1], -1)
            source_weight = probs.square().sum(dim=(1, 2))
            weighted_x = (x * source_weight.sqrt().unsqueeze(-1)).reshape(-1, x.shape[-1])
            flat_context = context.reshape(-1, context.shape[-1])
            part_v = weighted_x.t().matmul(weighted_x)
            part_o = flat_context.t().matmul(flat_context)
            v_metric = part_v if v_metric is None else v_metric + part_v
            o_metric = part_o if o_metric is None else o_metric + part_o
            tokens += flat_context.shape[0]
        v_metric = 0.5 * (v_metric + v_metric.t()) / tokens
        o_metric = 0.5 * (o_metric + o_metric.t()) / tokens
        wo = attn.o_proj.weight.detach().to(device, torch.float32)
        q_head_importance = wo.square().sum(dim=0).reshape(-1, attn.head_dim)
        kv_heads = attn.v_proj.out_features // attn.head_dim
        v_output = q_head_importance.reshape(kv_heads, attn.num_key_value_groups, attn.head_dim).sum(dim=1).reshape(-1)
        for suffix, weight, metric_m, output, offset in (
            ("v_proj", attn.v_proj.weight.detach().to(device, torch.float32), v_metric, v_output, 3011),
            ("o_proj", wo, o_metric, torch.ones(wo.shape[0], device=device), 4021),
        ):
            name = f"{layer_module_name(layer)}.{suffix}"
            fit(weight, metric="qk_full", scalar=None, imetric=metric_m,
                oimp=output, seed=args.seed + layer * 67 + offset, name=name)
        del v_metric, o_metric, wo
        gc.collect(); torch.cuda.empty_cache()
    print("[bg] done", flush=True)


if __name__ == "__main__":
    main()

