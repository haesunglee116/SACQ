#!/usr/bin/env python3
"""Stage 0 of the 2-budget fullstack ladder: capture aware-metrics once, save to disk.
Everything downstream (fit farm, eval) reads these files instead of re-capturing."""
from __future__ import annotations
import argparse, csv, gc, math, sys, json, os
from collections import defaultdict
from pathlib import Path
import torch

PROJ = Path("/data/workspace/libero_pro_eval")
sys.path.insert(0, str(PROJ)); sys.path.insert(0, str(PROJ / "Isaac-GR00T-N1.7"))
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
from gr00t.policy.gr00t_policy import Gr00tPolicy
from measure_dit_single_layer_downgrade_sensitivity import patched_env, replace_embedding
from measure_dit_scalar_vs_vector_2026_07_17 import load_batches
from measure_embedding_vision_quant_memory_v1 import quantize_visual_linears
from measure_gr00t_dit_fullstack_proxy_2026_07_21 import VLM_ENV
from measure_hotspot_protection_proxy import move_to_device, run_outputs

QA = PROJ / "quantization_analysis"
MODULE_CSV = QA / "gr00t_dit_scalar_vs_vector_2026-07-17/dit_modules.csv"


def gelu_derivative(x: torch.Tensor, approximate: str) -> torch.Tensor:
    """Derivative matching torch GELU's exact or tanh-approximate forward."""
    if approximate == "none":
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) + (
            x * torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        )
    if approximate == "tanh":
        alpha = math.sqrt(2.0 / math.pi)
        inner = alpha * (x + 0.044715 * x.pow(3))
        tanh_inner = torch.tanh(inner)
        inner_derivative = alpha * (1.0 + 3.0 * 0.044715 * x.square())
        return 0.5 * (1.0 + tanh_inner) + (
            0.5 * x * (1.0 - tanh_inner.square()) * inner_derivative
        )
    raise ValueError(f"unsupported GELU approximation: {approximate!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=str(PROJ / "models-gr00t-n17-libero/libero_spatial"))
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--calib-calls", type=int, default=48)
    ap.add_argument("--request-dir", default=str(PROJ / "request_replay_runs/calib_4suite_2026-07-02/val/requests"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--ff-up-activation-curvature",
        choices=("none", "diag_mean"),
        default="none",
        help=(
            "none은 기존 metric을 재현한다. diag_mean은 FF-up preactivation에서 "
            "E[sigma'(p)^2]를 수집해 downstream column energy와 결합한다."
        ),
    )
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    roles = {r["module_name"]: r["module_type"] for r in csv.DictReader(open(MODULE_CSV))}
    for i in range(32):
        roles[f"action_head.model.transformer_blocks.{i}.norm1.linear"] = "adaln"
    core = [n for n in sorted(roles) if "vl_self_attention" not in n]
    attn_prefixes = sorted({n[: n.index(".attn1") + 6] for n in core if ".attn1." in n})
    ffn = [n for n in core if "ff_" in roles[n]]
    adaln = [n for n in core if roles[n] == "adaln"]
    print(f"[m] core={len(core)} attn_blocks={len(attn_prefixes)} ffn={len(ffn)} adaln={len(adaln)}", flush=True)

    print(f"[m] checkpoint: {args.ckpt}", flush=True)
    policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"),
                         args.ckpt, device=args.device, strict=True)
    policy.model.eval()
    for p in policy.model.parameters(): p.requires_grad_(False)
    batches = load_batches(policy, Path(args.request_dir), args.limit)
    named = dict(policy.model.named_modules())
    with patched_env(VLM_ENV):
        quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device=args.device)
    replace_embedding(policy.model, 4, 64, True)
    quantize_visual_linears(policy.model, nbits=4, group_size=64, device=args.device)
    gc.collect(); torch.cuda.empty_cache()

    pair = defaultdict(list); handles = []
    ffn_g, ffn_t = {}, defaultdict(int)
    ffn_deriv_sq, ffn_deriv_n = {}, defaultdict(int)
    ada_g, ada_t, ada_sq, ada_n = {}, defaultdict(int), {}, defaultdict(int)

    def prehook(pfx):
        def h(mod, a, k):
            xq = a[0].detach(); xkv = k.get("encoder_hidden_states")
            pair[pfx].append((xq.to("cpu", torch.float16), (xq if xkv is None else xkv.detach()).to("cpu", torch.float16)))
        return h
    def gram(store, tok, name):
        def h(mod, i, o):
            x = i[0].detach().reshape(-1, i[0].shape[-1]).to(torch.float32)
            g = x.t() @ x
            store[name] = g if name not in store else store[name] + g
            tok[name] += x.shape[0]
        return h
    def ff_up_derivative(name, approximate):
        def h(mod, i, o):
            p = o.detach().reshape(-1, o.shape[-1]).to(torch.float32)
            derivative_sq = gelu_derivative(p, approximate).square().sum(dim=0)
            ffn_deriv_sq[name] = (
                derivative_sq
                if name not in ffn_deriv_sq
                else ffn_deriv_sq[name] + derivative_sq
            )
            ffn_deriv_n[name] += p.shape[0]
        return h
    def ada_pre(name):
        def h(mod, a, k):
            x = a[0].detach().to(torch.float32)
            ln = torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5).reshape(-1, x.shape[-1])
            s = ln.square().sum(0)
            ada_sq[name] = s if name not in ada_sq else ada_sq[name] + s
            ada_n[name] += ln.shape[0]
        return h

    for p_ in attn_prefixes: handles.append(named[p_].register_forward_pre_hook(prehook(p_), with_kwargs=True))
    for n in ffn:
        handles.append(named[n].register_forward_hook(gram(ffn_g, ffn_t, n)))
        if args.ff_up_activation_curvature == "diag_mean" and roles[n] == "ff_up":
            activation = named[n.rsplit(".proj", 1)[0]]
            approximate = getattr(activation, "approximate", None)
            if approximate not in ("none", "tanh"):
                raise ValueError(
                    f"FF-up activation for {n} is not a supported GELU: "
                    f"{type(activation).__name__}, approximate={approximate!r}"
                )
            handles.append(named[n].register_forward_hook(ff_up_derivative(n, approximate)))
    for n in adaln:
        handles.append(named[n].register_forward_hook(gram(ada_g, ada_t, n)))
        handles.append(named[n.rsplit(".", 1)[0]].register_forward_pre_hook(ada_pre(n), with_kwargs=True))

    consumption = []
    for i, (rid, bb, ai) in enumerate(batches):
        before = {p: len(pair[p]) for p in attn_prefixes}
        _ = run_outputs(policy, move_to_device(bb, device, torch.bfloat16),
                        move_to_device(ai, device, torch.bfloat16), seed=7000 + i)
        calls = {p: len(pair[p]) - before[p] for p in attn_prefixes}
        expected = int(policy.model.config.num_inference_timesteps)
        if any(n != expected for n in calls.values()):
            raise RuntimeError(f"attention coverage mismatch: {rid}: {calls}")
        consumption.append({"request_id": rid, "attention_calls_per_block": calls})
        print(f"[m] request {i+1}/{len(batches)} captured all {expected} calls/block", flush=True)
    (out / "consumption.json").write_text(json.dumps({"requests": consumption, "status": "COMPLETE"}, indent=2))
    for h in handles: h.remove()
    print(f"[m] captured {sum(len(v) for v in pair.values())} calls", flush=True)

    save = {}
    for pfx in attn_prefixes:
        attn = named[pfx]; heads = int(attn.heads)
        wq = named[f"{pfx}.to_q"].weight.detach().to(device, torch.float32)
        wk = named[f"{pfx}.to_k"].weight.detach().to(device, torch.float32)
        wv = named[f"{pfx}.to_v"].weight.detach().to(device, torch.float32)
        wo = named[f"{pfx}.to_out.0"].weight.detach().to(device, torch.float32)
        hd = wq.shape[0] // heads; sc = hd ** -0.5
        Gq = Gkv = Vm = Om = None; tq = 0
        for xq_c, xkv_c in pair[pfx]:
            xq = xq_c.to(device, torch.float32); xkv = xkv_c.to(device, torch.float32)
            fq = xq.reshape(-1, xq.shape[-1]); fkv = xkv.reshape(-1, xkv.shape[-1])
            Gq = fq.t() @ fq if Gq is None else Gq + fq.t() @ fq
            Gkv = fkv.t() @ fkv if Gkv is None else Gkv + fkv.t() @ fkv
            tq += fq.shape[0]
            B, Nq, _ = xq.shape; Nk = xkv.shape[1]
            q = (xq @ wq.t()).view(B, Nq, heads, hd).transpose(1, 2)
            k = (xkv @ wk.t()).view(B, Nk, heads, hd).transpose(1, 2)
            v = (xkv @ wv.t()).view(B, Nk, heads, hd).transpose(1, 2)
            pr = torch.softmax(q @ k.transpose(2, 3) * sc, dim=-1)
            ctx = (pr @ v).transpose(1, 2).reshape(B, Nq, heads * hd)
            wsrc = pr.square().sum(dim=(1, 2))
            wx = (xkv * wsrc.clamp_min(0).sqrt().unsqueeze(-1)).reshape(-1, xkv.shape[-1])
            fc = ctx.reshape(-1, ctx.shape[-1])
            Vm = wx.t() @ wx if Vm is None else Vm + wx.t() @ wx
            Om = fc.t() @ fc if Om is None else Om + fc.t() @ fc
        d = {"Gq": Gq, "Gkv": Gkv, "Vmet": Vm, "Omet": Om}
        d = {k_: (0.5 * (v_ + v_.t()) / max(tq, 1)).to("cpu", torch.float16) for k_, v_ in d.items()}
        d["q_imp"] = torch.diagonal(wk @ (d["Gkv"].to(device, torch.float32)) @ wk.t()).clamp_min(1e-12).cpu()
        d["k_imp"] = torch.diagonal(wq @ (d["Gq"].to(device, torch.float32)) @ wq.t()).clamp_min(1e-12).cpu()
        d["v_imp"] = wo.square().sum(dim=0).clamp_min(1e-12).cpu()
        save[pfx] = d
    for n in ffn:
        metric = {"G": (0.5 * (ffn_g[n] + ffn_g[n].t()) / max(ffn_t[n], 1)).to("cpu", torch.float16)}
        if args.ff_up_activation_curvature == "diag_mean" and roles[n] == "ff_up":
            if ffn_deriv_n[n] != ffn_t[n]:
                raise RuntimeError(
                    f"FF-up derivative/input sample mismatch for {n}: "
                    f"{ffn_deriv_n[n]} vs {ffn_t[n]}"
                )
            derivative_sq = ffn_deriv_sq[n] / max(ffn_deriv_n[n], 1)
            if derivative_sq.numel() != named[n].out_features:
                raise RuntimeError(
                    f"FF-up derivative width mismatch for {n}: "
                    f"{derivative_sq.numel()} vs {named[n].out_features}"
                )
            if not torch.isfinite(derivative_sq).all():
                raise RuntimeError(f"non-finite FF-up derivative statistic for {n}")
            metric["activation_derivative_sq"] = derivative_sq.to("cpu", torch.float32)
        save[f"ffn::{n}"] = metric
    for n in adaln:
        save[f"ada::{n}"] = {"G": (0.5 * (ada_g[n] + ada_g[n].t()) / max(ada_t[n], 1)).to("cpu", torch.float16),
                             "lnsq": (ada_sq[n] / max(ada_n[n], 1)).cpu()}
    torch.save(save, out / ".metrics.tmp.pt")
    os.replace(out / ".metrics.tmp.pt", out / "metrics.pt")
    print(f"[m] wrote {out}/metrics.pt", flush=True)

if __name__ == "__main__":
    main()

