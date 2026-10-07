#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
from measure_gr00t_qk_diag_action_proxy_2026_06_27 import (
    build_quantized_qk_weights,
    collect_qk_calibration,
    load_batches,
    run_outputs_with_blocks,
    write_csv,
)
from measure_gr00t_qk_diag_codebook_2026_06_27 import (
    layer_module_name,
    project_qk,
    quantize_vector_codebook,
    repeat_kv,
)
from measure_gr00t_swiglu_codebook_proxy_2026_07_11 import (
    apply_swiglu_overlay,
    build_swiglu_weights,
    collect_mlp_inputs,
    mlp_name,
    proj_name,
)
from measure_gr00t_top5_qkdiag_trainval_proxy_2026_07_02 import (
    aggregate,
    apply_qk_overlay,
    build_policy,
    build_settings,
    patched_env,
    percentile,
)
from measure_hotspot_protection_proxy import move_to_device, rel_metrics


PROJ = Path("@WORKSPACE@/libero_pro_eval")
DEFAULT_MODEL = "@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial"
DEFAULT_FIT_REQUESTS = PROJ / "request_replay_runs/calib_4suite_2026-07-02/train/requests"
DEFAULT_EVAL_REQUESTS = PROJ / "request_replay_runs/calib_4suite_2026-07-02/val/requests"
DEFAULT_OUT = PROJ / "quantization_analysis/gr00t_qkfull_swiglu_combo_proxy_2026-07-11"


@torch.no_grad()
def apply_filtered_swiglu_overlay(
    policy: Any,
    assignment: dict[str, int],
    weights: dict[str, dict[str, torch.Tensor]],
    aware: bool,
    device: torch.device,
) -> tuple[int, int]:
    selected = {name: value for name, value in weights.items() if assignment.get(name) == 2}
    return apply_swiglu_overlay(
        policy,
        selected,
        "gate_up_aware" if aware else "basic",
        device,
    )


@torch.no_grad()
def build_down_weights(
    policy: Any,
    caches: dict[int, Any],
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    named_modules = dict(policy.model.named_modules())
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for layer_idx, cache in caches.items():
        mlp = named_modules[mlp_name(layer_idx)]
        h = cache.tensor(device)
        gate = F.linear(h, mlp.gate_proj.weight.detach().to(device=device, dtype=torch.float32))
        up = F.linear(h, mlp.up_proj.weight.detach().to(device=device, dtype=torch.float32))
        z = F.silu(gate) * up
        z_sq_mean = z.square().mean(dim=0)
        ztz = z.t().matmul(z) / max(int(z.shape[0]), 1)
        ztz = 0.5 * (ztz + ztz.t())
        Wd = mlp.down_proj.weight.detach().to(device=device, dtype=torch.float32)
        name = proj_name(layer_idx, "down_proj")
        print(f"[down] layer={layer_idx} tokens={z.shape[0]} metric={tuple(ztz.shape)}", flush=True)
        weights[name] = {
            "basic": quantize_vector_codebook(
                Wd,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 53 + 2017,
                metric="activation",
                scalar_importance=z_sq_mean,
            ).to("cpu", torch.float32),
            "aware": quantize_vector_codebook(
                Wd,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 53 + 2017,
                metric="qk_full",
                scalar_importance=None,
                input_metric=ztz,
                output_importance=torch.ones(Wd.shape[0], device=device, dtype=torch.float32),
            ).to("cpu", torch.float32),
        }
        del h, gate, up, z, z_sq_mean, ztz, Wd
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return weights


@torch.no_grad()
def apply_filtered_down_overlay(
    policy: Any,
    assignment: dict[str, int],
    weights: dict[str, dict[str, torch.Tensor]],
    aware: bool,
    device: torch.device,
) -> int:
    named_modules = dict(policy.model.named_modules())
    applied = 0
    for name, candidates in weights.items():
        if assignment.get(name) != 2:
            continue
        module = named_modules[name]
        selected = candidates["aware" if aware else "basic"]
        module.weight.data.copy_(selected.to(device=device, dtype=module.weight.dtype))
        applied += int(aware)
    return applied


@torch.no_grad()
def build_vo_weights(
    policy: Any,
    caches: dict[int, Any],
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    named_modules = dict(policy.model.named_modules())
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for layer_idx, cache in caches.items():
        attn = named_modules[layer_module_name(layer_idx)]
        v_metric = None
        o_metric = None
        x_sq = None
        h_sq = None
        token_count = 0
        for x_cpu, cos_cpu, sin_cpu, mask_cpu in zip(
            cache.x_chunks, cache.cos_chunks, cache.sin_chunks, cache.mask_chunks
        ):
            x = x_cpu.to(device=device, dtype=torch.float32)
            cos = cos_cpu.to(device=device, dtype=torch.float32)
            sin = sin_cpu.to(device=device, dtype=torch.float32)
            q, _k, krep = project_qk(
                attn,
                x,
                cos,
                sin,
                attn.q_proj.weight.detach(),
                attn.k_proj.weight.detach(),
            )
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)
            v = F.linear(x, attn.v_proj.weight.detach().to(torch.float32)).view(hidden_shape).transpose(1, 2)
            vrep = repeat_kv(v, attn.num_key_value_groups)
            logits = torch.matmul(q.to(torch.float32), krep.to(torch.float32).transpose(2, 3)) * float(attn.scaling)
            if mask_cpu is not None:
                mask = mask_cpu.to(device=device)
                if mask.ndim == 4:
                    if mask.dtype.is_floating_point:
                        logits = logits + mask[..., : logits.shape[-2], : logits.shape[-1]].to(logits.dtype)
                    else:
                        logits = logits.masked_fill(
                            ~mask[..., : logits.shape[-2], : logits.shape[-1]].bool(),
                            torch.finfo(logits.dtype).min,
                        )
            else:
                causal = torch.ones(logits.shape[-2:], device=device, dtype=torch.bool).tril()
                logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1)
            context = torch.matmul(probs, vrep.to(torch.float32)).transpose(1, 2).contiguous()
            context = context.reshape(*input_shape, -1)

            source_weight = probs.square().sum(dim=(1, 2))
            weighted_x = x * source_weight.clamp_min(0).sqrt().unsqueeze(-1)
            flat_weighted_x = weighted_x.reshape(-1, weighted_x.shape[-1])
            flat_context = context.reshape(-1, context.shape[-1])
            part_v = flat_weighted_x.t().matmul(flat_weighted_x)
            part_o = flat_context.t().matmul(flat_context)
            part_x_sq = x.reshape(-1, x.shape[-1]).square().sum(dim=0)
            part_h_sq = flat_context.square().sum(dim=0)
            v_metric = part_v if v_metric is None else v_metric + part_v
            o_metric = part_o if o_metric is None else o_metric + part_o
            x_sq = part_x_sq if x_sq is None else x_sq + part_x_sq
            h_sq = part_h_sq if h_sq is None else h_sq + part_h_sq
            token_count += int(flat_context.shape[0])

        if v_metric is None or o_metric is None or x_sq is None or h_sq is None:
            raise ValueError(f"No V/O calibration data for layer {layer_idx}")
        v_metric = 0.5 * (v_metric + v_metric.t()) / max(token_count, 1)
        o_metric = 0.5 * (o_metric + o_metric.t()) / max(token_count, 1)
        input_imp = x_sq / max(token_count, 1)
        context_imp = h_sq / max(token_count, 1)
        Wo = attn.o_proj.weight.detach().to(device=device, dtype=torch.float32)
        q_head_importance = Wo.square().sum(dim=0).reshape(-1, attn.head_dim)
        num_kv_heads = attn.v_proj.out_features // attn.head_dim
        v_output_importance = q_head_importance.reshape(
            num_kv_heads, attn.num_key_value_groups, attn.head_dim
        ).sum(dim=1).reshape(-1)
        Wv = attn.v_proj.weight.detach().to(device=device, dtype=torch.float32)
        print(f"[vo] layer={layer_idx} tokens={token_count}", flush=True)
        v_name = f"{layer_module_name(layer_idx)}.v_proj"
        o_name = f"{layer_module_name(layer_idx)}.o_proj"
        weights[v_name] = {
            "basic": quantize_vector_codebook(
                Wv,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 67 + 3011,
                metric="activation",
                scalar_importance=input_imp,
            ).to("cpu", torch.float32),
            "aware": quantize_vector_codebook(
                Wv,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 67 + 3011,
                metric="qk_full",
                scalar_importance=None,
                input_metric=v_metric,
                output_importance=v_output_importance,
            ).to("cpu", torch.float32),
        }
        weights[o_name] = {
            "basic": quantize_vector_codebook(
                Wo,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 67 + 4021,
                metric="activation",
                scalar_importance=context_imp,
            ).to("cpu", torch.float32),
            "aware": quantize_vector_codebook(
                Wo,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 67 + 4021,
                metric="qk_full",
                scalar_importance=None,
                input_metric=o_metric,
                output_importance=torch.ones(Wo.shape[0], device=device, dtype=torch.float32),
            ).to("cpu", torch.float32),
        }
        del v_metric, o_metric, x_sq, h_sq, input_imp, context_imp, Wo, Wv
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return weights


@torch.no_grad()
def apply_filtered_vo_overlay(
    policy: Any,
    assignment: dict[str, int],
    weights: dict[str, dict[str, torch.Tensor]],
    use_v: bool,
    use_o: bool,
    device: torch.device,
) -> tuple[int, int]:
    named_modules = dict(policy.model.named_modules())
    n_v = 0
    n_o = 0
    for name, candidates in weights.items():
        if assignment.get(name) != 2:
            continue
        aware = (name.endswith(".v_proj") and use_v) or (name.endswith(".o_proj") and use_o)
        module = named_modules[name]
        module.weight.data.copy_(candidates["aware" if aware else "basic"].to(device=device, dtype=module.weight.dtype))
        n_v += int(aware and name.endswith(".v_proj"))
        n_o += int(aware and name.endswith(".o_proj"))
    return n_v, n_o


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--fit-request-dir", type=Path, default=DEFAULT_FIT_REQUESTS)
    parser.add_argument("--eval-request-dir", type=Path, default=DEFAULT_EVAL_REQUESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", default="2.05,2.1,2.3,2.5,2.7,2.9")
    parser.add_argument("--fit-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-tokens-per-layer", type=int, default=2048)
    parser.add_argument("--max-qk-tokens-per-layer", type=int, default=2048)
    parser.add_argument("--with-down", action="store_true")
    parser.add_argument("--with-vo", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bits_list = [float(x) for x in args.bits.split(",") if x.strip()]
    settings = build_settings(bits_list)
    all_layers = list(range(16))
    positions = [f"vlm_block_{idx}" for idx in all_layers] + [
        "dit_input",
        "final_dit_output",
        "final_action_pred",
        "final_decoded_action",
    ]

    print(f"[combo] loading BF16 policy on {device}", flush=True)
    ref_policy = build_policy(args.model_path, args.device)
    fit_batches = load_batches(ref_policy, args.fit_request_dir, args.fit_limit)
    eval_batches = load_batches(ref_policy, args.eval_request_dir, args.eval_limit)
    print(f"[combo] fit={len(fit_batches)} eval={len(eval_batches)}", flush=True)

    qk_caches = collect_qk_calibration(
        ref_policy,
        fit_batches,
        all_layers,
        device,
        max_cache_tokens_per_layer=args.max_qk_tokens_per_layer,
    )
    q_weights, k_weights = build_quantized_qk_weights(
        ref_policy,
        qk_caches,
        all_layers,
        group_size=args.group_size,
        vector_dim=args.vector_dim,
        codebook_size=args.codebook_size,
        sample_size=args.sample_size,
        iters=args.iters,
        seed=args.seed,
        device=device,
    )

    mlp_caches = collect_mlp_inputs(
        ref_policy,
        fit_batches,
        all_layers,
        device,
        max_tokens_per_layer=args.max_tokens_per_layer,
    )
    lowest_assignment = settings[0]["assignment"]
    swiglu_weights, importance_stats = build_swiglu_weights(
        ref_policy,
        mlp_caches,
        lowest_assignment,
        group_size=args.group_size,
        vector_dim=args.vector_dim,
        codebook_size=args.codebook_size,
        sample_size=args.sample_size,
        iters=args.iters,
        seed=args.seed,
        device=device,
    )
    down_weights = None
    if args.with_down:
        down_weights = build_down_weights(
            ref_policy,
            mlp_caches,
            group_size=args.group_size,
            vector_dim=args.vector_dim,
            codebook_size=args.codebook_size,
            sample_size=args.sample_size,
            iters=args.iters,
            seed=args.seed,
            device=device,
        )
    vo_weights = None
    if args.with_vo:
        vo_weights = build_vo_weights(
            ref_policy,
            qk_caches,
            group_size=args.group_size,
            vector_dim=args.vector_dim,
            codebook_size=args.codebook_size,
            sample_size=args.sample_size,
            iters=args.iters,
            seed=args.seed,
            device=device,
        )
    del qk_caches
    del mlp_caches

    print(f"[combo] building BF16 references for {len(eval_batches)} requests", flush=True)
    refs: dict[str, dict[str, torch.Tensor]] = {}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
        refs[req_id] = run_outputs_with_blocks(
            ref_policy,
            move_to_device(backbone_cpu, device, torch.bfloat16),
            move_to_device(action_cpu, device, torch.bfloat16),
            seed=3000 + req_idx,
            layer_indices=all_layers,
        )
    del ref_policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_request: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for setting in settings:
        assignment = setting["assignment"]
        variants = (
            (
                ("qk_gate_up_down", True, True, True, False, False),
                ("qk_gate_up_down_v", True, True, True, True, False),
                ("qk_gate_up_down_o", True, True, True, False, True),
                ("qk_gate_up_down_vo", True, True, True, True, True),
            )
            if args.with_vo
            else
            (
                ("basic", False, False, False, False, False),
                ("down_only", False, False, True, False, False),
                ("qkfull_down", True, False, True, False, False),
                ("gate_up_down", False, True, True, False, False),
                ("qkfull_gate_up_down", True, True, True, False, False),
            )
            if args.with_down
            else (
                ("basic", False, False, False, False, False),
                ("qkfull_only", True, False, False, False, False),
                ("gate_up_aware", False, True, False, False, False),
                ("qkfull_gate_up", True, True, False, False, False),
            )
        )
        for variant, use_qk, use_swiglu, use_down, use_v, use_o in variants:
            print(f"[combo] bits={setting['target_avg']} variant={variant}", flush=True)
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            policy = build_policy(args.model_path, args.device)
            with patched_env({k: str(v) for k, v in setting["env"].items()}):
                n_quant = quantize_gr00t_vlm27_dit4(
                    policy.model,
                    group_size=args.group_size,
                    device=args.device,
                )
            n_qk = apply_qk_overlay(
                policy,
                assignment,
                q_weights,
                k_weights,
                "qk_full",
                device,
            ) if use_qk else 0
            n_gate, n_up = apply_filtered_swiglu_overlay(
                policy,
                assignment,
                swiglu_weights,
                aware=use_swiglu,
                device=device,
            )
            n_down = apply_filtered_down_overlay(
                policy,
                assignment,
                down_weights or {},
                aware=use_down,
                device=device,
            )
            n_v, n_o = apply_filtered_vo_overlay(
                policy,
                assignment,
                vo_weights or {},
                use_v=use_v,
                use_o=use_o,
                device=device,
            )

            buckets: dict[str, list[dict[str, float]]] = {pos: [] for pos in positions}
            for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
                cand = run_outputs_with_blocks(
                    policy,
                    move_to_device(backbone_cpu, device, torch.bfloat16),
                    move_to_device(action_cpu, device, torch.bfloat16),
                    seed=3000 + req_idx,
                    layer_indices=all_layers,
                )
                for pos in positions:
                    metrics = rel_metrics(refs[req_id][pos], cand[pos])
                    buckets[pos].append(metrics)
                    per_request.append(
                        {
                            "setting": setting["name"],
                            "variant": variant,
                            "request_id": req_id,
                            "position": pos,
                            **metrics,
                        }
                    )
            for pos in positions:
                rels = [row["rel_rms_error"] for row in buckets[pos]]
                base = {
                    "setting": setting["name"],
                    "variant": variant,
                    "target_avg": setting["target_avg"],
                    "actual_avg": setting["stats"]["avg_bits"],
                    "n_quantized": n_quant,
                    "n_qkfull": n_qk,
                    "n_gate_aware": n_gate,
                    "n_up_aware": n_up,
                    "n_down_aware": n_down,
                    "n_v_aware": n_v,
                    "n_o_aware": n_o,
                    "position": pos,
                }
                summary_rows.append({**base, **aggregate(buckets[pos])})
                tail_rows.append(
                    {
                        **base,
                        "mean_rel_rms_error": float(sum(rels) / len(rels)),
                        "p90_rel_rms_error": percentile(rels, 0.90),
                        "p95_rel_rms_error": percentile(rels, 0.95),
                        "max_rel_rms_error": max(rels),
                    }
                )
            del policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.out_dir / "importance_stats.csv", importance_stats)
    write_csv(args.out_dir / "per_request_metrics.csv", per_request)
    write_csv(args.out_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.out_dir / "tail_metrics.csv", tail_rows)
    (args.out_dir / "settings.json").write_text(
        json.dumps(
            {"settings": settings, "args": {k: str(v) for k, v in vars(args).items()}},
            indent=2,
        )
        + "\n"
    )
    print(f"[combo] wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
