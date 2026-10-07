#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
from measure_gr00t_qk_diag_action_proxy_2026_06_27 import (
    decoder_layer_name,
    load_batches,
    run_outputs_with_blocks,
    write_csv,
)
from measure_gr00t_qk_diag_codebook_2026_06_27 import quantize_vector_codebook
from measure_gr00t_top5_qkdiag_trainval_proxy_2026_07_02 import (
    aggregate,
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
DEFAULT_OUT = PROJ / "quantization_analysis/gr00t_swiglu_codebook_proxy_2026-07-11"


def mlp_name(layer_idx: int) -> str:
    return f"{decoder_layer_name(layer_idx)}.mlp"


def proj_name(layer_idx: int, proj: str) -> str:
    return f"{mlp_name(layer_idx)}.{proj}"


@dataclass
class MlpInputCache:
    layer_idx: int
    max_tokens: int
    chunks: list[torch.Tensor] = field(default_factory=list)
    token_count: int = 0

    def add(self, h: torch.Tensor) -> None:
        flat = h.detach().reshape(-1, h.shape[-1])
        self.token_count += int(flat.shape[0])
        cached = sum(int(x.shape[0]) for x in self.chunks)
        keep = min(int(flat.shape[0]), max(0, self.max_tokens - cached))
        if keep:
            self.chunks.append(flat[:keep].to("cpu", dtype=torch.bfloat16))

    def tensor(self, device: torch.device) -> torch.Tensor:
        if not self.chunks:
            raise ValueError(f"No cached MLP inputs for layer {self.layer_idx}")
        return torch.cat(self.chunks, dim=0).to(device=device, dtype=torch.float32)


@torch.no_grad()
def collect_mlp_inputs(
    policy: Any,
    batches: list[tuple[str, Any, Any]],
    layer_indices: list[int],
    device: torch.device,
    max_tokens_per_layer: int,
) -> dict[int, MlpInputCache]:
    named_modules = dict(policy.model.named_modules())
    caches = {idx: MlpInputCache(idx, max_tokens_per_layer) for idx in layer_indices}
    handles = []
    for idx in layer_indices:
        mlp = named_modules[mlp_name(idx)]

        def make_hook(layer_idx: int):
            def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                caches[layer_idx].add(inputs[0])

            return hook

        handles.append(mlp.register_forward_pre_hook(make_hook(idx)))

    try:
        for req_idx, (_req_id, backbone_cpu, _action_cpu) in enumerate(batches, start=1):
            print(f"[swiglu] calibration forward {req_idx}/{len(batches)}", flush=True)
            backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
            _ = policy.model.backbone(backbone_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return caches


def silu_derivative(x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


@torch.no_grad()
def build_swiglu_weights(
    policy: Any,
    caches: dict[int, MlpInputCache],
    assignment: dict[str, int],
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    named_modules = dict(policy.model.named_modules())
    weights: dict[str, dict[str, torch.Tensor]] = {}
    stats: list[dict[str, Any]] = []

    for layer_idx, cache in caches.items():
        gate_name = proj_name(layer_idx, "gate_proj")
        up_name = proj_name(layer_idx, "up_proj")
        need_gate = assignment.get(gate_name) == 2
        need_up = assignment.get(up_name) == 2
        if not (need_gate or need_up):
            continue

        mlp = named_modules[mlp_name(layer_idx)]
        h = cache.tensor(device)
        Wg = mlp.gate_proj.weight.detach().to(device=device, dtype=torch.float32)
        Wu = mlp.up_proj.weight.detach().to(device=device, dtype=torch.float32)
        g = F.linear(h, Wg)
        u = F.linear(h, Wu)
        h_sq = h.square()

        print(
            f"[swiglu] layer={layer_idx} tokens={h.shape[0]} gate={need_gate} up={need_up}",
            flush=True,
        )
        if need_gate:
            sensitivity_sq = (silu_derivative(g) * u).square()
            importance = sensitivity_sq.t().matmul(h_sq) / max(int(h.shape[0]), 1)
            input_imp = h_sq.mean(dim=0)
            weights[gate_name] = {
                "basic": quantize_vector_codebook(
                    Wg,
                    group_size=group_size,
                    vector_dim=vector_dim,
                    codebook_size=codebook_size,
                    sample_size=sample_size,
                    iters=iters,
                    seed=seed + layer_idx * 37,
                    metric="activation",
                    scalar_importance=input_imp,
                ).to("cpu", torch.float32),
                "aware": quantize_vector_codebook(
                    Wg,
                    group_size=group_size,
                    vector_dim=vector_dim,
                    codebook_size=codebook_size,
                    sample_size=sample_size,
                    iters=iters,
                    seed=seed + layer_idx * 37,
                    metric="qk_diag",
                    scalar_importance=importance,
                ).to("cpu", torch.float32),
            }
            stats.append(
                {
                    "layer": layer_idx,
                    "projection": "gate_proj",
                    "cached_tokens": int(h.shape[0]),
                    "importance_mean": float(importance.mean().item()),
                    "importance_max_over_mean": float(importance.max().item() / importance.mean().clamp_min(1e-12).item()),
                }
            )
            del sensitivity_sq, importance

        if need_up:
            sensitivity_sq = F.silu(g).square()
            importance = sensitivity_sq.t().matmul(h_sq) / max(int(h.shape[0]), 1)
            input_imp = h_sq.mean(dim=0)
            weights[up_name] = {
                "basic": quantize_vector_codebook(
                    Wu,
                    group_size=group_size,
                    vector_dim=vector_dim,
                    codebook_size=codebook_size,
                    sample_size=sample_size,
                    iters=iters,
                    seed=seed + layer_idx * 37 + 1009,
                    metric="activation",
                    scalar_importance=input_imp,
                ).to("cpu", torch.float32),
                "aware": quantize_vector_codebook(
                    Wu,
                    group_size=group_size,
                    vector_dim=vector_dim,
                    codebook_size=codebook_size,
                    sample_size=sample_size,
                    iters=iters,
                    seed=seed + layer_idx * 37 + 1009,
                    metric="qk_diag",
                    scalar_importance=importance,
                ).to("cpu", torch.float32),
            }
            stats.append(
                {
                    "layer": layer_idx,
                    "projection": "up_proj",
                    "cached_tokens": int(h.shape[0]),
                    "importance_mean": float(importance.mean().item()),
                    "importance_max_over_mean": float(importance.max().item() / importance.mean().clamp_min(1e-12).item()),
                }
            )
            del sensitivity_sq, importance

        del h, h_sq, g, u, Wg, Wu
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return weights, stats


@torch.no_grad()
def apply_swiglu_overlay(
    policy: Any,
    weights: dict[str, dict[str, torch.Tensor]],
    variant: str,
    device: torch.device,
) -> tuple[int, int]:
    named_modules = dict(policy.model.named_modules())
    gate_aware = variant in ("gate_aware", "gate_up_aware")
    up_aware = variant in ("up_aware", "gate_up_aware")
    if variant not in ("basic", "gate_aware", "up_aware", "gate_up_aware"):
        raise ValueError(f"Unknown variant={variant}")
    n_gate = 0
    n_up = 0
    for name, candidates in weights.items():
        use_aware = (name.endswith("gate_proj") and gate_aware) or (name.endswith("up_proj") and up_aware)
        module = named_modules[name]
        selected = candidates["aware" if use_aware else "basic"]
        module.weight.data.copy_(selected.to(device=device, dtype=module.weight.dtype))
        n_gate += int(use_aware and name.endswith("gate_proj"))
        n_up += int(use_aware and name.endswith("up_proj"))
    return n_gate, n_up


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--fit-request-dir", type=Path, default=DEFAULT_FIT_REQUESTS)
    parser.add_argument("--eval-request-dir", type=Path, default=DEFAULT_EVAL_REQUESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", type=float, default=2.5)
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
    parser.add_argument("--variants", default="basic,gate_aware,up_aware,gate_up_aware")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setting = build_settings([args.bits])[0]
    assignment = setting["assignment"]
    target_layers = [
        idx
        for idx in range(16)
        if assignment.get(proj_name(idx, "gate_proj")) == 2 or assignment.get(proj_name(idx, "up_proj")) == 2
    ]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    positions = [f"vlm_block_{idx}" for idx in range(16)] + [
        "dit_input",
        "final_dit_output",
        "final_action_pred",
        "final_decoded_action",
    ]

    print(f"[swiglu] loading BF16 policy on {device}", flush=True)
    ref_policy = build_policy(args.model_path, args.device)
    fit_batches = load_batches(ref_policy, args.fit_request_dir, args.fit_limit)
    eval_batches = load_batches(ref_policy, args.eval_request_dir, args.eval_limit)
    print(
        f"[swiglu] fit_batches={len(fit_batches)} eval_batches={len(eval_batches)} target_layers={target_layers}",
        flush=True,
    )
    caches = collect_mlp_inputs(
        ref_policy,
        fit_batches,
        target_layers,
        device,
        max_tokens_per_layer=args.max_tokens_per_layer,
    )
    swiglu_weights, importance_stats = build_swiglu_weights(
        ref_policy,
        caches,
        assignment,
        group_size=args.group_size,
        vector_dim=args.vector_dim,
        codebook_size=args.codebook_size,
        sample_size=args.sample_size,
        iters=args.iters,
        seed=args.seed,
        device=device,
    )

    print(f"[swiglu] building BF16 references for {len(eval_batches)} requests", flush=True)
    refs: dict[str, dict[str, torch.Tensor]] = {}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
        refs[req_id] = run_outputs_with_blocks(
            ref_policy,
            move_to_device(backbone_cpu, device, torch.bfloat16),
            move_to_device(action_cpu, device, torch.bfloat16),
            seed=3000 + req_idx,
            layer_indices=list(range(16)),
        )
    del ref_policy, caches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_request: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[swiglu] evaluating variant={variant}", flush=True)
        policy = build_policy(args.model_path, args.device)
        with patched_env({k: str(v) for k, v in setting["env"].items()}):
            n_quant = quantize_gr00t_vlm27_dit4(policy.model, group_size=args.group_size, device=args.device)
        n_gate, n_up = apply_swiglu_overlay(policy, swiglu_weights, variant, device)
        buckets: dict[str, list[dict[str, float]]] = {pos: [] for pos in positions}
        for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
            cand = run_outputs_with_blocks(
                policy,
                move_to_device(backbone_cpu, device, torch.bfloat16),
                move_to_device(action_cpu, device, torch.bfloat16),
                seed=3000 + req_idx,
                layer_indices=list(range(16)),
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
                "target_avg": args.bits,
                "actual_avg": setting["stats"]["avg_bits"],
                "n_quantized": n_quant,
                "n_gate_aware": n_gate,
                "n_up_aware": n_up,
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
            {
                "setting": setting,
                "target_layers": target_layers,
                "target_modules": sorted(swiglu_weights),
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[swiglu] wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
