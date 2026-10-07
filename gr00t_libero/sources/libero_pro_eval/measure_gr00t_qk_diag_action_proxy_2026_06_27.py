#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import tree

from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import MsgSerializer
from measure_gr00t_qk_diag_codebook_2026_06_27 import (
    LayerCache,
    layer_module_name,
    project_qk,
    quantize_vector_codebook,
)
from measure_hotspot_protection_proxy import collate_request, move_to_device, rel_metrics, run_outputs


PROJ = Path("@WORKSPACE@/libero_pro_eval")
DEFAULT_MODEL = "@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial"
DEFAULT_REQUESTS = str(PROJ / "request_replay_runs/bf16_capture/requests")
DEFAULT_OUT = str(PROJ / "quantization_analysis/gr00t_qk_diag_action_proxy_2026-06-27")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def load_batches(policy: Gr00tPolicy, request_dir: Path, limit: int) -> list[tuple[str, Any, Any]]:
    batches = []
    for path in sorted(request_dir.glob("*.msgpack"))[:limit]:
        payload = MsgSerializer.from_bytes(path.read_bytes())
        collated = collate_request(policy, payload["observation"])
        backbone_inputs, action_inputs = policy.model.prepare_input(collated)
        batches.append(
            (
                payload["request_id"],
                tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, backbone_inputs),
                tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, action_inputs),
            )
        )
    return batches


def decoder_layer_name(layer_idx: int) -> str:
    return f"backbone.model.model.language_model.layers.{layer_idx}"


@torch.no_grad()
def run_outputs_with_blocks(
    policy: Gr00tPolicy,
    backbone_inputs: Any,
    action_inputs: Any,
    seed: int,
    layer_indices: list[int],
) -> dict[str, torch.Tensor]:
    named_modules = dict(policy.model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[f"vlm_block_{idx}"] = hidden.detach().cpu().float()

        return hook

    for idx in layer_indices:
        handles.append(named_modules[decoder_layer_name(idx)].register_forward_hook(make_hook(idx)))
    try:
        out = run_outputs(policy, backbone_inputs, action_inputs, seed)
    finally:
        for handle in handles:
            handle.remove()
    out.update(captured)
    return out


def collect_qk_calibration(
    policy: Gr00tPolicy,
    batches: list[tuple[str, Any, Any]],
    layer_indices: list[int],
    device: torch.device,
    max_cache_tokens_per_layer: int,
) -> dict[int, LayerCache]:
    named_modules = dict(policy.model.named_modules())
    caches = {idx: LayerCache(layer_idx=idx, name=layer_module_name(idx)) for idx in layer_indices}
    handles = []
    for idx in layer_indices:
        attn = named_modules[layer_module_name(idx)]

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x = kwargs["hidden_states"]
                cos, sin = kwargs["position_embeddings"]
                attention_mask = kwargs.get("attention_mask")
                q, _k, krep = project_qk(module, x, cos, sin, module.q_proj.weight.detach(), module.k_proj.weight.detach())
                caches[layer_idx].add(
                    x=x,
                    q=q,
                    krep=krep,
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
                    max_cache_tokens=max_cache_tokens_per_layer,
                )

            return hook

        handles.append(attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))

    for req_idx, (_req_id, backbone_cpu, _action_cpu) in enumerate(batches, start=1):
        print(f"[qk-action] calibration forward {req_idx}/{len(batches)}", flush=True)
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        _ = policy.model.backbone(backbone_inputs)

    for handle in handles:
        handle.remove()
    return caches


def input_metric_from_cache(cache: LayerCache, device: torch.device) -> torch.Tensor:
    xtx = None
    n_tokens = 0
    for x_cpu in cache.x_chunks:
        x = x_cpu.to(device=device, dtype=torch.float32).reshape(-1, x_cpu.shape[-1])
        part = x.t().matmul(x)
        xtx = part if xtx is None else xtx + part
        n_tokens += int(x.shape[0])
    if xtx is None:
        if cache.input_sq is None:
            raise ValueError(f"Layer {cache.layer_idx} has no cached input statistics")
        return torch.diag(cache.input_sq.to(device=device, dtype=torch.float32).clamp_min(1e-12))
    xtx = xtx / max(n_tokens, 1)
    xtx = 0.5 * (xtx + xtx.t())
    return xtx


@torch.no_grad()
def build_quantized_qk_weights(
    policy: Gr00tPolicy,
    caches: dict[int, LayerCache],
    layer_indices: list[int],
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    named_modules = dict(policy.model.named_modules())
    q_weights: dict[int, dict[str, torch.Tensor]] = {}
    k_weights: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in layer_indices:
        cache = caches[layer_idx]
        attn = named_modules[layer_module_name(layer_idx)]
        input_imp = (cache.input_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        input_metric = input_metric_from_cache(cache, device)
        k_imp = (cache.krep_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        q_imp = (cache.q_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        q_groups = attn.num_key_value_groups
        num_key_value_heads = attn.k_proj.out_features // attn.head_dim
        q_imp_grouped = q_imp.reshape(num_key_value_heads, q_groups, attn.head_dim).sum(dim=1)

        Wq = attn.q_proj.weight.detach().to(device=device, dtype=torch.float32)
        Wk = attn.k_proj.weight.detach().to(device=device, dtype=torch.float32)
        qk_q_importance = (k_imp.reshape(-1)[:, None] * input_imp[None, :]).contiguous()
        qk_k_importance = (q_imp_grouped.reshape(-1)[:, None] * input_imp[None, :]).contiguous()

        q_weights[layer_idx] = {
            "hqq2": quantize_weight_hqq(Wq, nbits=2, group_size=group_size, device=device).to("cpu", torch.float32)
        }
        k_weights[layer_idx] = {
            "hqq2": quantize_weight_hqq(Wk, nbits=2, group_size=group_size, device=device).to("cpu", torch.float32)
        }
        for metric in ("euclidean", "activation", "qk_diag", "qk_full"):
            print(f"[qk-action] quantizing layer {layer_idx} metric={metric}", flush=True)
            q_weights[layer_idx][metric] = quantize_vector_codebook(
                Wq,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 17,
                metric=metric,
                scalar_importance=input_imp if metric == "activation" else qk_q_importance if metric == "qk_diag" else None,
                input_metric=input_metric if metric == "qk_full" else None,
                output_importance=k_imp.reshape(-1) if metric == "qk_full" else None,
            ).to("cpu", torch.float32)
            k_weights[layer_idx][metric] = quantize_vector_codebook(
                Wk,
                group_size=group_size,
                vector_dim=vector_dim,
                codebook_size=codebook_size,
                sample_size=sample_size,
                iters=iters,
                seed=seed + layer_idx * 17 + 1009,
                metric=metric,
                scalar_importance=input_imp if metric == "activation" else qk_k_importance if metric == "qk_diag" else None,
                input_metric=input_metric if metric == "qk_full" else None,
                output_importance=q_imp_grouped.reshape(-1) if metric == "qk_full" else None,
            ).to("cpu", torch.float32)
    return q_weights, k_weights


def apply_variant(
    policy: Gr00tPolicy,
    layer_indices: list[int],
    q_weights: dict[int, dict[str, torch.Tensor]],
    k_weights: dict[int, dict[str, torch.Tensor]],
    variant: str,
    device: torch.device,
) -> None:
    named_modules = dict(policy.model.named_modules())
    for layer_idx in layer_indices:
        attn = named_modules[layer_module_name(layer_idx)]
        attn.q_proj.weight.data.copy_(q_weights[layer_idx][variant].to(device=device, dtype=attn.q_proj.weight.dtype))
        attn.k_proj.weight.data.copy_(k_weights[layer_idx][variant].to(device=device, dtype=attn.k_proj.weight.dtype))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--request-dir", default=DEFAULT_REQUESTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-cache-tokens-per-layer", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    positions = [f"vlm_block_{idx}" for idx in layer_indices] + ["dit_input", "final_decoded_action"]

    print(f"[qk-action] loading policy on {device}", flush=True)
    policy = Gr00tPolicy("LIBERO_PANDA", args.model_path, device=args.device)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)
    batches = load_batches(policy, Path(args.request_dir), args.limit)

    caches = collect_qk_calibration(
        policy,
        batches,
        layer_indices,
        device,
        max_cache_tokens_per_layer=args.max_cache_tokens_per_layer,
    )
    q_weights, k_weights = build_quantized_qk_weights(
        policy,
        caches,
        layer_indices,
        group_size=args.group_size,
        vector_dim=args.vector_dim,
        codebook_size=args.codebook_size,
        sample_size=args.sample_size,
        iters=args.iters,
        seed=args.seed,
        device=device,
    )

    print(f"[qk-action] building BF16 references for {len(batches)} requests", flush=True)
    refs: dict[str, dict[str, torch.Tensor]] = {}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        refs[req_id] = run_outputs_with_blocks(policy, backbone_inputs, action_inputs, seed=3000 + req_idx, layer_indices=layer_indices)

    variants = ["hqq2", "euclidean", "activation", "qk_diag", "qk_full"]
    per_request: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[qk-action] evaluating variant={variant}", flush=True)
        apply_variant(policy, layer_indices, q_weights, k_weights, variant, device)
        buckets: dict[str, list[dict[str, float]]] = {pos: [] for pos in positions}
        for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
            backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
            action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
            cand = run_outputs_with_blocks(
                policy,
                backbone_inputs,
                action_inputs,
                seed=3000 + req_idx,
                layer_indices=layer_indices,
            )
            ref = refs[req_id]
            for pos in positions:
                metrics = rel_metrics(ref[pos], cand[pos])
                buckets[pos].append(metrics)
                per_request.append({"variant": variant, "request_id": req_id, "position": pos, **metrics})
        for pos in positions:
            summary_rows.append({"variant": variant, "position": pos, **aggregate(buckets[pos])})
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "per_request_metrics.csv", per_request)
    write_csv(out_dir / "summary_metrics.csv", summary_rows)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[qk-action] wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
