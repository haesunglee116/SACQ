#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from build_mixed_bit_proxy_settings_v1 import allocate, assignment_stats, load_rows, specs_from_assignment
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
from gr00t.policy.gr00t_policy import Gr00tPolicy
from measure_gr00t_qk_diag_action_proxy_2026_06_27 import (
    build_quantized_qk_weights,
    collect_qk_calibration,
    load_batches,
    run_outputs_with_blocks,
    write_csv,
)
from measure_gr00t_qk_diag_codebook_2026_06_27 import layer_module_name
from measure_hotspot_protection_proxy import move_to_device, rel_metrics


PROJ = Path("@WORKSPACE@/libero_pro_eval")
DEFAULT_MODEL = "@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial"
DEFAULT_FIT_REQUESTS = PROJ / "request_replay_runs/calib_4suite_2026-07-02/train/requests"
DEFAULT_EVAL_REQUESTS = PROJ / "request_replay_runs/calib_4suite_2026-07-02/val/requests"
DEFAULT_OUT = PROJ / "quantization_analysis/gr00t_top5_qkdiag_trainval_proxy_2026-07-02"
SENS2 = PROJ / "quantization_analysis/dit_input_sensitivity_hqq2_v1/module_sensitivity_all.csv"
SENS3 = PROJ / "quantization_analysis/dit_input_sensitivity_hqq3_v1/module_sensitivity_all.csv"


CODEBOOK_ENV = {
    "VLM_BITS": "2",
    "DIT_BITS": "4",
    "VLM_QUANT_MODE": "uniform",
    "DIT_QUANT_MODE": "uniform",
    "VLM_ROTATION_MODE": "none",
    "VLM_2BIT_CODEBOOK_MODE": "act_vector_kmeans",
    "VLM_2BIT_CODEBOOK_SAMPLE": "200000",
    "VLM_2BIT_CODEBOOK_ITERS": "12",
    "VLM_2BIT_CODEBOOK_VECTOR_DIM": "4",
    "VLM_2BIT_CODEBOOK_SIZE": "256",
    "VLM_2BIT_CODEBOOK_SEED": "20260601",
}


@contextmanager
def patched_env(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return float(xs[idx])


def build_settings(bits_list: list[float]) -> list[dict[str, Any]]:
    rows = load_rows(SENS2, SENS3)
    settings = []
    for bits in bits_list:
        assignment = allocate(rows, target_avg=bits, rule="gain23", top4=5)
        stats = assignment_stats(rows, assignment)
        env = dict(CODEBOOK_ENV)
        env["VLM_FORCE_BITS_SPECS"] = specs_from_assignment(assignment)
        settings.append(
            {
                "name": f"gain23_{bits:.1f}_top5to4_act_d4_k256",
                "target_avg": bits,
                "stats": stats,
                "assignment": assignment,
                "env": env,
            }
        )
    return settings


def apply_qk_overlay(
    policy: Gr00tPolicy,
    assignment: dict[str, int],
    q_weights: dict[int, dict[str, torch.Tensor]],
    k_weights: dict[int, dict[str, torch.Tensor]],
    variant: str,
    device: torch.device,
) -> int:
    named_modules = dict(policy.model.named_modules())
    n = 0
    if variant not in q_weights[0]:
        raise ValueError(f"Unknown q/k overlay variant={variant}")
    for layer_idx in range(16):
        attn = named_modules[layer_module_name(layer_idx)]
        q_name = f"backbone.model.model.language_model.layers.{layer_idx}.self_attn.q_proj"
        k_name = f"backbone.model.model.language_model.layers.{layer_idx}.self_attn.k_proj"
        if assignment.get(q_name, 2) == 2:
            attn.q_proj.weight.data.copy_(
                q_weights[layer_idx][variant].to(device=device, dtype=attn.q_proj.weight.dtype)
            )
            n += 1
        if assignment.get(k_name, 2) == 2:
            attn.k_proj.weight.data.copy_(
                k_weights[layer_idx][variant].to(device=device, dtype=attn.k_proj.weight.dtype)
            )
            n += 1
    return n


def build_policy(model_path: str, device: str) -> Gr00tPolicy:
    policy = Gr00tPolicy("LIBERO_PANDA", model_path, device=device)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)
    return policy


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--fit-request-dir", type=Path, default=DEFAULT_FIT_REQUESTS)
    parser.add_argument("--eval-request-dir", type=Path, default=DEFAULT_EVAL_REQUESTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bits", default="2.7,2.5,2.3,2.1")
    parser.add_argument("--fit-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-cache-tokens-per-layer", type=int, default=8192)
    parser.add_argument("--variants", default="normal,qkdiag")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bits_list = [float(x) for x in args.bits.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    settings = build_settings(bits_list)
    layer_indices = list(range(16))
    positions = [f"vlm_block_{idx}" for idx in layer_indices] + [
        "dit_input",
        "final_dit_output",
        "final_action_pred",
        "final_decoded_action",
    ]

    print(f"[trainval-qkdiag] loading BF16 policy on {device}", flush=True)
    ref_policy = build_policy(args.model_path, args.device)
    fit_batches = load_batches(ref_policy, args.fit_request_dir, args.fit_limit)
    eval_batches = load_batches(ref_policy, args.eval_request_dir, args.eval_limit)
    print(
        f"[trainval-qkdiag] fit_batches={len(fit_batches)} eval_batches={len(eval_batches)}",
        flush=True,
    )

    caches = collect_qk_calibration(
        ref_policy,
        fit_batches,
        layer_indices,
        device,
        max_cache_tokens_per_layer=args.max_cache_tokens_per_layer,
    )
    q_weights, k_weights = build_quantized_qk_weights(
        ref_policy,
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

    print(f"[trainval-qkdiag] building BF16 eval references for {len(eval_batches)} requests", flush=True)
    refs: dict[str, dict[str, torch.Tensor]] = {}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        refs[req_id] = run_outputs_with_blocks(
            ref_policy,
            backbone_inputs,
            action_inputs,
            seed=3000 + req_idx,
            layer_indices=layer_indices,
        )
    del ref_policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_request: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for setting in settings:
        for variant in variants:
            print(
                f"[trainval-qkdiag] evaluating {setting['name']} variant={variant} stats={setting['stats']}",
                flush=True,
            )
            policy = build_policy(args.model_path, args.device)
            with patched_env({k: str(v) for k, v in setting["env"].items()}):
                n_quant = quantize_gr00t_vlm27_dit4(policy.model, group_size=args.group_size, device=args.device)
            n_qk_overlay = 0
            if variant == "qkdiag":
                n_qk_overlay = apply_qk_overlay(policy, setting["assignment"], q_weights, k_weights, "qk_diag", device)
            elif variant == "qkfull":
                n_qk_overlay = apply_qk_overlay(policy, setting["assignment"], q_weights, k_weights, "qk_full", device)
            elif variant != "normal":
                raise ValueError(f"Unknown variant={variant}")

            buckets: dict[str, list[dict[str, float]]] = {pos: [] for pos in positions}
            for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(eval_batches):
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
                    per_request.append(
                        {
                            "setting": setting["name"],
                            "variant": variant,
                            "target_avg": setting["target_avg"],
                            "actual_avg": setting["stats"]["avg_bits"],
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
                    "num_2bit": setting["stats"]["num_2bit"],
                    "num_3bit": setting["stats"]["num_3bit"],
                    "num_4bit": setting["stats"]["num_4bit"],
                    "n_quantized": n_quant,
                            "n_qk_overlay": n_qk_overlay,
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

    write_csv(args.out_dir / "per_request_metrics.csv", per_request)
    write_csv(args.out_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.out_dir / "tail_metrics.csv", tail_rows)
    (args.out_dir / "settings.json").write_text(
        json.dumps({"settings": settings, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2) + "\n"
    )
    print(f"[trainval-qkdiag] wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
