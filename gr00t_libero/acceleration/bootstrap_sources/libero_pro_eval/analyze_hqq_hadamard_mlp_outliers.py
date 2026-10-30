#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MLP_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard requires power-of-two size, got {size}")
    h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    while h.shape[0] < size:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / (size**0.5)


def quantize_weight_hqq(weight: torch.Tensor, nbits: int, group_size: int, device: torch.device) -> torch.Tensor:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    out_features, in_features = weight.shape
    eff_group = group_size
    if in_features % group_size != 0:
        for candidate in [64, 32, 16, 8, 4, 2, 1]:
            if candidate <= group_size and in_features % candidate == 0:
                eff_group = candidate
                break

    temp = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    temp.weight.data.copy_(weight.to(torch.bfloat16))
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=eff_group, axis=1)
    hqq = HQQLinear(
        temp,
        quant_config=cfg,
        compute_dtype=torch.bfloat16,
        device=device,
        del_orig=True,
    )
    return hqq.dequantize().to(torch.float32)


def collect_reference_pairs(reference_dir: Path) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    tensors_dir = reference_dir / "reference_tensors"
    per_module: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for path in sorted(tensors_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        full_inputs = payload.get("full_inputs", {})
        full_outputs = payload.get("full_outputs", {})
        for name, x in full_inputs.items():
            if not name.endswith(MLP_SUFFIXES):
                continue
            y = full_outputs.get(name)
            if y is None:
                continue
            per_module.setdefault(name, []).append((x.float(), y.float()))
    return per_module


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    flat = t.reshape(-1).to(torch.float64)
    abs_flat = flat.abs()
    mean = flat.mean()
    std = flat.std(unbiased=False)
    rms = torch.sqrt(torch.mean(flat.square())).item()
    p99 = torch.quantile(abs_flat, 0.99).item()
    p999 = torch.quantile(abs_flat, 0.999).item()
    centered = flat - mean
    var = torch.mean(centered.square())
    if var.item() <= 0:
        kurtosis = 0.0
    else:
        kurtosis = (torch.mean(centered.pow(4)) / (var * var)).item()
    if std.item() <= 0:
        outlier_frac = 0.0
    else:
        outlier_frac = (centered.abs() > (6.0 * std)).float().mean().item()
    return {
        "max_abs": abs_flat.max().item(),
        "p99_abs": p99,
        "p999_abs": p999,
        "rms": rms,
        "tail_ratio_p999": p999 / max(rms, 1e-12),
        "outlier_frac_6sigma": outlier_frac,
        "kurtosis": kurtosis,
    }


def relative_rms_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    diff_rms = torch.sqrt(torch.mean((cand - ref).square())).item()
    ref_rms = torch.sqrt(torch.mean(ref.square())).item()
    return diff_rms / max(ref_rms, 1e-12)


def aggregate_rel_rms(examples: list[tuple[torch.Tensor, torch.Tensor]], fn) -> float:
    diff_sq = 0.0
    ref_sq = 0.0
    elems = 0
    for x_cpu, y_cpu in examples:
        x = x_cpu
        y = y_cpu
        y_hat = fn(x)
        diff_sq += (y_hat - y).to(torch.float64).square().sum().item()
        ref_sq += y.to(torch.float64).square().sum().item()
        elems += y.numel()
    if elems == 0:
        return float("nan")
    return math.sqrt(diff_sq / elems) / max(math.sqrt(ref_sq / elems), 1e-12)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x.square()) * torch.sum(y.square())).item()
    if denom <= 0:
        return None
    return torch.sum(x * y).item() / denom


def module_kind(name: str) -> str:
    if name.endswith("mlp.gate_proj"):
        return "gate_proj"
    if name.endswith("mlp.up_proj"):
        return "up_proj"
    if name.endswith("mlp.down_proj"):
        return "down_proj"
    return "unknown"


def layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def mean_of_dicts(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {k: float("nan") for k in keys}
    return {k: sum(row[k] for row in rows) / len(rows) for k in keys}


def analyze(
    model_path: str,
    reference_dir: Path,
    out_dir: Path,
    nbits: int,
    group_size: int,
    layer_filter: set[int] | None,
) -> None:
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traces = collect_reference_pairs(reference_dir)
    if not traces:
        raise RuntimeError(f"No MLP full_inputs/full_outputs found under {reference_dir}")

    policy = Gr00tPolicy("LIBERO_PANDA", model_path, device=str(device))
    model = policy.model.eval()

    hidden_size = 2048
    r_hidden = hadamard_matrix(hidden_size, device)

    keys_stats = ["max_abs", "p99_abs", "p999_abs", "rms", "tail_ratio_p999", "outlier_frac_6sigma", "kurtosis"]
    per_module_rows: list[dict[str, Any]] = []

    for name in sorted(traces):
        idx = layer_index(name)
        if layer_filter is not None and idx not in layer_filter:
            continue

        module = model.get_submodule(name)
        weight = module.weight.detach().to(device=device, dtype=torch.float32)
        bias = module.bias.detach().to(device=device, dtype=torch.float32) if module.bias is not None else None
        kind = module_kind(name)

        if kind in {"gate_proj", "up_proj"}:
            weight_rot = torch.matmul(weight, r_hidden)

            def plain_fn(x: torch.Tensor) -> torch.Tensor:
                w_hat = quantize_weight_hqq(weight, nbits=nbits, group_size=group_size, device=device)
                return F.linear(x.to(device=device, dtype=torch.float32), w_hat, bias).cpu()

            def had_fn(x: torch.Tensor) -> torch.Tensor:
                w_rot_hat = quantize_weight_hqq(weight_rot, nbits=nbits, group_size=group_size, device=device)
                x_rot = torch.matmul(x.to(device=device, dtype=torch.float32), r_hidden)
                return F.linear(x_rot, w_rot_hat, bias).cpu()

        elif kind == "down_proj":
            weight_rot = torch.matmul(r_hidden.t(), weight)

            def plain_fn(x: torch.Tensor) -> torch.Tensor:
                w_hat = quantize_weight_hqq(weight, nbits=nbits, group_size=group_size, device=device)
                return F.linear(x.to(device=device, dtype=torch.float32), w_hat, bias).cpu()

            def had_fn(x: torch.Tensor) -> torch.Tensor:
                w_rot_hat = quantize_weight_hqq(weight_rot, nbits=nbits, group_size=group_size, device=device)
                y_rot = F.linear(x.to(device=device, dtype=torch.float32), w_rot_hat, bias)
                return torch.matmul(y_rot, r_hidden.t()).cpu()

        else:
            continue

        # Quantize once per module per setting.
        plain_q = quantize_weight_hqq(weight, nbits=nbits, group_size=group_size, device=device)
        had_q = quantize_weight_hqq(weight_rot, nbits=nbits, group_size=group_size, device=device)

        if kind in {"gate_proj", "up_proj"}:
            def plain_fn(x: torch.Tensor, q=plain_q):
                return F.linear(x.to(device=device, dtype=torch.float32), q, bias).cpu()

            def had_fn(x: torch.Tensor, q=had_q):
                x_rot = torch.matmul(x.to(device=device, dtype=torch.float32), r_hidden)
                return F.linear(x_rot, q, bias).cpu()
        else:
            def plain_fn(x: torch.Tensor, q=plain_q):
                return F.linear(x.to(device=device, dtype=torch.float32), q, bias).cpu()

            def had_fn(x: torch.Tensor, q=had_q):
                y_rot = F.linear(x.to(device=device, dtype=torch.float32), q, bias)
                return torch.matmul(y_rot, r_hidden.t()).cpu()

        plain_stats = tensor_stats(weight.detach().cpu())
        had_stats = tensor_stats(weight_rot.detach().cpu())

        row = {
            "module_name": name,
            "layer_idx": idx,
            "kind": kind,
            "plain_weight_rel_rms": relative_rms_error(weight.detach().cpu(), plain_q.detach().cpu()),
            "hadamard_weight_rel_rms": relative_rms_error(weight_rot.detach().cpu(), had_q.detach().cpu()),
            "plain_output_rel_rms": aggregate_rel_rms(traces[name], plain_fn),
            "hadamard_output_rel_rms": aggregate_rel_rms(traces[name], had_fn),
        }
        for key in keys_stats:
            row[f"plain_{key}"] = plain_stats[key]
            row[f"hadamard_{key}"] = had_stats[key]
            row[f"delta_{key}"] = had_stats[key] - plain_stats[key]

        row["reduction_tail_ratio_p999"] = plain_stats["tail_ratio_p999"] - had_stats["tail_ratio_p999"]
        row["reduction_max_abs"] = plain_stats["max_abs"] - had_stats["max_abs"]
        row["improvement_output_rel_rms"] = row["plain_output_rel_rms"] - row["hadamard_output_rel_rms"]
        row["improvement_weight_rel_rms"] = row["plain_weight_rel_rms"] - row["hadamard_weight_rel_rms"]
        per_module_rows.append(row)

    per_type: dict[str, list[dict[str, Any]]] = {}
    for row in per_module_rows:
        per_type.setdefault(row["kind"], []).append(row)

    type_summary = {}
    summary_keys = [
        "plain_weight_rel_rms",
        "hadamard_weight_rel_rms",
        "plain_output_rel_rms",
        "hadamard_output_rel_rms",
        "reduction_tail_ratio_p999",
        "reduction_max_abs",
        "improvement_output_rel_rms",
        "improvement_weight_rel_rms",
    ]
    for kind, rows in per_type.items():
        type_summary[kind] = mean_of_dicts(rows, summary_keys)

    corr_rows = per_module_rows
    correlations = {
        "corr_tail_ratio_reduction_vs_output_improvement": pearson(
            [row["reduction_tail_ratio_p999"] for row in corr_rows],
            [row["improvement_output_rel_rms"] for row in corr_rows],
        ),
        "corr_max_abs_reduction_vs_output_improvement": pearson(
            [row["reduction_max_abs"] for row in corr_rows],
            [row["improvement_output_rel_rms"] for row in corr_rows],
        ),
        "corr_tail_ratio_reduction_vs_weight_improvement": pearson(
            [row["reduction_tail_ratio_p999"] for row in corr_rows],
            [row["improvement_weight_rel_rms"] for row in corr_rows],
        ),
        "corr_max_abs_reduction_vs_weight_improvement": pearson(
            [row["reduction_max_abs"] for row in corr_rows],
            [row["improvement_weight_rel_rms"] for row in corr_rows],
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_module_summary.json").write_text(json.dumps(per_module_rows, indent=2))
    (out_dir / "type_summary.json").write_text(json.dumps(type_summary, indent=2))
    (out_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))

    md_lines = [
        "# HQQ Hadamard MLP Analysis",
        "",
        f"- model_path: `{model_path}`",
        f"- reference_dir: `{reference_dir}`",
        f"- nbits: `{nbits}`",
        f"- group_size: `{group_size}`",
        f"- analyzed_modules: `{len(per_module_rows)}`",
        "",
        "## Type Summary",
        "",
        "| Kind | Plain Output RMS | Hadamard Output RMS | Output Improvement | Tail Ratio Reduction | Max Abs Reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in sorted(type_summary):
        row = type_summary[kind]
        md_lines.append(
            f"| `{kind}` | {row['plain_output_rel_rms']:.6f} | {row['hadamard_output_rel_rms']:.6f} | "
            f"{row['improvement_output_rel_rms']:.6f} | {row['reduction_tail_ratio_p999']:.6f} | {row['reduction_max_abs']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Correlations",
            "",
            "| Metric | Pearson r |",
            "|---|---:|",
        ]
    )
    for key, value in correlations.items():
        formatted = "nan" if value is None else f"{value:.6f}"
        md_lines.append(f"| `{key}` | {formatted} |")
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="@LOCAL_RUNTIME@/models-gr00t-n17-libero/libero_spatial",
        help="GR00T checkpoint used to source the original VLM weights",
    )
    parser.add_argument(
        "--reference-dir",
        default="@WORKSPACE@/libero_pro_eval/request_replay_runs/spinquant_rotation_fit_v1/reference_capture",
    )
    parser.add_argument(
        "--out-dir",
        default="@WORKSPACE@/libero_pro_eval/quantization_analysis/hqq_hadamard_mlp_v1",
    )
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--layers",
        default="0,4,8,12,15",
        help="Comma-separated layer indices. Use 'all' for every captured MLP layer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layers.strip().lower() == "all":
        layer_filter = None
    else:
        layer_filter = {int(x) for x in args.layers.split(",") if x.strip()}
    analyze(
        model_path=args.model_path,
        reference_dir=Path(args.reference_dir),
        out_dir=Path(args.out_dir),
        nbits=args.nbits,
        group_size=args.group_size,
        layer_filter=layer_filter,
    )


if __name__ == "__main__":
    main()
