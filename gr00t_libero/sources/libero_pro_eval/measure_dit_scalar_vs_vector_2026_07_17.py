#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tree

from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq
from gr00t.eval.cosmos_quant_shim import NonUniform2BitLinear
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import MsgSerializer
from measure_codebook_mechanism_ab_2026_06_16 import (
    activation_weighted_rel_rms,
    effective_group_size,
    quantize_vector_codebook_weight,
    sample_flat,
    sample_stats,
    weight_metrics,
)
from measure_hotspot_protection_proxy import collate_request, move_to_device, run_outputs


PROJ = Path("@WORKSPACE@/libero_pro_eval")
DEFAULT_MODEL = "@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial"
DEFAULT_REQUESTS = str(PROJ / "request_replay_runs/bf16_capture/requests")
DEFAULT_MODULE_CSV = str(PROJ / "quantization_analysis/dit_input_sensitivity_hqq2_v1/module_sensitivity_all.csv")
DEFAULT_OUT = str(PROJ / "quantization_analysis/scalar_vs_vector_codebook_2026-06-16/shards")

VARIANTS = ["hqq2", "scalar_kmeans2", "vector_codebook_d4k256"]


def read_modules(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "module_name": row["module_name"],
                    "layer": int(float(row["layer"])),
                    "module_type": row["module_type"],
                    "rank_hqq2_dit_input": int(float(row["rank"])),
                    "hqq2_dit_input_err": float(row["dit_input_global_rel_rms_error"]),
                }
            )
    return rows


def load_batches(policy: Gr00tPolicy, request_dir: Path, limit: int) -> list[tuple[str, Any, Any]]:
    """Keep action_inputs. The VLM original discards them because
    `policy.model.backbone(...)` is all it needs; the DiT lives in the action
    head and never runs without them.
    """
    batches = []
    for path in sorted(request_dir.glob("*.msgpack"))[:limit]:
        payload = MsgSerializer.from_bytes(path.read_bytes())
        collated = collate_request(policy, payload["observation"])
        backbone_inputs, action_inputs = policy.model.prepare_input(collated)
        to_cpu = lambda s: tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, s)
        batches.append((payload["request_id"], to_cpu(backbone_inputs), to_cpu(action_inputs)))
    return batches


@torch.no_grad()
def quantize_scalar_kmeans_weight(
    W: torch.Tensor,
    *,
    group_size: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    """HQQ-like group scaling + learned scalar centroids shared per module.

    This isolates the effect of non-uniform scalar centroids. It keeps the same
    per-row/per-group max-abs normalization as vector codebook, but assigns each
    scalar independently to one of K=4 learned levels.
    """
    out_features, in_features = W.shape
    eff_group = effective_group_size(in_features, group_size)
    num_groups = in_features // eff_group
    Wg = W.reshape(out_features, num_groups, eff_group)
    scales = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    normalized = (Wg / scales).clamp(-1.0, 1.0)

    values = normalized.reshape(-1, 1).contiguous()
    total = values.shape[0]
    gen = torch.Generator(device=W.device)
    gen.manual_seed(seed)
    if total > sample_size:
        idx = torch.randperm(total, device=W.device, generator=gen)[:sample_size]
        samples = values[idx]
    else:
        samples = values

    codebook = NonUniform2BitLinear._weighted_kmeans_nd(
        samples,
        codebook_size,
        iters,
        weights=None,
        seed=seed,
    ).reshape(codebook_size, 1)
    indices = NonUniform2BitLinear._assign_vectors_chunked(values, codebook)
    Wq_norm = codebook[indices].reshape_as(normalized)
    return (Wq_norm * scales).reshape_as(W).detach()


@dataclass
class VariantAcc:
    ref_sq: float = 0.0
    cand_sq: float = 0.0
    diff_sq: float = 0.0
    dot: float = 0.0
    elems: int = 0
    sign_flip: int = 0
    sign_total: int = 0
    abs_samples: list[torch.Tensor] = field(default_factory=list)

    def update(self, ref: torch.Tensor, cand: torch.Tensor, max_samples: int) -> None:
        ref64 = ref.detach().reshape(-1).to(torch.float64)
        cand64 = cand.detach().reshape(-1).to(torch.float64)
        diff = cand64 - ref64
        self.ref_sq += float(ref64.square().sum().item())
        self.cand_sq += float(cand64.square().sum().item())
        self.diff_sq += float(diff.square().sum().item())
        self.dot += float((ref64 * cand64).sum().item())
        self.elems += int(ref64.numel())

        ref32 = ref.detach().reshape(-1).to(torch.float32)
        cand32 = cand.detach().reshape(-1).to(torch.float32)
        active = ref32.abs() > 1e-6
        self.sign_total += int(active.sum().item())
        self.sign_flip += int(((ref32 * cand32) < 0).logical_and(active).sum().item())
        self.abs_samples.append(sample_flat(diff.abs().to(torch.float32), max_samples))

    def metrics(self) -> dict[str, float]:
        ref_rms = math.sqrt(self.ref_sq / max(1, self.elems))
        diff_rms = math.sqrt(self.diff_sq / max(1, self.elems))
        samples = torch.cat(self.abs_samples) if self.abs_samples else torch.empty(0)
        tail = sample_stats(samples)
        return {
            "output_rel_rms": diff_rms / max(ref_rms, 1e-12),
            "output_p99_abs": tail["p99"],
            "output_p999_abs": tail["p999"],
            "output_sign_flip_rate": self.sign_flip / max(1, self.sign_total),
            "output_cosine": self.dot / max(1e-12, math.sqrt(self.ref_sq * self.cand_sq)),
        }


@dataclass
class ModuleData:
    row: dict[str, Any]
    params: int
    bias: torch.Tensor | None
    ref_weight: torch.Tensor
    weights: dict[str, torch.Tensor]
    weight_metrics: dict[str, dict[str, float]]
    accs: dict[str, VariantAcc] = field(default_factory=lambda: {name: VariantAcc() for name in VARIANTS})
    x_sq_sum: torch.Tensor | None = None
    x_count: int = 0

    def update_activation_diag(self, x: torch.Tensor) -> None:
        flat = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
        sq = flat.square().sum(dim=0)
        if self.x_sq_sum is None:
            self.x_sq_sum = sq.detach().clone()
        else:
            self.x_sq_sum += sq.detach()
        self.x_count += int(flat.shape[0])

    def activation_diag(self) -> torch.Tensor | None:
        if self.x_sq_sum is None or self.x_count <= 0:
            return None
        return self.x_sq_sum / float(self.x_count)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--request-dir", default=DEFAULT_REQUESTS)
    parser.add_argument("--module-csv", default=DEFAULT_MODULE_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-request", type=int, default=2048)
    parser.add_argument("--scalar-sample", type=int, default=200000)
    parser.add_argument("--scalar-iters", type=int, default=24)
    parser.add_argument("--codebook-sample", type=int, default=200000)
    parser.add_argument("--codebook-iters", type=int, default=12)
    parser.add_argument("--codebook-vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"scalar_vector_shard{args.shard_index:02d}_of{args.num_shards:02d}.csv"
    if out_csv.exists():
        print(f"[scalar-vector] skip existing {out_csv}", flush=True)
        return

    module_rows_all = read_modules(Path(args.module_csv))
    module_rows = [row for idx, row in enumerate(module_rows_all) if idx % args.num_shards == args.shard_index]
    module_names = {row["module_name"] for row in module_rows}
    print(f"[scalar-vector] shard {args.shard_index}/{args.num_shards}: modules={len(module_rows)}", flush=True)

    policy = Gr00tPolicy("LIBERO_PANDA", args.model_path, device=args.device)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)
    batches = load_batches(policy, Path(args.request_dir), args.limit)
    named_modules = dict(policy.model.named_modules())
    rows_by_name = {row["module_name"]: row for row in module_rows}
    qdata: dict[str, ModuleData] = {}

    for i, name in enumerate(sorted(module_names), start=1):
        module = named_modules.get(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear: {type(module)}")
        print(f"[scalar-vector] quantizing {i}/{len(module_names)} {name}", flush=True)
        W = module.weight.detach().to(device=device, dtype=torch.float32)
        bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=torch.float32)
        weights = {
            "hqq2": quantize_weight_hqq(W, nbits=2, group_size=args.group_size, device=device).to(torch.float32),
            "scalar_kmeans2": quantize_scalar_kmeans_weight(
                W,
                group_size=args.group_size,
                codebook_size=4,
                sample_size=args.scalar_sample,
                iters=args.scalar_iters,
                seed=args.seed,
            ),
            "vector_codebook_d4k256": quantize_vector_codebook_weight(
                W,
                group_size=args.group_size,
                vector_dim=args.codebook_vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.codebook_sample,
                iters=args.codebook_iters,
                seed=args.seed,
                act_scales=None,
                activation_weighted=False,
                shuffle_grouping=False,
            ),
        }
        wmetrics = {variant: weight_metrics(W, Wq, None) for variant, Wq in weights.items()}
        base = rows_by_name[name]
        base["in_features"] = int(module.in_features)
        base["out_features"] = int(module.out_features)
        qdata[name] = ModuleData(
            row=base,
            params=int(module.in_features) * int(module.out_features),
            bias=bias,
            ref_weight=W,
            weights=weights,
            weight_metrics=wmetrics,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    handles = []

    def make_hook(module_name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            data = qdata[module_name]
            x = inputs[0].detach().to(torch.float32)
            ref = output.detach().to(torch.float32)
            data.update_activation_diag(x)
            for variant, Wq in data.weights.items():
                cand = F.linear(x, Wq, data.bias)
                data.accs[variant].update(ref, cand, max_samples=args.samples_per_request)
        return hook

    for name in module_names:
        handles.append(named_modules[name].register_forward_hook(make_hook(name)))

    # run_outputs goes backbone -> action_head._encode_features -> DiT denoising,
    # so the DiT hooks actually fire. `policy.model.backbone(...)` alone leaves
    # every action-head hook untouched and silently yields all-zero output stats.
    for req_idx, (_req_id, backbone_cpu, action_cpu) in enumerate(batches, start=1):
        print(f"[dit-scalar-vector] forward {req_idx}/{len(batches)}", flush=True)
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        _ = run_outputs(policy, backbone_inputs, action_inputs, seed=7000 + req_idx)

    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    for name in sorted(qdata):
        data = qdata[name]
        base = data.row
        row: dict[str, Any] = {
            "module_name": name,
            "layer": base["layer"],
            "module_type": base["module_type"],
            "rank_hqq2_dit_input": base["rank_hqq2_dit_input"],
            "hqq2_dit_input_err": base["hqq2_dit_input_err"],
            "params": data.params,
            "in_features": base.get("in_features", ""),
            "out_features": base.get("out_features", ""),
        }
        actual_act_diag = data.activation_diag()
        for variant in VARIANTS:
            for key, value in data.weight_metrics[variant].items():
                row[f"{variant}_{key}"] = value
            row[f"{variant}_actual_activation_weighted_weight_rel_rms"] = activation_weighted_rel_rms(
                data.ref_weight,
                data.weights[variant],
                actual_act_diag,
            )
            for key, value in data.accs[variant].metrics().items():
                row[f"{variant}_{key}"] = value
        row["scalar_output_gain_vs_hqq"] = row["hqq2_output_rel_rms"] - row["scalar_kmeans2_output_rel_rms"]
        row["vector_output_gain_vs_hqq"] = row["hqq2_output_rel_rms"] - row["vector_codebook_d4k256_output_rel_rms"]
        row["vector_output_gain_vs_scalar"] = row["scalar_kmeans2_output_rel_rms"] - row[
            "vector_codebook_d4k256_output_rel_rms"
        ]
        rows.append(row)

    write_csv(out_csv, rows)
    (out_dir / f"scalar_vector_shard{args.shard_index:02d}_of{args.num_shards:02d}.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    print(f"[scalar-vector] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
