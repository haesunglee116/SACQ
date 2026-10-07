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
from measure_hotspot_protection_proxy import collate_request, move_to_device


PROJ = Path("@WORKSPACE@/libero_pro_eval")
DEFAULT_MODEL = "@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial"
DEFAULT_REQUESTS = str(PROJ / "request_replay_runs/bf16_capture/requests")
DEFAULT_MODULE_CSV = str(PROJ / "quantization_analysis/dit_input_sensitivity_hqq2_v1/module_sensitivity_all.csv")
DEFAULT_OUT = str(PROJ / "quantization_analysis/codebook_mechanism_ab_2026-06-16_v4/shards")
ACT_SCALES = PROJ / "quantization_analysis/act_scales.pt"


VARIANTS = [
    "hqq2",
    "cb_unweighted",
    "cb_act",
    "cb_act_shuffled",
]


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


def load_batches(policy: Gr00tPolicy, request_dir: Path, limit: int) -> list[tuple[str, Any]]:
    batches = []
    for path in sorted(request_dir.glob("*.msgpack"))[:limit]:
        payload = MsgSerializer.from_bytes(path.read_bytes())
        collated = collate_request(policy, payload["observation"])
        backbone_inputs, _ = policy.model.prepare_input(collated)
        batches.append(
            (
                payload["request_id"],
                tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, backbone_inputs),
            )
        )
    return batches


def act_scale_for_module(full: str, act_scales: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor | None:
    scale_key = full
    if scale_key.startswith("backbone.model."):
        scale_key = scale_key[len("backbone.model.") :]
    candidates = [scale_key]
    if full.startswith("backbone.model.model.language_model."):
        suffix = full[len("backbone.model.model.language_model.") :]
        candidates.append("model." + suffix)
        candidates.append(suffix)
    if scale_key.startswith("model.language_model."):
        candidates.append("model." + scale_key[len("model.language_model.") :])
    for key in candidates:
        value = act_scales.get(key)
        if value is not None:
            return value.to(device=device, dtype=torch.float32).flatten()
    return None


def sample_flat(t: torch.Tensor, max_samples: int) -> torch.Tensor:
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n <= max_samples:
        return flat.to("cpu", dtype=torch.float32)
    idx = torch.linspace(0, n - 1, steps=max_samples, device=flat.device).long().clamp_max(n - 1)
    return flat[idx].to("cpu", dtype=torch.float32)


def sample_stats(samples: torch.Tensor) -> dict[str, float]:
    x = samples.to(torch.float64)
    if x.numel() == 0:
        return {"p99": 0.0, "p999": 0.0, "kurtosis": 0.0}
    mean = x.mean()
    var = (x - mean).square().mean()
    kurt = 0.0 if var.item() <= 0 else ((x - mean).pow(4).mean() / (var * var)).item()
    return {
        "p99": torch.quantile(x, 0.99).item(),
        "p999": torch.quantile(x, 0.999).item(),
        "kurtosis": kurt,
    }


def effective_group_size(in_features: int, group_size: int) -> int:
    if in_features % group_size == 0:
        return group_size
    for candidate in [64, 32, 16, 8, 4, 2, 1]:
        if candidate <= group_size and in_features % candidate == 0:
            return candidate
    return group_size


@torch.no_grad()
def quantize_vector_codebook_weight(
    W: torch.Tensor,
    *,
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    act_scales: torch.Tensor | None,
    activation_weighted: bool,
    shuffle_grouping: bool,
) -> torch.Tensor:
    out_features, in_features = W.shape
    eff_group = effective_group_size(in_features, group_size)
    if eff_group % vector_dim != 0:
        raise ValueError(f"eff_group={eff_group} not divisible by vector_dim={vector_dim}")
    num_groups = in_features // eff_group
    Wg = W.reshape(out_features, num_groups, eff_group)
    scales = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    normalized = (Wg / scales).clamp(-1.0, 1.0)

    col_ids = torch.arange(in_features, device=W.device).reshape(1, num_groups, eff_group).expand(out_features, -1, -1)
    if shuffle_grouping:
        gen = torch.Generator(device=W.device)
        gen.manual_seed(seed + 9176)
        perm = torch.randperm(eff_group, device=W.device, generator=gen)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(eff_group, device=W.device)
        normalized_work = normalized[:, :, perm]
        col_ids_work = col_ids[:, :, perm]
    else:
        inv_perm = None
        normalized_work = normalized
        col_ids_work = col_ids

    vectors = normalized_work.reshape(-1, vector_dim).contiguous()
    vector_cols = col_ids_work.reshape(-1, vector_dim).contiguous()
    samples, weights = NonUniform2BitLinear._sample_vectors(
        vectors,
        vector_cols,
        sample_size=sample_size,
        act_scales=act_scales if activation_weighted else None,
        seed=seed,
    )
    codebook = NonUniform2BitLinear._weighted_kmeans_nd(samples, codebook_size, iters, weights=weights, seed=seed)
    indices = NonUniform2BitLinear._assign_vectors_chunked(vectors, codebook)
    Wq_norm_work = codebook[indices].reshape_as(normalized_work)
    if inv_perm is not None:
        Wq_norm = Wq_norm_work[:, :, inv_perm]
    else:
        Wq_norm = Wq_norm_work
    return (Wq_norm * scales).reshape_as(W).detach()


def weight_metrics(ref: torch.Tensor, cand: torch.Tensor, act_diag: torch.Tensor | None) -> dict[str, float]:
    ref64 = ref.detach().to(torch.float64)
    diff64 = (cand.detach().to(torch.float64) - ref64)
    ref_rms = torch.sqrt(ref64.square().mean()).item()
    diff_rms = torch.sqrt(diff64.square().mean()).item()
    abs_diff = sample_flat(diff64.abs().to(torch.float32), 262144)
    stats = sample_stats(abs_diff)
    out = {
        "weight_rel_rms": diff_rms / max(ref_rms, 1e-12),
        "weight_p999_abs": stats["p999"],
        "weight_max_abs_sample": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
    }
    if act_diag is not None and act_diag.numel() == ref.shape[1]:
        w = act_diag.to(device=ref.device, dtype=torch.float64).clamp_min(0)
        aw_ref = torch.sqrt((ref64.square() * w.view(1, -1)).sum() / max(1, ref.numel())).item()
        aw_diff = torch.sqrt((diff64.square() * w.view(1, -1)).sum() / max(1, ref.numel())).item()
        out["activation_weighted_weight_rel_rms"] = aw_diff / max(aw_ref, 1e-12)
    else:
        out["activation_weighted_weight_rel_rms"] = float("nan")
    return out


def activation_weighted_rel_rms(ref: torch.Tensor, cand: torch.Tensor, act_diag: torch.Tensor | None) -> float:
    if act_diag is None or act_diag.numel() != ref.shape[1]:
        return float("nan")
    ref64 = ref.detach().to(torch.float64)
    diff64 = cand.detach().to(torch.float64) - ref64
    w = act_diag.to(device=ref.device, dtype=torch.float64).clamp_min(0)
    aw_ref = torch.sqrt((ref64.square() * w.view(1, -1)).sum() / max(1, ref.numel())).item()
    aw_diff = torch.sqrt((diff64.square() * w.view(1, -1)).sum() / max(1, ref.numel())).item()
    return aw_diff / max(aw_ref, 1e-12)


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
        x32 = x.detach().to(torch.float32)
        flat = x32.reshape(-1, x32.shape[-1])
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
    parser.add_argument("--codebook-sample", type=int, default=200000)
    parser.add_argument("--codebook-iters", type=int, default=12)
    parser.add_argument("--codebook-vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--codebook-seed", type=int, default=20260601)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mechanism_shard{args.shard_index:02d}_of{args.num_shards:02d}.csv"
    if out_csv.exists():
        print(f"[mechanism] skip existing {out_csv}", flush=True)
        return

    module_rows_all = read_modules(Path(args.module_csv))
    module_rows = [row for idx, row in enumerate(module_rows_all) if idx % args.num_shards == args.shard_index]
    module_names = {row["module_name"] for row in module_rows}
    print(f"[mechanism] shard {args.shard_index}/{args.num_shards}: modules={len(module_rows)}", flush=True)

    policy = Gr00tPolicy("LIBERO_PANDA", args.model_path, device=args.device)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)
    batches = load_batches(policy, Path(args.request_dir), args.limit)
    act_scales = torch.load(ACT_SCALES, map_location=device) if ACT_SCALES.exists() else {}
    named_modules = dict(policy.model.named_modules())
    rows_by_name = {row["module_name"]: row for row in module_rows}
    qdata: dict[str, ModuleData] = {}

    for i, name in enumerate(sorted(module_names), start=1):
        module = named_modules.get(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear: {type(module)}")
        print(f"[mechanism] quantizing {i}/{len(module_names)} {name}", flush=True)
        W = module.weight.detach().to(device=device, dtype=torch.float32)
        bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=torch.float32)
        layer_scales = act_scale_for_module(name, act_scales, device)
        act_scale_numel = int(layer_scales.numel()) if layer_scales is not None else 0

        weights = {
            "hqq2": quantize_weight_hqq(W, nbits=2, group_size=args.group_size, device=device).to(torch.float32),
            "cb_unweighted": quantize_vector_codebook_weight(
                W,
                group_size=args.group_size,
                vector_dim=args.codebook_vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.codebook_sample,
                iters=args.codebook_iters,
                seed=args.codebook_seed,
                act_scales=None,
                activation_weighted=False,
                shuffle_grouping=False,
            ),
            "cb_act": quantize_vector_codebook_weight(
                W,
                group_size=args.group_size,
                vector_dim=args.codebook_vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.codebook_sample,
                iters=args.codebook_iters,
                seed=args.codebook_seed,
                act_scales=layer_scales,
                activation_weighted=True,
                shuffle_grouping=False,
            ),
            "cb_act_shuffled": quantize_vector_codebook_weight(
                W,
                group_size=args.group_size,
                vector_dim=args.codebook_vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.codebook_sample,
                iters=args.codebook_iters,
                seed=args.codebook_seed,
                act_scales=layer_scales,
                activation_weighted=True,
                shuffle_grouping=True,
            ),
        }
        wmetrics = {variant: weight_metrics(W, Wq, layer_scales) for variant, Wq in weights.items()}
        qdata[name] = ModuleData(
            row=rows_by_name[name],
            params=int(module.in_features) * int(module.out_features),
            bias=bias,
            ref_weight=W,
            weights=weights,
            weight_metrics=wmetrics,
        )
        qdata[name].row["in_features"] = int(module.in_features)
        qdata[name].row["out_features"] = int(module.out_features)
        qdata[name].row["act_scale_numel"] = act_scale_numel
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

    for req_idx, (_req_id, backbone_cpu) in enumerate(batches, start=1):
        print(f"[mechanism] forward {req_idx}/{len(batches)}", flush=True)
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        _ = policy.model.backbone(backbone_inputs)

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
            "act_scale_numel": base.get("act_scale_numel", 0),
            "act_scale_matched": int(base.get("act_scale_numel", 0) == base.get("in_features", -1)),
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
        row["act_codebook_output_gain_vs_hqq"] = row["hqq2_output_rel_rms"] - row["cb_act_output_rel_rms"]
        row["act_codebook_output_gain_vs_shuffled"] = row["cb_act_shuffled_output_rel_rms"] - row["cb_act_output_rel_rms"]
        row["act_weighting_output_gain_vs_unweighted"] = row["cb_unweighted_output_rel_rms"] - row["cb_act_output_rel_rms"]
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"mechanism_shard{args.shard_index:02d}_of{args.num_shards:02d}.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    print(f"[mechanism] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
