#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUT = str(PROJ / "quantization_analysis/gr00t_qk_diag_codebook_2026-06-27")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def effective_group_size(in_features: int, group_size: int) -> int:
    if in_features % group_size == 0:
        return group_size
    for candidate in [64, 32, 16, 8, 4, 2, 1]:
        if candidate <= group_size and in_features % candidate == 0:
            return candidate
    return group_size


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


@dataclass
class LayerCache:
    layer_idx: int
    name: str
    x_chunks: list[torch.Tensor] = field(default_factory=list)
    cos_chunks: list[torch.Tensor] = field(default_factory=list)
    sin_chunks: list[torch.Tensor] = field(default_factory=list)
    mask_chunks: list[torch.Tensor | None] = field(default_factory=list)
    input_sq: torch.Tensor | None = None
    q_sq: torch.Tensor | None = None
    krep_sq: torch.Tensor | None = None
    token_count: int = 0

    def add(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        krep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_cache_tokens: int,
    ) -> None:
        x32 = x.detach().to(torch.float32)
        q32 = q.detach().to(torch.float32)
        k32 = krep.detach().to(torch.float32)
        flat_x = x32.reshape(-1, x32.shape[-1])
        in_sq = flat_x.square().sum(dim=0)
        q_sq = q32.square().sum(dim=(0, 2))
        k_sq = k32.square().sum(dim=(0, 2))
        self.input_sq = in_sq if self.input_sq is None else self.input_sq + in_sq
        self.q_sq = q_sq if self.q_sq is None else self.q_sq + q_sq
        self.krep_sq = k_sq if self.krep_sq is None else self.krep_sq + k_sq
        self.token_count += int(flat_x.shape[0])

        # Keep full short calibration requests for exact post-RoPE QK evaluation.
        cached_tokens = sum(int(t.shape[0] * t.shape[1]) for t in self.x_chunks)
        if cached_tokens < max_cache_tokens:
            self.x_chunks.append(x.detach().to("cpu", dtype=torch.bfloat16))
            self.cos_chunks.append(cos.detach().to("cpu", dtype=torch.bfloat16))
            self.sin_chunks.append(sin.detach().to("cpu", dtype=torch.bfloat16))
            self.mask_chunks.append(None if attention_mask is None else attention_mask.detach().to("cpu"))


def sample_vectors_with_importance(
    vectors: torch.Tensor,
    importance_vectors: torch.Tensor,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = vectors.shape[0]
    if n <= sample_size:
        return vectors, importance_vectors
    gen = torch.Generator(device=vectors.device)
    gen.manual_seed(seed)
    idx = torch.randperm(n, device=vectors.device, generator=gen)[:sample_size]
    return vectors[idx].contiguous(), importance_vectors[idx].contiguous()


def sample_vectors_with_full_metric(
    vectors: torch.Tensor,
    row_ids: torch.Tensor,
    vector_cols: torch.Tensor,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = vectors.shape[0]
    if n <= sample_size:
        return vectors, row_ids, vector_cols
    gen = torch.Generator(device=vectors.device)
    gen.manual_seed(seed)
    idx = torch.randperm(n, device=vectors.device, generator=gen)[:sample_size]
    return vectors[idx].contiguous(), row_ids[idx].contiguous(), vector_cols[idx].contiguous()


def assign_vectors_diag(
    samples: torch.Tensor,
    centers: torch.Tensor,
    importance: torch.Tensor,
    chunk_size: int = 32768,
) -> torch.Tensor:
    assignments = []
    for start in range(0, samples.shape[0], chunk_size):
        x = samples[start : start + chunk_size]
        g = importance[start : start + chunk_size]
        dist = ((x[:, None, :] - centers[None, :, :]).square() * g[:, None, :]).sum(dim=-1)
        assignments.append(dist.argmin(dim=1))
    return torch.cat(assignments, dim=0)


def gather_full_block_metric(
    row_ids: torch.Tensor,
    vector_cols: torch.Tensor,
    output_importance: torch.Tensor,
    input_metric: torch.Tensor,
) -> torch.Tensor:
    out = output_importance[row_ids].to(dtype=torch.float32).view(-1, 1, 1)
    cols_i = vector_cols[:, :, None]
    cols_j = vector_cols[:, None, :]
    return input_metric[cols_i, cols_j].to(dtype=torch.float32) * out


def assign_vectors_full(
    samples: torch.Tensor,
    centers: torch.Tensor,
    full_metric: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    assignments = []
    for start in range(0, samples.shape[0], chunk_size):
        x = samples[start : start + chunk_size]
        g = full_metric[start : start + chunk_size]
        diff = x[:, None, :] - centers[None, :, :]
        dist = torch.einsum("nkd,nde,nke->nk", diff, g, diff)
        assignments.append(dist.argmin(dim=1))
    return torch.cat(assignments, dim=0)


def weighted_kmeans_diag(
    samples: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    n = samples.shape[0]
    if n < k:
        repeat = (k + n - 1) // max(n, 1)
        return samples.repeat(repeat, 1)[:k].contiguous()
    gen = torch.Generator(device=samples.device)
    gen.manual_seed(seed)
    centers = samples[torch.randperm(n, device=samples.device, generator=gen)[:k]].contiguous()
    for _ in range(iters):
        assign = assign_vectors_diag(samples, centers, importance)
        new_centers = torch.zeros_like(centers)
        denom = torch.zeros_like(centers)
        new_centers.index_add_(0, assign, samples * importance)
        denom.index_add_(0, assign, importance)
        nonempty = denom.sum(dim=1) > 0
        new_centers[nonempty] = new_centers[nonempty] / denom[nonempty].clamp_min(1e-8)
        new_centers[~nonempty] = centers[~nonempty]
        centers = new_centers.clamp(-1.0, 1.0).contiguous()
    return centers


def weighted_kmeans_full(
    samples: torch.Tensor,
    full_metric: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    n, d = samples.shape
    if n < k:
        repeat = (k + n - 1) // max(n, 1)
        return samples.repeat(repeat, 1)[:k].contiguous()
    gen = torch.Generator(device=samples.device)
    gen.manual_seed(seed)
    centers = samples[torch.randperm(n, device=samples.device, generator=gen)[:k]].contiguous()
    eye = torch.eye(d, device=samples.device, dtype=torch.float32)
    for _ in range(iters):
        assign = assign_vectors_full(samples, centers, full_metric)
        new_centers = torch.empty_like(centers)
        for j in range(k):
            mask = assign == j
            if not mask.any():
                new_centers[j] = centers[j]
                continue
            gj = full_metric[mask]
            xj = samples[mask]
            lhs = gj.sum(dim=0) + 1e-4 * eye
            rhs = torch.einsum("nde,ne->d", gj, xj)
            try:
                new_centers[j] = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                new_centers[j] = torch.linalg.lstsq(lhs, rhs[:, None]).solution[:, 0]
        centers = new_centers.clamp(-1.0, 1.0).contiguous()
    return centers


@torch.no_grad()
def quantize_vector_codebook(
    W: torch.Tensor,
    *,
    group_size: int,
    vector_dim: int,
    codebook_size: int,
    sample_size: int,
    iters: int,
    seed: int,
    metric: str,
    scalar_importance: torch.Tensor | None,
    input_metric: torch.Tensor | None = None,
    output_importance: torch.Tensor | None = None,
    return_packed: bool = False,
    scale_squared_weighting: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    out_features, in_features = W.shape
    eff_group = effective_group_size(in_features, group_size)
    if eff_group % vector_dim != 0:
        raise ValueError(f"eff_group={eff_group} not divisible by vector_dim={vector_dim}")
    num_groups = in_features // eff_group
    Wg = W.reshape(out_features, num_groups, eff_group)
    scales = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    normalized = (Wg / scales).clamp(-1.0, 1.0)
    vectors = normalized.reshape(-1, vector_dim).contiguous()
    # Each normalized vector belongs to one (output row, input group), whose
    # reconstructed weight is ``scale * codeword``.  For a quadratic fitting
    # objective the corresponding normalized-space loss is therefore weighted
    # by ``scale**2``.  The legacy fitter intentionally treats every normalized
    # group equally; keep that behavior as the default for exact reproduction.
    vectors_per_group = eff_group // vector_dim
    vector_scale_sq = (
        scales.squeeze(-1)
        .square()
        .unsqueeze(-1)
        .expand(out_features, num_groups, vectors_per_group)
        .reshape(-1)
        .to(device=W.device, dtype=torch.float32)
    )

    if metric in ("euclidean", "vector_kmeans"):
        if scale_squared_weighting:
            samples, sample_scale_sq = sample_vectors_with_importance(
                vectors, vector_scale_sq[:, None], sample_size=sample_size, seed=seed,
            )
            weights = sample_scale_sq[:, 0]
            weights = weights / weights.mean().clamp_min(1e-12)
        else:
            samples, weights = NonUniform2BitLinear._sample_vectors(
                vectors,
                torch.zeros_like(vectors, dtype=torch.long),
                sample_size=sample_size,
                act_scales=None,
                seed=seed,
            )
        codebook = NonUniform2BitLinear._weighted_kmeans_nd(samples, codebook_size, iters, weights=weights, seed=seed)
        indices = NonUniform2BitLinear._assign_vectors_chunked(vectors, codebook)
    elif metric == "activation":
        if scalar_importance is None:
            raise ValueError("activation metric requires scalar_importance")
        col_ids = torch.arange(in_features, device=W.device).reshape(1, num_groups, eff_group).expand(out_features, -1, -1)
        vector_cols = col_ids.reshape(-1, vector_dim).contiguous()
        if scale_squared_weighting:
            act_scales = scalar_importance.to(
                device=W.device, dtype=torch.float32
            ).flatten()
            if act_scales.numel() < int(vector_cols.max().item()) + 1:
                raise ValueError("activation importance is shorter than vector columns")
            vector_weights = act_scales[vector_cols].mean(dim=1).clamp_min(1e-8)
            vector_weights = vector_weights * vector_scale_sq
            samples, sampled_weights = sample_vectors_with_importance(
                vectors, vector_weights[:, None], sample_size=sample_size, seed=seed,
            )
            weights = sampled_weights[:, 0]
            weights = weights / weights.mean().clamp_min(1e-12)
        else:
            samples, weights = NonUniform2BitLinear._sample_vectors(
                vectors,
                vector_cols,
                sample_size=sample_size,
                act_scales=scalar_importance,
                seed=seed,
            )
        codebook = NonUniform2BitLinear._weighted_kmeans_nd(samples, codebook_size, iters, weights=weights, seed=seed)
        indices = NonUniform2BitLinear._assign_vectors_chunked(vectors, codebook)
    elif metric == "qk_diag":
        if scalar_importance is None:
            raise ValueError("qk_diag metric requires scalar_importance")
        g = scalar_importance.reshape(out_features, num_groups, eff_group).reshape(-1, vector_dim).contiguous()
        g = g.to(device=W.device, dtype=torch.float32).clamp_min(1e-12)
        if scale_squared_weighting:
            g = g * vector_scale_sq[:, None]
        g = g / g.mean().clamp_min(1e-12)
        g = g.clamp(1e-4, 1e4)
        samples, sample_g = sample_vectors_with_importance(vectors, g, sample_size=sample_size, seed=seed)
        codebook = weighted_kmeans_diag(samples, sample_g, codebook_size, iters, seed=seed)
        indices = assign_vectors_diag(vectors, codebook, g)
    elif metric == "qk_full":
        if input_metric is None or output_importance is None:
            raise ValueError("qk_full metric requires input_metric and output_importance")
        row_ids = torch.arange(out_features, device=W.device).reshape(out_features, 1, 1).expand(-1, num_groups, eff_group)
        row_ids = row_ids.reshape(-1, vector_dim)[:, 0].contiguous()
        col_ids = torch.arange(in_features, device=W.device).reshape(1, num_groups, eff_group).expand(out_features, -1, -1)
        vector_cols = col_ids.reshape(-1, vector_dim).contiguous()
        input_metric = input_metric.to(device=W.device, dtype=torch.float32)
        output_importance = output_importance.to(device=W.device, dtype=torch.float32).clamp_min(1e-12)
        output_importance = output_importance / output_importance.mean().clamp_min(1e-12)
        diag_mean = input_metric.diag().mean().clamp_min(1e-12)
        input_metric = input_metric / diag_mean
        sample_vecs, sample_rows, sample_cols = sample_vectors_with_full_metric(
            vectors,
            row_ids,
            vector_cols,
            sample_size=sample_size,
            seed=seed,
        )
        sample_g = gather_full_block_metric(sample_rows, sample_cols, output_importance, input_metric)
        if scale_squared_weighting:
            sample_scale_sq = vector_scale_sq[
                sample_rows * (num_groups * vectors_per_group)
                + (sample_cols[:, 0] // vector_dim)
            ]
            sample_g = sample_g * sample_scale_sq[:, None, None]
        g_diag_mean = sample_g.diagonal(dim1=-2, dim2=-1).mean().clamp_min(1e-12)
        sample_g = sample_g / g_diag_mean
        codebook = weighted_kmeans_full(sample_vecs, sample_g, codebook_size, iters, seed=seed)
        assignments = []
        for start in range(0, vectors.shape[0], 8192):
            chunk_rows = row_ids[start : start + 8192]
            chunk_cols = vector_cols[start : start + 8192]
            chunk_g = gather_full_block_metric(chunk_rows, chunk_cols, output_importance, input_metric) / g_diag_mean
            if scale_squared_weighting:
                chunk_scale_sq = vector_scale_sq[start : start + chunk_rows.shape[0]]
                chunk_g = chunk_g * chunk_scale_sq[:, None, None]
            assignments.append(assign_vectors_full(vectors[start : start + 8192], codebook, chunk_g, chunk_size=8192))
        indices = torch.cat(assignments, dim=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    Wq_norm = codebook[indices].reshape_as(normalized)
    Wq = (Wq_norm * scales).reshape_as(W).detach()
    if not return_packed:
        return Wq
    packed = {
        "format": "packed_vector_codebook_v1",
        "codebook": codebook.detach().to("cpu", torch.bfloat16),
        "indices": indices.reshape(out_features, num_groups, eff_group // vector_dim)
        .detach()
        .to("cpu", torch.uint8 if codebook.shape[0] <= 256 else torch.int32),
        "scales": scales.squeeze(-1).detach().to("cpu", torch.bfloat16),
        "shift": torch.zeros((), dtype=torch.bfloat16),
        "in_features": in_features,
        "out_features": out_features,
        "group_size": eff_group,
        "vector_dim": vector_dim,
        "codebook_fit_scale_squared": bool(scale_squared_weighting),
    }
    return Wq, packed


def project_qk(
    attn: nn.Module,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(torch.float32)
    Wq = Wq.to(torch.float32)
    Wk = Wk.to(torch.float32)
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_norm(F.linear(x, Wq).view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(F.linear(x, Wk).view(hidden_shape)).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    krep = repeat_kv(k, attn.num_key_value_groups)
    return q, k, krep


def relative_mse(diff: torch.Tensor, ref: torch.Tensor) -> float:
    return float(diff.to(torch.float64).square().sum().item() / max(ref.to(torch.float64).square().sum().item(), 1e-12))


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
    return 0.5 * (xtx + xtx.t())


@torch.no_grad()
def evaluate_layer(
    attn: nn.Module,
    cache: LayerCache,
    Wq_ref: torch.Tensor,
    Wk_ref: torch.Tensor,
    Wq_cand: torch.Tensor,
    Wk_cand: torch.Tensor,
    device: torch.device,
    max_query_positions: int,
) -> dict[str, float]:
    q_ref_sq = q_diff_sq = 0.0
    k_ref_sq = k_diff_sq = 0.0
    z_ref_sq = z_diff_sq = 0.0
    scaling = float(attn.head_dim) ** -0.5

    for x_cpu, cos_cpu, sin_cpu, mask_cpu in zip(cache.x_chunks, cache.cos_chunks, cache.sin_chunks, cache.mask_chunks):
        x = x_cpu.to(device=device, dtype=torch.bfloat16)
        cos = cos_cpu.to(device=device, dtype=torch.bfloat16)
        sin = sin_cpu.to(device=device, dtype=torch.bfloat16)
        q_ref, k_ref, krep_ref = project_qk(attn, x, cos, sin, Wq_ref, Wk_ref)
        q_hat, k_hat, krep_hat = project_qk(attn, x, cos, sin, Wq_cand, Wk_cand)

        q_ref64 = q_ref.to(torch.float64)
        k_ref64 = k_ref.to(torch.float64)
        q_ref_sq += float(q_ref64.square().sum().item())
        k_ref_sq += float(k_ref64.square().sum().item())
        q_diff_sq += float((q_hat.to(torch.float64) - q_ref64).square().sum().item())
        k_diff_sq += float((k_hat.to(torch.float64) - k_ref64).square().sum().item())

        t = q_ref.shape[-2]
        if t > max_query_positions:
            q_idx = torch.linspace(0, t - 1, steps=max_query_positions, device=device).round().long().unique()
        else:
            q_idx = torch.arange(t, device=device)
        z_ref = torch.matmul(q_ref[:, :, q_idx, :].to(torch.float32), krep_ref.to(torch.float32).transpose(2, 3)) * scaling
        z_hat = torch.matmul(q_hat[:, :, q_idx, :].to(torch.float32), krep_hat.to(torch.float32).transpose(2, 3)) * scaling
        if mask_cpu is not None:
            mask = mask_cpu.to(device=device)
            if mask.ndim == 4:
                valid = mask[:, :, q_idx, : z_ref.shape[-1]]
                valid = valid == 0 if valid.dtype.is_floating_point else valid.bool()
                z_ref = z_ref[valid.expand_as(z_ref)]
                z_hat = z_hat[valid.expand_as(z_hat)]
        z_ref64 = z_ref.to(torch.float64)
        z_ref_sq += float(z_ref64.square().sum().item())
        z_diff_sq += float((z_hat.to(torch.float64) - z_ref64).square().sum().item())

    return {
        "q_post_rope_rel_mse": q_diff_sq / max(q_ref_sq, 1e-12),
        "k_post_rope_rel_mse": k_diff_sq / max(k_ref_sq, 1e-12),
        "qk_logit_rel_mse": z_diff_sq / max(z_ref_sq, 1e-12),
    }


def layer_module_name(layer_idx: int) -> str:
    return f"backbone.model.model.language_model.layers.{layer_idx}.self_attn"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--request-dir", default=DEFAULT_REQUESTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--layers", default="0,4,8,12,15")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--vector-dim", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-cache-tokens-per-layer", type=int, default=8192)
    parser.add_argument("--max-query-positions", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    print(f"[qk-diag] loading policy on {device}", flush=True)
    policy = Gr00tPolicy("LIBERO_PANDA", args.model_path, device=args.device)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)
    batches = load_batches(policy, Path(args.request_dir), args.limit)
    named_modules = dict(policy.model.named_modules())

    caches: dict[int, LayerCache] = {}
    handles = []
    for layer_idx in layers:
        name = layer_module_name(layer_idx)
        attn = named_modules.get(name)
        if attn is None:
            raise KeyError(name)
        caches[layer_idx] = LayerCache(layer_idx=layer_idx, name=name)

        def make_hook(idx: int):
            def hook(module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                x = kwargs["hidden_states"]
                cos, sin = kwargs["position_embeddings"]
                attention_mask = kwargs.get("attention_mask")
                q, _k, krep = project_qk(module, x, cos, sin, module.q_proj.weight.detach(), module.k_proj.weight.detach())
                caches[idx].add(q=q, krep=krep, x=x, cos=cos, sin=sin, attention_mask=attention_mask, max_cache_tokens=args.max_cache_tokens_per_layer)

            return hook

        handles.append(attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))

    for req_idx, (_req_id, backbone_cpu) in enumerate(batches, start=1):
        print(f"[qk-diag] calibration forward {req_idx}/{len(batches)}", flush=True)
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        _ = policy.model.backbone(backbone_inputs)

    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    metrics = ["euclidean", "activation", "qk_diag", "qk_full"]
    for layer_idx in layers:
        cache = caches[layer_idx]
        attn = named_modules[cache.name]
        input_imp = (cache.input_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        input_metric = input_metric_from_cache(cache, device)
        k_imp = (cache.krep_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        q_imp = (cache.q_sq / max(cache.token_count, 1)).to(device=device, dtype=torch.float32)
        q_groups = attn.num_key_value_groups
        num_key_value_heads = attn.k_proj.out_features // attn.head_dim
        q_imp_grouped = q_imp.reshape(num_key_value_heads, q_groups, attn.head_dim).sum(dim=1)

        Wq = attn.q_proj.weight.detach().to(device=device, dtype=torch.float32)
        Wk = attn.k_proj.weight.detach().to(device=device, dtype=torch.float32)
        q_out_imp = k_imp.reshape(-1)
        k_out_imp = q_imp_grouped.reshape(-1)
        qk_q_importance = (q_out_imp[:, None] * input_imp[None, :]).contiguous()
        qk_k_importance = (k_out_imp[:, None] * input_imp[None, :]).contiguous()

        q_weights: dict[str, torch.Tensor] = {
            "hqq2": quantize_weight_hqq(Wq, nbits=2, group_size=args.group_size, device=device).to(torch.float32)
        }
        k_weights: dict[str, torch.Tensor] = {
            "hqq2": quantize_weight_hqq(Wk, nbits=2, group_size=args.group_size, device=device).to(torch.float32)
        }
        for metric in metrics:
            print(f"[qk-diag] layer {layer_idx} quantizing metric={metric}", flush=True)
            q_weights[metric] = quantize_vector_codebook(
                Wq,
                group_size=args.group_size,
                vector_dim=args.vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.sample_size,
                iters=args.iters,
                seed=args.seed + layer_idx * 17,
                metric=metric,
                scalar_importance=input_imp if metric == "activation" else qk_q_importance if metric == "qk_diag" else None,
                input_metric=input_metric if metric == "qk_full" else None,
                output_importance=q_out_imp if metric == "qk_full" else None,
            )
            k_weights[metric] = quantize_vector_codebook(
                Wk,
                group_size=args.group_size,
                vector_dim=args.vector_dim,
                codebook_size=args.codebook_size,
                sample_size=args.sample_size,
                iters=args.iters,
                seed=args.seed + layer_idx * 17 + 1009,
                metric=metric,
                scalar_importance=input_imp if metric == "activation" else qk_k_importance if metric == "qk_diag" else None,
                input_metric=input_metric if metric == "qk_full" else None,
                output_importance=k_out_imp if metric == "qk_full" else None,
            )

        variants = ["hqq2", *metrics]
        for variant in variants:
            for scope, Wq_cand, Wk_cand in (
                ("q_only", q_weights[variant], Wk),
                ("k_only", Wq, k_weights[variant]),
                ("qk", q_weights[variant], k_weights[variant]),
            ):
                print(f"[qk-diag] layer {layer_idx} evaluate {variant}/{scope}", flush=True)
                eval_metrics = evaluate_layer(
                    attn,
                    cache,
                    Wq,
                    Wk,
                    Wq_cand,
                    Wk_cand,
                    device,
                    max_query_positions=args.max_query_positions,
                )
                q_weight_rel_mse = relative_mse(q_weights[variant] - Wq, Wq) if scope in ("q_only", "qk") else 0.0
                k_weight_rel_mse = relative_mse(k_weights[variant] - Wk, Wk) if scope in ("k_only", "qk") else 0.0
                rows.append(
                    {
                        "layer": layer_idx,
                        "module": cache.name,
                        "variant": variant,
                        "scope": scope,
                        "requests": len(batches),
                        "cached_tokens": sum(int(t.shape[0] * t.shape[1]) for t in cache.x_chunks),
                        "calib_tokens": cache.token_count,
                        "q_weight_rel_mse": q_weight_rel_mse,
                        "k_weight_rel_mse": k_weight_rel_mse,
                        **eval_metrics,
                    }
                )

    csv_path = out_dir / "qk_diag_codebook_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[qk-diag] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
