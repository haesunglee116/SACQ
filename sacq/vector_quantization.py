from __future__ import annotations
import math
from typing import Any
import torch

class NonUniform2BitLinear:
    @staticmethod
    def _sample_vectors(
        vectors: torch.Tensor,
        vector_cols: torch.Tensor,
        sample_size: int,
        act_scales: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = vectors.shape[0]
        if n > sample_size:
            gen = torch.Generator(device=vectors.device)
            gen.manual_seed(seed)
            idx = torch.randperm(n, device=vectors.device, generator=gen)[:sample_size]
            vectors = vectors[idx]
            vector_cols = vector_cols[idx]
        weights = None
        if act_scales is not None:
            scales = act_scales.to(device=vectors.device, dtype=torch.float32).flatten()
            if scales.numel() >= int(vector_cols.max().item()) + 1:
                weights = scales[vector_cols].mean(dim=1).clamp_min(1e-8)
                weights = weights / weights.mean().clamp_min(1e-8)
        return vectors, weights

    @staticmethod
    def _assign_vectors_chunked(samples: torch.Tensor, centers: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
        # Bound the temporary distance matrix for large codebooks (for example,
        # d8/K65536 at a nominal 2 bits per weight).
        max_distance_elements = 64 * 1024 * 1024
        chunk_size = min(chunk_size, max(1, max_distance_elements // centers.shape[0]))
        assignments = []
        for start in range(0, samples.shape[0], chunk_size):
            chunk = samples[start : start + chunk_size]
            dist = torch.cdist(chunk, centers, p=2)
            assignments.append(dist.argmin(dim=1))
        return torch.cat(assignments, dim=0)

    @classmethod
    def _weighted_kmeans_nd(
        cls,
        samples: torch.Tensor,
        k: int,
        iters: int,
        weights: torch.Tensor | None = None,
        seed: int = 20260611,
    ) -> torch.Tensor:
        n = samples.shape[0]
        if n < k:
            repeat = (k + n - 1) // max(n, 1)
            return samples.repeat(repeat, 1)[:k].contiguous()

        gen = torch.Generator(device=samples.device)
        gen.manual_seed(seed)
        init_idx = torch.randperm(n, device=samples.device, generator=gen)[:k]
        centers = samples[init_idx].contiguous()
        for _ in range(iters):
            assign = cls._assign_vectors_chunked(samples, centers)
            new_centers = torch.zeros_like(centers)
            if weights is None:
                counts = torch.bincount(assign, minlength=k).to(dtype=torch.float32, device=samples.device)
                new_centers.index_add_(0, assign, samples)
                nonempty = counts > 0
                new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None].clamp_min(1e-8)
            else:
                ww = weights.to(dtype=torch.float32)
                counts = torch.bincount(assign, weights=ww, minlength=k).to(dtype=torch.float32, device=samples.device)
                new_centers.index_add_(0, assign, samples * ww[:, None])
                nonempty = counts > 0
                new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None].clamp_min(1e-8)
            new_centers[~nonempty] = centers[~nonempty]
            centers = new_centers.clamp(-1.0, 1.0).contiguous()
        return centers


def effective_group_size(in_features: int, group_size: int) -> int:
    if in_features % group_size == 0:
        return group_size
    for candidate in [64, 32, 16, 8, 4, 2, 1]:
        if candidate <= group_size and in_features % candidate == 0:
            return candidate
    return group_size


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

