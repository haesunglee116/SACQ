from __future__ import annotations
import math
import torch

def apply_swiglu_down_diag_curvature(
    gate_importance: torch.Tensor,
    up_importance: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate diagonal gate/up sensitivity through ``down_proj``.

    For y = W_down (SiLU(g) * u), a perturbation of intermediate row r is
    weighted by diag(W_down^T W_down)_r = ||W_down[:, r]||_2^2.  The full
    W_down^T W_down would couple different intermediate rows; this helper
    deliberately retains its diagonal so the existing per-weight ``qk_diag``
    fitter remains applicable.
    """
    if gate_importance.shape != up_importance.shape:
        raise ValueError(
            "gate/up importance shape mismatch: "
            f"{tuple(gate_importance.shape)} vs {tuple(up_importance.shape)}"
        )
    if down_weight.ndim != 2:
        raise ValueError(f"down_weight must be a matrix, got {tuple(down_weight.shape)}")
    if down_weight.shape[1] != gate_importance.shape[0]:
        raise ValueError(
            "down_proj input width must match gate/up output rows: "
            f"{down_weight.shape[1]} vs {gate_importance.shape[0]}"
        )
    down_column_energy = down_weight.square().sum(dim=0)
    if not torch.isfinite(down_column_energy).all():
        raise ValueError("non-finite down_proj diagonal curvature")
    row_weight = down_column_energy.unsqueeze(1)
    return gate_importance * row_weight, up_importance * row_weight


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

