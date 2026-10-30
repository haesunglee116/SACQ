#!/usr/bin/env python3

from __future__ import annotations

import random

import torch
from torch.optim.optimizer import Optimizer


def _norm(v: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return v.norm(p=2, dim=dim, keepdim=True)


def _unit(v: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    vnorm = _norm(v, dim)
    return v / vnorm.add(eps), vnorm


def _matrix_norm_one(w: torch.Tensor) -> torch.Tensor:
    out = torch.abs(w)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def _cayley_loop(x: torch.Tensor, w: torch.Tensor, tan_vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    y = x + t * tan_vec
    for _ in range(5):
        y = x + t * torch.matmul(w, 0.5 * (x + y))
    return y.t()


def _qr_retraction(tan_vec: torch.Tensor) -> torch.Tensor:
    p, n = tan_vec.size()
    del p, n
    tan_vec = tan_vec.t()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    return q.t()


class SGDG(Optimizer):
    """SpinQuant-style Stiefel optimizer for orthogonal matrices."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        stiefel: bool = False,
        grad_clip=None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            grad_clip=grad_clip,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        epsilon = 1e-8
        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                unity, _ = _unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size(0) <= unity.size(1):
                    g = p.grad.data.view(p.size()[0], -1)
                    lr = group["lr"]
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(
                            g.t().size(), device=g.device, dtype=g.dtype
                        )
                    v = param_state["momentum_buffer"]
                    if random.randint(1, 101) == 1:
                        unity = _qr_retraction(unity)

                    v = momentum * v - g.t()
                    mx = torch.mm(v, unity)
                    xmx = torch.mm(unity, mx)
                    xxmx = torch.mm(unity.t(), xmx)
                    w_hat = mx - 0.5 * xxmx
                    w = w_hat - w_hat.t()
                    t = 0.5 * 2.0 / (_matrix_norm_one(w) + epsilon)
                    alpha = min(float(t), lr)
                    p_new = _cayley_loop(unity.t(), w, v, alpha)
                    v_new = torch.mm(w, unity.t())
                    p.data.copy_(p_new.view(p.size()))
                    param_state["momentum_buffer"] = v_new
                    continue

                d_p = p.grad.data
                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = d_p.clone()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - group["dampening"])
                    if group["nesterov"]:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha=-group["lr"])
        return loss
