"""Existing scalar GPTQ numerical core; fixed damping, no adaptive fallback."""
import math
import torch

def quantize_tensor(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class RowGroupQuantizer:
    """Reference GPTQ affine quantizer: one scale/zero per output row and group."""

    def __init__(self, bits: int, symmetric: bool=False):
        self.maxq = (1 << bits) - 1
        self.symmetric = symmetric
        self.scale: torch.Tensor | None = None
        self.zero: torch.Tensor | None = None

    def find_params(self, x: torch.Tensor) -> None:
        flat = x.flatten(1)
        zeros = torch.zeros(flat.shape[0], device=flat.device)
        xmin = torch.minimum(flat.min(1).values, zeros)
        xmax = torch.maximum(flat.max(1).values, zeros)
        if self.symmetric:
            xmax = torch.maximum(xmin.abs(), xmax)
            xmin = torch.where(xmin < 0, -xmax, xmin)
        dead = (xmin == 0) & (xmax == 0)
        xmin = torch.where(dead, torch.full_like(xmin, -1), xmin)
        xmax = torch.where(dead, torch.ones_like(xmax), xmax)
        scale = (xmax - xmin) / self.maxq
        if self.symmetric:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)
        self.scale = scale.unsqueeze(1)
        self.zero = zero.unsqueeze(1)

    def apply(self, column: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero is None:
            raise RuntimeError('quantizer parameters not initialized')
        return quantize_tensor(column.unsqueeze(1), self.scale, self.zero, self.maxq).flatten()

class RowGroupTernaryQuantizer:
    """Symmetric row/group ternary quantizer with codes {-1, 0, +1}.

    The threshold and reconstruction scale are fitted once from the original
    64-column row group, exactly where the affine GPTQ baseline fits its
    scale/zero.  The GPTQ column-error update is otherwise unchanged.
    """

    def __init__(self) -> None:
        self.scale: torch.Tensor | None = None
        self.threshold: torch.Tensor | None = None

    def find_params(self, x: torch.Tensor) -> None:
        flat = x.flatten(1)
        threshold = 0.75 * flat.abs().mean(dim=1)
        code = torch.where(flat.abs() > threshold[:, None], flat.sign(), torch.zeros_like(flat))
        count = code.abs().sum(dim=1)
        scale = (code * flat).sum(dim=1) / count.clamp_min(1.0)
        scale = torch.where(count > 0, scale, torch.zeros_like(scale))
        self.scale = scale
        self.threshold = threshold

    def apply(self, column: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.threshold is None:
            raise RuntimeError('ternary quantizer parameters not initialized')
        code = torch.where(column.abs() > self.threshold, column.sign(), torch.zeros_like(column))
        return self.scale * code

class GPTQAccumulator:
    """Reference sequential GPTQ update, specialized to torch.nn.Linear."""

    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        self.columns = layer.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        inp2 = inp.reshape(-1, inp.shape[-1]).t()
        self.H *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        inp2 = math.sqrt(2.0 / self.nsamples) * inp2.float()
        self.H += inp2.matmul(inp2.t())

    @torch.no_grad()
    def fasterquant(self, bits: int, groupsize: int=64, blocksize: int=128, percdamp: float=0.01, symmetric: bool=False, ternary: bool=False) -> tuple[torch.Tensor, float, float]:
        W = self.layer.weight.detach().float().clone()
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        base_damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=H.device)
        chol = None
        used_damp = None
        for multiplier in (1.0,):
            trial = H.clone()
            trial[diag, diag] += base_damp * multiplier
            try:
                chol = torch.linalg.cholesky(trial)
                used_damp = float(percdamp * multiplier)
                break
            except torch.linalg.LinAlgError:
                continue
        if chol is None or used_damp is None:
            raise RuntimeError('GPTQ Hessian remained non-positive after damping retries')
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
        Q = torch.zeros_like(W)
        total_loss = 0.0
        quantizer = RowGroupTernaryQuantizer() if ternary else RowGroupQuantizer(bits, symmetric=symmetric)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                absolute_i = i1 + i
                if absolute_i % groupsize == 0:
                    quantizer.find_params(W[:, absolute_i:absolute_i + groupsize])
                w = W1[:, i]
                d = Hinv1[i, i]
                q = quantizer.apply(w)
                if not torch.isfinite(q).all():
                    raise RuntimeError(f'GPTQ quantizer produced non-finite output at column {absolute_i}')
                Q1[:, i] = q
                total_loss += float(((w - q) ** 2 / d ** 2 / 2).sum().item())
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            if not torch.isfinite(W[:, i2:]).all():
                raise RuntimeError(f'GPTQ error propagation produced non-finite weights after column {i2}')
        return (Q.to(torch.bfloat16), total_loss, used_damp)
