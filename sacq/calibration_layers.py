from __future__ import annotations
import math
import torch
import torch.nn as nn

# Source defaults, not a claim about every experiment launcher.
WCACHE = True
FP32_CODEBOOK_GRAD = False

class _FP32CodebookGather(torch.autograd.Function):
    """BF16 codebook gather whose codebook gradient is accumulated in FP32.

    ``self.codebook.to(bfloat16)[idx]`` makes the stock gather backward scatter
    into a temporary BF16 tensor before casting that gradient back to the FP32
    AdamW master parameter.  For small codebooks, many assignments collide on
    each entry and BF16 atomics become both slow and unnecessarily lossy.  The
    forward below is value-identical; only the linear gradient reduction is
    accumulated directly into FP32.  It is mathematically the same derivative,
    but (like any parallel reduction change) not bitwise identical.
    """

    @staticmethod
    def forward(ctx, codebook, idx):
        ctx.save_for_backward(idx)
        ctx.codebook_shape = codebook.shape
        return codebook.to(torch.bfloat16)[idx]

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        grad_codebook = torch.zeros(
            ctx.codebook_shape,
            device=grad_output.device,
            dtype=torch.float32,
        )
        grad_codebook.index_add_(
            0,
            idx.reshape(-1),
            grad_output.reshape(-1, grad_output.shape[-1]).float(),
        )
        return grad_codebook, None


def pack_indices(indices: torch.Tensor, index_bits: int) -> torch.Tensor:
    """gr00t.eval.cosmos_quant_shim.PrecomputedPackedVectorCodebookLinear의
    packed_indices 언패킹과 정확히 맞물리는 역함수 (1--16bit CPU/GPU
    round-trip 검증됨, 2026-08-22).
    byte 정렬 "indices"(uint8 고정)는 k>256 층에서 값이 잘리는 버그가 있어
    항상 이 비트팩킹 경로로 저장한다."""
    if index_bits < 1 or index_bits > 56:
        raise ValueError(f"index_bits must be in [1, 56], got {index_bits}")
    flat = indices.reshape(-1).long()
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= (1 << index_bits)):
        raise ValueError(f"indices do not fit in {index_bits} bits")
    count = flat.numel()
    n_bytes = (count * index_bits + 7) // 8
    # An index can start at bit offset 7, so arbitrary widths may span more
    # than two bytes (11-bit indices need three). Keep enough temporary guard
    # bytes for the widest possible alignment and trim them before saving.
    max_offset = 8 - math.gcd(index_bits, 8)
    window_bytes = (index_bits + max_offset + 7) // 8
    out = torch.zeros(n_bytes + window_bytes - 1, dtype=torch.int64)
    positions = torch.arange(count, dtype=torch.long) * index_bits
    byte_ids = torch.div(positions, 8, rounding_mode="floor")
    offsets = positions.remainder(8)
    shifted = flat << offsets
    for byte_offset in range(window_bytes):
        out.index_put_(
            (byte_ids + byte_offset,),
            (shifted >> (8 * byte_offset)) & 0xFF,
            accumulate=True,
        )
    assert (out <= 255).all(), "byte overflow -- 인덱스 겹침"
    return out[:n_bytes].to(torch.uint8)


class QuantLinear(nn.Module):
    """W reconstructed on the fly from trainable codebook + scales, frozen idx."""

    def __init__(self, packed: dict, bias, device):
        super().__init__()
        self.out_features = packed["out_features"]
        self.in_features = packed["in_features"]
        self.gs = packed["group_size"]
        self.vd = packed["vector_dim"]
        self.codebook = nn.Parameter(packed["codebook"].to(device, torch.float32))
        self.scales = nn.Parameter(packed["scales"].to(device, torch.float32))
        self.register_buffer(
            "shift",
            torch.as_tensor(packed.get("shift", 0.0), device=device, dtype=torch.bfloat16),
        )
        idx = packed["indices"].to(torch.long)
        self.register_buffer("idx", idx.to(device))
        self._wc = None          # 스텝 내 W 캐시 (매 샘플 시작 시 무효화)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device, torch.bfloat16))
        else:
            self.bias = None

    def weight_(self):
        # Match packed deployment arithmetic exactly while retaining FP32 master
        # parameters for AdamW.  Casting remains differentiable, so gradients
        # flow to the master codebook/scales through the BF16 storage simulation.
        # Runtime order: BF16 gather -> reshape -> BF16 scale -> reshape -> shift.
        scales = self.scales.to(torch.bfloat16)
        if FP32_CODEBOOK_GRAD and torch.is_grad_enabled():
            vec = _FP32CodebookGather.apply(self.codebook, self.idx)
        else:
            codebook = self.codebook.to(torch.bfloat16)
            vec = codebook[self.idx]                        # (out, g, gs//vd, vd)
        Wn = vec.reshape(self.out_features, -1, self.gs)    # (out, g, gs)
        W = Wn * scales.unsqueeze(-1)
        return W.reshape(self.out_features, self.in_features) + self.shift

    def forward(self, x):
        if getattr(self, "_leaf", None) is not None:
            return nn.functional.linear(x, self._leaf.to(torch.bfloat16), self.bias)
        # 한 샘플의 forward 안에서 코드북/스케일은 불변인데, DiT 층은 디노이징
        # 4스텝 때문에 같은 W 를 4번 재구성한다. 첫 호출 결과를 캐시해 재사용한다.
        # 같은 그래프 노드를 재사용하므로 backward 는 두 경로의 기울기를 그대로
        # 누적한다 — 수치적으로 재계산과 동일(무손실). 샘플 시작 시 _wc 를 비운다.
        if not WCACHE:
            return nn.functional.linear(x, self.weight_().to(torch.bfloat16), self.bias)
        W = self._wc
        if W is None:
            W = self.weight_().to(torch.bfloat16)
            self._wc = W
        return nn.functional.linear(x, W, self.bias)

