"""Tiny synthetic demonstration, not a VLA benchmark or paper preset.

Run from the repository root after installation: python -m examples.cpu_pipeline
"""
import json
import torch
import torch.nn.functional as F
from sacq.vector_quantization import quantize_vector_codebook
from sacq.index_correction import compensate_packed, decode_fixed
from sacq.calibration_layers import QuantLinear
from sacq.cache import reconstruction_cache_scope


def main():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    weight = torch.randn(4, 64)
    inputs = torch.randn(64, 64)
    second_moment = inputs.T @ inputs / len(inputs)
    _, packed = quantize_vector_codebook(
        weight, group_size=64, vector_dim=4, codebook_size=16,
        sample_size=64, iters=5, seed=123, metric='qk_full',
        input_metric=second_moment, output_importance=torch.ones(4),
        scalar_importance=None, return_packed=True, scale_squared_weighting=False)
    corrected, correction = compensate_packed(
        packed, weight, second_moment, block_columns=64, damping=0.1)
    layer = QuantLinear(corrected, bias=None, device='cpu')
    # Distinct learning rates are demonstration choices, not paper settings.
    optimizer = torch.optim.AdamW([
        {'params': [layer.codebook], 'lr': 1e-3},
        {'params': [layer.scales], 'lr': 1e-4},
    ], weight_decay=0)
    x = inputs.to(torch.bfloat16)
    target = F.linear(x, weight.to(torch.bfloat16)).float()
    def loss():
        with reconstruction_cache_scope(layer):
            return F.mse_loss(layer(x).float(), target)
    before = float(loss().detach())
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()
    after = float(loss().detach())
    print(json.dumps(dict(
        scope='synthetic CPU demonstration only',
        weight_shape=list(weight.shape), calibration_updates=8,
        index_report=correction, mse_before_calibration=before,
        mse_after_calibration=after,
    ), indent=2))


if __name__ == '__main__':
    main()
