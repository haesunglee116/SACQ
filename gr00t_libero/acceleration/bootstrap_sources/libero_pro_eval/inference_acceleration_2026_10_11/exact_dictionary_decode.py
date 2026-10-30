"""Fuse the final model's exact GPTQ level-dictionary decode into one launch.

No requantization, persistent dense weights, precision change, or new packing.
The output is a temporary BF16 weight used by the original F.linear. Inference
only. This path targets ExactPackedLinear, not an official GPTQ affine codec.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(packed, levels, out, COUNT: tl.constexpr, GROUP: tl.constexpr,
            BITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    bit = i * BITS
    byte = bit // 8
    word = tl.load(packed + byte, valid, 0).to(tl.int32)
    word |= tl.load(packed + byte + 1, valid, 0).to(tl.int32) << 8
    index = (word >> (bit % 8)) & ((1 << BITS) - 1)
    value = tl.load(levels + (i // GROUP) * (1 << BITS) + index, valid, 0)
    tl.store(out + i, value, valid)


def reconstruct(module):
    count = module.out_features * module.in_features
    if (module.packed.device.type != "cuda" or module.packed.dtype != torch.uint8
            or module.levels.dtype != torch.bfloat16
            or module.levels.device != module.packed.device
            or not module.packed.is_contiguous() or not module.levels.is_contiguous()
            or module.bits not in (2, 3, 4) or module.in_features % module.group_size
            or tuple(module.levels.shape) != (count // module.group_size, 1 << module.bits)
            or module.packed.numel() < (count * module.bits + 7) // 8 + 1):
        raise ValueError("Unsupported exact dictionary layout")
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("Inference-only decode requires eval + no_grad")
    out = torch.empty((module.out_features, module.in_features),
                      dtype=torch.bfloat16, device=module.packed.device)
    _decode[(triton.cdiv(count, 256),)](
        module.packed, module.levels, out, count, module.group_size, module.bits, 256)
    return out


def install(model):
    from exact_packed_linear import ExactPackedLinear
    changed = []
    for name, module in model.named_modules():
        if type(module) is ExactPackedLinear:
            if hasattr(module, "_decode_original"):
                raise RuntimeError("Dictionary decode already installed")
            module._decode_original = module.reconstruct
            module.reconstruct = lambda m=module: reconstruct(m)
            changed.append(name)
    return changed


def restore(model):
    for module in model.modules():
        if hasattr(module, "_decode_original"):
            module.reconstruct = module._decode_original
            del module._decode_original
