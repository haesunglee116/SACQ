import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Keep native FP32 RMS statistics; fuse its tail and original FP8 amax.

No new quantization recipe. Norm BF16 output, AQ bytes and Sa must match.
Only input_layernorm -> Q and post_attention_layernorm -> gate are connected.
"""
import torch
import triton
import triton.language as tl
from a8_rtn_r2 import Quantizer
from omni_a8_scope_ffdown_runtime_r1 import exact

@triton.jit
def _bf16_square_f32(X, SQUARE, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < COUNT, 0).to(tl.float32)
    tl.store(SQUARE + i, value * value, i < COUNT)

@triton.jit
def _rms_bf16_tail_absmax(X, VAR, W, Y, PARTIAL, COUNT: tl.constexpr, K: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < COUNT, 0).to(tl.float32)
    variance = tl.load(VAR + i // K, i < COUNT, 0)
    inverse = tl.rsqrt(variance + EPS)
    normalized = (value * inverse).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W + i % K).to(tl.float32)
    output = (normalized * weight).to(tl.bfloat16)
    tl.store(Y + i, output, i < COUNT)
    maximum = tl.max(tl.where(i < COUNT, tl.abs(output.to(tl.float32)), 0), 0)
    tl.store(PARTIAL + tl.program_id(0), maximum)

def install(policy, c):
    import a8_joint_pair_r1 as fp8
    import inspect
    spec = c['rms_tail_fusion']
    s = dict(enabled=True, norm_checks={}, operand_checks={}, output_checks={}, square_checks={}, owners=[], runtime_source_files={})
    language = policy.model.backbone.model.model.language_model
    groups = policy.model._language_joint_fusion_state['groups']
    by_first = {g.names[0]: g for g in groups}
    for index, layer in enumerate(language.layers):
        prefix = f'backbone.model.model.language_model.layers.{index}.'
        for norm_name, suffix in [('input_layernorm', 'self_attn.q_proj'), ('post_attention_layernorm', 'mlp.gate_proj')]:
            norm = getattr(layer, norm_name)
            name = prefix + norm_name
            group = by_first[prefix + suffix]
            member = group.members[0]
            original_norm = norm.forward
            original_first = member.forward
            source = inspect.getsource(type(norm).forward)
            assert 'hidden_states.pow(2).mean(-1, keepdim=True)' in source
            assert 'self.weight * hidden_states.to(input_dtype)' in source
            assert norm.weight.dtype == torch.bfloat16 and tuple(norm.weight.shape) == (2048,)
            owner = dict(pending=None)
            s['owners'].append(owner)
            s['norm_checks'][name] = s['operand_checks'][name] = s['output_checks'][name] = s['square_checks'][name] = 0

            def forward_norm(x, _norm=norm, _original=original_norm, _owner=owner, _name=name):
                assert _owner['pending'] is None and tuple(x.shape) == (1, ROLLOUT_SHAPE.rows, 2048)
                reference = _original(x) if s['enabled'] else None
                squares = torch.empty(x.shape, dtype=torch.float32, device=x.device)
                _bf16_square_f32[triton.cdiv(x.numel(), 4096),](x, squares, x.numel(), 4096, num_warps=4, enable_fp_fusion=False)
                variance = squares.mean(-1, keepdim=True)
                if s['enabled']:
                    assert exact(squares, x.to(torch.float32).pow(2)), (_name, 'nativeFP32squares')
                    s['square_checks'][_name] += 1
                y = torch.empty_like(x)
                parts = triton.cdiv(x.numel(), 4096)
                partial = torch.empty(parts, dtype=torch.float32, device=x.device)
                aq = torch.empty((ROLLOUT_SHAPE.padded, 2048), dtype=torch.float8_e4m3fn, device=x.device)
                sa = torch.empty((), dtype=torch.float32, device=x.device)
                _rms_bf16_tail_absmax[parts,](x, variance, _norm.weight, y, partial, x.numel(), 2048, _norm.variance_epsilon, 4096, num_warps=4, enable_fp_fusion=False)
                fp8._finish_quantize[triton.cdiv(ROLLOUT_SHAPE.padded * 2048, 1024),](y, partial, aq, sa, x.numel(), ROLLOUT_SHAPE.padded * 2048, parts, triton.next_power_of_2(parts), 1024, num_warps=4)
                if s['enabled']:
                    assert exact(y, reference), (_name, 'BF16_norm_tail', (y.float() - reference.float()).abs().max().item())
                    q, scale = Quantizer('fp8_e4m3fn').quantize(reference.reshape(ROLLOUT_SHAPE.rows, 2048))
                    padded = torch.zeros_like(aq)
                    padded[:ROLLOUT_SHAPE.rows].copy_(q)
                    assert exact(aq, padded) and exact(sa, scale), (_name, 'AQ/Sa')
                    s['norm_checks'][_name] += 1
                    s['operand_checks'][_name] += 1
                _owner['pending'] = (y.data_ptr(), tuple(y.shape), aq, sa)
                return y

            def forward_first(x, _group=group, _member=member, _original=original_first, _owner=owner, _name=name):
                pending = _owner['pending']
                assert pending is not None
                pointer, shape, aq, sa = pending
                assert x.data_ptr() == pointer and tuple(x.shape) == shape
                assert _group.pending is None
                reference = _original(x) if s['enabled'] else None
                if s['enabled']:
                    assert _group.pending is not None
                    _group.pending = None
                _member._a8_shared_activation = (pointer, shape, aq, sa)
                try:
                    y = fp8.linear(x, _member, _group.scales[_group.names[0]], True)
                finally:
                    del _member._a8_shared_activation
                sig = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype)
                _group.pending = (sig, aq, sa, 1)
                _owner['pending'] = None
                if s['enabled']:
                    assert exact(y, reference), (_name, 'native_projection')
                    s['output_checks'][_name] += 1
                return y
            norm.forward = forward_norm
            member.forward = forward_first
    assert len(s['owners']) == spec['norm_count'] == 32
    policy.model._rms_tail_fusion_state = s
    return dict(norm_count=32, original_native_FP32_mean=True, fused_cast_square=True, BF16_roundings_preserved=True, FP8_quantizer_unchanged=True, persistent_expanded_weights=False, implementation='Fused BF16->FP32square; SAME nativeFP32mean; fused rsqrt/mul/BF16/weight/amax; original finish quantize and native GEMM')

def finish_validation(policy, c):
    s = policy.model._rms_tail_fusion_state
    for key in ['norm_checks', 'operand_checks', 'output_checks', 'square_checks']:
        assert len(s[key]) == 32 and set(s[key].values()) == {25}, (key, s[key])
    assert all((o['pending'] is None for o in s['owners']))
    s['enabled'] = False
    return dict(status='ALL800_RMS_TAIL_BF16_AQ_SCALE_NATIVE_OUTPUT_CHECKS_EXACT', **{key: dict(s[key]) for key in ['norm_checks', 'operand_checks', 'output_checks', 'square_checks']}, validation_disabled_before_timing=True, pending_empty=True)
