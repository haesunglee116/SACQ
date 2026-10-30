import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Original Language SiLU BF16 -> multiply BF16 -> original A8; no recipe change."""
import torch
import triton
from a8_rtn_r2 import Quantizer
from a8_swiglu_joint_kernel_r1 import _decode_codebook_and_swiglu_absmax
from omni_a8_scope_ffdown_runtime_r1 import exact
from fused_ffup_cutlass_probe_r1b import bitsha

def install(policy, c):
    import a8_joint_pair_r1 as fp8
    s = dict(enabled=True, checks={}, weight_checks={})
    scales = policy.model._a8_weight_scales_ref
    metadata = policy.model._omni_a8_language_state['metadata']
    for index, layer in enumerate(policy.model.backbone.model.model.language_model.layers):
        mlp = layer.mlp
        assert mlp.__class__.__name__ == 'Qwen3VLTextMLP' and mlp.act_fn.__class__.__name__ == 'SiLU'
        name = f'backbone.model.model.language_model.layers.{index}.mlp.down_proj'
        member = mlp.down_proj
        assert member.in_features == 6144 and member.out_features == 2048 and (member.bias is None)
        s['checks'][name] = s['weight_checks'][name] = 0

        def forward(x, _mlp=mlp, _m=member, _name=name):
            gate = _mlp.gate_proj(x)
            up = _mlp.up_proj(x)
            assert tuple(gate.shape) == tuple(up.shape) == (1, ROLLOUT_SHAPE.rows, 6144)
            y = torch.empty_like(gate)
            n, k = (_m.out_features, _m.in_features)
            mp = ROLLOUT_SHAPE.padded
            wq = torch.empty((n, k), device=x.device, dtype=torch.float8_e4m3fn)
            aq = torch.empty((mp, k), device=x.device, dtype=torch.float8_e4m3fn)
            parts = triton.cdiv(y.numel(), 4096)
            partial = torch.empty(parts, device=x.device, dtype=torch.float32)
            sa = torch.empty((), device=x.device, dtype=torch.float32)
            sw = scales[_name]
            bn, bv, warps = fp8.DECODE_TILES[fp8._decode_key(_m)]
            blocks = triton.cdiv(n, bn) * triton.cdiv(k // 4, bv)
            _decode_codebook_and_swiglu_absmax[blocks + parts,](_m.packed_indices, _m.codebook, _m.scales, _m.shift, wq, sw, gate, up, y, partial, y.numel(), N=n, K=k, NUM_GROUPS=k // _m.group_size, GROUP_SIZE=_m.group_size, VECTOR_DIM=4, INDEX_BITS=_m.index_bits, INDEX_WINDOW_BYTES=_m.index_window_bytes, RECIP=_m._a8_fast_weight_div, BLOCK_N=bn, BLOCK_V=bv, num_warps=warps, enable_fp_fusion=False)
            fp8._finish_quantize[triton.cdiv(mp * k, 1024),](y, partial, aq, sa, y.numel(), mp * k, parts, triton.next_power_of_2(parts), 1024, num_warps=4)
            out = torch._scaled_mm(aq, wq.t(), sa, sw, bias=_m.bias, out_dtype=torch.bfloat16, use_fast_accum=False)
            if isinstance(out, tuple):
                out = out[0]
            out = out[:ROLLOUT_SHAPE.rows].reshape(1, ROLLOUT_SHAPE.rows, n)
            if s['enabled']:
                refy = _mlp.act_fn(gate) * up
                assert exact(y, refy), (_name, 'SiLU_BF16_mul_BF16', int((y != refy).sum().item()), (y.float() - refy.float()).abs().max().item())
                q, scale = Quantizer('fp8_e4m3fn').quantize(refy.reshape(ROLLOUT_SHAPE.rows, k))
                padded = torch.zeros_like(aq)
                padded[:ROLLOUT_SHAPE.rows].copy_(q)
                assert exact(aq, padded) and exact(sa, scale), (_name, 'AQ/Sa')
                assert bitsha(wq) == metadata[_name]['fp8_weight_sha256'], (_name, 'weightbytes')
                reference = _mlp.down_proj(refy)
                assert exact(out, reference), (_name, 'native_output')
                s['checks'][_name] += 1
                s['weight_checks'][_name] += 1
            return out
        mlp.forward = forward
    assert len(s['checks']) == c['swiglu_fusion']['module_count'] == 16
    policy.model._swiglu_fusion_state = s
    return dict(module_count=16, original_SiLU_BF16_round=True, original_mul_BF16_round=True, original_global_per_tensor_E4M3_AQ=True, native_scaled_mm=True, persistent_expanded_weights=False, source='PyTorch2.11SiLU x/(1+exp(-x)); libdeviceexp anddiv_rn; originaldecoder')

def finish_validation(policy, c):
    s = policy.model._swiglu_fusion_state
    assert len(s['checks']) == 16 and set(s['checks'].values()) == {25}
    assert s['checks'] == s['weight_checks']
    s['enabled'] = False
    return dict(status='ALL400_SWIGLU_BF16_AQ_SCALE_WEIGHT_NATIVE_OUTPUT_CHECKS_EXACT', checks=dict(s['checks']), weight_checks=dict(s['weight_checks']), validation_disabled_before_timing=True)
