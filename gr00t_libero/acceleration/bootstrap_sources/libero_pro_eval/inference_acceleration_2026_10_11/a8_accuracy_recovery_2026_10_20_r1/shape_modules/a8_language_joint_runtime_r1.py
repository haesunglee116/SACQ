import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Original Language112 FP8 math, shared inputs and joint decode/absmax.

All GEMMs remain sequential with each original matrix scale. AQ is shared only
inside actual Q/K/V or gate/up calls. No future projection output is computed.
"""
import torch
import triton
from a8_rtn_r2 import Quantizer
from omni_a8_scope_ffdown_runtime_r1 import exact, verified
from fused_ffup_cutlass_probe_r1b import bitsha

def joint(x, m, sw, debug):
    import a8_joint_pair_r1 as fp8
    a = x.reshape(-1, m.in_features).contiguous()
    rows, k = a.shape
    mp = (rows + 15) // 16 * 16
    assert tuple(x.shape) == (1, ROLLOUT_SHAPE.rows, k) and k in (2048, 6144)
    n = m.out_features
    wq = torch.empty((n, k), device=x.device, dtype=torch.float8_e4m3fn)
    aq = torch.empty((mp, k), device=x.device, dtype=torch.float8_e4m3fn)
    parts = triton.cdiv(a.numel(), 4096)
    partial = torch.empty(parts, device=x.device, dtype=torch.float32)
    sa = torch.empty((), device=x.device, dtype=torch.float32)
    bn, bv, warps = fp8.DECODE_TILES[fp8._decode_key(m)]
    blocks = triton.cdiv(n, bn) * triton.cdiv(k // 4, bv)
    fp8._decode_codebook_and_absmax[blocks + parts,](m.packed_indices, m.codebook, m.scales, m.shift, wq, sw, a, partial, a.numel(), N=n, K=k, NUM_GROUPS=k // m.group_size, GROUP_SIZE=m.group_size, VECTOR_DIM=4, INDEX_BITS=m.index_bits, INDEX_WINDOW_BYTES=m.index_window_bytes, RECIP=m._a8_fast_weight_div, BLOCK_N=bn, BLOCK_V=bv, num_warps=warps, enable_fp_fusion=False)
    fp8._finish_quantize[triton.cdiv(mp * k, 1024),](a, partial, aq, sa, a.numel(), mp * k, parts, triton.next_power_of_2(parts), 1024, num_warps=4)
    weight_sha = bitsha(wq) if debug else None
    y = torch._scaled_mm(aq, wq.t(), sa, sw, bias=m.bias, out_dtype=torch.bfloat16, use_fast_accum=False)
    if isinstance(y, tuple):
        y = y[0]
    return (y[:rows].reshape(*x.shape[:-1], n), aq, sa, weight_sha)

class Group:

    def __init__(self, names, members, scales, metadata, state):
        self.names = names
        self.members = members
        self.scales = scales
        self.metadata = metadata
        self.state = state
        self.originals = [m.forward for m in members]
        self.pending = None

    def forward(self, slot, x):
        import a8_joint_pair_r1 as fp8
        s = self.state
        n = self.names[slot]
        m = self.members[slot]
        assert fp8.ACTIVATION_OBSERVER is None and getattr(m, '_a8_shared_activation', None) is None
        sig = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype)
        reference = self.originals[slot](x) if s['enabled'] else None
        if slot == 0:
            assert self.pending is None
            y, aq, sa, sha = joint(x, m, self.scales[n], s['enabled'])
            if s['enabled']:
                a = x.reshape(-1, m.in_features).contiguous()
                q, scale = Quantizer('fp8_e4m3fn').quantize(a)
                padded = torch.zeros_like(aq)
                padded[:a.shape[0]].copy_(q)
                assert exact(aq, padded) and exact(sa, scale), (n, 'AQ/Sa')
                assert sha == self.metadata[n]['fp8_weight_sha256'], (n, 'FP8weight')
                s['activation_checks'] += 1
                s['joint_weight_checks'] += 1
            self.pending = (sig, aq, sa, 1)
        else:
            assert self.pending is not None and self.pending[0] == sig and (self.pending[3] == slot), (n, 'sharedinput/order')
            _, aq, sa, _ = self.pending
            m._a8_shared_activation = (x.data_ptr(), tuple(x.shape), aq, sa)
            try:
                y = fp8.linear(x, m, self.scales[n], True)
            finally:
                del m._a8_shared_activation
            self.pending = (sig, aq, sa, slot + 1)
        if s['enabled']:
            assert exact(y, reference), (n, 'native_output')
            s['checks'][n] += 1
        if slot == len(self.members) - 1:
            self.pending = None
        return y

def install(policy, c):
    spec = c['language_joint_fusion']
    priority = verified(spec['priority_manifest'])
    names = priority['ranks'][4]['layers']
    assert len(names) == 112
    md = policy.model._omni_a8_language_state['metadata']
    assert sorted(md) == names
    scales = policy.model._a8_weight_scales_ref
    import a8_joint_pair_r1 as fp8
    s = dict(enabled=True, checks={n: 0 for n in names}, groups=[], weights={}, activation_checks=0, joint_weight_checks=0)
    for n in names:
        m = policy.model.get_submodule(n)
        assert fp8._decode_key(m) == tuple(md[n]['decode_key'])
        wq = fp8.decode(m, scales[n], True)
        assert bitsha(wq) == md[n]['fp8_weight_sha256'] and bitsha(scales[n]) == md[n]['fp8_weight_scale_sha256']
        s['weights'][n] = md[n]
        del wq
    for index, b in enumerate(policy.model.backbone.model.model.language_model.layers):
        prefix = f'backbone.model.model.language_model.layers.{index}.'
        for suffixes in [('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'), ('mlp.gate_proj', 'mlp.up_proj'), ('self_attn.o_proj',), ('mlp.down_proj',)]:
            ns = [prefix + v for v in suffixes]
            members = [policy.model.get_submodule(n) for n in ns]
            g = Group(ns, members, scales, md, s)
            for slot, m in enumerate(members):

                def forward(x, _g=g, _slot=slot):
                    return _g.forward(_slot, x)
                m.forward = forward
            s['groups'].append(g)
    assert len(s['groups']) == 64 and sum((len(g.names) for g in s['groups'])) == 112
    policy.model._language_joint_fusion_state = s
    return dict(status='INSTALLED_PENDING_ALL2800_LANGUAGE_NATIVE_OUTPUT_GATES', weights=s['weights'], original_projection_order=True, original_per_matrix_scales=True, persistent_expanded_weights=False, groups=64, shared_input_groups=32, joint_decode_absmax_projections=64, original_AQ_padding='logical156_physical160', accuracy_acceptance=False, adoption=False)

def finish_validation(policy, c):
    s = policy.model._language_joint_fusion_state
    assert c['warmup'] + c['num_iterations'] == 25
    assert len(s['checks']) == 112 and set(s['checks'].values()) == {25}
    assert s['activation_checks'] == s['joint_weight_checks'] == 1600
    assert all((g.pending is None for g in s['groups']))
    s['enabled'] = False
    return dict(status='ALL2800_LANGUAGE_JOINT_NATIVE_OUTPUT_AND1600_AQ_WEIGHT_CHECKS_EXACT', checks_per_layer=dict(s['checks']), activation_checks=1600, joint_weight_checks=1600, validation_disabled_before_timing=True, temporary_input_groups_empty=True, persistent_expanded_weights=False)
