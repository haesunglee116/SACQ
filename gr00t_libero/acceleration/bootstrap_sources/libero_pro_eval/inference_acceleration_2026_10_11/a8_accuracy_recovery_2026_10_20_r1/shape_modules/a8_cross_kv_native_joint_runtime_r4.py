import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Same Cross-KV math; joint K decode/AQ and native compact-M GEMMs."""
import hashlib
import json
from pathlib import Path
import torch
from a8_rtn_r2 import Quantizer
from a8_cross_kv_native_joint_r4 import evaluate as native_joint
from fused_ffup_cutlass_probe_r1b import bitsha
from omni_a8_scope_ffdown_runtime_r1 import exact

def install(policy, c):
    spec = c['cross_kv_fusion']
    assert spec['implementation'] == 'native_joint_K_decode_AQ_compact_M156'
    evidence = spec['native_exact_evidence']
    assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
    assert json.loads(Path(evidence['path']).read_text())['all3200_outputs_exact']
    import a8_joint_pair_r1 as fp8
    scales = policy.model._a8_weight_scales_ref
    metadata = policy.model._omni_a8_attention_state['metadata']
    state = dict(enabled=True, checks={}, output_checks={}, weights={}, groups=[], compact_storage_checks=0, activation_checks=0, weight_byte_checks=0)
    for g in policy.model._omni_a8_shared_activation_groups:
        if len(g.names) != 2:
            continue
        assert all(((m.in_features, m.out_features) == (2048, 1536) for m in g.members))
        for n, m in zip(g.names, g.members):
            assert fp8._decode_key(m) == tuple(metadata[n]['decode_key'])
            decoded = fp8.decode(m, scales[n], True)
            assert bitsha(decoded) == metadata[n]['fp8_weight_sha256']
            assert bitsha(scales[n]) == metadata[n]['fp8_weight_scale_sha256']
            state['weights'][n] = dict(fp8_weight_sha256=metadata[n]['fp8_weight_sha256'], fp8_weight_scale_sha256=metadata[n]['fp8_weight_scale_sha256'], payload_sha256=metadata[n]['payload_sha256'])
            state['output_checks'][n] = 0
            m._a8_compact_crosskv = True
            del decoded
        name = g.names[0]
        state['checks'][name] = 0
        original = g.evaluate

        def evaluate(x, _original=original, _members=g.members, _names=g.names, _name=name):
            assert tuple(x.shape) == (1, ROLLOUT_SHAPE.rows, 2048) and x.dtype == torch.bfloat16
            reference = _original(x) if state['enabled'] else None
            value = native_joint(x, _members, _names, scales, debug=state['enabled'])
            if state['enabled']:
                result, details = value
                aq, sa = Quantizer('fp8_e4m3fn').quantize(x.reshape(ROLLOUT_SHAPE.rows, 2048))
                assert exact(details['aq'], aq) and exact(details['sa'], sa)
                assert details['weight_FP8_sha256'] == {n: metadata[n]['fp8_weight_sha256'] for n in _names}
                assert result[0].untyped_storage().data_ptr() != result[1].untyped_storage().data_ptr()
                for n, y, ref in zip(_names, result, reference):
                    assert y.is_contiguous() and y.untyped_storage().nbytes() == y.numel() * y.element_size()
                    assert exact(y, ref), ('NativeJointCrossKVMismatch', n)
                    state['output_checks'][n] += 1
                    state['compact_storage_checks'] += 1
                state['checks'][_name] += 1
                state['activation_checks'] += 1
                state['weight_byte_checks'] += 2
            else:
                result = value
            return result
        g.evaluate = evaluate
        state['groups'].append(g)
    assert len(state['groups']) == 16 and len(state['weights']) == 32
    policy.model._cross_kv_fusion_r2_state = state
    return dict(status='INSTALLED_PENDING_ALL3200_NATIVE_CROSS_KV_OUTPUTS', weights=state['weights'], implementation=spec['implementation'], independent_compact_outputs=True, original_per_matrix_scales=True, persistent_expanded_weights=False, transient_FP8_weight_logical_bytes_at_a_time=1536 * 2048, query_KV_recomputation_and_hoisting_unchanged=True, accuracy_acceptance=False, adoption=False)

def finish_validation(policy, c):
    from a8_cross_kv_fusion_runtime_r2 import finish_validation as finish
    s = policy.model._cross_kv_fusion_r2_state
    assert s['activation_checks'] == 1600 and s['weight_byte_checks'] == 3200
    result = finish(policy, c)
    result.update(activation_checks=s['activation_checks'], weight_byte_checks=s['weight_byte_checks'], native_GEMMs_preserved=True)
    return result
