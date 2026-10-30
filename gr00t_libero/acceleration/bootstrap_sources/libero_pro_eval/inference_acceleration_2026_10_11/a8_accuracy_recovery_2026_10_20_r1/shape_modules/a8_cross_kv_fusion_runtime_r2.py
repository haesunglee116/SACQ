import rollout_bridge_r1 as ROLLOUT_SHAPE
"""Rank3: shared AQ followed by grouped native-exact compact Cross-KV."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from fused_ffup_cutlass_probe_r1b import bitsha
from omni_a8_scope_ffdown_runtime_r1 import exact

def install(policy, c):
    spec = c['cross_kv_fusion']
    assert spec['config_id'] in (0, 1) and spec['supply_mode'] in (1, 2)
    path = Path(spec['extension_loader']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['extension_loader']['sha256']
    receipt = json.loads(Path(spec['compile_receipt']['path']).read_text())
    assert receipt['status'] == 'CPU_COMPILE_PASS_NO_GPU_NUMERICAL_EXECUTION'
    assert hashlib.sha256(Path(receipt['binary']['path']).read_bytes()).hexdigest() == receipt['binary']['sha256']
    modspec = importlib.util.spec_from_file_location('a8_cross_kv_extension_r2', path)
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    ext = module.extension()
    assert ext.__file__ == receipt['binary']['path']
    resource = ext.resources(spec['config_id'], spec['supply_mode'])
    assert resource[0] + resource[1] <= resource[5]
    import a8_joint_pair_r1 as fp8
    scales = policy.model._a8_weight_scales_ref
    metadata = policy.model._omni_a8_attention_state['metadata']
    state = dict(enabled=True, checks={}, output_checks={}, weights={}, groups=[], compact_storage_checks=0)
    for g in policy.model._omni_a8_shared_activation_groups:
        if len(g.names) != 2:
            continue
        assert all(((m.in_features, m.out_features) == (2048, 1536) for m in g.members))
        assert all((metadata[n]['decode_key'][-1] is True for n in g.names))
        for n, m in zip(g.names, g.members):
            decoded = ext.decode_check(m.packed_indices, m.codebook, m.scales, m.shift, scales[n], m.index_bits)
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
            aq, sa = fp8.prepare_activation(x.reshape(ROLLOUT_SHAPE.rows, 2048).contiguous(), ROLLOUT_SHAPE.padded)
            args = [[getattr(m, key) for m in _members] for key in ('packed_indices', 'codebook', 'scales', 'shift')]
            args += [[scales[n] for n in _names], [m.bias for m in _members], [m.index_bits for m in _members]]
            outputs = ext.grouped(aq, sa, *args, spec['config_id'], spec['supply_mode'])
            assert len(outputs) == 2
            result = [v.reshape(1, ROLLOUT_SHAPE.rows, 1536) for v in outputs]
            if state['enabled']:
                assert result[0].untyped_storage().data_ptr() != result[1].untyped_storage().data_ptr()
                for n, y, ref in zip(_names, result, reference):
                    assert y.is_contiguous() and y.untyped_storage().nbytes() == y.numel() * y.element_size()
                    if not exact(y, ref):
                        d = y.float() - ref.float()
                        raise AssertionError(('CrossKVNativeMismatch', n, d.abs().max().item(), (d.norm() / ref.float().norm().clamp_min(1e-12)).item()))
                    state['output_checks'][n] += 1
                    state['compact_storage_checks'] += 1
                state['checks'][_name] += 1
            return result
        g.evaluate = evaluate
        state['groups'].append(g)
    assert len(state['groups']) == 16 and len(state['weights']) == 32
    policy.model._cross_kv_fusion_r2_state = state
    return dict(status='INSTALLED_PENDING_ALL3200_NATIVE_CROSS_KV_OUTPUTS', weights=state['weights'], config_id=spec['config_id'], supply_mode=spec['supply_mode'], resources=resource, independent_compact_outputs=True, original_per_matrix_scales=True, persistent_expanded_weights=False, global_weight_scratch_bytes=0, query_KV_recomputation_and_hoisting_unchanged=True, accuracy_acceptance=False, adoption=False)

def finish_validation(policy, c):
    s = policy.model._cross_kv_fusion_r2_state
    expected = (c['warmup'] + c['num_iterations']) * c['a8_layer_probe']['denoising_steps']
    assert expected == 100 and len(s['checks']) == 16 and (set(s['checks'].values()) == {expected})
    assert len(s['output_checks']) == 32 and set(s['output_checks'].values()) == {expected}
    assert s['compact_storage_checks'] == 3200 and all((not g.pending for g in s['groups']))
    s['enabled'] = False
    return dict(status='ALL3200_CROSS_KV_NATIVE_OUTPUT_AND_COMPACT_STORAGE_CHECKS_EXACT', checks_per_group=s['checks'], output_checks=s['output_checks'], compact_storage_checks=s['compact_storage_checks'], validation_disabled_before_timing=True, persistent_expanded_weights=False, scientific_changes=[])
