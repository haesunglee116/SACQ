"""Priority1 trial. Replace only the16 existing shared-AQ self-QKV groups.

Every original per-matrix payload/FP8 scale/bias is retained. The original
shared native evaluator is the actual-trajectory oracle before timing.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from fused_ffup_cutlass_probe_r1b import bitsha
from omni_a8_scope_ffdown_runtime_r1 import exact


def install(policy, c):
    spec = c['self_qkv_fusion']
    assert (spec['config_id'],spec['supply_mode'],spec['promotion_mode'],spec['epilogue_mode']) == (0,2,1,1)
    path = Path(spec['extension_loader']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['extension_loader']['sha256']
    receipt = json.loads(Path(spec['compile_receipt']['path']).read_text())
    assert receipt['status'] == 'CPU_COMPILE_PASS_NO_GPU_NUMERICAL_EXECUTION'
    assert hashlib.sha256(Path(receipt['binary']['path']).read_bytes()).hexdigest() == receipt['binary']['sha256']
    module_spec = importlib.util.spec_from_file_location('a8_self_qkv_fusion_extension_r1', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    ext = module.extension()
    assert ext.__file__ == receipt['binary']['path']
    import a8_joint_pair_r1 as fp8
    state = dict(enabled=True, checks={}, output_checks={}, weights={}, groups=[])
    scales = policy.model._a8_weight_scales_ref
    metadata = policy.model._omni_a8_attention_state['metadata']
    for g in policy.model._omni_a8_shared_activation_groups:
        if len(g.names) != 3:
            continue
        assert all(m.in_features == m.out_features == 1536 for m in g.members)
        assert all(list(m.shift.shape) in ([], [1]) for m in g.members)
        assert all(metadata[n]['decode_key'][-1] is True for n in g.names)
        for n, m in zip(g.names, g.members):
            decoded = ext.decode_check(m.packed_indices, m.codebook, m.scales, m.shift, scales[n], m.index_bits)
            assert bitsha(decoded) == metadata[n]['fp8_weight_sha256'], (n, 'nativeFP8weightbytes')
            assert bitsha(scales[n]) == metadata[n]['fp8_weight_scale_sha256']
            state['weights'][n] = dict(fp8_weight_sha256=metadata[n]['fp8_weight_sha256'],
                                       fp8_weight_scale_sha256=metadata[n]['fp8_weight_scale_sha256'],
                                       payload_sha256=metadata[n]['payload_sha256'], index_bits=m.index_bits)
            state['output_checks'][n] = 0
            del decoded
        name = g.names[0]
        state['checks'][name] = 0
        original = g.evaluate

        def evaluate(x, _original=original, _members=g.members, _names=g.names, _name=name):
            assert tuple(x.shape) == (1, 41, 1536) and x.dtype == torch.bfloat16
            reference = _original(x) if state['enabled'] else None
            aq, sa = fp8.prepare_activation(x.reshape(41, 1536).contiguous(), 48)
            # Resolve current module views after compaction/scale relocation;
            # retaining pre-compaction Tensor objects would duplicate payloads.
            args = [[getattr(m, key) for m in _members] for key in ('packed_indices', 'codebook', 'scales', 'shift')]
            args += [[scales[n] for n in _names], [m.bias for m in _members], [m.index_bits for m in _members]]
            y = ext.grouped(aq, sa, *args, spec['config_id'], spec['supply_mode'], spec['promotion_mode'], spec['epilogue_mode'])
            result = [y[i].reshape(1, 41, 1536) for i in range(3)]
            if state['enabled']:
                diagnostics = []
                for n, a, b in zip(_names, result, reference):
                    ok = exact(a, b)
                    if not ok:
                        d = a.float() - b.float()
                        diagnostics.append(dict(layer=n, max_abs=d.abs().max().item(),
                                                relative_l2=(d.norm()/b.float().norm().clamp_min(1e-12)).item()))
                    else:
                        state['output_checks'][n] += 1
                assert not diagnostics, ('GroupedFP8nativeoutputmismatch', diagnostics)
                state['checks'][_name] += 1
            return result

        g.evaluate = evaluate
        state['groups'].append(g)
    assert len(state['groups']) == 16 and len(state['weights']) == 48
    policy.model._self_qkv_fusion_r1_state = state
    return dict(status='INSTALLED_PENDING_ALL_ORIGINAL25_NATIVE_GROUP_GATES', groups=16, projections=48,
                config_id=spec['config_id'], supply_mode=spec['supply_mode'],
                weights=state['weights'], persistent_expanded_weights=False, global_weight_scratch_bytes=0,
                original_per_matrix_FP8_scales=True, compact_contiguous_outputs=True,
                activation_preparation='Unchangedexistingprepare_activation; oncepergroup',
                expected_grouped_kernels_per_action=64, expected_remaining_native_FP8_MMA=568,
                accuracy_acceptance=False, adoption=False)


def finish_validation(policy, c):
    s = policy.model._self_qkv_fusion_r1_state
    expected = (c['warmup'] + c['num_iterations']) * c['a8_layer_probe']['denoising_steps']
    assert len(s['checks']) == 16 and set(s['checks'].values()) == {expected}
    assert len(s['output_checks']) == 48 and set(s['output_checks'].values()) == {expected}
    assert all(not g.pending for g in s['groups'])
    s['enabled'] = False
    return dict(status='ALL1600_GROUP_AND4800_NATIVE_OUTPUT_CHECKS_EXACT', checks_per_group=s['checks'],
                output_checks=s['output_checks'], validation_disabled_before_timing=True,
                scientific_values_changed=[], persistent_expanded_weights=False)
