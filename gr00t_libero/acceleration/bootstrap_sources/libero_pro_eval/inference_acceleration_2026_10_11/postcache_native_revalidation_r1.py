"""Validate the actual cached Action schedule without disabling its caches.

Cached modulation executes zero times; CrossKV is computed once per query.
All other existing numerical oracles remain active for their original calls.
Counter snapshots are restored for independent prior receipts/scope audits.
"""
import copy
import hashlib
import importlib
from pathlib import Path
import numpy as np
import torch


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def cache_hashes(stages):
    from fused_ffup_cutlass_probe_r1b import bitsha
    result = {}
    for layer, row in enumerate(stages.cache.outputs):
        for step, value in enumerate(row):
            result[f'AdaLN/{layer}/{step}'] = bitsha(value)
    for entry in stages.schedule.entries:
        for step, value in enumerate(entry['outputs']):
            result[f'schedule/{entry["name"]}/{step}'] = bitsha(value)
    return result


def run(policy, body, references, c, rng_snapshot, expected_all_rng):
    import benchmark_inference as official
    from a8_sage_attention_adapter_r4 import begin_projection_revalidation
    from omni_a8_scope_ffdown_runtime_r1 import exact
    from benchmark_a8_joint_pair_confirm_r1 import durable_numpy_save

    queries = c['warmup']+c['num_iterations']
    assert len(references) == queries == 25
    stages = policy.model.action_head._omni_action_graph_owner['stages']
    assert stages.cache.steps == stages.schedule.steps == c['a8_layer_probe']['denoising_steps'] == 4
    assert stages.kv is not None and stages.kv.entries[0] == 32
    before_cache = cache_hashes(stages)
    attrs = ['_self_qkv_fusion_r1_state', '_cross_kv_fusion_r2_state',
             '_language_joint_fusion_state', '_vision_joint_fusion_state',
             '_omni_a8_scope_extra_state', '_omni_a8_modulation_state',
             '_omni_fp8_norm_producer_state']
    for feature, attr in [('rms_tail_fusion', '_rms_tail_fusion_state'),
                          ('mask_bias_fusion', '_mask_bias_fusion_state'),
                          ('swiglu_fusion', '_swiglu_fusion_state')]:
        if c[feature]['enabled_by_arm'][c['_arm']]:
            attrs.append(attr)
    snapshots = {}
    for attr in attrs:
        state = getattr(policy.model, attr)
        snapshots[attr] = {key: copy.deepcopy(value) for key, value in state.items()
            if key in ['enabled', 'validation_enabled', 'activation_checks', 'joint_weight_checks',
                       'operand_checks', 'compact_storage_checks', 'weight_byte_checks', 'shapes']
            or 'check' in key}
    begin_projection_revalidation(policy, c)
    gates = {}
    actual_actions = []
    try:
        official.set_seed(c['seed'])
        for query, reference in enumerate(references):
            actual = body()
            assert exact(actual, reference), ('PostcacheNativeOwnActionMismatch', query)
            actual_actions.append(actual.float().cpu().numpy().copy())
        assert rng_snapshot() == expected_all_rng, 'PostcacheNativeAll4RNG'
        del actual
        # Source: ActionStages resets KV each action; the installed hoist
        # computes K/V at step0 and reuses only within that action. Counters
        # below therefore cover25 calls per group, not the pre-cache100.
        cross = policy.model._cross_kv_fusion_r2_state
        assert len(cross['checks']) == 16 and set(cross['checks'].values()) == {queries}
        assert len(cross['output_checks']) == 32 and set(cross['output_checks'].values()) == {queries}
        assert cross['activation_checks'] == 16*queries
        assert cross['weight_byte_checks'] == cross['compact_storage_checks'] == 32*queries
        assert all(not group.pending for group in cross['groups'])
        cross['enabled'] = False
        gates['CrossKV'] = dict(status='ALL800_ACTUAL_POSTCACHE_CROSS_KV_NATIVE_OPERANDS_OUTPUTS_COMPACT_EXACT',
            checks_per_group=copy.deepcopy(cross['checks']), output_checks=copy.deepcopy(cross['output_checks']),
            activation_checks=cross['activation_checks'], weight_byte_checks=cross['weight_byte_checks'],
            compact_storage_checks=cross['compact_storage_checks'], validation_disabled_before_timing=True)
        modulation = policy.model._omni_a8_modulation_state
        assert len(modulation['checks']) == 33 and set(modulation['checks'].values()) == {0}
        modulation['validation_enabled'] = False
        assert cache_hashes(stages) == before_cache, 'FixedScheduleCacheChangedDuringPostcacheValidation'
        gates['CachedModulation33'] = dict(status='ZERO_NATIVE_MODULATION_CALLS_AND_ALL_FIXED_CACHE_BYTES_UNCHANGED',
            checks=copy.deepcopy(modulation['checks']), fixed_cache_sha256=before_cache,
            interpretation='Original immutable pre-cache and fixed-output equivalence receipts remain separate; cached operations are not claimed as newly executed native checks.')
        functions = [('SelfQKV', 'a8_self_qkv_fusion_runtime_r3'),
                     ('Language', 'a8_language_joint_runtime_r1'),
                     ('Vision', 'a8_vision_joint_runtime_r1b'),
                     ('FFdown', 'omni_a8_scope_ffdown_runtime_r1'),
                     ('FFup', c['lowbit_runtime_modules']['FP8']['producer'])]
        for name, module in functions:
            gates[name] = importlib.import_module(module).finish_validation(policy, c)
        attention = policy.model._omni_a8_attention_state
        names = {f'action_head.model.transformer_blocks.{i}.attn1.to_out.0' for i in range(32)}
        names |= {f'action_head.model.transformer_blocks.{i}.attn1.to_q' for i in range(0, 32, 2)}
        active = {n: value for n, value in attention['checks'].items() if value}
        assert set(active) == names and set(active.values()) == {queries*stages.cache.steps}
        gates['remaining_Attention48'] = dict(status='ALL4800_ACTUAL_NATIVE_ATTENTION_OUTPUTS_EXACT', checks=active)
        rms_module = ('a8_rms_tail_square_runtime_r2' if c['rms_square_fusion']['enabled_by_arm'][c['_arm']]
                      else 'a8_rms_tail_joint_runtime_r1')
        for feature, module in [('rms_tail_fusion', rms_module),
                                ('mask_bias_fusion', 'a8_attention_mask_bias_runtime_r1'),
                                ('swiglu_fusion', 'a8_swiglu_joint_runtime_r1b')]:
            if c[feature]['enabled_by_arm'][c['_arm']]:
                gates[feature] = importlib.import_module(module).finish_validation(policy, c)
    finally:
        attention = policy.model._omni_a8_attention_state
        attention['validation_enabled'] = False
        attention['checks'] = policy.model._attention8_prior_native_checks
        for module, original in policy.model._attention8_native_remaining_holders:
            module.forward = original
        del policy.model._attention8_native_remaining_holders
        del policy.model._attention8_prior_native_checks
        for attr, values in snapshots.items():
            state = getattr(policy.model, attr)
            for key, value in values.items():
                state[key] = value
    path = Path(c['output_directory'])/(c['_arm']+'_postcache_native_own25_actual_actions.npy')
    durable_numpy_save(path, np.stack(actual_actions))
    return dict(status='ALL_ACTUAL_POSTCACHE_NATIVE_OPERANDS_OUTPUTS_AND_OWN25_EXACT',
        source=ref(__file__), gates=gates, actual_actions=ref(path),
        own25_action_bytes_exact=True, all4_RNG_exact=True,
        cached_schedule_kept_enabled=True, prior_counter_receipts_preserved=True,
        source_contracts=[ref(Path(__file__).parent/name) for name in
            ['omni_action_stages_r3.py', 'kv_hoist_a8_logical_r2.py', 'fixed_adaln_r2.py',
             'fixed_schedule_outputs.py', 'a8_sage_attention_adapter_r4.py']],
        scientific_delta={})
