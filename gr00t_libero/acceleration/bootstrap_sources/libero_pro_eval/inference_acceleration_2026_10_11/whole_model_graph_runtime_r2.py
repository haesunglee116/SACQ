"""Experimental eager-body whole-model capture; not enabled without trial gates.

Keep the Action-only graph for Action-only measurements. Replace the Backbone
graph and its retained outputs, so a second copy of those outputs is not kept
merely to time the whole model. Every whole-model call validates the original
metadata, copies current pixels/AI, and recomputes Backbone plus Action.
"""
import gc
import hashlib
from pathlib import Path
import torch
from transformers.feature_extraction_utils import BatchFeature


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def install(policy, bi, ai, references, c, arm, rng_snapshot, expected_all_rng,
            postcache_native_validator=None):
    import benchmark_inference as official
    import omni_remaining_cost_profile_r2 as measurement
    from omni_a8_scope_ffdown_runtime_r1 import exact
    from benchmark_a8_joint_pair_confirm_r1 import durable_numpy_save

    assert len(references) == c['warmup']+c['num_iterations'] == 25
    fp8 = arm.startswith('sacq') and c['precision_by_arm'][arm] == 'FP8'
    # The existing adapter validates the pre-cache schedule: cached modulation
    # no longer executes, and hoisted CrossKV executes once per query. A new
    # source/trace-supported post-cache validator is required; silently using
    # the old pre-cache counters would audit a different execution schedule.
    if fp8 and postcache_native_validator is None:
        raise RuntimeError('Whole graph prototype requires the post-cache actual-native validator before GPU use')
    bb = policy.model.backbone
    ah = policy.model.action_head
    bb_owner = bb._omni_backbone_graph_owner
    ah_owner = ah._omni_action_graph_owner
    bb_original = bb_owner['original']
    static, frozen = bb_owner['static'], bb_owner['frozen']
    static_ai, static_bo = ah_owner['static_ai'], ah_owner['static_bo']
    stream = ah_owner['stream']
    assert stream.cuda_stream == torch.cuda.current_stream().cuda_stream
    assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    before = torch.cuda.memory_allocated()
    from postcache_native_revalidation_r1 import cache_hashes
    original_cache_hashes = cache_hashes(ah_owner['stages'])
    metadata_keys = list(frozen)
    old_output_bytes = sum(v.numel()*v.element_size() for v in bb_owner['captured'].values())

    def copy_backbone_input(inputs):
        for key, target in static.items():
            value = inputs[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('Backbone graph layout changed: '+key)
        for key, target in frozen.items():
            if not torch.equal(inputs[key], target):
                raise ValueError('Fixed-demo tokens/mask/grid changed: '+key)
        static['pixel_values'].copy_(inputs['pixel_values'])

    def copy_action_input(inputs):
        assert set(inputs) == set(static_ai)
        if 'action' in inputs:
            raise ValueError('Fixed-demo adapter excludes RTC')
        for key, target in static_ai.items():
            value = inputs[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('Graph layout mismatch: '+key)
            target.copy_(value)

    def eager_backbone(inputs):
        # A standalone Backbone call still computes current features. It is
        # never replaced by cached/previous features or a deferred placeholder.
        copy_backbone_input(inputs)
        return bb_original(BatchFeature(data=dict(static)))

    def body():
        bo = bb_original(BatchFeature(data=dict(static)))
        assert set(bo) == set(static_bo)
        for key, target in static_bo.items():
            source = bo[key]
            assert source.shape == target.shape and source.dtype == target.dtype
            target.copy_(source)
        del bo
        return ah_owner['body']().action_pred

    copy_backbone_input(bi)
    copy_action_input(ai)
    torch.cuda.synchronize()
    # Do not retain the old forward closure: it closes over its graph/output.
    bb.forward = eager_backbone
    del bb_owner['graph']
    del bb_owner['captured']
    gc.collect()
    torch.cuda.synchronize()

    parent_revalidation = None
    if fp8:
        parent_revalidation = postcache_native_validator(
            policy, body, references, c, rng_snapshot, expected_all_rng)
        assert parent_revalidation['status'] == 'ALL_ACTUAL_POSTCACHE_NATIVE_OPERANDS_OUTPUTS_AND_OWN25_EXACT'

    # Same three startup warm calls as the pinned ActionStages.graph source;
    # the scientific 5+20 stream is reset and checked independently below.
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=ah_owner['graph'].pool()):
        output = body()

    state = dict(graph=graph, output=output, calls=0, metadata_keys=metadata_keys,
                 body=body, parent_revalidation=parent_revalidation)

    def call(inputs, action_inputs):
        copy_backbone_input(inputs)
        copy_action_input(action_inputs)
        graph.replay()
        state['calls'] += 1
        return output

    state['call'] = call
    policy.model._whole_model_graph_owner = state
    original_prepared = measurement.prepared

    def prepared(target_policy, backbone_inputs, action_inputs):
        owner = getattr(target_policy.model, '_whole_model_graph_owner', None)
        if owner is None:
            return original_prepared(target_policy, backbone_inputs, action_inputs)
        return owner['call'](backbone_inputs, action_inputs)

    measurement.prepared = prepared
    actual_actions = []
    official.set_seed(c['seed'])
    for query, reference in enumerate(references):
        actual = prepared(policy, bi, ai)
        assert exact(actual, reference), ('WholeGraphOwnActionMismatch', query)
        actual_actions.append(actual.float().cpu().numpy().copy())
    assert rng_snapshot() == expected_all_rng, 'WholeGraphAll4RNG'
    import numpy as np
    path = Path(c['output_directory'])/(arm+'_whole_graph_own25_actual_actions.npy')
    durable_numpy_save(path, np.stack(actual_actions))
    del actual, actual_actions
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert cache_hashes(ah_owner['stages']) == original_cache_hashes, 'WholeGraphChangedFixedCacheBytes'
    state['fixed_cache_sha256'] = original_cache_hashes
    # The caller still holds its GPU reference trajectory here. The final
    # resident gate belongs after those references are released and GC, at
    # the same point as the control arm; setup size is diagnostic only.
    result = dict(status='WHOLE_GRAPH_OWN25_RAW_ACTION_ALL4RNG_PASS_PENDING_FINAL_RESIDENT_AND_DISPATCH',
        source=ref(__file__), actual_actions=ref(path), action_checks=25,
        parent_projection_revalidation=parent_revalidation,
        all4_RNG_exact=True, all25_action_bytes_exact=True,
        resident_before_bytes=before, resident_after_setup_bytes=after,
        old_backbone_output_logical_bytes_released=old_output_bytes,
        metadata_keys=metadata_keys, original_metadata_checks_preserved=True,
        fixed_cache_sha256=original_cache_hashes,
        current_pixels_AI_and_all_features_recomputed=True,
        eager_bodies_captured=True, nested_graph_replay_captured=False,
        standalone_backbone_returns_fresh_eager_features=True,
        action_only_graph_preserved=True, original_action_loop=True,
        no_persistent_expanded_weight_cache=True, scientific_delta={},
        final_resident_dispatch_and_speed_validation_pending=True)
    state['validation'] = result
    return result
