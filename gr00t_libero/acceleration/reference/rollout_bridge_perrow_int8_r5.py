"""Same final A8 arithmetic, current task layout and explicit canonical noise."""
import copy
import hashlib
import inspect
import json
import re
import sys
import types
from pathlib import Path

rows = padded = 0
first_observation = first_noise = None
noise_proxy = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested(flat, policy):
    result = {name: {} for name in ['video', 'state', 'language']}
    for modality in result:
        for key in policy.modality_configs[modality].modality_keys:
            name = ('annotation.human.coarse_action' if key == 'task' and 'annotation.human.coarse_action' in flat else key) if modality == 'language' else modality + '.' + key
            value = flat[name]
            result[modality][key] = [[str(item)] for item in value] if modality == 'language' else value
    return result


def set_rows(c, actual):
    global rows, padded
    assert rows == padded == 0 and actual > 0
    rows, padded = actual, (actual + 15) // 16 * 16
    # Reported native configs are shape-specific. Only row dimensions change;
    # their algorithm attributes, FP32 scales and accumulator flags do not.
    cfg = c['native_fp8_heuristic_selector']
    def dimensions(value):
        if isinstance(value, list):
            return [dimensions(x) for x in value]
        if type(value) is int:
            return rows if value == 156 else padded if value == 160 else value
        return value
    for geometry in cfg['target_geometries']:
        for key in ['A_shape', 'B_shape']:
            geometry[key] = dimensions(geometry[key])
    original = cfg['frozen_selected_config_by_descriptor']
    cfg['frozen_selected_config_by_descriptor'] = {
        re.sub(r'(?<!\d)(156|160)(?!\d)', lambda m: str(rows if m[0] == '156' else padded), key): value
        for key, value in original.items()}
    c['rollout_actual_layout'] = dict(language_rows=rows, original_physical_padding=padded,
                                     numeric_recipe_changed=False, original_rows=156)


class CanonicalNoise:
    def __init__(self, head, value):
        import torch
        assert 'GR00T_NOISE_SEED' not in __import__('os').environ
        self.buffer = value.to(device='cuda:0').clone()
        assert self.buffer.dtype == torch.bfloat16
        original = head.get_action_with_features
        raw = inspect.unwrap(original.__func__)
        self.torch = raw.__globals__['torch']
        self.original = original
        namespace = dict(raw.__globals__, torch=self)
        patched = types.FunctionType(raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__)
        patched.__kwdefaults__ = raw.__kwdefaults__
        head.get_action_with_features = types.MethodType(torch.no_grad()(patched), head)
        self.draws = 0

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def randn(self, *args, **kwargs):
        shape = kwargs.get('size', args[0] if len(args) == 1 else args)
        assert tuple(shape) == tuple(self.buffer.shape)
        assert kwargs['dtype'] == self.buffer.dtype and 'generator' not in kwargs
        self.draws += 1
        return self.buffer.clone()

    def select(self, value):
        assert value.shape == self.buffer.shape and value.dtype == self.buffer.dtype
        self.buffer.copy_(value)


def install_noise(policy):
    global noise_proxy
    assert noise_proxy is None and first_noise is not None
    noise_proxy = CanonicalNoise(policy.model.action_head, first_noise)


def complete(policy,c,observation,receipt,bi,ai,expected_rng):
    from omni_a8_all_rng_snapshot_r1 import snapshot
    assert receipt['denoising_steps']==4 and receipt['a8_scope']['actual_8bit_projection_count']==441
    assert receipt['whole_model_graph_validation']['all25_action_bytes_exact'] and snapshot()==expected_rng
    receipt['status']='PERROW_BOOTSTRAP_ALL441_NATIVE_PAYLOAD_OWN25_AND_GRAPH_PASS'
    receipt['actual_rollout_layout']=c['rollout_actual_layout']
    receipt['noise']=dict(source='Canonical currentepisode/firstquery repeatedonlyforstartupchecks',global_random_draw_replaced_by_explicit_noise=True,noise_shape=list(noise_proxy.buffer.shape))
    target=Path(c['output_directory'])/'BOOTSTRAP_NATIVE_PASS.json';target.write_text(json.dumps(receipt,indent=2))
    return dict(policy=policy,contract=c,bootstrap=receipt,bootstrap_path=str(target))


def build(contract_path, observation, noise):
    global first_observation, first_noise
    first_observation, first_noise = observation, noise
    import importlib.util
    spec = importlib.util.spec_from_file_location('rollout_bootstrap_policy', Path(__file__).with_name('bootstrap_policy_perrow_int8_r5.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv
    try:
        sys.argv = [module.__file__, '--protocol', str(contract_path), '--arm', json.loads(Path(contract_path).read_text())['isolation']['source_arm']]
        return module.main()
    finally:
        sys.argv = old_argv


def native_current_call(policy,callback):
    row=policy.model._perrow_a8_candidate_state;att=policy.model._attention_A8_expansion_state;prep=policy.model._sage_single_preparation_state
    assert not row['validation_enabled'] and not att['enabled'] and not prep['enabled']
    old=row['forward_calls'];row['forward_calls']={n:0 for n in old};row['validation_enabled']=True;att['enabled']=True;prep['enabled']=True
    pointwise=[getattr(policy.model,k) for k in ['_common_vision_rotary_state','_common_language_qk_norm_state','_adaln_norm_modulate_state']]
    assert all(not s['enabled'] for s in pointwise)
    for s in pointwise:s['enabled']=True
    try:
        value=callback();counts=row['forward_calls'];shapes=policy._rollout_contract['perrow_method']['projection_shapes']
        for name,count in counts.items():
            if name=='action_head.model.proj_out_1' or (name.startswith('action_head.') and '.norm1.linear' in name):expected=0
            elif name.startswith('action_head.') and (name.endswith('.attn1.to_k') or name.endswith('.attn1.to_v')) and shapes[name][1]==2048:expected=1
            else:expected=4 if name.startswith('action_head.') else 1
            assert count==expected,('ActualRow8ModuleCoverage',name,count,expected)
        assert sum(counts.values())==888
        policy._perrow_last_native_counts=dict(counts)
        return value
    finally:
        row['forward_calls']=old;row['validation_enabled']=False;att['enabled']=False;prep['enabled']=False
        for s in pointwise:s['enabled']=False


def bind_policy(bundle, profile):
    """Integrate the shared DefaultAcceleration factory with existing whole Graph."""
    import time
    import torch
    from rollout_acceleration_default_20260920 import DefaultAcceleration, exact, rng_snapshot, rng_restore
    from omni_remaining_cost_profile_r2 import save
    policy, c = bundle['policy'], bundle['contract']
    policy._rollout_contract = c
    owner = policy.model._whole_model_graph_owner
    original_get_action = policy.model.get_action
    bindings=inspect.getclosurevars(owner['call']).nonlocals
    copy_backbone=bindings['copy_backbone_input'];copy_action=bindings['copy_action_input']
    route = dict(mode='graph', checks=0, model_wall_ms=[], calls=0, canaries=[], eager_seconds=0., graph_seconds=0., record=False)

    def model_action(**kwargs):
        inputs = kwargs['inputs'] if set(kwargs) == {'inputs'} else kwargs
        bi, ai = policy.model.prepare_input(inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if route['mode'] == 'graph':
            value = owner['call'](bi, ai)
        else:
            copy_backbone(bi);copy_action(ai)
            value = native_current_call(policy, owner['body']) if route['mode']=='eager' else owner['body']()
        value = value.clone()
        torch.cuda.synchronize()
        if route['record']:route['model_wall_ms'].append((time.perf_counter() - start) * 1000)
        return dict(action_pred=value)

    policy.model.get_action = model_action

    class FinalA8Acceleration(DefaultAcceleration):
        def __init__(self):
            super().__init__(profile)
            # The accepted packed A8 path already has fixed timestep outputs
            # and whole-model Graph; a second dense-weight cache would change
            # its specified residency and bypass current decode oracles.
            self.cache.allow_cache = False
            self.cache.state = 'DISABLED'
            self.state = 'CANARY'

        def __call__(self, observation, noise):
            assert not policy.model.training and not any(p.requires_grad for p in policy.model.parameters())
            noise_proxy.select(noise)
            current = nested(observation, policy)
            before = rng_snapshot()
            if route['checks'] < 3:
                route['mode'] = 'eager'
                baseline = policy.get_action(current)
                after = rng_snapshot()
                rng_restore(before)
                route['mode'] = 'graph'
                candidate = policy.get_action(current)
                assert exact(candidate, baseline) and exact(rng_snapshot(), after), 'Actual rollout own-A8 action/RNG canary mismatch'
                assert exact(before, after), 'Explicit canonical-noise invocation consumed a global RNG'
                timings={}
                for mode in ['eager_timing','graph']:
                    route['mode']=mode;rng_restore(before);torch.cuda.synchronize();start=time.perf_counter()
                    timed_value=policy.get_action(current);torch.cuda.synchronize();timings[mode]=time.perf_counter()-start
                    assert exact(timed_value,baseline) and exact(rng_snapshot(),after),'Uninstrumented eager/graph timed action or RNG mismatch'
                route['eager_seconds']+=timings['eager_timing'];route['graph_seconds']+=timings['graph']
                route['checks'] += 1
                route['canaries'].append(dict(actual_call=route['calls'], action_exact=True, all4_RNG_exact=True, current_native_projection_checks=True, uninstrumented_timings_seconds=timings))
                if route['checks']==3:
                    # Observe actual production dispatch on the same third input/noise.
                    # This diagnostic call never advances simulator or canonical noise.
                    from torch.profiler import profile as torch_profile, ProfilerActivity
                    from collections import Counter
                    route['mode']='graph' if route['graph_seconds']<route['eager_seconds'] else 'eager_production'
                    rng_restore(before)
                    with torch_profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
                        dispatched=policy.get_action(current)
                        torch.cuda.synchronize()
                    assert exact(dispatched,baseline) and exact(rng_snapshot(),after)
                    trace_path=Path(c['output_directory'])/'ACTUAL_ROLLOUT_DISPATCH.json'
                    temp=trace_path.with_suffix('.tmp');trace.export_chrome_trace(str(temp));temp.replace(trace_path)
                    events=json.loads(trace_path.read_text())['traceEvents']
                    histogram=Counter(e['name'] for e in events if e.get('cat')=='kernel' and e.get('ph')=='X')
                    assert histogram,'Actual rollout CUDA kernel events missing'
                    encoding='cutlass_80_tensorop_i16832gemm_s8_'
                    native=sum(v for n,v in histogram.items() if 'cutlass::Kernel2' in n and encoding in n)
                    expected_native=888
                    assert native==expected_native,('Actual whole-model INT8 projection dispatch count',native,expected_native)
                    sage=sum(v for n,v in histogram.items() if 'qk_int_sv_f8_attn' in n)
                    expected_sage=sum(c['attention_A8_expansion']['calls_per_module_for_25queries'].values())//25 if c['isolation']['attention_lowbit'] else 0
                    assert sage==expected_sage,('Actual lowbit attention calls',sage,expected_sage)
                    launches=sum(e.get('cat')=='cuda_runtime' and e.get('ph')=='X' and e.get('name')=='cudaGraphLaunch' for e in events)
                    assert launches==(1 if route['mode']=='graph' else 0),('Actual graph launches',launches)
                    route['dispatch']=dict(status='ACTUAL_ROLLOUT_PRODUCTION_INT8_DISPATCH_ACTION_RNG_PASS',native_INT8_GEMMs=native,
                        sage_attention_calls=sage,expected_projection_INT8_GEMMs=expected_native,expected_sage_calls=expected_sage,graph_launches=launches,trace=dict(path=str(trace_path),sha256=sha(trace_path)),kernel_histogram=dict(histogram),action_exact=True,all4_RNG_exact=True)
                    save(Path(c['output_directory'])/'ACTUAL_ROLLOUT_DISPATCH_RECEIPT.json',route['dispatch'])
                    del trace,events,dispatched
                result = baseline
                rng_restore(after)
            else:
                route['mode'] = 'graph' if self.state=='ENABLED' else 'eager_production'
                route['record']=True
                result = policy.get_action(current)
                assert exact(before, rng_snapshot()), 'Canonical noise graph changed global RNG'
            route['calls'] += 1
            if route['checks'] == 3:
                self.state = 'ENABLED' if route['graph_seconds']<route['eager_seconds'] else 'DISABLED_SLOW_GRAPH'
            if route['calls'] <= 3:
                save(Path(c['output_directory'])/'ACTUAL_ROLLOUT_CANARIES.json', dict(status=self.state, checks=route['canaries'], inherited_factory_source=str(Path(inspect.getfile(DefaultAcceleration))), dense_cache=False, whole_graph=self.state=='ENABLED', uninstrumented_eager_seconds=route['eager_seconds'], graph_seconds=route['graph_seconds']))
            resident=torch.cuda.memory_allocated()/1e9
            if c['candidate_details']['memory_bound_GB'] is not None:assert resident<=c['candidate_details']['memory_bound_GB'],('Declaredresidentboundexceeded',resident)
            actions, info = result
            return {f'action.{key}': value for key, value in actions.items()}, info

        def audit(self):
            import numpy as np
            values = route['model_wall_ms']
            return dict(status=self.state, full_action_checks=route['checks'], production_dispatch=route.get('dispatch'), calls=route['calls'], canaries=route['canaries'],
                        explicit_noise=True, current_pixels_state_language_copied=True, language_rows=rows,
                        no_persistent_expanded_weight_cache=True, default_factory_integrated=True, graph_enabled=self.state=='ENABLED', eager_canary_seconds=route['eager_seconds'], graph_canary_seconds=route['graph_seconds'],
                        model_wall_median_ms=float(np.median(values)) if values else 0.,
                        model_wall_p95_ms=float(np.percentile(values,95)) if values else 0.,
                        resident_allocated_GB=torch.cuda.memory_allocated()/1e9,
                        peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,projection_format=c['perrow_method']['projection_format'],row_scale_logical_bytes=policy.model._perrow_a8_candidate_state['scale_logical_bytes'],historical_W1_5_resident_bound_GB=1.011404288,original_resident_bound_pass_claim=False)

    return FinalA8Acceleration()
