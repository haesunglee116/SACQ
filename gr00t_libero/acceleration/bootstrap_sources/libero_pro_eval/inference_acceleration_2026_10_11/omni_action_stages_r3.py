"""Same-action/RNG gated fixed-demo acceleration for the official Omni port.

Uses the stock action loop, not a rewritten numerical algorithm. No decoded
weight cache. Fixed outputs are prepared for the loaded timestep schedule.
"""
import torch
from transformers.feature_extraction_utils import BatchFeature
from fixed_adaln_r2 import FixedAdaLN
from fixed_schedule_outputs import FixedScheduleOutputs


class ActionStages:
    def __init__(self, policy, bi, ai):
        self.policy = policy
        self.bi = bi
        self.ai = ai
        self.ah = policy.model.action_head
        self.original = self.ah.get_action
        self.cache = None
        self.kv = None
        self.counter = 0

    def fixed_outputs(self):
        ah = self.ah
        self.cache = FixedAdaLN([b.norm1 for b in ah.model.transformer_blocks])
        self.schedule = FixedScheduleOutputs([
            ('dit_timestep_encoder', ah.model.timestep_encoder),
            ('dit_output_modulation', ah.model.proj_out_1),
            ('action_sinusoidal_time', ah.action_encoder.pos_encoding),
            ('action_position_embedding', ah.position_embedding)])
        bo = self.policy.model.backbone(self.bi)
        self.schedule.prepare(lambda: self.cache.prepare(lambda: self.original(bo, self.ai)))
        assert self.cache.steps == self.schedule.steps == ah.num_inference_timesteps
        self.cache.enable()
        self.schedule.enable()
        def select_step(module, args):
            # Stock Omni encodes action time before DiT timestep_encoder.
            # Select at action_encoder entry so all fixed outputs see this step.
            self.cache.select_step(self.counter)
            self.schedule.select_step(self.counter)
            if self.kv is not None:
                self.kv.step = self.counter
            self.counter += 1
        self.step_hook = ah.action_encoder.register_forward_pre_hook(select_step)

        def get_action(bo, ai, options=None):
            if options or 'action' in ai:
                raise ValueError('Fixed-demo adapter excludes RTC')
            self.counter = 0
            if self.kv is not None:
                self.kv.begin_action()
            y = self.original(bo, ai, options)
            assert self.counter == self.cache.steps
            return y
        ah.get_action = get_action
        return dict(adaln_steps=self.cache.steps, adaln_bytes=self.cache.bytes,
                    schedule=self.schedule.receipt(), stock_action_loop=True,
                    decoded_weights_cached=False)

    def cross_kv(self):
        from kv_hoist_a8_logical_r2 import install
        self.kv = install(self.policy)
        return dict(projections=self.kv.entries[0], recompute_every_action=True,
                    decoded_weights_cached=False)

    def graph(self):
        # Preserve pre-encoder tensors. The stock encoder mutates its mapping.
        bo = self.policy.model.backbone(self.bi)
        static_bo = {k: v.clone() for k, v in bo.items()}
        static_ai = {k: v.clone() for k, v in self.ai.items()}
        eager = self.ah.get_action

        def body():
            return eager(BatchFeature(data=dict(static_bo)), BatchFeature(data=dict(static_ai)))
        stream = torch.cuda.current_stream()
        assert stream.cuda_stream != torch.cuda.default_stream().cuda_stream, "Use one dedicated model stream"
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            output = body()
        captured = dict(output)

        def get_action(bo, ai, options=None):
            if options or 'action' in ai:
                raise ValueError('Fixed-demo adapter excludes RTC')
            for source, dest in [(bo, static_bo), (ai, static_ai)]:
                assert set(source) == set(dest)
                for k, v in dest.items():
                    x = source[k]
                    if x.shape != v.shape or x.dtype != v.dtype:
                        raise ValueError('Graph layout mismatch: ' + k)
                    v.copy_(x)
            g.replay()
            return BatchFeature(data=dict(captured))
        self.ah.get_action = get_action
        self.ah._omni_action_graph_owner = dict(graph=g, static_bo=static_bo,
            static_ai=static_ai, captured=captured, body=body, stream=stream, stages=self)
        return dict(cuda_graph=True, stock_action_loop=True, default_generator_in_graph=True,
                    graph_inputs_rewritten_each_action=True, fixed_input_only=True,
                    decoded_weight_cache=False)


def profile_action(policy, bi, ai, contract):
    """CUDA activity events, no privileged counters or profiled latency claim."""
    import re
    from collections import defaultdict
    from torch.profiler import profile, ProfilerActivity, record_function
    import benchmark_inference as official
    ah = policy.model.action_head
    # Backbone is deterministic at this fixed input; profile action only.
    bo = policy.model.backbone(bi)
    raw = dict(bo)
    official.set_seed(contract['seed'])
    for _ in range(contract['warmup']):
        ah.get_action(BatchFeature(data=dict(raw)), ai)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(contract['num_iterations']):
            with record_function('omni_action'):
                ah.get_action(BatchFeature(data=dict(raw)), ai)
        torch.cuda.synchronize()
    groups = defaultdict(lambda: dict(kernel_us=0.0, count=0))
    names = defaultdict(lambda: dict(kernel_us=0.0, count=0))
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.name
        if name == 'omni_action':
            continue
        if re.search(r'_dequantize_packed_codebook_weight|_hqq4_dequant_kernel|_decode', name):
            group = 'weight_decode'
        elif re.search(r'gemm|xmma|mma|matmul', name, re.I):
            group = 'gemm'
        elif re.search(r'flash|fmha|attention', name, re.I):
            group = 'attention'
        else:
            group = 'other_compute_copy'
        t = float(event.device_time_total)
        groups[group]['kernel_us'] += t; groups[group]['count'] += 1
        names[name]['kernel_us'] += t; names[name]['count'] += 1
    assert sum(v['count'] for v in groups.values()) > 0
    ranges = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU and e.name == 'omni_action']
    assert len(ranges) == contract['num_iterations']
    operators = []
    for e in prof.key_averages():
        if e.self_cpu_time_total > 0:
            operators.append(dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                                  self_device_us=e.self_device_time_total))
    return dict(groups=dict(groups), top_kernels=sorted([dict(name=k, **v) for k,v in names.items()],
                key=lambda x:x['kernel_us'], reverse=True)[:40],
                top_cpu_operators=sorted(operators,key=lambda x:x['self_cpu_us'],reverse=True)[:30],
                profiled_action_calls=len(ranges), warmup_excluded=contract['warmup'],
                interpretation='Kernel activity sum / calls, NOT guaranteed removable wall time; profiler changes launch overhead')
