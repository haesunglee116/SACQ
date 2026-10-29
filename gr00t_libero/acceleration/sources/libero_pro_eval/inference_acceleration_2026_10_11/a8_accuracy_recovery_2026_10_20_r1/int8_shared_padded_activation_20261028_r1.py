"""Reuse exact activation bytes only within one sibling projection parent call.

Cache lifetime is one parent forward, never one model query or denoising step.
Retaining the input view prevents allocation-address reuse within that call.
Every validation hit recomputes the original producer and checks both outputs.
"""
import torch
import perrow_int8_fast_policy_r3 as fast
import perrow_int8_padded_producer_r1 as producer


def enable():
    original_install = fast.install
    original_activation = producer.activation
    frames = []
    state = {'parents': [], 'hits': 0, 'misses': 0, 'validated_hits': 0}
    model_state = None

    def activation(x, mode):
        if not frames:
            return original_activation(x, mode)
        key = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, mode)
        cache = frames[-1]
        if key not in cache:
            result = original_activation(x, mode)
            cache[key] = (x, result)
            state['misses'] += 1
            return result
        source, (q, scale) = cache[key]
        assert source.device == x.device
        if model_state['validation_enabled']:
            expected_q, expected_scale = original_activation(x, mode)
            assert torch.equal(q, expected_q), 'SharedActivationINT8Bytes'
            assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32)), 'SharedActivationScaleBits'
            state['validated_hits'] += 1
        state['hits'] += 1
        return q, scale

    def install(policy, names, contract):
        nonlocal model_state
        result = original_install(policy, names, contract)
        model_state = policy.model._perrow_a8_candidate_state
        names = set(names)
        # These modules evaluate sibling projections on an unchanged input.
        # Actual input/producer-byte gates remain enabled during bootstrap.
        groups = [('q_proj', 'k_proj', 'v_proj'), ('gate_proj', 'up_proj'),
                  ('to_q', 'to_k', 'to_v')]
        parents = sorted({name.rsplit('.', 1)[0] for name in names})
        for parent_name in parents:
            group = next((g for g in groups if all(parent_name + '.' + n in names for n in g)), None)
            if group is None:
                continue
            parent = policy.model.get_submodule(parent_name)
            def enter(this, args):
                frames.append({})

            def leave(this, args, output):
                frames.pop()

            # Hooks survive the existing attention forward replacements.
            parent.register_forward_pre_hook(enter)
            parent.register_forward_hook(leave, always_call=True)
            state['parents'].append({'name': parent_name, 'children': list(group)})
        assert state['parents'], 'No eligible sibling projection groups'
        producer.activation = activation
        result['shared_activation'] = state
        policy.model._int8_shared_activation_state = state
        return result

    fast.install = install
    return state
