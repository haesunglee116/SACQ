"""Share exact FP8 input within sequential Q/K/V or gate/up projections.

Projection order and GEMMs stay sequential. The temporary quantized input is
created at the first projection and released at the last, never retained across
groups, policy calls or graph replays. BF16 modules have no quantization to share.
"""
import torch


class _InputGroup:
    def __init__(self, modules):
        self.modules = modules
        self.prepared = None

    def clear(self):
        for module in self.modules:
            if hasattr(module, '_a8_shared_activation'):
                del module._a8_shared_activation
        self.prepared = None

    def run(self, slot, x):
        if slot == 0:
            if self.prepared is not None:
                raise RuntimeError('Previous input group did not finish')
            first = self.modules[0]
            a = x.reshape(-1, first.in_features).contiguous()
            mp = max(first._a8_minimum_rows, ((a.shape[0] + 15) // 16) * 16)
            aq, scale = first._a8_prepare_activation(a, mp)
            self.prepared = (x.data_ptr(), tuple(x.shape), aq, scale)
            for module in self.modules:
                module._a8_shared_activation = self.prepared
        elif self.prepared is None:
            raise RuntimeError('Input group projection order changed')
        try:
            out = self.modules[slot]._a8_language_stock(x)
        except Exception:
            self.clear()
            raise
        if slot == len(self.modules) - 1:
            self.clear()
        return out


def install(policy):
    groups = []
    language = policy.model.backbone.model.model.language_model
    for block in language.layers:
        for container, names in ((block.self_attn, ('q_proj', 'k_proj', 'v_proj')),
                                 (block.mlp, ('gate_proj', 'up_proj'))):
            modules = [getattr(container, name) for name in names]
            if not all(hasattr(module, '_a8_prepare_activation') for module in modules):
                continue
            assert len({module.in_features for module in modules}) == 1
            group = _InputGroup(modules)
            for slot, module in enumerate(modules):
                module._a8_language_stock = module.forward
                def forward(x, _group=group, _slot=slot):
                    return _group.run(_slot, x)
                module.forward = forward
            groups.append(group)
    assert len(groups) == 2 * len(language.layers), 'Language A8 scope mismatch'
    # Only CPU ownership for wrappers; their prepared tensors must be None
    # after each complete forward. No permanent activation cache is allocated.
    policy.model._a8_language_input_groups = groups
    return {'groups': len(groups), 'projection_order_unchanged': True,
            'persistent_activation_cache': False}
