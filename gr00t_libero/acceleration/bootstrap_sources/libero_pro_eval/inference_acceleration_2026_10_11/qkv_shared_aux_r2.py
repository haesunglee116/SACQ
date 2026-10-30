"""Reuse one auxiliary stream for sequential attention K/V projection pairs.

Each pair still forks V onto the auxiliary stream, runs K on the caller's
stream, then joins before either result is consumed. The original group creates
one auxiliary stream per pair; native cuBLAS retains a workspace for every
handle/stream combination. Reuse does not alter projection arithmetic or inputs.
"""
import torch


def _logical_output(out):
    # FP8 GEMM pads M to16; retaining cropped views keeps padding alive.
    # Copy only when storage exceeds the logical output. Arithmetic is unchanged.
    if out.untyped_storage().nbytes() > out.numel()*out.element_size():
        return out.clone()
    return out


class _ProjectionGroup:
    def __init__(self, members, stream):
        assert len(members) == 2
        self.members = members
        self.stream = stream
        self.results = [None, None]

    def run(self, x):
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self.results[1] = _logical_output(self.members[1]._aq26_stock(x))
        self.results[0] = _logical_output(self.members[0]._aq26_stock(x))
        current.wait_stream(self.stream)


def _install_group(members, stream):
    group = _ProjectionGroup(members, stream)
    for slot, module in enumerate(members):
        module._aq26_stock = module.forward

        def forward(x, _group=group, _slot=slot):
            if _slot == 0:
                _group.run(x)
            return _group.results[_slot]

        module.forward = forward
    return group


def install(policy, verbose=True):
    device = next(policy.model.parameters()).device
    stream = torch.cuda.Stream(device=device)
    groups, counts = [], {}
    for block in policy.model.action_head.model.transformer_blocks:
        for name in ("attn1", "attn2"):
            attn = getattr(block, name, None)
            if attn is None or not all(hasattr(attn, n) for n in ("to_k", "to_v")):
                continue
            if attn.to_k.in_features != attn.to_v.in_features:
                continue
            groups.append(_install_group([attn.to_k, attn.to_v], stream))
            counts[f"{name}_kv"] = counts.get(f"{name}_kv", 0) + 1
    assert groups, "No K/V pairs installed"
    assert len({g.stream.cuda_stream for g in groups}) == 1
    if verbose:
        print(f"[qkv] shared auxiliary stream, groups: {counts}", flush=True)
    return {"groups": len(groups), "auxiliary_streams": 1,
            "crop_retained_output_storage": True, "pair_order": "V auxiliary, K caller, join", **counts}
