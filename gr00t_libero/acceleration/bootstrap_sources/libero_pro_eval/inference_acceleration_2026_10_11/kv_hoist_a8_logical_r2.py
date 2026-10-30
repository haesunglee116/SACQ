#!/usr/bin/env python3
"""Project the vision-language features into keys and values once, not four times.

`AlternateVLDiT.forward` is called once per denoising step with a fresh
`hidden_states` and `timestep` but with the *same* `encoder_hidden_states`: the
vision-language features are the conditioning and are computed before the loop
starts. Its even-numbered blocks are the cross-attention ones, and their `to_k`
and `to_v` read only those features. Four denoising steps therefore compute the
same 32 projections four times and get the same answer four times.

Hoisting them is worth more than it sounds. Those are the M=149 keys, 3.64 ms of
the loop, and three quarters of that is recomputation. It is also the same
medicine as everything else that has worked here: the bottleneck is an idle
device, and this removes 96 launches outright rather than making any of them
cheaper.

The cache is indexed by a step counter the caller resets at the top of each
action, not by tensor identity -- under a captured graph the input buffer is
static, so identity would say "unchanged" forever and reuse across actions.
Unrolled into a graph this becomes: step 0 computes, steps 1..3 read.
"""
from __future__ import annotations

import torch
from qkv_shared_aux_r2 import _logical_output


class KVHoist:
    def __init__(self):
        self.step = 0
        self.entries = []

    def begin_action(self):
        """Call once before a denoising loop; the next projection recomputes."""
        self.step = 0

    def advance(self):
        self.step += 1


def install(policy, verbose: bool = True) -> dict:
    state = KVHoist()
    blocks = policy.model.action_head.model.transformer_blocks
    patched = 0
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn1", None)
        if attn is None:
            continue
        for name in ("to_k", "to_v"):
            module = getattr(attn, name, None)
            if module is None:
                continue
            # Only the cross-attention projections read the encoder states; the
            # self-attention blocks project the action tokens, which do change.
            if module.in_features == blocks[0].attn1.to_q.in_features:
                continue
            module._aq26_kv_stock = module.forward
            module._a8_kv_hoist_state = state
            cache = {}

            def forward(x, _m=module, _c=cache, _s=state):
                if _s.step == 0 or "v" not in _c:
                    _c["v"] = _logical_output(_m._aq26_kv_stock(x))
                return _c["v"]

            module.forward = forward
            patched += 1
    if verbose:
        print(f"[kvh] hoisted {patched} cross-attention key/value projections "
              f"out of the denoising loop", flush=True)
    state.entries = [patched]
    return state
