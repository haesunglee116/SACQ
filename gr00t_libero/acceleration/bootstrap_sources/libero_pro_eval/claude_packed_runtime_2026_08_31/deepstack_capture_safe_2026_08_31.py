#!/usr/bin/env python3
"""Make Qwen3VL's deepstack merge CUDA-graph capturable, without changing its math.

`Qwen3VLTextModel._deepstack_process` adds visual features back into the hidden
states at layers 5, 11 and 17 using boolean-mask indexing:

    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    hidden_states[visual_pos_masks, :] = local_this

Boolean-mask indexing lowers to `nonzero()`, whose output shape depends on the
tensor's contents, so it synchronises with the host. That is illegal during graph
capture and is why the VLM text stack refuses to capture even as the first
capture in a clean process.

The replacement below is the same operation written with an integer index that is
resolved once, outside capture, and then reused:

    idx  = mask.view(-1).nonzero().squeeze(-1)       # computed eagerly, cached
    flat = hidden_states.view(-1, H)
    flat.index_copy_(0, idx, flat.index_select(0, idx) + visual_embeds)

`hidden_states[mask, :]` enumerates positions in row-major order over (B, T),
which is exactly the order of `mask.view(-1).nonzero()`, so the two forms select
and write the same rows in the same order. Values are untouched.

The cached index is keyed on the mask's shape and its number of set positions,
and is refreshed whenever either changes or whenever we are not capturing, so a
different prompt layout can never silently reuse a stale index.
"""
from __future__ import annotations

import torch


def install(text_model: torch.nn.Module, verbose: bool = True) -> dict:
    """Patch this instance's `_deepstack_process`. Returns a status/restore dict."""
    cls = type(text_model)
    original = cls._deepstack_process
    state = {"cache": {}, "rebuilds": 0, "calls": 0, "capture_calls": 0}

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        state["calls"] += 1
        capturing = torch.cuda.is_current_stream_capturing()
        state["capture_calls"] += int(capturing)
        mask = visual_pos_masks.to(hidden_states.device)
        key = (tuple(mask.shape), hidden_states.shape[-1])
        entry = state["cache"].get(key)
        if entry is None and capturing:
            # Never fall back to the boolean path mid-capture; that is the bug.
            raise RuntimeError(
                "deepstack index not primed before capture; call prime() first")
        if not capturing:
            # nonzero() is legal here, so refresh the cached index every eager call.
            idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            if entry is None or entry.shape != idx.shape:
                state["cache"][key] = idx.clone()
                state["rebuilds"] += 1
                entry = state["cache"][key]
            else:
                entry.copy_(idx)
        idx = entry
        h = hidden_states.reshape(-1, hidden_states.shape[-1])
        embeds = visual_embeds.to(h.device, h.dtype)
        h.index_copy_(0, idx, h.index_select(0, idx) + embeds)
        return hidden_states

    cls._deepstack_process = _deepstack_process
    state["restore"] = lambda: setattr(cls, "_deepstack_process", original)
    if verbose:
        print(f"[ds] patched {cls.__name__}._deepstack_process "
              "(boolean mask -> cached integer index)", flush=True)
    return state
