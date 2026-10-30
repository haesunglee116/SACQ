#!/usr/bin/env python3
"""Let Flash Attention 2 run inside a CUDA graph, without changing its result.

`transformers.modeling_flash_attention_utils._is_packed_sequence` ends with

    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()

which is a 0-dim CUDA tensor. `_flash_attention_forward` then uses it in

    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:

and evaluating a CUDA tensor in a boolean context copies it to the host. That is
illegal during capture, and it is the reason a single Qwen3VL decoder layer
refuses to capture -- confirmed by a stack trace, not inferred.

The predicate asks whether `position_ids` is a flattened pack of several
sequences. For a fixed prompt layout it is a constant, so this wrapper computes
it eagerly, caches the resulting Python bool per (shape, batch), and serves the
cached value while capturing. Same answer, no host round trip. The cache key
includes the tensor's own values via a cheap checksum, so a genuinely different
layout re-evaluates instead of silently reusing the previous verdict.
"""
from __future__ import annotations

import torch


def install(verbose: bool = True) -> dict:
    import transformers.modeling_flash_attention_utils as fa

    original = fa._is_packed_sequence
    state: dict = {"cache": {}, "eager_calls": 0, "capture_hits": 0, "misses": 0}

    def _key(position_ids, batch_size):
        if position_ids is None:
            return None
        # Shape plus a checksum of the ids: cheap, and it changes if the packing
        # layout changes. The checksum itself is only read outside capture.
        return (tuple(position_ids.shape), int(batch_size), position_ids.dtype)

    def _is_packed_sequence(position_ids, batch_size):
        if position_ids is None:
            return False
        key = _key(position_ids, batch_size)
        if torch.cuda.is_current_stream_capturing():
            hit = state["cache"].get(key)
            if hit is None:
                state["misses"] += 1
                # Nothing cached: fall back to the packed path being off, which is
                # what a single unpacked prompt uses. Recorded so it is visible.
                return False
            state["capture_hits"] += 1
            return hit
        value = original(position_ids, batch_size)
        value = bool(value.item()) if torch.is_tensor(value) else bool(value)
        state["cache"][key] = value
        state["eager_calls"] += 1
        return value

    fa._is_packed_sequence = _is_packed_sequence
    state["restore"] = lambda: setattr(fa, "_is_packed_sequence", original)
    if verbose:
        print("[fa] patched _is_packed_sequence (tensor bool -> cached python bool)",
              flush=True)
    return state


def install_varlen_cache(verbose: bool = True) -> dict:
    """Let the *vision* tower's Flash Attention path capture, unchanged.

    `Qwen3VLVisionAttention.forward` computes

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

    and hands that 0-dim CUDA tensor to `flash_varlen_fn` as `max_seqlen_q`.
    Flash Attention wants a Python int, so it calls `Tensor.__index__`, which
    copies to the host -- the stack trace that stops the vision blocks from
    capturing, at `modeling_flash_attention_utils.py:647`.

    The value is a property of the image grid, so for a fixed input resolution it
    is a constant. This intercepts the attention interface, converts the tensor to
    an int eagerly, and serves the cached int while capturing. The kernel receives
    exactly the number it received before, so the result is unchanged.

    Keyed on the `cu_seqlens` shape, which is all that can be read during capture.
    Every eager call re-derives the value and reports a conflict if a key ever
    resolves differently, so a genuinely varying grid is visible rather than
    silently served a stale length.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "flash_attention_2"
    original = ALL_ATTENTION_FUNCTIONS[name]
    state: dict = {"cache": {}, "eager_calls": 0, "capture_hits": 0,
                   "misses": 0, "conflicts": 0}

    def _as_int(value, cu_seqlens):
        if not torch.is_tensor(value) or value.numel() != 1:
            return value
        key = (tuple(cu_seqlens.shape), str(cu_seqlens.dtype))
        if torch.cuda.is_current_stream_capturing():
            hit = state["cache"].get(key)
            if hit is None:
                # Leave the tensor alone: the sync will fire and name itself
                # rather than this cache inventing a length.
                state["misses"] += 1
                return value
            state["capture_hits"] += 1
            return hit
        resolved = int(value.item())
        if state["cache"].get(key, resolved) != resolved:
            state["conflicts"] += 1
        state["cache"][key] = resolved
        state["eager_calls"] += 1
        return resolved

    def wrapper(*args, **kwargs):
        cu = kwargs.get("cu_seq_lens_q")
        if cu is not None:
            for field in ("max_length_q", "max_length_k"):
                if field in kwargs:
                    kwargs[field] = _as_int(kwargs[field], cu)
        return original(*args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[name] = wrapper

    def restore():
        ALL_ATTENTION_FUNCTIONS[name] = original

    state["restore"] = restore
    if verbose:
        print("[fa] patched flash_attention_2 varlen kwargs "
              "(max_seqlen tensor -> cached python int)", flush=True)
    return state


def install_posid_varlen_cache(verbose: bool = True) -> dict:
    """The text stack's own varlen blocker, one level below the vision tower's.

    Qwen3-VL feeds Flash Attention 3-D mrope position ids, which its packed-
    sequence probe reads as a flattened multi-sequence batch, so the text stack
    takes the varlen path even for a single prompt. That path calls
    `prepare_fa_kwargs_from_position_ids`, which ends with

        max_length_q = cu_seq_lens_q.diff().max()
        max_length_q = max_length_q.item()

    -- a host copy, and the reason `Qwen3VLTextModel` still refuses to capture
    after the vision tower's `max_seqlen` is cached.

    Both outputs are functions of the prompt layout, constant for a fixed one.
    This caches the whole return value per position-ids shape: during capture the
    graph reads the cached `cu_seq_lens` tensors at their captured addresses and
    receives the cached ints, so Flash Attention is handed exactly what it was
    handed eagerly. Eager calls re-derive and count a conflict if a key ever
    resolves differently.
    """
    import transformers.modeling_flash_attention_utils as fa

    original = fa.prepare_fa_kwargs_from_position_ids
    state: dict = {"cache": {}, "eager_calls": 0, "capture_hits": 0,
                   "misses": 0, "conflicts": 0}

    def patched(position_ids):
        key = (tuple(position_ids.shape), str(position_ids.dtype))
        if torch.cuda.is_current_stream_capturing():
            hit = state["cache"].get(key)
            if hit is None:
                state["misses"] += 1
                return original(position_ids)
            state["capture_hits"] += 1
            return hit
        value = original(position_ids)
        previous = state["cache"].get(key)
        if previous is not None:
            same = (previous[1] == value[1]
                    and all(bool(torch.equal(a, b))
                            for a, b in zip(previous[0], value[0])))
            if not same:
                state["conflicts"] += 1
        state["cache"][key] = value
        state["eager_calls"] += 1
        return value

    fa.prepare_fa_kwargs_from_position_ids = patched

    def restore():
        fa.prepare_fa_kwargs_from_position_ids = original

    state["restore"] = restore
    if verbose:
        print("[fa] patched prepare_fa_kwargs_from_position_ids "
              "(cu_seqlens + max_length cached per prompt layout)", flush=True)
    return state
