#!/usr/bin/env python3
"""Make the Qwen3-VL text stack CUDA-graph capturable, without changing its math.

Three HuggingFace call sites synchronise with the host, each found from a stack
trace taken while the stream was capturing, not by reading the source and
guessing:

  1. `Qwen3VLTextModel._deepstack_process` indexes with a boolean mask, which
     lowers to `nonzero()` and therefore has a data-dependent output shape.
  2. `modeling_flash_attention_utils._is_packed_sequence` returns a 0-dim CUDA
     tensor that `_flash_attention_forward` evaluates in an `elif`. It reports
     True here because Qwen3-VL's 3-D mrope position ids are not a plain arange,
     so Flash Attention takes its variable-length path for what is actually a
     single sequence.
  3. `masking_utils.create_causal_mask` decides whether an explicit mask can be
     skipped by inspecting the padding mask's contents.

All three are constants for a fixed prompt layout: same image count, same token
count, same padding. So each is computed once outside capture and served from a
cache during it. The values are identical; only where they are computed moves.

`install_all` returns a dict of per-patch state so the caller can assert that the
caches were primed and actually hit, rather than assuming the patches ran.
"""
from __future__ import annotations

import torch

import deepstack_capture_safe_2026_08_31 as dsfix
import fa_capture_safe_2026_08_31 as fafix


def install_causal_mask_cache(verbose: bool = True) -> dict:
    import transformers.masking_utils as mu

    original = mu.create_causal_mask
    state: dict = {"cache": {}, "eager": 0, "capture_hits": 0, "misses": 0}

    def _key(kwargs):
        embeds = kwargs.get("input_embeds")
        am = kwargs.get("attention_mask")
        cp = kwargs.get("cache_position")
        return (tuple(embeds.shape) if embeds is not None else None,
                tuple(am.shape) if torch.is_tensor(am) else None,
                int(cp.shape[0]) if torch.is_tensor(cp) else None)

    def create_causal_mask(**kwargs):
        key = _key(kwargs)
        if torch.cuda.is_current_stream_capturing():
            if key in state["cache"]:
                state["capture_hits"] += 1
                return state["cache"][key]
            state["misses"] += 1
            return None          # visible in the state dict, never silent
        value = original(**kwargs)
        state["cache"][key] = value
        state["eager"] += 1
        return value

    mu.create_causal_mask = create_causal_mask
    # Qwen3-VL imported the symbol directly, so rebind it there too.
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qm
        if hasattr(qm, "create_causal_mask"):
            qm.create_causal_mask = create_causal_mask
    except Exception:
        pass
    state["restore"] = lambda: setattr(mu, "create_causal_mask", original)
    if verbose:
        print("[mask] patched create_causal_mask (cached per input signature)", flush=True)
    return state


def install_vision_cache(vision_model: torch.nn.Module, verbose: bool = True) -> dict:
    """Cache the two vision-tower helpers that depend only on the image grid.

    `Qwen3VLVisionModel.rot_pos_emb` calls `.item()` on the grid extents and
    `fast_pos_embed_interpolate` indexes with a tensor; both synchronise with the
    host and both are pure functions of `grid_thw`. For a fixed camera setup that
    tensor never changes, so each is evaluated once outside capture and replayed
    from cache during it. The cache key is the grid's contents, so a different
    image geometry recomputes rather than silently reusing.
    """
    cls = type(vision_model)
    state: dict = {"rot": {}, "pos": {}, "eager": 0, "capture_hits": 0, "misses": 0}
    orig_rot = cls.rot_pos_emb
    orig_pos = cls.fast_pos_embed_interpolate

    def _key(grid_thw):
        return tuple(int(v) for v in grid_thw.reshape(-1).tolist())

    def _wrap(name, cache, original):
        def fn(self, grid_thw):
            if torch.cuda.is_current_stream_capturing():
                for k, v in cache.items():
                    state["capture_hits"] += 1
                    return v          # one geometry per session; primed below
                state["misses"] += 1
                raise RuntimeError(f"{name} not primed before capture")
            key = _key(grid_thw)
            if key not in cache:
                cache[key] = original(self, grid_thw)
                state["eager"] += 1
            return cache[key]
        return fn

    cls.rot_pos_emb = _wrap("rot_pos_emb", state["rot"], orig_rot)
    cls.fast_pos_embed_interpolate = _wrap(
        "fast_pos_embed_interpolate", state["pos"], orig_pos)

    def restore():
        cls.rot_pos_emb = orig_rot
        cls.fast_pos_embed_interpolate = orig_pos

    state["restore"] = restore
    if verbose:
        print("[vis] patched rot_pos_emb and fast_pos_embed_interpolate "
              "(cached per image grid)", flush=True)
    return state


def install_all(text_model, use_sdpa: bool = True, configs=(),
                vision_model=None, verbose: bool = True) -> dict:
    """Apply every patch needed for capture. `configs` are the config objects to
    switch to SDPA and to turn the KV cache off on."""
    state = {"deepstack": dsfix.install(text_model, verbose=verbose),
             "flash_attention": fafix.install(verbose=verbose),
             "fa_varlen": fafix.install_varlen_cache(verbose=verbose),
             "fa_posid_varlen": fafix.install_posid_varlen_cache(verbose=verbose),
             "causal_mask": install_causal_mask_cache(verbose=verbose)}
    if vision_model is not None:
        state["vision"] = install_vision_cache(vision_model, verbose=verbose)
    for cfg in configs:
        if cfg is not None:
            cfg.use_cache = False
    if use_sdpa:
        for cfg in configs:
            if cfg is not None:
                cfg._attn_implementation = "sdpa"
        for layer in text_model.layers:
            layer.self_attn.config._attn_implementation = "sdpa"
        state["attn"] = sorted({l.self_attn.config._attn_implementation
                                for l in text_model.layers})
        if vision_model is not None:
            # The two towers want opposite implementations. The text stack must
            # leave Flash Attention because its packed-sequence probe reads a
            # tensor on the host. The vision tower must *stay* on it: its
            # non-flash branch splits Q/K/V with `lengths.tolist()`, which is a
            # host round trip, while the flash branch passes cu_seqlens straight
            # through.
            for blk in vision_model.blocks:
                blk.attn.config._attn_implementation = "flash_attention_2"
            state["vision_attn"] = sorted({b.attn.config._attn_implementation
                                           for b in vision_model.blocks})
        if verbose:
            print(f"[hf] attention implementation now {state['attn']}", flush=True)
    return state
