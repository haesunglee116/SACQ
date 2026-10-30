#!/usr/bin/env python3
"""A shape-guarded CUDA graph wrapper for one nn.Module's forward.

Used to graph the VLM text stack, which holds all 111 VLM codebook modules and is
capture-clean, while leaving the capture-hostile prologue (the vision tower and
`get_rope_index`, both of which call `.item()`/`.tolist()`) eager.

The wrapper records the tensor arguments of the first post-warmup call, allocates
static buffers for them, captures the forward against those buffers, and on later
calls copies the incoming tensors in and replays. Any change of argument
signature -- shape, dtype, or which arguments are tensors -- falls back to the
original eager forward, so a shape it has never captured can never silently get
stale results.
"""
from __future__ import annotations

import torch


def _walk(value, path, out):
    """Collect (path, tensor) for tensors at any depth in lists and tuples.

    Qwen3VL passes `deepstack_visual_embeds` as a *list* of tensors produced fresh
    by the vision tower on every call. Treating only top-level tensor arguments
    would leave the graph reading the addresses those tensors happened to occupy
    at capture time, which is a silent correctness bug rather than a crash.
    """
    if torch.is_tensor(value):
        out.append((tuple(path), value))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _walk(item, path + [i], out)


def _tensor_slots(args, kwargs):
    """(container, key, nested path) for every tensor argument, deterministic order."""
    slots = []
    for i, v in enumerate(args):
        found = []
        _walk(v, [], found)
        slots += [("a", i, path) for path, _ in found]
    for k in sorted(kwargs):
        found = []
        _walk(kwargs[k], [], found)
        slots += [("k", k, path) for path, _ in found]
    return slots


def _get(container, path):
    for step in path:
        container = container[step]
    return container


def _signature(args, kwargs):
    sig = []
    for kind, key, path in _tensor_slots(args, kwargs):
        base = args[key] if kind == "a" else kwargs[key]
        t = _get(base, path)
        sig.append((kind, key, path, tuple(t.shape), t.dtype))
    # non-tensor arguments are part of the signature too, so a changed flag or a
    # cache object of a different type falls back instead of replaying blindly
    for i, v in enumerate(args):
        if not torch.is_tensor(v) and not isinstance(v, (list, tuple)):
            sig.append(("a", i, type(v).__name__))
    for k in sorted(kwargs):
        v = kwargs[k]
        if not torch.is_tensor(v) and not isinstance(v, (list, tuple)):
            sig.append(("k", k, type(v).__name__))
    return tuple(sig)


def graph_wrap_forward(module: torch.nn.Module, warmup: int = 3, verbose: bool = True):
    """Replace module.forward with a capture-on-first-stable-call version.

    Returns a status dict that the caller can inspect after a few calls:
    `captured`, `fallbacks`, `error`, `replays`.
    """
    original = module.forward
    st = {"graph": None, "sig": None, "statics": None, "out": None,
          "calls": 0, "replays": 0, "fallbacks": 0, "captured": False, "error": None}

    def fwd(*args, **kwargs):
        st["calls"] += 1
        sig = _signature(args, kwargs)

        if st["captured"]:
            if sig != st["sig"]:
                st["fallbacks"] += 1
                return original(*args, **kwargs)
            for (kind, key, path), buf in zip(_tensor_slots(args, kwargs),
                                              st["statics"]):
                base = args[key] if kind == "a" else kwargs[key]
                buf.copy_(_get(base, path))
            st["graph"].replay()
            st["replays"] += 1
            return st["out"]

        if st["calls"] <= warmup or st["error"] is not None:
            return original(*args, **kwargs)

        # Capture against static copies of exactly these arguments.
        slots = _tensor_slots(args, kwargs)
        statics = []
        for kind, key, path in slots:
            base = args[key] if kind == "a" else kwargs[key]
            statics.append(_get(base, path).detach().clone())

        def _rebuild(value, path, out, buf):
            """Copy of `value` with the tensor at `path` replaced by `buf`."""
            if not path:
                return buf
            idx = path[0]
            items = list(value)
            items[idx] = _rebuild(items[idx], path[1:], out, buf)
            return type(value)(items) if isinstance(value, tuple) else items

        c_args, c_kwargs = list(args), dict(kwargs)
        for (kind, key, path), buf in zip(slots, statics):
            target = c_args if kind == "a" else c_kwargs
            target[key] = _rebuild(target[key], list(path), None, buf)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    original(*c_args, **c_kwargs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                out = original(*c_args, **c_kwargs)
            torch.cuda.synchronize()
            st.update({"graph": g, "sig": sig, "statics": statics,
                       "out": out, "captured": True})
            if verbose:
                print(f"[gw] captured {type(module).__name__}.forward "
                      f"({len(statics)} tensor args)", flush=True)
            # Replay once so the caller receives graph output, not the capture's.
            for (kind, key, path), buf in zip(slots, statics):
                base = args[key] if kind == "a" else kwargs[key]
                buf.copy_(_get(base, path))
            g.replay()
            st["replays"] += 1
            return out
        except Exception as exc:
            st["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            if verbose:
                print(f"[gw] capture FAILED for {type(module).__name__}: {st['error']}",
                      flush=True)
            return original(*args, **kwargs)

    module.forward = fwd
    st["restore"] = lambda: setattr(module, "forward", original)
    return st
