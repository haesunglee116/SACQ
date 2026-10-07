"""Explicit lifetime management for the source QuantLinear reconstruction cache."""
from contextlib import contextmanager
from .calibration_layers import QuantLinear


def clear_reconstruction_cache(model):
    """Drop reconstructed weights without changing parameters or gradients.

    Call between independent forwards/backwards, after an optimizer step, and
    when switching gradient mode or device. Traverses registered child modules.
    This does not clear the optional research trainer's ``_leaf`` override.
    """
    for layer in model.modules():
        if isinstance(layer, QuantLinear):
            layer._wc = None


@contextmanager
def reconstruction_cache_scope(model):
    """Reuse reconstructions within one request; clear them on entry and exit.

    Parameters, indices, device and gradient mode must remain unchanged inside
    the scope. Put optimizer steps outside it. Multiple policy/denoising calls
    belonging to the same differentiable request may share the scope. Backward
    can run before exit or on the returned loss afterward; clearing the module's
    reference does not detach the autograd graph held by that loss.
    """
    clear_reconstruction_cache(model)
    try:
        yield model
    finally:
        clear_reconstruction_cache(model)
