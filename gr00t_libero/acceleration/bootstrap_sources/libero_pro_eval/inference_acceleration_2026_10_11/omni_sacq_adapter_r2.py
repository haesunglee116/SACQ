"""Load the unchanged final SACQ overlay into the official Omni model port.

No fitting, allocation, index change, activation quantization or dense cache.
The environment is the pinned final gr00t_rollout_model.py installation recipe.
"""
from contextlib import contextmanager
import hashlib
import os
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def patched_env(values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def category_compatibility():
    import gr00t.eval.cosmos_quant_shim as shim
    from vllm_omni.diffusion.models.gr00t.modeling.modules.embodiment_conditioned_mlp import CategorySpecificLinear
    original = shim.materialize_single_embodiment_action_head
    def materialize(model, embodiment_id, device='cuda:0'):
        names=[name for name,layer in model.action_head.named_modules() if type(layer) is CategorySpecificLinear]
        assert len(names)==7, names
        for name in names:
            layer=model.action_head.get_submodule(name)
            assert 0 <= embodiment_id < layer.W.shape[0]
            replacement=shim.SingleEmbodimentCategoryLinear(layer,embodiment_id)
            assert torch.equal(replacement.W,layer.W[embodiment_id])
            assert torch.equal(replacement.b,layer.b[embodiment_id])
            parent,leaf=name.rsplit('.',1)
            setattr(model.action_head.get_submodule(parent),leaf,replacement)
        print('OMNI_SINGLE_EMBODIMENT_EXACT',embodiment_id,len(names),flush=True)
        return len(names)
    shim.materialize_single_embodiment_action_head = materialize
    return original


def install(policy, plan, suite):
    from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4, PrecomputedPackedVectorCodebookLinear
    from pack_gptq_reference import pack_installed_gptq
    from pack_sacq_reference import pack_installed_sacq
    m = plan['workshop_model']
    assert plan['condition'] == 'sacq1.5' and plan['selected_suites'] == [suite]
    for r in [m[k] for k in ('vlm', 'dit', 'vlm_spec', 'dit_spec')] + m['gptq']:
        assert sha(r['path']) == r['sha256'], r['path']
    env = dict(VLM_BITS='2', DIT_BITS='4', VLM_QUANT_MODE='uniform', DIT_QUANT_MODE='uniform',
        VLM_ROTATION_MODE='none', VLM_FORCE_BITS_SPECS_FILE=m['vlm_spec']['path'],
        DIT_FORCE_BITS_SPECS_FILE=m['dit_spec']['path'], VLM_CODEBOOK_OVERLAY_PATH=m['vlm']['path'],
        VLM_CODEBOOK_OVERLAY_SHA256=m['vlm']['sha256'], DIT_CODEBOOK_OVERLAY_PATH=m['dit']['path'],
        DIT_INCLUDE_ADALN='1', DIT_SKIP_VL_SELF_ATTENTION='0', VLM_EMBED_BITS='4', VLM_VISION_BITS='4',
        VLM_DROP_LM_HEAD='1', ACTION_HEAD_SINGLE_EMBODIMENT_ID='2',
        GR00T_CODEBOOK_LINEAR_BACKEND='dense_reconstruct', GR00T_CODEBOOK_ACCELERATE_VLM='0', GR00T_STRICT_QUANT_LAYOUT='1')
    original_type = category_compatibility()
    try:
        with patched_env(env):
            quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device='cuda:0')
    finally:
        import gr00t.eval.cosmos_quant_shim as shim
        shim.materialize_single_embodiment_action_head = original_type
    assert len(m['gptq']) == 105
    for r in m['gptq']:
        p = torch.load(r['path'], map_location='cpu', weights_only=False)
        name = r['layer']; w = p['weight']
        assert p['layer'] == name and p['bits'] == (3 if name == 'action_head.model.proj_out_2' else 4)
        old = policy.model.get_submodule(name); bias = getattr(old, 'bias', None)
        layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device='cuda:0', dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:
                layer.bias.copy_(bias)
        parent, leaf = name.rsplit('.', 1)
        setattr(policy.model.get_submodule(parent), leaf, layer.requires_grad_(False))
        del old, layer, p, w
    scalar = pack_installed_gptq(policy.model, m['gptq'])
    sacq = pack_installed_sacq(policy.model)
    names = [n for n, layer in policy.model.named_modules() if type(layer) is PrecomputedPackedVectorCodebookLinear]
    assert len(names) == 337, len(names)
    assert len(scalar['layers']) == 105
    assert all(policy.model.get_submodule(n).packed_indices is not None for n in names)
    policy.model.eval().requires_grad_(False)
    return dict(sacq=sacq, scalar=scalar, installed_codebook_layers=len(names),
                checkpoint_unchanged=True, activation_dtype='bfloat16', dense_weight_cache=False,
                background='unchanged approved final embedding/vision/scalar payload recipe')


def install_kernels(policy, contract):
    """Reuse existing A16 exact decoders; do not install graphs or model fusion."""
    import json
    import decoder_tuning as decoder
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear
    from exact_dictionary_decode import install as dictionary
    import hqq_fused_dequant_2026_08_31 as hqq
    for group in json.load(open(contract['decoder_tuning']['path']))['groups']:
        decoder.SELECTED[tuple(group['geometry'])] = tuple(group['selected'])
    checked = []
    for name, layer in policy.model.named_modules():
        if type(layer) is PrecomputedPackedVectorCodebookLinear:
            reference = layer.dequantize_weight()
            actual = decoder.dequantize(layer)
            assert torch.equal(reference.view(torch.int16), actual.view(torch.int16)), name
            layer.forward = lambda x, _layer=layer: decoder.linear(x, _layer)
            checked.append(name)
            del reference, actual
    assert len(checked) == 337
    state = hqq.install(policy.model)
    assert not state['mismatch'], state
    return dict(weight_exact_layers=checked, dictionary_layers=dictionary(policy.model),
                hqq=state, cuda_graph=False, optimization_scope='existing exact A16 decoders only')
