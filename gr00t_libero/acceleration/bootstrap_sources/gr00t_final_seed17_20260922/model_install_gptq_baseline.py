"""Pure L/A GPTQ with the exact approved scalar background; no requantization."""
from io_utils import sha

def install(model, plan, suite):
    import torch
    from measure_dit_single_layer_downgrade_sensitivity import patched_env, replace_module
    from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
    assert plan['condition'] == 'gptq2_baseline'
    m = plan['baseline_model']
    for key in ['vlm_spec', 'dit_spec']:
        assert sha(m[key]['path']) == m[key]['sha256']
    # Preserve all L/A as BF16 during background construction. Install their
    # completed GPTQ tensors below, without a second quantization operation.
    env = dict(VLM_BITS='16', DIT_BITS='16', VLM_QUANT_MODE='uniform',
        DIT_QUANT_MODE='uniform', VLM_ROTATION_MODE='none',
        VLM_FORCE_BITS_SPECS_FILE=m['vlm_spec']['path'],
        DIT_FORCE_BITS_SPECS_FILE=m['dit_spec']['path'],
        DIT_INCLUDE_ADALN='1', DIT_SKIP_VL_SELF_ATTENTION='0',
        VLM_EMBED_BITS='4', VLM_VISION_BITS='4', VLM_DROP_LM_HEAD='1',
        ACTION_HEAD_SINGLE_EMBODIMENT_ID='2',
        GR00T_CODEBOOK_LINEAR_BACKEND='dense_reconstruct',
        GR00T_CODEBOOK_ACCELERATE_VLM='0', GR00T_STRICT_QUANT_LAYOUT='1')
    with patched_env(env):
        quantize_gr00t_vlm27_dit4(model, group_size=64, device='cuda:0')
    assert len(m['weights']) == 442
    assert sum(r['bits'] == 2 for r in m['weights']) == 338
    for r in m['weights']:
        assert sha(r['path']) == r['sha256']
        p = torch.load(r['path'], map_location='cpu', weights_only=False)
        assert p['layer'] == r['layer'] and p['bits'] == r['bits']
        w = p['weight']; old = model.get_submodule(r['layer'])
        bias = getattr(old, 'bias', None)
        layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias is not None,
                                device='cuda:0', dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None: layer.bias.copy_(bias)
        layer.requires_grad_(False)
        replace_module(model, r['layer'], layer)
    model.eval().requires_grad_(False)
    return dict(method='pure_LA_GPTQ2', count=338, vision_gptq4=104,
                vlsa_hqq4=24, embedding='affine4', packed_memory_claim=False,
                diagnostic_only=True, held_out=False)
