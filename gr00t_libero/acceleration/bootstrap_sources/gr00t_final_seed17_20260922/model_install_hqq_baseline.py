"""Uniform HQQ L/A338, approved fixed GPTQ4/HQQ4/affine4 background."""
from io_utils import sha

def install(model, plan, suite):
    import torch
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
    from measure_dit_single_layer_downgrade_sensitivity import patched_env, replace_module
    from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
    m=plan['baseline_model']; bits=m['bits']
    assert bits in (2,3) and len(m['targets'])==338
    for key in ['vlm_spec','dit_spec']:
        assert sha(m[key]['path'])==m[key]['sha256']
    env=dict(VLM_BITS='16',DIT_BITS='4',VLM_QUANT_MODE='uniform',DIT_QUANT_MODE='uniform',
        VLM_ROTATION_MODE='none',VLM_FORCE_BITS_SPECS_FILE=m['vlm_spec']['path'],
        DIT_FORCE_BITS_SPECS_FILE=m['dit_spec']['path'],DIT_INCLUDE_ADALN='1',
        DIT_SKIP_VL_SELF_ATTENTION='0',VLM_EMBED_BITS='4',VLM_VISION_BITS='4',
        VLM_DROP_LM_HEAD='1',ACTION_HEAD_SINGLE_EMBODIMENT_ID='2',
        GR00T_CODEBOOK_LINEAR_BACKEND='dense_reconstruct',GR00T_CODEBOOK_ACCELERATE_VLM='0',
        GR00T_STRICT_QUANT_LAYOUT='1')
    with patched_env(env): quantize_gr00t_vlm27_dit4(model,group_size=64,device='cuda:0')
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    config=BaseQuantizeConfig(nbits=bits,group_size=64,axis=1,quant_zero=False,
        quant_scale=False,offload_meta=False,view_as_float=False)
    assert config['weight_quant_params']['optimize'] and not config['weight_quant_params']['round_zero']
    audits=[]
    for row in m['targets']:
        name=row['layer']; old=model.get_submodule(name)
        assert type(old) is torch.nn.Linear and old.weight.dtype==torch.bfloat16
        shape=list(old.weight.shape)
        new=HQQLinear(old,quant_config=config,compute_dtype=torch.bfloat16,device='cuda:0',del_orig=True)
        assert new.meta['nbits']==bits and new.meta['group_size']==64 and new.meta['axis']==1
        assert list(new.dequantize().shape)==shape and torch.isfinite(new.dequantize()).all()
        replace_module(model,name,new);audits.append(dict(layer=name,bits=bits,shape=shape))
    assert len(m['vision'])==104
    for row in m['vision']:
        assert sha(row['path'])==row['sha256']
        data=torch.load(row['path'],map_location='cpu',weights_only=False)
        assert data['layer']==row['layer'] and data['bits']==4
        old=model.get_submodule(row['layer']);bias=getattr(old,'bias',None);w=data['weight']
        new=torch.nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device='cuda:0',dtype=torch.bfloat16)
        with torch.no_grad():
            new.weight.copy_(w)
            if bias is not None:new.bias.copy_(bias)
        replace_module(model,row['layer'],new)
    model.eval().requires_grad_(False)
    return dict(method='HQQ'+str(bits),count=338,layers=audits,vision_gptq4=104,
        vlsa_hqq4=24,embedding='affine4',diagnostic_only=True,held_out=False)
