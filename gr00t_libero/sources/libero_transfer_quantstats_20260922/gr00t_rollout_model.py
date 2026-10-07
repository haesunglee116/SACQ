"""Install calibrated corrected SACQ overlay and identical frozen GPTQ background."""
from pathlib import Path
from oft_index_worker import sha
def validate_manifest(manifest):
    for key in ('expected_sacq_count','expected_gptq_count'):
        if type(manifest.get(key)) is not int or manifest[key]<0:
            raise ValueError(f'Explicit nonnegative {key} required')
    rows=manifest['gptq']
    if len(rows)!=manifest['expected_gptq_count']:
        raise ValueError('Frozen background count differs from calibration manifest')
    names=[row['layer'] for row in rows]
    if len(set(names))!=len(names):
        raise ValueError('Duplicate frozen background layer')
    for row in rows:
        if type(row.get('bits')) is not int or not 1<=row['bits']<=16:
            raise ValueError('Explicit background precision required')

def install(model,plan,suite):
    import torch
    from measure_dit_single_layer_downgrade_sensitivity import patched_env,replace_module
    from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
    m=plan['workshop_model'];assert suite==plan['selected_suites'][0]
    validate_manifest(m)
    for key in ('vlm','dit','vlm_spec','dit_spec'):
        assert sha(m[key]['path'])==m[key]['sha256']
    env=dict(VLM_BITS='2',DIT_BITS='4',VLM_QUANT_MODE='uniform',DIT_QUANT_MODE='uniform',
        VLM_ROTATION_MODE='none',VLM_FORCE_BITS_SPECS_FILE=m['vlm_spec']['path'],
        DIT_FORCE_BITS_SPECS_FILE=m['dit_spec']['path'],VLM_CODEBOOK_OVERLAY_PATH=m['vlm']['path'],
        VLM_CODEBOOK_OVERLAY_SHA256=m['vlm']['sha256'],DIT_CODEBOOK_OVERLAY_PATH=m['dit']['path'],
        DIT_INCLUDE_ADALN='1',DIT_SKIP_VL_SELF_ATTENTION='0',VLM_EMBED_BITS='4',VLM_VISION_BITS='4',
        VLM_DROP_LM_HEAD='1',ACTION_HEAD_SINGLE_EMBODIMENT_ID='2',
        GR00T_CODEBOOK_LINEAR_BACKEND='dense_reconstruct',GR00T_CODEBOOK_ACCELERATE_VLM='0',GR00T_STRICT_QUANT_LAYOUT='1')
    with patched_env(env):quantize_gr00t_vlm27_dit4(model,group_size=64,device='cuda:0')
    for r in m['gptq']:
        assert sha(r['path'])==r['sha256']
        p=torch.load(r['path'],map_location='cpu',weights_only=False);name=r['layer'];w=p['weight']
        assert p['layer']==name and p['bits']==r['bits']
        old=model.get_submodule(name);bias=getattr(old,'bias',None)
        layer=torch.nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device='cuda:0',dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:layer.bias.copy_(bias)
        layer.requires_grad_(False);replace_module(model,name,layer)
    model.eval().requires_grad_(False)
    return dict(method='quantstats_gptq_cal',count=m['expected_sacq_count'],gptq_frozen=m['expected_gptq_count'],overlays=m,diagnostic_only=True,held_out=False)
