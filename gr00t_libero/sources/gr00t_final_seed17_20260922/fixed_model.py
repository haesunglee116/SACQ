"""Install original selected SACQ bank plus approved same-bit GPTQ background."""
import csv,json
from pathlib import Path
def install(policy,ft,allocation,artifacts,gptq):
    import torch
    rows=list(csv.DictReader(open(allocation)));assert len(rows)==338
    named=dict(policy.model.named_modules());bias={r['layer']:named[r['layer']].bias for r in rows}
    vf={r['layer']:16 for r in rows if r['scope']=='vlm'}
    df={r['layer']:(3 if int(r['k'])==80000 else 4 if int(r['k'])==90000 else 16) for r in rows if r['scope']=='dit'}
    assert len(vf)==112 and len(df)==226 and len(ft.dit_vlsa)==24
    df.update({n:4 for n in ft.dit_vlsa})
    env=dict(ft.VLM_ENV);env.update(VLM_BITS='2',DIT_BITS='4',VLM_QUANT_MODE='uniform',DIT_QUANT_MODE='uniform',
        VLM_ROTATION_MODE='none',VLM_FORCE_BITS_SPECS=';'.join(f'{n}={b}' for n,b in sorted(vf.items())),VLM_FORCE_BITS_SPECS_FILE='',
        DIT_FORCE_BITS_SPECS=';'.join(f'{n}={b}' for n,b in sorted(df.items())),DIT_FORCE_BITS_SPECS_FILE='',DIT_INCLUDE_ADALN='1',
        DIT_SKIP_VL_SELF_ATTENTION='0',VLM_CODEBOOK_OVERLAY_PATH='',DIT_CODEBOOK_OVERLAY_PATH='',GR00T_STRICT_QUANT_LAYOUT='1')
    with ft.patched_env(env):ft.quantize_gr00t_vlm27_dit4(policy.model,group_size=64,device='cuda:0')
    assert all('HQQ' in type(policy.model.get_submodule(n)).__name__ for n in ft.dit_vlsa)
    ft.replace_embedding(policy.model,4,64,True)
    def frozen(name,w,b):
        m=torch.nn.Linear(w.shape[1],w.shape[0],bias=b is not None,device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():
            m.weight.copy_(w)
            if b is not None:m.bias.copy_(b)
        m.requires_grad_(False);ft.replace_module(policy.model,name,m)
    scalar={r['layer']:(3 if int(r['k'])==80000 else 4) for r in rows if int(r['k'])>=80000}
    assert len(artifacts)==338-len(scalar) and len(gptq)==104+len(scalar)
    assert {r['layer'] for r in artifacts}=={r['layer'] for r in rows if int(r['k'])<80000}
    for r in artifacts:
        p=torch.load(r['path'],map_location='cpu',weights_only=False)['packed'];name=r['layer']
        q=ft.QuantLinear(p,bias[name],'cuda:0')
        with torch.no_grad():w=q.weight_()
        frozen(name,w,bias[name]);del q,w,p
    for r in gptq:
        p=torch.load(r['path'],map_location='cpu',weights_only=False);name=r['layer']
        assert p['layer']==name and p['bits']==scalar.get(name,4)
        old=policy.model.get_submodule(name);frozen(name,p['weight'].to('cuda'),getattr(old,'bias',None))
    policy.model.eval().requires_grad_(False);torch.cuda.empty_cache()

