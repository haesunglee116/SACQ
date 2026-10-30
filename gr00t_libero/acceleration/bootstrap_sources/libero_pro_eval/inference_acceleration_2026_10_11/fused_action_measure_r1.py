"""Actual-input FFN microbenchmarks; reported errors are not acceptance gates."""
import torch
from bottleneck_attribution_r3 import graph_measure
import fused_action_candidates_r1 as fused

def capture(model):
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    samples={};handles=[]
    def hook(name):
        def collect(m,args):
            x=args[0]
            key=(m.out_features,m.in_features,getattr(m,'index_bits',None),getattr(m,'index_window_bytes',None),x.numel()//x.shape[-1])
            if key not in samples:samples[key]=(name,m,x.detach().cpu().clone())
        return collect
    for name,m in model.named_modules():
        if name.startswith('action_head.model.transformer_blocks.') and '.norm1.linear' not in name and type(m) in (Packed,torch.nn.Linear):
            handles.append(m.register_forward_pre_hook(hook(name)))
    return samples,handles

def error(y,ref):
    assert torch.isfinite(y).all(), 'Non-finite candidate output'
    d=y.float()-ref.float()
    return dict(max_abs=d.abs().max().item(),relative_l2=(d.norm()/ref.float().norm().clamp_min(1e-12)).item(),
                exact=torch.equal(y.view(torch.int16),ref.view(torch.int16)))

def measure(samples,c,emit,packed):
    import decoder_tuning as decoder
    rows=[]
    pressure=torch.empty(4*torch.cuda.get_device_properties(0).L2_cache_size,device='cuda',dtype=torch.uint8)
    for key,(name,m,cpu_x) in sorted(samples.items()):
        x=cpu_x.to('cuda');w=decoder.dequantize(m) if packed else m.weight
        ref=torch.nn.functional.linear(x,w,m.bias)
        dense=lambda:torch.nn.functional.linear(x,w,m.bias)
        row=dict(name=name,geometry=key,input_shape=list(x.shape),candidates=[])
        for label,fn in [('dense',dense)]+([('current_sacq',lambda:decoder.linear(x,m))] if packed else []):
            hot,y=graph_measure(fn,c);cold,z=graph_measure(fn,c,pressure)
            row[label]=dict(repeated=hot,cache_pressure=cold,error=error(y,ref))
            del y,z
        if packed:
            for cfg in fused.CONFIGS:
                fn=lambda cfg=cfg:fused.linear(x,m,cfg)
                hot,y=graph_measure(fn,c)
                row['candidates'].append(dict(config=cfg,repeated=hot,error=error(y,ref)))
                del y
            finalists=sorted(row['candidates'],key=lambda r:r['repeated']['median_ms'])[:2]
            for r in finalists:
                cold,y=graph_measure(lambda:fused.linear(x,m,r['config']),c,pressure)
                r['cache_pressure']=cold;del y
                r['score_ms']=(r['repeated']['median_ms']+cold['median_ms'])/2
            winner=min(finalists,key=lambda r:r['score_ms'])
            base=row['current_sacq'];base_score=(base['repeated']['median_ms']+base['cache_pressure']['median_ms'])/2
            row['selected']=winner['config'] if winner['score_ms']<base_score else None
            row['selection_basis']='Mean of repeated/cache-pressure time; finite outputs only; NO accuracy acceptance claim'
            if row['selected'] is not None:fused.SELECTED[key]=tuple(row['selected'])
        rows.append(row);emit(rows)
        print('FFN_MICRO',name,key,'dense_us',row['dense']['repeated']['median_ms']*1000,
              'selected',row.get('selected'),flush=True)
        del x,w,ref
    return rows
