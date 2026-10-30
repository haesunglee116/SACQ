"""Actual-input candidate selection with matched existing runtime controls."""
import gc
import torch
import priority_fused_r3 as fused
from bottleneck_fused_detail_r1 import graph_measure


def capture(model,stage):
    samples={};handles=[]
    def hook(name,region):
        def call(m,args):
            x=args[0];k=fused.key(m,x,region)
            if k not in samples:samples[k]=dict(name=name,module=m,x_cpu=x.detach().cpu().clone())
        return call
    for name,m in model.named_modules():
        region='vision' if '.visual.' in name else 'language' if '.language_model.' in name else 'action'
        eligible=(hasattr(m,'packed_indices') and '.norm1.linear' not in name) or (region=='vision' and type(m).__name__=='ExactPackedLinear' and m.bits==4)
        if region==stage and eligible:handles.append(m.register_forward_pre_hook(hook(name,region)))
    return samples,handles


def tune(samples,c,emit):
    import decoder_tuning as dec
    import exact_dictionary_decode as dic
    rows=[];pressure=torch.empty(4*torch.cuda.get_device_properties(0).L2_cache_size,device='cuda',dtype=torch.uint8)
    for k,s in sorted(samples.items(),key=lambda a:str(a[0])):
        m=s['module'];x=s['x_cpu'].to('cuda');region=k[0]
        base=(lambda:torch.nn.functional.linear(x,dic.reconstruct(m),m.bias)) if region=='vision' else lambda:dec.linear(x,m)
        w=dic.reconstruct(m) if region=='vision' else dec.dequantize(m)
        ref=torch.nn.functional.linear(x,w,m.bias)
        def measure(fn):
            t,y=graph_measure(fn,c);torch.cuda.synchronize()
            assert torch.isfinite(y).all()
            delta=y.float()-ref.float()
            t['error']=dict(relative_l2=(delta.norm()/ref.float().norm().clamp_min(1e-12)).item(),max_abs=delta.abs().max().item())
            del y,delta
            return t
        row=dict(geometry=k,name=s['name'],candidates=[])
        row['dense_repeated']=measure(lambda:torch.nn.functional.linear(x,w,m.bias))
        row['decode_repeated']=measure_decode(dic,m,c) if region=='vision' else measure_decode(dec,m,c)
        row['baseline_repeated']=measure(base)
        t,y=graph_measure(base,c,pressure);del y
        row['baseline_pressure']=t
        row['baseline_score_ms']=(row['baseline_repeated']['median_ms']+t['median_ms'])/2
        configs=[('triton',)+cfg for cfg in fused.CONFIGS]
        if region=='action':
            # New CTA budgets; preserve frozen old configuration as control.
            configs=[('cuda',ctas,sp,True) for ctas in (128,192,768,1024) for sp in (2,8)]+configs[:2]
        for cfg in configs:
            fn=(lambda cfg=cfg:fused.linear(x,m,cfg[1:])) if cfg[0]=='triton' else lambda cfg=cfg:fused.cuda_linear(x,m,cfg[1:])
            hot=measure(fn);cold,y=graph_measure(fn,c,pressure);del y
            entry=dict(config=cfg,repeated=hot,cache_pressure=cold,score_ms=(hot['median_ms']+cold['median_ms'])/2)
            if cfg[0]=='triton':entry['compiled_resources']=fused.COMPILED[(region=='vision',m.out_features,m.in_features,k[3],k[4],tuple(cfg[1:]))]
            row['candidates'].append(entry)
        best=min(row['candidates'],key=lambda r:r['score_ms'])
        row['selected']=best['config'] if best['score_ms']<row['baseline_score_ms'] else None
        if row['selected'] is not None:fused.SELECTED[k]=tuple(row['selected'])
        rows.append(row);emit(rows)
        print('PRIORITY_GEOMETRY',k,'base_us',row['baseline_score_ms']*1000,'best_us',best['score_ms']*1000,'selected',row['selected'],flush=True)
        del x,w,ref
    del pressure
    gc.collect();torch.cuda.empty_cache()
    return rows


def measure_decode(dec,m,c):
    fn=(lambda:dec.reconstruct(m)) if hasattr(m,"packed") else lambda:dec.dequantize(m)
    timing,y=graph_measure(fn,c);del y
    return timing
