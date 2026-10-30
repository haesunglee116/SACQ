"""Only fixed-schedule Norm1+modulation; preserve all native rounding points."""
import torch
import hashlib
from pathlib import Path
from omni_a8_scope_ffdown_runtime_r1 import exact

def install(policy,c):
 from adaln_norm_modulate_r1 import extension
 from fixed_adaln_r2 import FixedAdaLN
 ext=extension();original_enable=FixedAdaLN.enable
 selected=[n.split('.ff.net.0.proj')[0]+'.norm1' for n in c['a8_layer_probe']['projection_layers']]
 assert len(selected)==len(set(selected))==32
 byid={id(policy.model.get_submodule(n)):n for n in selected}
 state=dict(enabled=True,checks={n:0 for n in selected},native_norm_checks={n:0 for n in selected})
 empty=torch.empty(0,dtype=torch.bfloat16,device='cuda:0')
 def ref(path):
  path=Path(path).resolve();return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 binary_receipt=ref(ext.__file__)
 sources=[ref(Path(__file__).resolve().parent/'adaln_norm_modulate_r1'/name) for name in ['kernel.cu','binding.cpp']]
 def enable(cache):
  original_enable(cache)
  assert len(cache.norms)==32 and {id(n) for n in cache.norms}==set(byid)
  for index,norm in enumerate(cache.norms):
   original=norm.forward;name=byid[id(norm)]
   def forward(x,temb,_i=index,_n=norm,_old=original,_name=name):
    assert not _n.training and not torch.is_grad_enabled() and cache.step is not None
    modulation=cache.outputs[_i][cache.step]
    reference=_old(x,temb) if state['enabled'] else None
    weight=_n.norm.weight if _n.norm.weight is not None else empty
    bias=_n.norm.bias if _n.norm.bias is not None else empty
    out,native=ext.norm_modulate(x,weight,bias,modulation,float(_n.norm.eps),state['enabled'])
    if state['enabled']:
     assert exact(native,_n.norm(x)),(_name,'NativeNorm1BF16Changed')
     assert exact(out,reference),(_name,'OriginalCachedAdaLNBF16Changed')
     state['checks'][_name]+=1;state['native_norm_checks'][_name]+=1
    return out
   norm.forward=forward
 FixedAdaLN.enable=enable
 policy.model._adaln_norm_modulate_state=state
 return dict(status='FIXED_ADALN_ENABLE_WRAPPER_PENDING_ALL_ACTUAL_NATIVE_OUTPUTS',modules=selected,
  original_FP32_Welford=True,original_BF16_norm_factor_product_sum_roundings=True,
  persistent_extra_GPU_storage=False,scientific_delta={},extension_binary=binary_receipt,extension_sources=sources)

def finish_validation(policy,c):
 state=policy.model._adaln_norm_modulate_state
 expected=(c['warmup']+c['num_iterations'])*c['a8_layer_probe']['denoising_steps']
 for key in ['checks','native_norm_checks']:
  assert len(state[key])==32 and set(state[key].values())=={expected},(key,state[key],expected)
 state['enabled']=False
 return dict(status='ALL3200_NATIVE_NORM1_AND_CACHED_ADALN_BF16_OUTPUTS_EXACT',checks=state['checks'],
  native_norm_checks=state['native_norm_checks'],validation_disabled_before_timing=True)
