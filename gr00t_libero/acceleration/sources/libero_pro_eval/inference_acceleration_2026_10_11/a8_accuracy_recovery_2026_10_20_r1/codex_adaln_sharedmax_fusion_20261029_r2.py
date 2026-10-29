"""Exact native AdaLN output plus per-row INT8 producer; immediate consumer only."""
from pathlib import Path
import torch
EXT=None

def extension():
 global EXT
 if EXT is None:
  from torch.utils.cpp_extension import load
  p=Path(__file__).resolve().parent/'codex_adaln_sharedmax_20261029_r2'
  b=Path('@LOCAL_RUNTIME@/codex_adaln_sharedmax_20261029_r2_sm89');b.mkdir(exist_ok=True)
  EXT=load(name='codex_adaln_sharedmax_20261029_r2',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],build_directory=str(b),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v','-gencode=arch=compute_89,code=sm_89'],verbose=True)
 return EXT

def enable():
 import adaln_norm_modulate_r1 as native_module
 import perrow_int8_fast_policy_r3 as fast
 import perrow_int8_padded_producer_r1 as producer
 original_install=fast.install
 ext=extension()
 pending={}
 summary=dict(produced=0,hits=0,validated_hits=0,extra_persistent_weight_bytes=0)
 class Adapter:
  __file__=ext.__file__
  def norm_modulate(self,*args):
   y,native,q,scale=ext.norm_modulate(*args)
   pending.clear();pending.update(source=y,q=q,scale=scale,validate=bool(args[-1]))
   summary['produced']+=1
   return y,native
 native_module.extension=lambda:Adapter()
 def install(policy,names,contract):
  result=original_install(policy,names,contract)
  old=producer.activation;state=policy.model._perrow_a8_candidate_state
  def activation(x,mode):
   source=pending.get('source')
   if source is None or mode!=0 or x.data_ptr()!=source.data_ptr() or x.numel()!=source.numel() or not x.is_contiguous():
    return old(x,mode)
   q,scale=pending['q'],pending['scale'];validate=pending['validate'];pending.clear()
   if state['validation_enabled'] or validate:
    expected_q,expected_scale=old(x,mode)
    assert torch.equal(q,expected_q),'AdaLNProducerINT8Bytes'
    assert torch.equal(scale.view(torch.int32),expected_scale.view(torch.int32)),'AdaLNProducerScaleBytes'
    summary['validated_hits']+=1
   summary['hits']+=1
   return q,scale
  producer.activation=activation
  return result
 fast.install=install
 return summary
