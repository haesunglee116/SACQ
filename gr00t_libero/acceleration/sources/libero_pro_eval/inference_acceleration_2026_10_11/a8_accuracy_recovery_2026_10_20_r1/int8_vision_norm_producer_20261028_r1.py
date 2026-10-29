"""Native-order Vision LayerNorm and exact per-row INT8 producer."""
from pathlib import Path
import torch
EXT=None
def extension():
 global EXT
 if EXT is None:
  from torch.utils.cpp_extension import load
  p=Path(__file__).parent/'vision_norm_int8_20261028_r1';b=Path('@LOCAL_RUNTIME@/vision_norm_int8_20261028_r1_sm89');b.mkdir(exist_ok=True)
  EXT=load(name='vision_norm_int8_20261028_r1',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],build_directory=str(b),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v','-gencode=arch=compute_89,code=sm_89'],verbose=True)
 return EXT
def enable(full_contract):
 import perrow_int8_fast_policy_r3 as fast
 import perrow_int8_padded_producer_r1 as producer
 previous_install=fast.install;ext=extension();pending={}
 summary=dict(modules=[],native_norm_checks=0,validated_hits=0,hits=0,produced=0,layouts={})
 def install(policy,names,contract):
  result=previous_install(policy,names,contract);old_activation=producer.activation;state=policy.model._perrow_a8_candidate_state
  selected=set(names)
  from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
  for attn_name in full_contract['common_vision_rotary']['modules']:
   base=attn_name.rsplit('.attn',1)[0];block=policy.model.get_submodule(base);assert type(block) is Qwen3VLVisionBlock
   assert base+'.attn.qkv' in selected and base+'.mlp.linear_fc1' in selected
   for suffix in ['norm1','norm2']:
    name=base+'.'+suffix;norm=policy.model.get_submodule(name);assert type(norm) is torch.nn.LayerNorm and tuple(norm.normalized_shape)==(1024,)
    original=norm.forward
    def forward(x,_norm=norm,_old=original,_name=name):
     assert not _norm.training and not torch.is_grad_enabled() and x.dtype==torch.bfloat16 and x.shape[-1]==1024 and x.is_contiguous()
     a=x.reshape(-1,1024)
     y,unused,q,scale=ext.norm_modulate(a,_norm.weight,_norm.bias,a,float(_norm.eps),False);y=y.reshape(x.shape)
     if state['validation_enabled']:
      expected=_old(x);assert torch.equal(y.view(torch.int16),expected.view(torch.int16)),('VisionNativeNormBytes',_name)
      summary['native_norm_checks']+=1;summary['layouts'][_name]=dict(shape=list(x.shape),stride=list(x.stride()),eps=_norm.eps)
     pending.clear();pending.update(source=y,q=q,scale=scale);summary['produced']+=1
     return y
    norm.forward=forward;summary['modules'].append(name)
  assert len(summary['modules'])==48
  def activation(x,mode):
   source=pending.get('source')
   if source is None or mode!=0 or x.data_ptr()!=source.data_ptr() or x.numel()!=source.numel() or not x.is_contiguous():return old_activation(x,mode)
   q,scale=pending['q'],pending['scale'];pending.clear()
   if state['validation_enabled']:
    eq,es=old_activation(x,mode)
    assert torch.equal(q,eq),'VisionNormINT8Bytes'
    assert torch.equal(scale.view(torch.int32),es.view(torch.int32)),'VisionNormScaleBytes'
    summary['validated_hits']+=1
   summary['hits']+=1
   return q,scale
  producer.activation=activation
  return result
 fast.install=install
 return summary
