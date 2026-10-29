"""Eliminate only Vision SDPA's output contiguous copy before unchanged cat."""
import ast,inspect,hashlib
from pathlib import Path
import torch

def enable(full_contract):
 import perrow_int8_fast_policy_r3 as fast
 previous=fast.install
 summary=dict(calls=0,validation_layout_checks=0,modules=[],source_sha256=None)
 def install(policy,names,contract):
  result=previous(policy,names,contract)
  from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
  from transformers.integrations.sdpa_attention import sdpa_attention_forward as native
  current=qwen.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',qwen.eager_attention_forward)
  assert current is native,('UnexpectedSDPAInterface',current)
  source=inspect.getsource(native)
  expected=Path(__file__).with_name('codex_vision_sdpa_source_20261028_r1.txt').read_text()
  assert ast.dump(ast.parse(source))==ast.dump(ast.parse(expected)),'InstalledSDPASourceChanged'
  summary['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
  line='attn_output = attn_output.transpose(1, 2).contiguous()'
  assert source.count(line)==1
  source=source.replace(line,'attn_output = attn_output.transpose(1, 2)')
  namespace=dict(native.__globals__)
  exec(compile('from __future__ import annotations\n'+source,__file__+':view','exec'),namespace)
  view_forward=namespace['sdpa_attention_forward']
  selected=set()
  for name in full_contract['common_vision_rotary']['modules']:
   module=policy.model.get_submodule(name)
   assert type(module) is qwen.Qwen3VLVisionAttention
   assert module.config._attn_implementation=='sdpa',module.config._attn_implementation
   selected.add(id(module));summary['modules'].append(name)
  assert len(selected)==24
  state=policy.model._perrow_a8_candidate_state
  def forward(module,*args,**kwargs):
   if id(module) not in selected:return current(module,*args,**kwargs)
   y,weights=view_forward(module,*args,**kwargs)
   assert y.ndim==4 and y.dtype==torch.bfloat16 and weights is None
   summary['calls']+=1
   if state['validation_enabled']:
    # Original wrapper's only removed operation: materialization of this view.
    reference=y.contiguous()
    assert torch.equal(y.view(torch.int16),reference.view(torch.int16)),'VisionSDPAViewBytes'
    summary['validation_layout_checks']+=1
   return y,weights
  qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',forward)
  return result
 fast.install=install
 return summary
