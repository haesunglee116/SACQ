"""Exact native AdaLN output plus per-row INT8 producer; immediate consumer only."""
from pathlib import Path
import hashlib,inspect,json,types
import torch
EXT=None

def extension():
 global EXT
 if EXT is None:
  from torch.utils.cpp_extension import load
  p=Path(__file__).resolve().parent/'codex_norm3_regpack_20261029_r1'
  b=Path('@LOCAL_RUNTIME@/codex_norm3_regpack_20261029_r1_sm89');b.mkdir(exist_ok=True)
  EXT=load(name='codex_norm3_regpack_20261029_r1',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],build_directory=str(b),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v','-gencode=arch=compute_89,code=sm_89'],verbose=True)
 return EXT

def enable():
 import perrow_int8_fast_policy_r3 as fast
 import perrow_int8_padded_producer_r1 as producer
 original_install=fast.install
 ext=extension();pending={}
 summary=dict(produced=0,hits=0,validated_hits=0,native_norm_checks=0,residual_checks=0,modules=[])
 def install(policy,names,contract):
  result=original_install(policy,names,contract)
  old=producer.activation;state=policy.model._perrow_a8_candidate_state
  selected=[n.rsplit('.ff.net.0.proj',1)[0]+'.norm3' for n in names if n.endswith('.ff.net.0.proj') and n.startswith('action_head.')]
  assert len(selected)==len(set(selected))==32
  empty=torch.empty(0,dtype=torch.bfloat16,device='cuda:0')
  provenance_path=Path(__file__).parent.parent/'omni_fp8_norm_gemm_parent_inspection_r7.json'
  provenance=json.loads(provenance_path.read_text())
  def block_forward(this,hidden_states,attention_mask=None,encoder_hidden_states=None,encoder_attention_mask=None,temb=None):
      # Pinned installed BasicTransformerBlock.forward, changing only the
      # residual-add/Norm3 boundary to return the same rounded residual too.
      if this.norm_type=='ada_norm':norm_hidden_states=this.norm1(hidden_states,temb)
      else:norm_hidden_states=this.norm1(hidden_states)
      if this.pos_embed is not None:norm_hidden_states=this.pos_embed(norm_hidden_states)
      attn_output=this.attn1(norm_hidden_states,encoder_hidden_states=encoder_hidden_states,
          attention_mask=(encoder_attention_mask if encoder_hidden_states is not None else attention_mask))
      if this.final_dropout:attn_output=this.final_dropout(attn_output)
      hidden_states,norm_hidden_states=this.norm3(attn_output,hidden_states)
      import int8_ffdown_residual_epilogue_20261028_r2 as ffdown_fusion
      hidden_states=ffdown_fusion.run_ff(this.ff,norm_hidden_states,hidden_states)
      if hidden_states.ndim==4:hidden_states=hidden_states.squeeze(1)
      return hidden_states
  for name in selected:
   module=policy.model.get_submodule(name);assert isinstance(module,torch.nn.LayerNorm)
   block=policy.model.get_submodule(name.rsplit('.norm3',1)[0])
   source=Path(inspect.getfile(type(block)))
   assert type(block).__name__=='BasicTransformerBlock' and not block.training
   assert str(source)==provenance['node_source'] and hashlib.sha256(source.read_bytes()).hexdigest()==provenance['sha256']
   block.forward=types.MethodType(block_forward,block)
   original=module.forward
   def forward(attention,hidden,_m=module,_old=original,_name=name):
    assert attention.shape==hidden.shape and list(attention.shape) in ([1,41,1536],[1,1,41,1536])
    x=attention.reshape(1,41,1536).contiguous();h=hidden.reshape(1,41,1536).contiguous()
    assert not _m.training and not torch.is_grad_enabled()
    gamma=_m.weight if _m.weight is not None else empty
    beta=_m.bias if _m.bias is not None else empty
    y,native,q,scale=ext.norm_modulate(x,gamma,beta,h,float(_m.eps),False)
    if state['validation_enabled']:
     expected_sum=(attention+hidden).reshape(1,41,1536)
     assert torch.equal(native.view(torch.int16),expected_sum.view(torch.int16)),('ResidualBF16Bytes',_name)
     summary['residual_checks']+=1
     expected=_old(expected_sum)
     assert torch.equal(y.view(torch.int16),expected.view(torch.int16)),('Norm3BF16Bytes',_name)
     summary['native_norm_checks']+=1
    pending.clear();pending.update(source=y,q=q,scale=scale)
    summary['produced']+=1
    return native,y
   module.forward=forward;summary['modules'].append(name)
  def activation(x,mode):
   source=pending.get('source')
   if source is None or mode!=0 or x.data_ptr()!=source.data_ptr() or x.numel()!=source.numel() or not x.is_contiguous():return old(x,mode)
   q,scale=pending['q'],pending['scale'];pending.clear()
   if state['validation_enabled']:
    expected_q,expected_scale=old(x,mode)
    assert torch.equal(q,expected_q),'Norm3ProducerINT8Bytes'
    assert torch.equal(scale.view(torch.int32),expected_scale.view(torch.int32)),'Norm3ProducerScaleBytes'
    summary['validated_hits']+=1
   summary['hits']+=1
   return q,scale
  producer.activation=activation
  return result
 fast.install=install
 return summary
