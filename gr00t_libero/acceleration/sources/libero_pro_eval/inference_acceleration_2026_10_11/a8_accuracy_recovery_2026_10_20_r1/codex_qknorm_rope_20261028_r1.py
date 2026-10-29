"""Fuse QK RMS128 and RoPE with every original BF16 rounding preserved."""
import ast,inspect,textwrap
from pathlib import Path
import torch,triton
import triton.language as tl

@triton.jit
def _codex_qk_rms_rope128(X,W,C,S,Y,Z,V,ROWS:tl.constexpr,H:tl.constexpr,EPS:tl.constexpr,CN:tl.constexpr,CD:tl.constexpr,SN:tl.constexpr,SD:tl.constexpr,VALIDATE:tl.constexpr):
 row=tl.program_id(0)*4+tl.arange(0,4);lane=tl.arange(0,32)
 i=row[:,None]*128+lane[None,:]*4;ok=row[:,None]<ROWS
 a=tl.load(X+i,ok,0).to(tl.float32);b=tl.load(X+i+1,ok,0).to(tl.float32)
 c=tl.load(X+i+2,ok,0).to(tl.float32);d=tl.load(X+i+3,ok,0).to(tl.float32)
 acc=((a*a+b*b)+c*c)+d*d
 for offset in tl.static_range(5):
  shift=16>>offset
  ix=tl.broadcast_to(((lane+shift)%32)[None,:],(4,32));acc=acc+tl.gather(acc,ix,1)
 var=tl.sum(tl.where(lane[None,:]==0,acc,0.),1)*(1./128.)
 if VALIDATE:tl.store(V+row,var,row<ROWS)
 inv=tl.rsqrt(var+EPS)[:,None]
 wa=tl.load(W+lane*4).to(tl.float32)[None,:];wb=tl.load(W+lane*4+1).to(tl.float32)[None,:]
 wc=tl.load(W+lane*4+2).to(tl.float32)[None,:];wd=tl.load(W+lane*4+3).to(tl.float32)[None,:]
 a=((a*inv).to(tl.bfloat16).to(tl.float32)*wa).to(tl.bfloat16).to(tl.float32)
 b=((b*inv).to(tl.bfloat16).to(tl.float32)*wb).to(tl.bfloat16).to(tl.float32)
 c=((c*inv).to(tl.bfloat16).to(tl.float32)*wc).to(tl.bfloat16).to(tl.float32)
 d=((d*inv).to(tl.bfloat16).to(tl.float32)*wd).to(tl.bfloat16).to(tl.float32)
 ix=tl.broadcast_to(((lane+16)%32)[None,:],(4,32))
 for j in tl.static_range(4):
  if j==0:value=a
  elif j==1:value=b
  elif j==2:value=c
  else:value=d
  other=tl.gather(value,ix,1);other=tl.where(lane[None,:]<16,-other,other)
  co=tl.load(C+(row//H)[:,None]*CN+(lane[None,:]*4+j)*CD,ok,0).to(tl.float32)
  si=tl.load(S+(row//H)[:,None]*SN+(lane[None,:]*4+j)*SD,ok,0).to(tl.float32)
  prod=(value*co).to(tl.bfloat16).to(tl.float32)
  rotated=(other*si).to(tl.bfloat16).to(tl.float32)
  tl.store(Y+i+j,(prod+rotated).to(tl.bfloat16),ok)
  if VALIDATE:tl.store(Z+i+j,value.to(tl.bfloat16),ok)

def enable():
 import perrow_int8_fast_policy_r3 as fast
 previous=fast.install
 qsummary=dict(checks={},variance_checks={})
 rsummary=dict(checks=0,calls=0,source_guard=False)
 pending={}
 def outer(policy,names,contract):
  result=previous(policy,names,contract)
  import common_language_qk_norm_fusion_r2 as base
  from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
  native_rope=qwen.apply_rotary_pos_emb
  expected=Path(__file__).with_name('codex_language_rope_reference_20261028_r2.txt').read_text().strip()
  assert inspect.getsource(native_rope).strip()==expected,'RoPESourceChanged'
  expected=Path(__file__).with_name('codex_qknorm_rope_attention_source_20261028_r1.txt').read_text()
  assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(qwen.Qwen3VLTextAttention.forward))))==ast.dump(ast.parse(expected)),'AttentionConsumerChanged'
  original_install=base.install;rsummary['source_guard']=True
  def install(policy,contract):
   result=original_install(policy,contract)
   state=policy.model._common_language_qk_norm_state
   astat=policy.model._perrow_a8_candidate_state
   for name in contract['common_language_qk_norm']['modules']:
    norm=policy.model.get_submodule(name);old=norm.forward
    qsummary['checks'][name]=0;qsummary['variance_checks'][name]=0
    def defer(x,_name=name,_norm=norm,_old=old):
     assert x.is_contiguous() and x.ndim==4 and x.shape[0]==1 and x.shape[-1]==128 and x.dtype==torch.bfloat16
     assert x.data_ptr() not in pending,'DuplicatePendingQKNorm'
     pending[x.data_ptr()]=(_name,_norm,_old,x)
     return x
    norm.forward=defer
   def rope(q,k,cos,sin,unsqueeze_dim=1):
    assert unsqueeze_dim==1 and cos.shape==sin.shape==(1,q.shape[2],128)
    assert cos.dtype==sin.dtype==q.dtype==k.dtype==torch.bfloat16
    assert len(pending)==2 and q.data_ptr() in pending and k.data_ptr() in pending
    validate=state['enabled'] or astat['validation_enabled'];outputs=[];refs=[]
    for x in (q,k):
     name,norm,old,raw=pending.pop(x.data_ptr())
     h,n=x.shape[1:3];assert x.shape==(1,h,n,128) and x.stride()[1:]==(128,h*128,1)
     y=torch.empty_like(x);assert y.stride()==x.stride()
     z=torch.empty_like(raw) if validate else y
     var=torch.empty(raw.shape[:-1],device=x.device,dtype=torch.float32) if validate else y
     _codex_qk_rms_rope128[(triton.cdiv(n*h,4),)](x,norm.weight,cos,sin,y,z,var,n*h,h,norm.variance_epsilon,*cos.stride()[1:],*sin.stride()[1:],validate,num_warps=4,enable_fp_fusion=False)
     if validate:
      ref=old(raw)
      assert torch.equal(z.view(torch.int16),ref.view(torch.int16)),('FusedQKNormBytes',name)
      expected_var=raw.float().pow(2).mean(-1)
      assert torch.equal(var.view(torch.int32),expected_var.view(torch.int32)),('FusedQKVarianceBytes',name)
      refs.append(ref.transpose(1,2))
      if state['enabled']:
       qsummary['checks'][name]+=1;qsummary['variance_checks'][name]+=1
     outputs.append(y)
    assert not pending
    if validate:
     expected=native_rope(*refs,cos,sin,unsqueeze_dim)
     for y,ref in zip(outputs,expected):assert torch.equal(y.view(torch.int16),ref.view(torch.int16)),'FusedQKRoPEBytes'
     rsummary['checks']+=1
    rsummary['calls']+=1
    return tuple(outputs)
   qwen.apply_rotary_pos_emb=rope
   return result
  base.install=install
  return result
 fast.install=outer
 return qsummary,rsummary
