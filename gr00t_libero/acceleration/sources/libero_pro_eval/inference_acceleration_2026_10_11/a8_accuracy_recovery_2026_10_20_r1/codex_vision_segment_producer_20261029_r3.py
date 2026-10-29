"""Read two Vision attention segments directly into exact row-INT8 operands."""
import ast,inspect,textwrap,types
from pathlib import Path
import torch,triton
import triton.language as tl
from int8_direct_integer_round_20261028_r1 import round_int8

@triton.jit
def _vision_segment_int8(X0,X1,Q,S,N0:tl.constexpr,N1:tl.constexpr,H:tl.constexpr,D:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);col=tl.arange(0,BLOCK);h=col//D;d=col%D
 if row<N0:
  value=tl.load(X0+row*A0+h*A1+d*A2,col<H*D,0).to(tl.float32)
 else:
  value=tl.load(X1+(row-N0)*B0+h*B1+d*B2,(row<N0+N1)&(col<H*D),0).to(tl.float32)
 maximum=tl.max(tl.abs(value),0)
 scale=tl.where(maximum>0,maximum*(1.0/127.0),1.0)
 encoded=round_int8(tl.minimum(tl.maximum(tl.div_rn(value,scale),-127.0),127.0))
 tl.store(Q+row*(H*D)+col,encoded,col<H*D);tl.store(S+row,scale)

def enable(full_contract):
 import perrow_int8_fast_policy_r3 as fast
 previous=fast.install
 summary=dict(modules=[],checks=0,calls=0,source_guard=False,layouts=[])
 def install(policy,names,contract):
  result=previous(policy,names,contract)
  from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
  import int8_refined_policy_20261028_r1 as refined
  import int8_torch_epilogue_runtime_r2 as runtime
  state=policy.model._perrow_a8_candidate_state
  native=qwen.Qwen3VLVisionAttention.forward
  source=textwrap.dedent(inspect.getsource(native))
  expected=Path(__file__).with_name('codex_vision_segment_source_20261028_r1.txt').read_text()
  assert ast.dump(ast.parse(source))==ast.dump(ast.parse(expected)),'VisionAttentionSourceChanged'
  line='attn_output = torch.cat(attn_outputs, dim=1)'
  assert source.count(line)==1
  source=source.replace(line,'return _codex_segment_project(self.proj, attn_outputs, seq_length)')
  metadata={}
  def direct(proj,segments,seq_length):
   assert len(segments)==2
   x0,x1=segments
   assert x0.ndim==x1.ndim==4 and x0.shape[0]==x1.shape[0]==1
   assert x0.shape[2:]==x1.shape[2:] and x0.dtype==x1.dtype==torch.bfloat16
   assert x0.stride(-1)==x1.stride(-1)==1
   n0,n1=x0.shape[1],x1.shape[1];h,d=x0.shape[2:];m=n0+n1;k=h*d
   assert m==seq_length and k==proj.in_features
   name,mode,old=metadata[id(proj)];mp=max(32,triton.cdiv(m,16)*16)
   aq=torch.empty((mp,k),device=x0.device,dtype=torch.int8)
   sa=torch.empty((mp,1),device=x0.device,dtype=torch.float32)
   _vision_segment_int8[(mp,)](x0,x1,aq,sa,n0,n1,h,d,*x0.stride()[1:],*x1.stride()[1:],triton.next_power_of_2(k),num_warps=16,enable_fp_fusion=False)
   summary['calls']+=1
   if state['validation_enabled']:
    original=torch.cat(segments,dim=1).reshape(m,k).contiguous()
    qr,sr=refined.original.quantize_rows(original,'int8_symmetric')
    assert torch.equal(aq[:m],qr),('VisionSegmentINT8Bytes',name)
    assert torch.equal(sa[:m].view(torch.int32),sr.view(torch.int32)),('VisionSegmentScaleBytes',name)
    summary['checks']+=1
    layout=dict(shapes=[list(x0.shape),list(x1.shape)],strides=[list(x0.stride()),list(x1.stride())])
    if layout not in summary['layouts']:summary['layouts'].append(layout)
    return proj(original)
   wq=refined.decoder.decode(proj,mode)
   return runtime.linear(aq,wq,sa,proj._a8_perrow_weight_scale,proj.bias,True,actual_rows=m).reshape(m,proj.out_features)
  import omni_backbone_graph_r1 as graph_runtime
  graph_native=graph_runtime.install
  graph_source=textwrap.dedent(inspect.getsource(graph_native))
  graph_expected=Path(__file__).with_name('codex_vision_segment_graph_source_20261029_r1.txt').read_text()
  assert ast.dump(ast.parse(graph_source))==ast.dump(ast.parse(graph_expected)),'BackboneGraphSourceChanged'
  graph_line='return _m.proj(torch.cat(out, dim=1).reshape(seq, -1).contiguous())'
  assert graph_source.count(graph_line)==1
  graph_source=graph_source.replace(graph_line,'return _codex_segment_project(_m.proj, out, seq)')
  graph_namespace=dict(graph_native.__globals__);graph_namespace['_codex_segment_project']=direct
  exec(compile(graph_source,__file__+':graph_install','exec'),graph_namespace)
  graph_runtime.install=graph_namespace['install']
  namespace=dict(native.__globals__);namespace['_codex_segment_project']=direct
  # Resolve the installed rotary at call time; bootstrap installs its exact gate later.
  namespace['apply_rotary_pos_emb_vision']=lambda *args,**kwargs:qwen.apply_rotary_pos_emb_vision(*args,**kwargs)
  exec(compile('from __future__ import annotations\n'+source,__file__+':forward','exec'),namespace)
  replacement=namespace['forward']
  for name in full_contract['common_vision_rotary']['modules']:
   module=policy.model.get_submodule(name)
   assert type(module) is qwen.Qwen3VLVisionAttention and module.forward.__func__ is native
   assert module.config._attn_implementation=='sdpa'
   pname=name+'.proj';proj=module.proj
   assert pname in state['weight_division_modes']
   metadata[id(proj)]=(pname,state['weight_division_modes'][pname],proj.forward)
   module.forward=types.MethodType(replacement,module);summary['modules'].append(name)
  assert len(metadata)==24
  summary['source_guard']=True
  return result
 fast.install=install
 return summary
