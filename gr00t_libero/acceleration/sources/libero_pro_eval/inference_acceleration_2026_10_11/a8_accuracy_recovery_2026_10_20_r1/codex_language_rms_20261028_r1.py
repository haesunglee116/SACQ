"""Native 2048-wide FP32 mean order: 32 lanes, four accumulators, sixteen vector loads.
Torch Reduce.cuh on this runtime uses (32,16) blocks, independent rows per warp.
No reassociation/FMA; preserve BF16 normalization and final weight product roundings.
"""
import torch,triton
import triton.language as tl
@triton.jit
def _codex_native_rms2048(X,W,Y,V,EPS:tl.constexpr,VALIDATE:tl.constexpr):
 row=tl.program_id(0);lane=tl.arange(0,32)
 a=tl.full((32,),0,tl.float32);b=tl.full((32,),0,tl.float32)
 c=tl.full((32,),0,tl.float32);d=tl.full((32,),0,tl.float32)
 for j in range(16):
  i=row*2048+j*128+lane*4
  x0=tl.load(X+i).to(tl.float32);x1=tl.load(X+i+1).to(tl.float32)
  x2=tl.load(X+i+2).to(tl.float32);x3=tl.load(X+i+3).to(tl.float32)
  a=a+x0*x0;b=b+x1*x1;c=c+x2*x2;d=d+x3*x3
 v=tl.sum(((a+b)+c)+d,0)*(1.0/2048.0)
 if VALIDATE:tl.store(V+row,v)
 i=tl.arange(0,2048)
 x=tl.load(X+row*2048+i).to(tl.float32)
 normalized=(x*tl.rsqrt(v+EPS)).to(tl.bfloat16).to(tl.float32)
 w=tl.load(W+i).to(tl.float32)
 tl.store(Y+row*2048+i,(normalized*w).to(tl.bfloat16))
def enable(full_contract):
 import perrow_int8_fast_policy_r3 as fast
 previous=fast.install
 cfg=full_contract['common_BF16_fusions'];assert cfg['rms_tail'] and cfg['rms_square']
 summary=dict(checks={},squares={},variance_checks={},native_fp32_mean=True,activation_quantization_unchanged=True)
 def install(policy,names,contract):
  result=previous(policy,names,contract);state=policy.model._perrow_a8_candidate_state
  for index,layer in enumerate(policy.model.backbone.model.model.language_model.layers):
   for suffix in ['input_layernorm','post_attention_layernorm']:
    norm=getattr(layer,suffix);old=norm.forward;name=f'{index}.{suffix}'
    for key in ['checks','squares','variance_checks']:summary[key][name]=0
    def forward(x,_norm=norm,_old=old,_name=name):
     import rollout_bridge_r1 as bridge
     assert tuple(x.shape)==(1,bridge.rows,2048) and x.dtype==torch.bfloat16 and x.is_contiguous()
     validate=state['validation_enabled'];y=torch.empty_like(x)
     variance=torch.empty((bridge.rows,),device=x.device,dtype=torch.float32) if validate else y
     _codex_native_rms2048[(bridge.rows,)](x,_norm.weight,y,variance,_norm.variance_epsilon,validate,num_warps=4,enable_fp_fusion=False)
     if validate:
      reference_variance=x.float().pow(2).mean(-1).reshape(-1)
      assert torch.equal(variance.view(torch.int32),reference_variance.view(torch.int32)),('RMS2048Variance',_name,(variance-reference_variance).abs().max().item())
      reference=_old(x)
      assert torch.equal(y.view(torch.int16),reference.view(torch.int16)),('RMS2048Output',_name)
      for key in ['checks','squares','variance_checks']:summary[key][_name]+=1
     return y
    norm.forward=forward
  assert len(summary['checks'])==32
  return result
 fast.install=install
 return summary
