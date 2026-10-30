"""Approved cumulative441: native FP8 on all33 cached modulation projections.

Independent matrix scales are retained even for former grouped QKV. The first
implementation bypasses BF16 QKV fusion; its latency is not an FP8 ceiling.
"""
import types
import torch
from a8_rtn_r2 import Quantizer
from fused_ffup_cutlass_probe_r1b import bitsha as _tensor_bitsha

def bitsha(x):
 # None is an absent optional payload, never invented zero-valued bytes.
 return None if x is None else _tensor_bitsha(x)
from omni_a8_scope_ffdown_runtime_r1 import exact,verified


def install(policy,c):
 spec=c['a8_scope_expansion'];approval=verified(spec['approval']);diff=verified(spec['protocol_diff'])
 assert approval['status']=='USER_APPROVED_EXACT_A8_SCOPE_EXPANSION' and 5 in approval['approved_stages']
 assert approval['approved_protocol_diff_sha256']==spec['protocol_diff']['sha256']
 stage=verified(diff['proposed_override'])['stages'][5]
 assert stage['stage']==5 and stage['total_FP8_projection_count']==441
 names=stage['additional_layers'];assert len(names)==33 and names==sorted(names)
 scales=policy.model._a8_weight_scales_ref;assert len(scales)==408
 from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
 import a8_joint_pair_r1 as fp8
 assert fp8.ACTIVATION_OBSERVER is None
 q=Quantizer('fp8_e4m3fn');s=dict(validation_enabled=True,checks={n:0 for n in names},counts={n:0 for n in names},metadata={},shapes={n:{} for n in names})
 for n in names:
  m=policy.model.get_submodule(n)
  assert type(m) is Packed and m.vector_dim==4 and m.group_size==64 and m.out_features%16==m.in_features%16==0
  payload={k:bitsha(getattr(m,k)) for k in ('packed_indices','codebook','scales','shift','bias')}
  w=m.dequantize_weight();qw,sw=q.quantize(w);m._a8_fast_weight_div=True
  assert fp8._decode_key(m) in fp8.DECODE_TILES,(n,'missingfrozendecodetile')
  decoded=fp8.decode(m,sw,True)
  if not exact(decoded,qw):
   m._a8_fast_weight_div=False;key=fp8._decode_key(m)
   if key not in fp8.DECODE_TILES:fp8.DECODE_TILES[key]=fp8.DECODE_TILES[key[:-1]+(True,)]
   del decoded;decoded=fp8.decode(m,sw,True)
  assert exact(decoded,qw),(n,'nativeFP8weightbytes')
  scales[n]=sw
  s['metadata'][n]=dict(payload_sha256=payload,original_weight_sha256=bitsha(w),fp8_weight_sha256=bitsha(qw),fp8_weight_scale_sha256=bitsha(sw),index_bits=m.index_bits,shape=[m.out_features,m.in_features],decode_key=list(fp8._decode_key(m)),decode_tile=list(fp8.DECODE_TILES[fp8._decode_key(m)]))
  del w,qw,decoded
  def forward(this,x,_n=n,_m=m):
   s['counts'][_n]+=1
   if not s['validation_enabled']:return fp8.linear(x,_m,scales[_n],True)
   assert x.dtype==torch.bfloat16 and x.is_cuda and x.shape[-1]==_m.in_features
   a=x.reshape(-1,_m.in_features).contiguous();rows=a.shape[0];mp=(rows+15)//16*16
   aq,sa=q.quantize(a);padded=torch.zeros((mp,_m.in_features),dtype=aq.dtype,device=aq.device);padded[:rows].copy_(aq)
   wq=fp8.decode(_m,scales[_n],True)
   reference=torch._scaled_mm(padded,wq.t(),sa,scales[_n],bias=_m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
   if isinstance(reference,tuple):reference=reference[0]
   reference=reference[:rows].reshape(*x.shape[:-1],_m.out_features)
   observed=[0]
   def observer(actual,actual_q,actual_s):
    assert exact(actual,a) and exact(actual_q,padded) and exact(actual_s,sa),(_n,'nativeAQ/Sa')
    observed[0]+=1
   assert fp8.ACTIVATION_OBSERVER is None;fp8.ACTIVATION_OBSERVER=observer
   try:y=fp8.linear(x,_m,scales[_n],True)
   finally:fp8.ACTIVATION_OBSERVER=None
   # Both joint decoding and separate activation preparation invoke observer.
   assert observed[0]==1 and exact(y,reference),(_n,'nativeFP8projectionoutput',tuple(x.shape))
   s['checks'][_n]+=1;shape=str(tuple(x.shape));s['shapes'][_n][shape]=s['shapes'][_n].get(shape,0)+1
   return y
  m.forward=types.MethodType(forward,m)
 assert len(scales)==441
 policy.model._omni_a8_modulation_state=s
 return dict(status='INSTALLED_PENDING_NATIVE_ACTUAL_TRAJECTORY_GATES',additional_layers=names,additional_count=33,total_FP8_projection_count=441,metadata=s['metadata'],payload_unchanged=True,persistent_expanded_weights=False,extra_scale_logical_bytes=132,implementation='Independentper-originalModulationMatrixFP8; nativeFP32accum; retainsoriginal33payloads/scales; all33outputsremainfixedschedulecached',performance_claim='InitialimplementationnotFP8ceiling')


def finish_validation(policy,c):
 s=policy.model._omni_a8_modulation_state;expected=(c['warmup']+c['num_iterations'])*c['a8_layer_probe']['denoising_steps']
 assert expected==100 and len(s['checks'])==33 and set(s['checks'].values())=={100}
 s['validation_enabled']=False
 return dict(status='ALL3300_MODULATION_NATIVE_FP8_OPERANDS_SCALE_OUTPUT_EXACT',checks_per_layer=dict(s['checks']),actual_input_shapes=s['shapes'],validation_disabled_before_timing=True,persistent_expanded_weights=False)


def recheck_payloads(policy):
 s=policy.model._omni_a8_modulation_state
 for n,md in s['metadata'].items():
  m=policy.model.get_submodule(n)
  for key,digest in md['payload_sha256'].items():assert bitsha(getattr(m,key))==digest,(n,key)
  assert bitsha(policy.model._a8_weight_scales_ref[n])==md['fp8_weight_scale_sha256']
 return dict(status='ALL33_MODULATION_PAYLOAD_AND_SCALE_IDENTITIES_UNCHANGED')
