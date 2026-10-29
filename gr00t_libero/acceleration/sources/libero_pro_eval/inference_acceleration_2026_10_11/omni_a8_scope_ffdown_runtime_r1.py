"""Proposed first scope expansion: 32 existing FF-up + 32 native FP8 FF-down.

No GPU work is authorized by this module alone. Caller must supply an explicitly
approved immutable scope contract. Native FP8 path is a measured starting point,
not a claim about the fastest achievable expanded-scope implementation.
"""
import hashlib,json,types
from pathlib import Path
import torch
from a8_rtn_r2 import Quantizer
from fused_ffup_cutlass_probe_r1b import bitsha


def exact(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))


def verified(ref):
 p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'],str(p)
 return json.loads(p.read_text())


def install(policy,c):
 spec=c['a8_scope_expansion'];approval=verified(spec['approval'])
 assert approval['status']=='USER_APPROVED_EXACT_A8_SCOPE_EXPANSION'
 assert approval['approved_protocol_diff_sha256']==spec['protocol_diff']['sha256']
 diff=verified(spec['protocol_diff']);proposal=verified(diff['proposed_override'])
 stage=proposal['stages'][spec['stage']];assert stage['stage']==1 and stage['total_FP8_projection_count']==64
 names=stage['additional_layers'];assert len(names)==32 and names==sorted(names)
 assert all(n.startswith('action_head.model.transformer_blocks.') and n.endswith('.ff.net.2') for n in names)
 scales=policy.model._a8_weight_scales_ref;assert sorted(scales)==sorted(c['a8_layer_probe']['projection_layers'])
 from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
 import a8_joint_pair_r1 as fp8
 assert fp8.ACTIVATION_OBSERVER is None
 frozen=verified(c['a8_layer_probe']['fp8_decode_selection'])
 assert frozen['scope']==441
 for row in frozen['groups']:
  key=tuple(row['key']);tile=tuple(row['selected']['tile'])
  assert key in fp8.DECODE_TILES and fp8.DECODE_TILES[key]==tile
 state=dict(validation_enabled=True,checks={n:0 for n in names},counts={n:0 for n in names},samples_cpu={n:[] for n in names},originals={},metadata={})
 q=Quantizer('fp8_e4m3fn')
 for n in names:
  m=policy.model.get_submodule(n)
  assert type(m) is Packed and (m.in_features,m.out_features,m.vector_dim,m.group_size)==(6144,1536,4,64)
  assert m.index_bits in (5,6,7)
  payload_sha={key:bitsha(getattr(m,key)) for key in ('packed_indices','codebook','scales','shift','bias')}
  w=m.dequantize_weight();qw,sw=q.quantize(w);m._a8_fast_weight_div=True
  assert fp8._decode_key(m) in fp8.DECODE_TILES,(n,'missingfrozendecodetile')
  decoded=fp8.decode(m,sw,True)
  if not exact(decoded,qw):
   m._a8_fast_weight_div=False
   key=fp8._decode_key(m)
   if key not in fp8.DECODE_TILES:fp8.DECODE_TILES[key]=fp8.DECODE_TILES[key[:-1]+(True,)]
   del decoded;decoded=fp8.decode(m,sw,True)
  assert exact(decoded,qw),(n,'nativeFP8weightbytes')
  scales[n]=sw
  state['metadata'][n]=dict(payload_sha256=payload_sha,original_weight_sha256=bitsha(w),fp8_weight_sha256=bitsha(qw),fp8_weight_scale_sha256=bitsha(sw),index_bits=m.index_bits,decode_key=list(fp8._decode_key(m)),decode_tile=list(fp8.DECODE_TILES[fp8._decode_key(m)]),shape=[1536,6144],input_shape=[1,41,6144])
  state['originals'][n]=m.forward
  del w,qw,decoded
  def forward(this,x,_n=n,_m=m):
   assert tuple(x.shape)==(1,41,6144) and x.dtype==torch.bfloat16 and x.is_cuda
   state['counts'][_n]+=1
   if not state['validation_enabled']:return fp8.linear(x,_m,scales[_n],True)
   a=x.reshape(41,6144);aq,sa=q.quantize(a)
   padded=torch.zeros((48,6144),dtype=aq.dtype,device=aq.device);padded[:41].copy_(aq)
   wq=fp8.decode(_m,scales[_n],True)
   reference=torch._scaled_mm(padded,wq.t(),sa,scales[_n],bias=_m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
   if isinstance(reference,tuple):reference=reference[0]
   reference=reference[:41].reshape(1,41,1536)
   seen=[0]
   def observer(actual,actual_q,actual_s):
    assert exact(actual,a) and exact(actual_q,padded) and exact(actual_s,sa),(_n,'nativeAQ/Sa')
    seen[0]+=1
   assert fp8.ACTIVATION_OBSERVER is None
   fp8.ACTIVATION_OBSERVER=observer
   try:y=fp8.linear(x,_m,scales[_n],True)
   finally:fp8.ACTIVATION_OBSERVER=None
   assert seen[0]==1 and exact(y,reference),(_n,'nativeFP8FFdownoutput')
   state['checks'][_n]+=1
   if len(state['samples_cpu'][_n])<c['a8_layer_probe']['denoising_steps']:
    state['samples_cpu'][_n].append(x.detach().cpu().clone())
   return y
  m.forward=types.MethodType(forward,m)
 assert len(scales)==64
 policy.model._omni_a8_scope_extra_state=state
 return dict(status='PROPOSED_FFDOWN_NATIVE_FP8_INSTALLED_PENDING_ACTUAL_TRAJECTORY_GATES',stage=1,additional_layers=names,additional_count=32,total_FP8_projection_count=64,metadata=state['metadata'],weight_payload_unchanged=True,persistent_expanded_weights=False,extra_scale_logical_bytes=128,native_reference='OriginalBF16->nativeTorchE4M3fnquant; frozenexactTritonweightdecode+absmax; native_scaled_mm FP32accum/BF16biasoutput. NoGELUonFFdown.',performance_claim='Measuredstartingpointonly; notFP8maximum',accuracy_acceptance=False)


def finish_validation(policy,c):
 s=policy.model._omni_a8_scope_extra_state
 expected=(c['warmup']+c['num_iterations'])*c['a8_layer_probe']['denoising_steps']
 assert expected==100 and len(s['checks'])==32 and set(s['checks'].values())=={expected}
 assert all(len(x)==4 for x in s['samples_cpu'].values())
 s['validation_enabled']=False
 return dict(status='ALL3200_FFDOWN_NATIVE_FP8_OPERANDS_SCALE_OUTPUT_EXACT',checks_per_layer=dict(s['checks']),validation_disabled_before_timing=True,persistent_expanded_weights=False,samples='Firstfouroriginalown-stageFFdowninputsperlayerkeptonCPUonlyfortheapprovedlayerdiagnostic',accuracy_acceptance=False)


def recheck_payloads(policy):
 s=policy.model._omni_a8_scope_extra_state
 for n,md in s['metadata'].items():
  m=policy.model.get_submodule(n)
  for key,digest in md['payload_sha256'].items():assert bitsha(getattr(m,key))==digest,(n,key)
  assert bitsha(policy.model._a8_weight_scales_ref[n])==md['fp8_weight_scale_sha256']
 return dict(status='ALL32_ADDITIONAL_PAYLOAD_AND_SCALE_IDENTITIES_UNCHANGED')
