"""Approved cumulative408: exact existing lossless Vision payload -> native FP8.

Payload codec stays unchanged. No original dictionary or expanded weight cache
is retained. Each invocation restores and converts the original BF16 weights.
"""
import types
import torch
import triton
import triton.language as tl
from a8_rtn_r2 import Quantizer
from a8_joint_pair_r1 import _fp8_rtn
from fused_ffup_cutlass_probe_r1b import bitsha as _tensor_bitsha

def bitsha(x):
 # None is an absent optional payload, never invented zero-valued bytes.
 return None if x is None else _tensor_bitsha(x)
from omni_a8_scope_ffdown_runtime_r1 import exact,verified


@triton.jit
def _affine_fp8(packed,scales,offsets,out,sw,COUNT:tl.constexpr,BITS:tl.constexpr,RECIP:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<COUNT
 bit=i*BITS;position=bit%8
 word=tl.load(packed+bit//8,valid,0).to(tl.uint32)
 q=((word>>position)&((1<<BITS)-1)).to(tl.int32)
 offset=tl.load(offsets+i//64,valid,0).to(tl.int32)
 scale=tl.load(scales+i//64,valid,0)
 w=((q+offset).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
 s=tl.load(sw)
 if RECIP:v=w*tl.div_rn(1.0,s)
 else:v=tl.div_rn(w,s)
 v=_fp8_rtn(tl.minimum(tl.maximum(v,-448.0),448.0))
 tl.store(out+i,v,valid)


def decode(m,sw):
 assert m.bits==4 and m.group_size==64
 q=torch.empty((m.out_features,m.in_features),device=sw.device,dtype=torch.float8_e4m3fn)
 _affine_fp8[(triton.cdiv(q.numel(),1024),)](m.packed,m.scales,m.offsets,q,sw,q.numel(),m.bits,m._a8_fast_weight_div,1024,num_warps=4,enable_fp_fusion=False)
 return q


def linear(x,m,sw):
 import a8_joint_pair_r1 as fp8
 a=x.reshape(-1,m.in_features).contiguous();rows=a.shape[0];mp=(rows+15)//16*16
 aq,sa=fp8.prepare_activation(a,mp);wq=decode(m,sw)
 y=torch._scaled_mm(aq,wq.t(),sa,sw,bias=m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
 if isinstance(y,tuple):y=y[0]
 return y[:rows].reshape(*x.shape[:-1],m.out_features)


def install(policy,c):
 spec=c['a8_scope_expansion'];approval=verified(spec['approval']);diff=verified(spec['protocol_diff'])
 assert approval['status']=='USER_APPROVED_EXACT_A8_SCOPE_EXPANSION' and 4 in approval['approved_stages']
 assert approval['approved_protocol_diff_sha256']==spec['protocol_diff']['sha256']
 stage=verified(diff['proposed_override'])['stages'][4]
 assert stage['stage']==4 and stage['total_FP8_projection_count']==408
 names=stage['additional_layers'];assert len(names)==104 and names==sorted(names)
 scales=policy.model._a8_weight_scales_ref;assert len(scales)==304
 from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear as Packed
 import a8_joint_pair_r1 as fp8
 assert fp8.ACTIVATION_OBSERVER is None
 q=Quantizer('fp8_e4m3fn');s=dict(validation_enabled=True,checks={n:0 for n in names},counts={n:0 for n in names},metadata={},shapes={n:{} for n in names})
 for n in names:
  m=policy.model.get_submodule(n)
  assert type(m) is Packed and m.bits==4 and m.group_size==64 and m.out_features%16==m.in_features%16==0
  payload={k:bitsha(getattr(m,k)) for k in ('packed','scales','offsets','bias')}
  w=m.reconstruct();qw,sw=q.quantize(w);m._a8_fast_weight_div=True;decoded=decode(m,sw)
  if not exact(decoded,qw):
   m._a8_fast_weight_div=False;del decoded;decoded=decode(m,sw)
  assert exact(decoded,qw),(n,'nativeFP8weightbytes')
  scales[n]=sw
  s['metadata'][n]=dict(payload_sha256=payload,original_weight_sha256=bitsha(w),fp8_weight_sha256=bitsha(qw),fp8_weight_scale_sha256=bitsha(sw),shape=[m.out_features,m.in_features],index_bits=m.bits,precise_weight_division=not m._a8_fast_weight_div)
  del w,qw,decoded
  def forward(this,x,_n=n,_m=m):
   s['counts'][_n]+=1
   if not s['validation_enabled']:return linear(x,_m,scales[_n])
   assert x.dtype==torch.bfloat16 and x.is_cuda and x.shape[-1]==_m.in_features
   a=x.reshape(-1,_m.in_features).contiguous();rows=a.shape[0];mp=(rows+15)//16*16
   aq,sa=q.quantize(a);padded=torch.zeros((mp,_m.in_features),dtype=aq.dtype,device=aq.device);padded[:rows].copy_(aq)
   wq=decode(_m,scales[_n]);reference=torch._scaled_mm(padded,wq.t(),sa,scales[_n],bias=_m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
   if isinstance(reference,tuple):reference=reference[0]
   reference=reference[:rows].reshape(*x.shape[:-1],_m.out_features)
   observed=[0]
   def observer(actual,actual_q,actual_s):
    assert exact(actual,a) and exact(actual_q,padded) and exact(actual_s,sa),(_n,'nativeAQ/Sa')
    observed[0]+=1
   assert fp8.ACTIVATION_OBSERVER is None;fp8.ACTIVATION_OBSERVER=observer
   try:y=linear(x,_m,scales[_n])
   finally:fp8.ACTIVATION_OBSERVER=None
   assert observed[0]==1 and exact(y,reference),(_n,'nativeFP8projectionoutput',tuple(x.shape))
   s['checks'][_n]+=1;shape=str(tuple(x.shape));s['shapes'][_n][shape]=s['shapes'][_n].get(shape,0)+1
   return y
  m.forward=types.MethodType(forward,m)
 assert len(scales)==408;policy.model._omni_a8_vision_state=s
 return dict(status='INSTALLED_PENDING_NATIVE_ACTUAL_TRAJECTORY_GATES',additional_layers=names,additional_count=104,total_FP8_projection_count=408,metadata=s['metadata'],payload_unchanged=True,persistent_expanded_weights=False,extra_scale_logical_bytes=416,implementation='OriginalLosslessAffineBF16rounding->E4M3direct; nativeFP32accum',performance_claim='InitialimplementationnotFP8ceiling')


def finish_validation(policy,c):
 s=policy.model._omni_a8_vision_state;expected=c['warmup']+c['num_iterations']
 assert expected==25 and len(s['checks'])==104 and set(s['checks'].values())=={25}
 s['validation_enabled']=False
 return dict(status='ALL2600_VISION_NATIVE_FP8_OPERANDS_SCALE_OUTPUT_EXACT',checks_per_layer=dict(s['checks']),actual_input_shapes=s['shapes'],validation_disabled_before_timing=True,persistent_expanded_weights=False)


def recheck_payloads(policy):
 s=policy.model._omni_a8_vision_state
 for n,md in s['metadata'].items():
  m=policy.model.get_submodule(n)
  for key,digest in md['payload_sha256'].items():assert bitsha(getattr(m,key))==digest,(n,key)
  assert bitsha(policy.model._a8_weight_scales_ref[n])==md['fp8_weight_scale_sha256']
 return dict(status='ALL104_VISION_PAYLOAD_AND_SCALE_IDENTITIES_UNCHANGED')
