"""Prepared FP32+FP32 exact Sage preparation: pad/QK/subtract/V in one launch.

Original arithmetic from SageAttention Apache-2.0 component sources; all actual
native operands and original Sage output bytes must pass before timing.
"""
import torch
import json
from pathlib import Path
_MEAN_DIAGNOSTICS=[]
_MEAN_DIAGNOSTIC_ROOT=None
import triton
import triton.language as tl

@triton.jit
def _joint_qk_padded(Q,K,KM,QO,KO,QS,KS,
                   Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
                   K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                   M0:tl.constexpr,M1:tl.constexpr,
                   NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                   D:tl.constexpr,QP:tl.constexpr,KP:tl.constexpr,QT:tl.constexpr,DO:tl.constexpr,RNK:tl.constexpr,INTERNAL_MEAN:tl.constexpr):
    pid=tl.program_id(0)
    if pid<QT:
        qb=QP//8
        p0=pid%qb
        bh=pid//qb;b=bh//HQ;h=bh%HQ
        qr=tl.arange(0,4);qg=tl.arange(0,8);qd=tl.arange(0,D)
        ns=p0*32+qr[:,None]*8+qg[None,:]
        x=tl.load(Q+b*Q0+h*Q1+ns[:,:,None]*Q2+qd[None,None,:],mask=(ns[:,:,None]<NQ)&(qd[None,None,:]<DO),other=0).to(tl.float32)
        scale=tl.max(tl.max(tl.abs(x),2),0)/127.+0.0000001
        qi=x/scale[None,:,None]
        qi+=0.5*tl.where(qi>=0,1,-1)
        tl.store(QO+(b*HQ+h)*NQ*D+ns[:,:,None]*D+qd[None,None,:],qi.to(tl.int8),mask=ns[:,:,None]<NQ)
        tl.store(QS+(b*HQ+h)*QP+p0*8+qg,scale)
    else:
        kpid=pid-QT;kbp=KP//4;p0=kpid%kbp
        kbh=kpid//kbp;kb=kbh//HK;kh=kbh%HK
        kd=tl.arange(0,D)
        if INTERNAL_MEAN:
            mean_n=tl.arange(0,RNK)
            mean_x=tl.load(K+kb*K0+kh*K1+mean_n[:,None]*K2+kd[None,:],mask=(mean_n[:,None]<NK)&(kd[None,:]<DO),other=0).to(tl.float32)
            km=(tl.sum(mean_x,0)/NK).to(tl.bfloat16).to(tl.float32)
            if p0==0:tl.store(KM+kb*M0+kh*M1+kd,km.to(tl.bfloat16))
        else:
            km=tl.load(KM+kb*M0+kh*M1+kd,mask=kd<DO,other=0).to(tl.float32)
        kr=tl.arange(0,8);kg=tl.arange(0,4);pair=tl.arange(0,2)
        kn=p0*64+kr[:,None,None]*8+kg[None,:,None]*2+pair[None,None,:]
        raw=tl.load(K+kb*K0+kh*K1+kn[:,:,:,None]*K2+kd[None,None,None,:],mask=(kn[:,:,:,None]<NK)&(kd[None,None,None,:]<DO),other=0).to(tl.float32)
        kx=tl.where(kn[:,:,:,None]<NK,(raw-km[None,None,None,:]).to(tl.bfloat16).to(tl.float32),0.)
        scale_k=tl.max(tl.max(tl.max(tl.abs(kx),3),2),0)/127.+0.0000001
        ki=kx/scale_k[None,:,None,None]
        ki+=0.5*tl.where(ki>=0,1,-1)
        tl.store(KO+(kb*HK+kh)*NK*D+kn[:,:,:,None]*D+kd[None,None,None,:],ki.to(tl.int8),mask=kn[:,:,:,None]<NK)
        tl.store(KS+(kb*HK+kh)*KP+p0*4+kg,scale_k)


@triton.jit
def _v_flat_prepare(V,O,S,V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
                         N:tl.constexpr,D:tl.constexpr,PN:tl.constexpr,
                         RN:tl.constexpr,BD:tl.constexpr,MODE:tl.constexpr,DO:tl.constexpr,HK:tl.constexpr,VPID):
    vd=tl.cdiv(D,BD)
    dh=(VPID%vd)*BD+tl.arange(0,BD)
    h=(VPID//vd)%HK;b=VPID//(vd*HK)
    out_n=tl.arange(0,RN)
    # Exact inverse of the native shared-load sequence permutation.
    in_n=(out_n//16)*16+((out_n%16)//4)*2+((out_n//2)%2)*8+(out_n%2)
    x=tl.load(V+b*V0+h*V1+in_n[None,:]*V2+dh[:,None],
              mask=(dh[:,None]<DO)&(in_n[None,:]<N)&(out_n[None,:]<PN),other=0).to(tl.float32)
    amax=tl.max(tl.abs(x),1)
    # Pinned CUDA is compiled with --use_fast_math. Keep its actual division
    # and multiply instructions, including zero-channel NaN encoding.
    scale=tl.inline_asm_elementwise('div.approx.ftz.f32 $0, $1, $2;',constraints='=f,f,f',args=[amax,448.0],dtype=tl.float32,is_pure=True,pack=1)
    if MODE==0:
        recip=tl.inline_asm_elementwise('div.approx.ftz.f32 $0, $1, $2;',constraints='=f,f,f',args=[448.0,amax],dtype=tl.float32,is_pure=True,pack=1)
    elif MODE==1:
        inv=tl.inline_asm_elementwise('rcp.approx.ftz.f32 $0, $1;',constraints='=f,f',args=[amax],dtype=tl.float32,is_pure=True,pack=1)
        recip=inv*448.0
    elif MODE==2:
        recip=tl.div_rn(448.0,amax)
    else:
        recip=tl.div_rn(1.,amax)*448.0
    y=tl.inline_asm_elementwise('mul.ftz.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,recip[:,None]],dtype=tl.float32,is_pure=True,pack=1)
    packed=tl.inline_asm_elementwise('cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;',constraints='=h,f',args=[y],dtype=tl.uint16,is_pure=True,pack=1)
    z=packed.to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    tl.store(O+((b*HK+h)*D+dh[:,None])*PN+out_n[None,:],z,
             mask=(dh[:,None]<D)&(out_n[None,:]<PN))
    tl.store(S+(b*HK+h)*D+dh,scale,mask=dh<D)


@triton.jit
def _packed_qkv_preparation(Q,K,V,KM,QO,KO,QS,KS,VO,VS,
 Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
 V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,M0:tl.constexpr,M1:tl.constexpr,
 NQ:tl.constexpr,NK:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,DO:tl.constexpr,D:tl.constexpr,
 QP:tl.constexpr,KP:tl.constexpr,QT:tl.constexpr,QKT:tl.constexpr,PN:tl.constexpr,RN:tl.constexpr,RNK:tl.constexpr,INTERNAL_MEAN:tl.constexpr):
 pid=tl.program_id(0)
 if pid<QKT:
  _joint_qk_padded(Q,K,KM,QO,KO,QS,KS,Q0,Q1,Q2,K0,K1,K2,M0,M1,NQ,NK,HQ,HK,D,QP,KP,QT,DO,RNK,INTERNAL_MEAN)
 else:
  _v_flat_prepare(V,VO,VS,V0,V1,V2,NK,D,PN,RN,32,0,DO,HK,pid-QKT)


def prepare(q,k,v,internal_mean=True):
 assert q.dtype==k.dtype==v.dtype==torch.bfloat16
 b,hq,nq,do=q.shape;bk,hk,nk,dk=k.shape
 assert b==bk==1 and dk==do and tuple(v.shape)==tuple(k.shape) and do in (48,64,128)
 assert q.stride(-1)==k.stride(-1)==v.stride(-1)==1
 d=64 if do<64 else 128 if do>64 else 64
 qp=triton.cdiv(nq,128)*32;kp=triton.cdiv(nk,64)*4;qt=b*hq*(qp//8);qkt=qt+b*hk*(kp//4)
 qo=torch.empty((b,hq,nq,d),dtype=torch.int8,device=q.device);ko=torch.empty((b,hk,nk,d),dtype=torch.int8,device=k.device)
 qs=torch.empty((b,hq,qp),dtype=torch.float32,device=q.device);ks=torch.empty((b,hk,kp),dtype=torch.float32,device=k.device)
 pn=triton.cdiv(nk,64)*64;rn=triton.next_power_of_2(pn);rnk=triton.next_power_of_2(nk)
 vo=torch.empty((b,hk,d,pn),dtype=torch.float8_e4m3fn,device=v.device);vs=torch.empty((b,hk,d),dtype=torch.float32,device=v.device)
 if internal_mean:km=torch.empty((b,hk,1,d),dtype=torch.bfloat16,device=k.device)
 else:
  from int8_sage_mean_parallel_y_20261028_r1 import mean
  km=mean(k)
  # Keep the exact native mean; masked kernel loads supply padded +0.
 _packed_qkv_preparation[(qkt+b*hk*triton.cdiv(d,32),)](q,k,v,km,qo,ko,qs,ks,vo,vs,*q.stride()[:3],*k.stride()[:3],*v.stride()[:3],*km.stride()[:2],nq,nk,hq,hk,do,d,qp,kp,qt,qkt,pn,rn,rnk,internal_mean,num_warps=4)
 return qo,qs,ko,ks,vo,vs,km


def operator(q,k,v,tensor_layout='HND',is_causal=False,qk_quant_gran='per_thread',sm_scale=None,pv_accum_dtype='fp32+fp32',smooth_k=True,smooth_v=False,return_lse=False,internal_mean=True,validate=False):
 assert tensor_layout=='HND' and qk_quant_gran=='per_thread' and pv_accum_dtype=='fp32+fp32'
 assert smooth_k is True and smooth_v is False and return_lse is False
 import sageattention.core as core
 torch.cuda.set_device(v.device);do=q.shape[-1]
 if sm_scale is None:sm_scale=do**-0.5
 qo,qs,ko,ks,vo,vs,km=prepare(q,k,v,internal_mean=internal_mean)
 if validate:
  from omni_a8_scope_ffdown_runtime_r1 import exact
  pad=qo.shape[-1]-do
  qq,kk,vv=[torch.nn.functional.pad(x,(0,pad)) if pad else x for x in (q,k,v)]
  refkm=kk.mean(dim=2,keepdim=True)
  checked_km=torch.nn.functional.pad(km,(0,refkm.shape[-1]-km.shape[-1])) if km.shape[-1]<refkm.shape[-1] else km
  assert exact(checked_km,refkm),('SinglePreparationMeanMismatch',q.shape)
  # Diagnose the rejected internal reduction separately. Production operands
  # always use the same torch.mean as the pinned official Sage implementation.
  legacy=None;legacy_record=None
  if not internal_mean:
   legacy=prepare(q,k,v,internal_mean=True)
   if not exact(legacy[-1],refkm):
    previous=legacy[-1];mask=previous.view(torch.int16)!=refkm.view(torch.int16)
    legacy_record=dict(shape=list(q.shape),mean_bit_mismatches=int(mask.sum()),mean_values_equal=bool(torch.equal(previous,refkm)),mean_max_abs=float((previous.float()-refkm.float()).abs().max()),old_vs_reference_prepared_operand_bytes=[exact(x,y) for x,y in zip(legacy[:-1],(qo,qs,ko,ks,vo,vs))])
    if _MEAN_DIAGNOSTIC_ROOT is not None and len(_MEAN_DIAGNOSTICS)<8:
     root=Path(_MEAN_DIAGNOSTIC_ROOT);root.mkdir(parents=True,exist_ok=True)
     cap=root/('mean_mismatch_'+str(len(_MEAN_DIAGNOSTICS))+'.pt')
     torch.save(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),internal_mean=previous.detach().cpu(),official_mean=refkm.detach().cpu()),cap)
     legacy_record['saved_input']=str(cap)

  refs=core.per_thread_int8_triton(qq,kk,refkm,tensor_layout='HND',BLKQ=128,WARPQ=32,BLKK=64,WARPK=64)
  for name,x,y in zip(('Qint8','Qscale','Kint8','Kscale'),(qo,qs,ko,ks),refs):assert exact(x,y),('SinglePreparationOperandMismatch',name,q.shape)
  refs=core.per_channel_fp8(vv,tensor_layout='HND',scale_max=448.0,smooth_v=False)
  for name,x,y in zip(('Vfp8','Vscale'),(vo,vs),refs[:2]):assert exact(x,y),('SinglePreparationOperandMismatch',name,q.shape)
 # Original FP32 backend requires contiguous padded HND output.
 o=torch.empty(qo.shape,dtype=q.dtype,device=q.device)
 core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(qo,ko,vo,o,qs,ks,vs,1,int(is_causal),3,sm_scale,0)
 if validate and legacy_record is not None:
  lq,lqs,lk,lks,lv,lvs,lkm=legacy
  old=torch.empty_like(o)
  core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(lq,lk,lv,old,lqs,lks,lvs,1,int(is_causal),3,sm_scale,0)
  legacy_record['old_vs_official_attention_output_bytes_exact']=exact(old,o)
  _MEAN_DIAGNOSTICS.append(legacy_record)
  if _MEAN_DIAGNOSTIC_ROOT is not None:
   p=Path(_MEAN_DIAGNOSTIC_ROOT)/'MEAN_REDUCTION_DIAGNOSTICS_R2.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(_MEAN_DIAGNOSTICS,indent=2));tmp.replace(p)
 return o[...,:do]


def install(policy,c):
 global _MEAN_DIAGNOSTIC_ROOT
 assert c['sage_single_preparation']['internal_mean'] is False
 _MEAN_DIAGNOSTIC_ROOT=str(Path(c['output_directory'])/'mean_diagnostics_r2')
 import sageattention
 original=sageattention.sageattn_qk_int8_pv_fp8_cuda
 state=dict(enabled=True,calls=0,operand_checks=0,output_checks=0,original_padded_HND_output_checks=0,original=original,recipe='fp32+fp32',V_quant_max=448.0)
 def wrapped(q,k,v,**kw):
  y=operator(q,k,v,**kw,internal_mean=c['sage_single_preparation']['internal_mean'],validate=state['enabled'])
  if state['enabled']:
   expected=original(q,k,v,**kw)
   from omni_a8_scope_ffdown_runtime_r1 import exact
   assert exact(y,expected),('SinglePreparationSageOutputMismatch',q.shape)
   # The returned 48D crop retains the original 64D HND stride.
   assert y.stride(-1)==1 and y.stride(-2)==(64 if q.shape[-1]<64 else q.shape[-1]),('OriginalHNDOutputStrideMismatch',y.shape,y.stride())
   state['original_padded_HND_output_checks']+=1
   state['calls']+=1;state['operand_checks']+=7;state['output_checks']+=1
  return y
 sageattention.sageattn_qk_int8_pv_fp8_cuda=wrapped
 policy.model._sage_single_preparation_state=state
 return dict(status='INSTALLED_PENDING_ALL_ACTUAL_NATIVE_OPERAND_AND_SAGE_OUTPUT_CHECKS',internal_mean=c['sage_single_preparation']['internal_mean'],scientific_delta={})


def finish_validation(policy,c):
 s=policy.model._sage_single_preparation_state
 expected=sum(c['attention_A8_expansion']['calls_per_module_for_25queries'].values())
 assert s['calls']==expected and s['operand_checks']==7*expected and s['output_checks']==expected
 s['enabled']=False
 return dict(status='ALL_ACTUAL_SINGLE_PREPARATION_MEAN_QK_V_AND_SAGE_OUTPUT_BYTES_EXACT',calls=s['calls'],operand_checks=s['operand_checks'],output_checks=s['output_checks'],original_padded_HND_output_checks=s['original_padded_HND_output_checks'],validation_disabled_before_timing=True)


def enable():
 import recovery_sage_single_preparation_fp32_r2 as original
 original.operator=operator
 return dict(native_mean_preserved=True,masked_zero_padding=True)
