"""Fuse original affine-to-FP8 decode with original activation partial absmax."""
import torch
import triton
import triton.language as tl
from a8_joint_pair_r1 import _fp8_rtn
from a8_rtn_r2 import Quantizer
from omni_a8_scope_ffdown_runtime_r1 import exact,verified
from fused_ffup_cutlass_probe_r1b import bitsha


@triton.jit
def _affine_fp8_and_absmax(packed,scales,offsets,out,sw,a,partial,
                           COUNT:tl.constexpr,ACOUNT:tl.constexpr,BITS:tl.constexpr,
                           RECIP:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    blocks:tl.constexpr=triton.cdiv(COUNT,BLOCK)
    if pid<blocks:
        i=pid*BLOCK+tl.arange(0,BLOCK);valid=i<COUNT
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
    else:
        part=pid-blocks;activation_offsets=part*4096+tl.arange(0,4096)
        value=tl.load(a+activation_offsets,activation_offsets<ACOUNT,0).to(tl.float32)
        tl.store(partial+part,tl.max(tl.abs(value),0))


def linear(x,m,sw,debug):
    import a8_joint_pair_r1 as fp8
    from a8_dynamic_scale_r2 import _finish
    a=x.reshape(-1,m.in_features).contiguous();rows,k=a.shape;mp=(rows+15)//16*16
    assert rows in (128,512) and m.bits==4 and m.group_size==64
    n=m.out_features;count=n*k;parts=triton.cdiv(a.numel(),4096)
    wq=torch.empty((n,k),device=x.device,dtype=torch.float8_e4m3fn)
    aq=torch.empty((mp,k),device=x.device,dtype=torch.float8_e4m3fn)
    partial=torch.empty(parts,device=x.device,dtype=torch.float32);sa=torch.empty((),device=x.device,dtype=torch.float32)
    _affine_fp8_and_absmax[(triton.cdiv(count,1024)+parts,)](
        m.packed,m.scales,m.offsets,wq,sw,a,partial,count,a.numel(),m.bits,m._a8_fast_weight_div,1024,num_warps=4,enable_fp_fusion=False)
    # Preserve the existing large-input three-launch quantizer structure.
    if parts<=256:
        fp8._finish_quantize[(triton.cdiv(mp*k,1024),)](
            a,partial,aq,sa,a.numel(),mp*k,parts,triton.next_power_of_2(parts),1024,num_warps=4)
    else:
        _finish[(1,)](partial,sa,parts,triton.next_power_of_2(parts),num_warps=4)
        fp8._quantize_activation[(triton.cdiv(mp*k,1024),)](a,aq,sa,rows,k,mp,1024,num_warps=4)
    sha=bitsha(wq) if debug else None
    y=torch._scaled_mm(aq,wq.t(),sa,sw,bias=m.bias,out_dtype=torch.bfloat16,use_fast_accum=False)
    if isinstance(y,tuple):y=y[0]
    return y[:rows].reshape(*x.shape[:-1],n),aq,sa,sha


def install(policy,c):
    spec=c['vision_joint_fusion'];p=verified(spec['priority_manifest']);names=p['ranks'][5]['layers'];assert len(names)==104
    md=policy.model._omni_a8_vision_state['metadata'];assert names==sorted(md)
    scales=policy.model._a8_weight_scales_ref
    from omni_a8_scope_vision_runtime_r1b import decode
    s=dict(enabled=True,checks={n:0 for n in names},weights={},operand_checks=0)
    for n in names:
        m=policy.model.get_submodule(n);wq=decode(m,scales[n])
        assert bitsha(wq)==md[n]['fp8_weight_sha256'] and bitsha(scales[n])==md[n]['fp8_weight_scale_sha256']
        s['weights'][n]=md[n];del wq
        original=m.forward
        def forward(x,_n=n,_m=m,_original=original):
            reference=_original(x) if s['enabled'] else None
            y,aq,sa,sha=linear(x,_m,scales[_n],s['enabled'])
            if s['enabled']:
                a=x.reshape(-1,_m.in_features).contiguous();q,sw=Quantizer('fp8_e4m3fn').quantize(a)
                assert exact(aq,q) and exact(sa,sw) and sha==md[_n]['fp8_weight_sha256'],(_n,'FP8operands')
                assert exact(y,reference),(_n,'native_output')
                s['checks'][_n]+=1;s['operand_checks']+=1
            return y
        m.forward=forward
    policy.model._vision_joint_fusion_state=s
    return dict(status='INSTALLED_PENDING_ALL2600_VISION_NATIVE_OUTPUT_GATES',weights=s['weights'],
                original_FP8_primitive=True,original_per_matrix_scales=True,persistent_expanded_weights=False,
                joint_affine_decode_absmax_layers=104,large_activation_quantizer_structure_preserved=True,
                accuracy_acceptance=False,adoption=False)


def finish_validation(policy,c):
    s=policy.model._vision_joint_fusion_state;assert c['warmup']+c['num_iterations']==25
    assert len(s['checks'])==104 and set(s['checks'].values())=={25} and s['operand_checks']==2600
    s['enabled']=False
    return dict(status='ALL2600_VISION_JOINT_NATIVE_OUTPUT_AQ_SCALE_WEIGHT_CHECKS_EXACT',checks_per_layer=dict(s['checks']),
                operand_checks=2600,validation_disabled_before_timing=True,persistent_expanded_weights=False)
