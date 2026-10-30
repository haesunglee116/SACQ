"""Fuse the installed Vision RoPE's FP32 operations with original BF16 rounding."""
import torch
import triton
import triton.language as tl
from omni_a8_scope_ffdown_runtime_r1 import exact


@triton.jit
def _vision_rotary_qk(Q,K,C,S,QO,KO,
                     Q0:tl.constexpr,Q1:tl.constexpr,K0:tl.constexpr,K1:tl.constexpr,
                     C0:tl.constexpr,C1:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
                     N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,TILES:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    j=(pid%TILES)*BLOCK+tl.arange(0,BLOCK)
    dim=j%D
    head=(j//D)%H
    seq=j//(H*D)
    partner=(dim+D//2)%D
    if pid<TILES:
        x=tl.load(Q+seq*Q0+head*Q1+dim,seq<N,0).to(tl.float32)
        other=tl.load(Q+seq*Q0+head*Q1+partner,seq<N,0).to(tl.float32)
    else:
        x=tl.load(K+seq*K0+head*K1+dim,seq<N,0).to(tl.float32)
        other=tl.load(K+seq*K0+head*K1+partner,seq<N,0).to(tl.float32)
    cos=tl.load(C+seq*C0+dim*C1,seq<N,0).to(tl.float32)
    sin=tl.load(S+seq*S0+dim*S1,seq<N,0).to(tl.float32)
    rotated=tl.where(dim<D//2,-other,other)
    # Original PyTorch performs separate FP32 multiplies and FP32 add.
    result=(x*cos)+(rotated*sin)
    if pid<TILES:
        tl.store(QO+j,result.to(tl.bfloat16),seq<N)
    else:
        tl.store(KO+j,result.to(tl.bfloat16),seq<N)


def install(policy, contract):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    native=qwen.apply_rotary_pos_emb_vision
    state=dict(enabled=True,active=[],checks={},handles=[])
    names=contract['common_vision_rotary']['modules']
    for name in names:
        module=policy.model.get_submodule(name)
        state['checks'][name]=0
        def before(module,args,_name=name):state['active'].append(_name)
        def after(module,args,output,_name=name):
            assert state['active'] and state['active'].pop()==_name
        state['handles'].append(module.register_forward_pre_hook(before))
        state['handles'].append(module.register_forward_hook(after,always_call=True))

    def rotary(q,k,cos,sin):
        assert q.dtype==k.dtype==torch.bfloat16 and q.shape==k.shape and q.ndim==3
        n,h,d=q.shape
        assert q.stride(-1)==k.stride(-1)==1 and d%2==0
        assert cos.shape==sin.shape==(n,d)
        qo=torch.empty(q.shape,dtype=q.dtype,device=q.device)
        ko=torch.empty(k.shape,dtype=k.dtype,device=k.device)
        tiles=triton.cdiv(q.numel(),1024)
        _vision_rotary_qk[(2*tiles,)](q,k,cos,sin,qo,ko,*q.stride()[:2],*k.stride()[:2],
            *cos.stride(),*sin.stride(),n,h,d,tiles,1024,num_warps=4,enable_fp_fusion=False)
        if state['enabled']:
            expected_q,expected_k=native(q,k,cos,sin)
            assert exact(qo,expected_q) and exact(ko,expected_k),'VisionRotaryChangedNativeOutputBytes'
            assert state['active']
            state['checks'][state['active'][-1]]+=1
        return qo,ko

    qwen.apply_rotary_pos_emb_vision=rotary
    policy.model._common_vision_rotary_state=state
    return dict(status='INSTALLED_PENDING_ALL_ACTUAL_VISION_ROTARY_NATIVE_BYTES',
                modules=names,original_FP32_multiplies_and_add=True,original_BF16_output_rounding=True,
                new_quantization=False,weight_cache=False)


def finish_validation(policy, contract):
    state=policy.model._common_vision_rotary_state
    assert not state['active']
    expected=contract['common_vision_rotary']['calls_per_module_for_validation']
    assert state['checks']==expected,(state['checks'],expected)
    state['enabled']=False
    return dict(status='ALL_ACTUAL_VISION_ROTARY_NATIVE_Q_AND_K_BYTES_EXACT',
                checks=state['checks'],validation_disabled_before_timing=True)
