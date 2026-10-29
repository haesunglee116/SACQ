"""Apply eligible common Language fusions to the optimized BF16 control too."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from omni_a8_scope_ffdown_runtime_r1 import exact


@triton.jit
def _bf16_rms_tail_only(X,VAR,W,Y,COUNT:tl.constexpr,K:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<COUNT,0).to(tl.float32)
    variance=tl.load(VAR+i//K,i<COUNT,0)
    normalized=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    weight=tl.load(W+i%K).to(tl.float32)
    tl.store(Y+i,(normalized*weight).to(tl.bfloat16),i<COUNT)


@triton.jit
def _bf16_silu_mul_only(G,U,Y,COUNT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<COUNT,0).to(tl.float32)
    u=tl.load(U+i,i<COUNT,0).to(tl.float32)
    silu=tl.div_rn(g,1.0+libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,(silu*u).to(tl.bfloat16),i<COUNT)


def install(policy,c):
    cfg=c['common_BF16_fusions'];s=dict(enabled=True,norm_checks={},silu_checks={})
    for index,layer in enumerate(policy.model.backbone.model.model.language_model.layers):
        if cfg['rms_tail']:
            for suffix in ['input_layernorm','post_attention_layernorm']:
                norm=getattr(layer,suffix);old=norm.forward;name=f'{index}.{suffix}';s['norm_checks'][name]=0
                def forward_norm(x,_norm=norm,_old=old,_name=name):
                    assert tuple(x.shape)==(1,156,2048) and x.dtype==torch.bfloat16
                    reference=_old(x) if s['enabled'] else None
                    if cfg['rms_square']:
                        from a8_rms_tail_square_runtime_r2 import _bf16_square_f32
                        square=torch.empty(x.shape,dtype=torch.float32,device=x.device)
                        _bf16_square_f32[(triton.cdiv(x.numel(),4096),)](x,square,x.numel(),4096,num_warps=4,enable_fp_fusion=False)
                        variance=square.mean(-1,keepdim=True);source=x
                        if s['enabled']:assert exact(square,x.float().pow(2)),(_name,'BF16controlsquares')
                    else:
                        source=x.float();variance=source.pow(2).mean(-1,keepdim=True)
                    y=torch.empty_like(x)
                    _bf16_rms_tail_only[(triton.cdiv(x.numel(),4096),)](source,variance,_norm.weight,y,x.numel(),2048,_norm.variance_epsilon,4096,num_warps=4,enable_fp_fusion=False)
                    if s['enabled']:
                        assert exact(y,reference),(_name,'BF16controlnativeRMS')
                        s['norm_checks'][_name]+=1
                    return y
                norm.forward=forward_norm
        if cfg['silu_mul']:
            mlp=layer.mlp;name=f'{index}.mlp';s['silu_checks'][name]=0
            def forward_mlp(x,_m=mlp,_name=name):
                gate=_m.gate_proj(x);up=_m.up_proj(x);y=torch.empty_like(gate)
                _bf16_silu_mul_only[(triton.cdiv(gate.numel(),4096),)](gate,up,y,gate.numel(),4096,num_warps=4,enable_fp_fusion=False)
                if s['enabled']:
                    reference=_m.act_fn(gate)*up
                    assert exact(y,reference),(_name,'BF16controlnativeSiLUmul')
                    s['silu_checks'][_name]+=1
                return _m.down_proj(y)
            mlp.forward=forward_mlp
    assert len(s['norm_checks'])==(32 if cfg['rms_tail'] else 0)
    assert len(s['silu_checks'])==(16 if cfg['silu_mul'] else 0)
    policy.model._common_BF16_fusion_state=s
    return dict(configuration=dict(cfg),FP8_casts_or_quantization=False,original_native_FP32_mean=True,original_BF16_roundings=True)


def finish_validation(policy,c):
    s=policy.model._common_BF16_fusion_state
    assert all(v==25 for key in ['norm_checks','silu_checks'] for v in s[key].values())
    s['enabled']=False
    return dict(status='ALL_SELECTED_BF16_COMMON_FUSIONS_EXACT',norm_checks=dict(s['norm_checks']),
                silu_checks=dict(s['silu_checks']),validation_disabled_before_timing=True,
                BF16_precision_preserved=True)
