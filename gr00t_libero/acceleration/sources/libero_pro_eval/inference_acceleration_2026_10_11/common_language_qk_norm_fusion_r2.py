"""Preserve native FP32 RMS mean and BF16 roundings in Language Q/K norms."""
import torch
import triton
import triton.language as tl
from a8_rms_tail_square_runtime_r2 import _bf16_square_f32
from common_bf16_language_fusions_r1 import _bf16_rms_tail_only
from omni_a8_scope_ffdown_runtime_r1 import exact


@triton.jit
def _strided_square(X,Y,COUNT:tl.constexpr,D1:tl.constexpr,D2:tl.constexpr,D3:tl.constexpr,
                    S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    offset=(i//(D1*D2*D3))*S0+(i//(D2*D3)%D1)*S1+(i//D3%D2)*S2+(i%D3)*S3
    x=tl.load(X+offset,i<COUNT,0).to(tl.float32)
    tl.store(Y+i,x*x,i<COUNT)


@triton.jit
def _strided_rms_tail(X,VAR,W,Y,COUNT:tl.constexpr,D1:tl.constexpr,D2:tl.constexpr,D3:tl.constexpr,
                     S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    offset=(i//(D1*D2*D3))*S0+(i//(D2*D3)%D1)*S1+(i//D3%D2)*S2+(i%D3)*S3
    x=tl.load(X+offset,i<COUNT,0).to(tl.float32)
    variance=tl.load(VAR+i//D3,i<COUNT,0)
    normalized=(x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    weight=tl.load(W+i%D3).to(tl.float32)
    tl.store(Y+i,(normalized*weight).to(tl.bfloat16),i<COUNT)


def install(policy, contract):
    state=dict(enabled=True,checks={},square_checks={},variance_checks={},input_layouts={})
    for name in contract['common_language_qk_norm']['modules']:
        norm=policy.model.get_submodule(name)
        original=norm.forward
        for key in ['checks','square_checks','variance_checks']:state[key][name]=0
        state['input_layouts'][name]=set()
        def forward(x,_norm=norm,_original=original,_name=name):
            assert x.dtype==torch.bfloat16 and x.ndim==4 and all(v>=0 for v in x.stride())
            if state['enabled']:state['input_layouts'][_name].add((tuple(x.shape),tuple(x.stride())))
            k=x.shape[-1]
            assert _norm.weight.numel()==k
            reference=_original(x) if state['enabled'] else None
            squares=torch.empty(x.shape,dtype=torch.float32,device=x.device)
            if x.is_contiguous():
                _bf16_square_f32[(triton.cdiv(x.numel(),4096),)](x,squares,x.numel(),4096,num_warps=4,enable_fp_fusion=False)
            else:
                _strided_square[(triton.cdiv(x.numel(),4096),)](x,squares,x.numel(),*x.shape[1:],*x.stride(),4096,num_warps=4,enable_fp_fusion=False)
            variance=squares.mean(-1,keepdim=True)
            y=torch.empty(x.shape,dtype=x.dtype,device=x.device)
            if x.is_contiguous():
                _bf16_rms_tail_only[(triton.cdiv(x.numel(),4096),)](x,variance,_norm.weight,y,x.numel(),k,_norm.variance_epsilon,4096,num_warps=4,enable_fp_fusion=False)
            else:
                _strided_rms_tail[(triton.cdiv(x.numel(),4096),)](x,variance,_norm.weight,y,x.numel(),*x.shape[1:],*x.stride(),_norm.variance_epsilon,4096,num_warps=4,enable_fp_fusion=False)
            if state['enabled']:
                expected_squares=x.float().pow(2)
                assert exact(squares,expected_squares),(_name,'NativeQKNormSquareChanged')
                assert exact(variance,expected_squares.mean(-1,keepdim=True)),(_name,'NativeQKNormFP32MeanChanged')
                assert exact(y,reference),(_name,'NativeQKNormBF16OutputChanged')
                for key in ['checks','square_checks','variance_checks']:state[key][_name]+=1
            return y
        norm.forward=forward
    policy.model._common_language_qk_norm_state=state
    return dict(status='INSTALLED_PENDING_ALL_ACTUAL_NATIVE_QK_RMS_VALUES',modules=contract['common_language_qk_norm']['modules'],
                native_FP32_mean=True,original_BF16_roundings=True,new_quantization=False)


def finish_validation(policy, contract):
    state=policy.model._common_language_qk_norm_state
    expected=contract['common_language_qk_norm']['calls_per_module_for_validation']
    for key in ['checks','square_checks','variance_checks']:assert state[key]==expected,(key,state[key],expected)
    state['enabled']=False
    return dict(status='ALL_ACTUAL_LANGUAGE_QK_RMS_FP32_SQUARE_MEAN_AND_BF16_OUTPUT_BYTES_EXACT',
                checks=state['checks'],square_checks=state['square_checks'],variance_checks=state['variance_checks'],
                validation_disabled_before_timing=True,input_layouts={n:sorted(v) for n,v in state['input_layouts'].items()},strided_sources_read_directly_without_copy=True)
