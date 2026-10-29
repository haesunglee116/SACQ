"""Prepared accuracy-first FP8 or INT8 per-row candidate; neither auto-enabled.

Inspired by official TorchAO PerRow / ModelOpt per-channel-per-token recipes.
Retains SACQ compressed payloads; no persistent decoded/FP8 weight cache. The
zero-scale rule, RTN cast and E4M3fn range inherit a8_rtn_r2.Quantizer; only the
reduction axis changes. Requires an explicit future method contract and source-
verified ungrouped parent forwards; installing this alone cannot prove coverage.
"""
import types
import torch
from fused_ffup_cutlass_probe_r1b import bitsha as _tensor_bitsha

def bitsha(x):
    return None if x is None else _tensor_bitsha(x)


def quantize_rows(x,mode):
    assert x.ndim==2 and x.dtype==torch.bfloat16
    assert mode in ['fp8_e4m3fn','int8_symmetric']
    limit=448.0 if mode=='fp8_e4m3fn' else 127.0
    z=x.float();maximum=z.abs().amax(dim=1,keepdim=True)
    scale=torch.where(maximum>0,maximum/limit,torch.ones_like(maximum))
    value=(z/scale).clamp(-limit,limit)
    encoded=value.to(torch.float8_e4m3fn) if mode=='fp8_e4m3fn' else value.round().to(torch.int8)
    return encoded,scale


def install(policy,names,contract):
    mode=contract['projection_format'];assert mode in ['fp8_e4m3fn','int8_symmetric']
    assert contract['projection_recipe']=='dynamic_per_token_activation_per_output_row_weight'
    assert contract['persistent_expanded_weight_cache'] is False
    assert contract['source_payload']=='unchanged_SACQ_W1.5_plus_original_scalar_background'
    assert getattr(policy.model,'_row_candidate_parent_fusions_disabled',False),'Grouped/fused parent bodies must not bypass new module forwards'
    assert names==sorted(set(names)) and len(names)==441
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    from lossless_scalar_affine_runtime_r1 import LosslessAffineLinear as Affine
    import decoder_tuning
    from vllm import _custom_ops as ops
    assert torch.cuda.get_device_capability()==(8,9)
    if mode=='fp8_e4m3fn':assert torch.ops._C.cutlass_scaled_mm_supports_fp8(89),'No requested native SM89 FP8 backend; fail closed'
    state=dict(validation_enabled=True,forward_calls={name:0 for name in names},metadata={},scale_logical_bytes=0)
    for name in names:
        module=policy.model.get_submodule(name)
        if type(module) is Packed:
            decode=lambda m=module:decoder_tuning.dequantize(m)
            reference=module.dequantize_weight()
            fields=['packed_indices','codebook','scales','shift','bias']
        else:
            assert type(module) is Affine
            decode=module.reconstruct;reference=decode();fields=['packed','scales','offsets','bias']
        w=decode();assert bitsha(w)==bitsha(reference),(name,'OriginalBF16WeightBytes')
        weight_q,weight_scale=quantize_rows(w,mode)
        module.register_buffer('_a8_perrow_weight_scale',weight_scale)
        state['scale_logical_bytes']+=weight_scale.numel()*weight_scale.element_size()
        state['metadata'][name]=dict(payload_sha256={key:bitsha(getattr(module,key)) for key in fields},original_BF16_weight_sha256=bitsha(w),encoded8bit_weight_sha256=bitsha(weight_q),weight_row_scale_sha256=bitsha(weight_scale),shape=list(w.shape),scale_shape=list(weight_scale.shape),native_backend='vllm torch.ops._C.cutlass_scaled_mm; actual dispatch proof pending',projection_format=mode)
        del reference,w,weight_q,weight_scale
        def forward(this,x,_name=name,_module=module,_decode=decode):
            assert x.dtype==torch.bfloat16 and x.shape[-1]==_module.in_features
            a=x.reshape(-1,_module.in_features).contiguous();aq,sa=quantize_rows(a,mode)
            w=_decode();sw=_module._a8_perrow_weight_scale
            # Frozen weights: recompute only normalized FP8 bytes, scale stays
            # a small immutable row vector counted in resident/bit accounting.
            limit=448.0 if mode=='fp8_e4m3fn' else 127.0
            normalized=(w.float()/sw).clamp(-limit,limit)
            wq=normalized.to(torch.float8_e4m3fn) if mode=='fp8_e4m3fn' else normalized.round().to(torch.int8)
            if mode=='int8_symmetric':assert _module.in_features*127*127<=torch.iinfo(torch.int32).max
            del w
            y=ops.cutlass_scaled_mm(aq,wq.t(),sa,sw.t(),out_dtype=torch.bfloat16,bias=_module.bias)
            if state['validation_enabled']:
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                if state['forward_calls'][_name]==0:assert bitsha(wq)==state['metadata'][_name]['encoded8bit_weight_sha256']
                state['forward_calls'][_name]+=1
            return y.reshape(*x.shape[:-1],_module.out_features)
        module.forward=types.MethodType(forward,module)
    policy.model._perrow_a8_candidate_state=state
    return dict(status='INSTALLED_PENDING_FULL_NATIVE_DISPATCH_COVERAGE_RNG_AND_ACTION_GATES',module_count=441,scale_logical_bytes=state['scale_logical_bytes'],metadata=state['metadata'],persistent_expanded_weight_cache=False,new_training_or_calibration=False,native_speed_claim=False,accuracy_acceptance=False)


def finish_validation(policy):
    state=policy.model._perrow_a8_candidate_state
    assert len(state['forward_calls'])==441 and min(state['forward_calls'].values())>0,'Unexecuted requested module'
    for name,metadata in state['metadata'].items():
        module=policy.model.get_submodule(name)
        for key,digest in metadata['payload_sha256'].items():assert bitsha(getattr(module,key))==digest,(name,key)
        assert bitsha(module._a8_perrow_weight_scale)==metadata['weight_row_scale_sha256']
    state['validation_enabled']=False
    return dict(status='REQUESTED441_EXECUTED_PAYLOAD_AND_ROW_SCALE_IDENTITIES_PASS',counts=dict(state['forward_calls']),scale_logical_bytes=state['scale_logical_bytes'],production_dispatch_proof_required_separately=True)
