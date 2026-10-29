"""Keep exact successful INT8 operands, fuse dynamic producers and packed decode."""
import types
import torch
import perrow_a8_candidate_r3 as original
import perrow_int8_padded_producer_r1 as producer
import int8_torch_epilogue_runtime_r2 as integer_mm
import perrow_int8_fused_decode_r1 as decoder

def install(policy,names,contract):
    assert contract['projection_format']=='int8_symmetric'
    baseline=original.install(policy,names,contract)
    state=policy.model._perrow_a8_candidate_state
    state['native_MMA_output_checks']={n:0 for n in names};state['fused_weight_checks']={};state['weight_division_modes']={};state['actual_activation_checks']={n:0 for n in names}
    from vllm import _custom_ops as ops
    for name in names:
        module=policy.model.get_submodule(name)
        assert module.in_features*127*127<=torch.iinfo(torch.int32).max
        candidate=decoder.decode(module,0)
        assert original.bitsha(candidate)==state['metadata'][name]['encoded8bit_weight_sha256'],('FusedPackedPerrowINT8WeightBytes',name)
        state['fused_weight_checks'][name]=True
        del candidate
        reciprocal=decoder.decode(module,2)
        mode=2 if original.bitsha(reciprocal)==state['metadata'][name]['encoded8bit_weight_sha256'] else 0
        state['weight_division_modes'][name]=mode
        del reciprocal
        def forward(this,x,_name=name,_module=module,_mode=mode):
            assert x.dtype==torch.bfloat16 and x.shape[-1]==_module.in_features
            a=x.reshape(-1,_module.in_features).contiguous()
            aq,sa=producer.activation(a,0);wq=decoder.decode(_module,_mode);sw=_module._a8_perrow_weight_scale
            y=integer_mm.linear(aq,wq,sa,sw,_module.bias,True,actual_rows=a.shape[0])
            if state['validation_enabled']:
                expected,scale=original.quantize_rows(a,'int8_symmetric')
                assert torch.equal(aq[:a.shape[0]],expected),('ActualActivationINT8Bytes',_name)
                assert torch.equal(sa[:a.shape[0]].view(torch.int32),scale.view(torch.int32)),('ActualActivationFP32RowScaleBits',_name)
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                native=ops.cutlass_scaled_mm(expected,wq.t(),scale,sw.t(),out_dtype=torch.bfloat16,bias=_module.bias)
                assert torch.equal(y.view(torch.int16),native.view(torch.int16)),('NativeScaledINT8MMAOutputBytes',_name)
                state['native_MMA_output_checks'][_name]+=1
                state['actual_activation_checks'][_name]+=1
                state['forward_calls'][_name]+=1
            return y.reshape(*x.shape[:-1],_module.out_features)
        module.forward=types.MethodType(forward,module)
    baseline.update(weight_division_modes=state['weight_division_modes'],fused_decode_INT8_weight_checks=state['fused_weight_checks'],fused_activation=True,division_instruction='correctly_rounded_FP32_div',rounding='nearbyint_ties_even',source_weight_rounding='unchanged_original_BF16_before_INT8',no_persistent_expanded_weights=True)
    return baseline

def finish_validation(policy):
    state=policy.model._perrow_a8_candidate_state
    assert len(state['fused_weight_checks'])==441 and all(state['fused_weight_checks'].values())
    assert state['actual_activation_checks']==state['forward_calls']
    assert state['native_MMA_output_checks']==state['forward_calls']
    result=original.finish_validation(policy)
    result.update(all_actual_native_MMA_output_bytes_exact=True,native_MMA_output_checks=dict(state['native_MMA_output_checks']),weight_division_modes=dict(state['weight_division_modes']),fused_weight_INT8_bytes_all441_exact=True,actual_activation_INT8_and_FP32_scale_checks=dict(state['actual_activation_checks']))
    return result
