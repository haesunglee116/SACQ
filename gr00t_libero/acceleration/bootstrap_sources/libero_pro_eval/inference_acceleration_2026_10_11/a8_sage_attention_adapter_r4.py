"""Prepared adapter for the explicit 8+8 attention proposal; not auto-enabled.

Only selected module calls can use Sage. Original masks, scale, causal flag,
BF16 interfaces, GQA shapes and RNG are preserved. Never discard a mask.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F


def install(policy,c):
    spec=c['attention_A8_expansion']
    assert spec['approval']['status']=='USER_APPROVED_EXACT_ATTENTION_A8_EXPANSION'
    receipt=json.loads(Path(spec['runtime_receipt']['path']).read_text())
    assert hashlib.sha256(Path(spec['runtime_receipt']['path']).read_bytes()).hexdigest()==spec['runtime_receipt']['sha256']
    assert receipt['status']=='CPU_BUILD_COMPLETE_NO_GPU_NUMERICS'
    assert receipt['source_commit']==spec['source_commit']
    sys.path.insert(0,receipt['pythonpath'])
    import sageattention
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    assert str(Path(sageattention.__file__).resolve()).startswith(receipt['pythonpath']+'/')
    sys.path.insert(0,str(Path(__file__).resolve().parent/'a8_attention_expansion_goal_2026_10_18_r1'))
    from load_sage_raw_original_oracle_r1 import load as load_raw_original
    _current_native=sageattention.core.sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
    legacy=load_raw_original(spec['original_unpatched_runtime_receipt'])
    assert sageattention.core.sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf is _current_native
    legacy_sage=legacy.sageattn_qk_int8_pv_fp8_cuda
    original=F.scaled_dot_product_attention
    s=dict(enabled=True,active=[],checks={n:[] for n in spec['modules']},
           shapes={n:set() for n in spec['modules']},handles=[],call_counts={n:0 for n in spec['modules']},
           stream_oracle_checks={n:0 for n in spec['modules']},captured=set())
    for name in spec['modules']:
        module=policy.model.get_submodule(name)
        def before(module,args,_name=name):s['active'].append(_name)
        def after(module,args,output,_name=name):
            assert s['active'] and s['active'].pop()==_name
        s['handles'].append(module.register_forward_pre_hook(before))
        s['handles'].append(module.register_forward_hook(after,always_call=True))

    def sdpa(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
        if not s['active']:
            return original(q,k,v,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
        name=s['active'][-1]
        assert attn_mask is None,(name,'ExternalCUDA8+8doesnotacceptarbitrarymask; failclosed')
        assert dropout_p==0.0 and q.dtype==k.dtype==v.dtype==torch.bfloat16
        assert q.ndim==k.ndim==v.ndim==4 and q.shape[0]==k.shape[0]==v.shape[0]==1
        assert q.stride(-1)==k.stride(-1)==v.stride(-1)==1
        assert q.size(1)%k.size(1)==0 and k.shape==v.shape
        # Installed GR00T DiT config has head_dim48; the pinned external API
        # explicitly zero-pads dimensions below64 and crops its output back.
        assert q.size(-1) in (48,64,128) and (not is_causal or q.size(2)==k.size(2))
        kwargs=dict(spec['api_arguments'])
        assert kwargs==dict(tensor_layout='HND',qk_quant_gran='per_thread',
                            pv_accum_dtype='fp32+fp16',smooth_k=True,smooth_v=False,return_lse=False)
        kwargs.update(is_causal=is_causal,sm_scale=scale if scale is not None else q.size(-1)**-0.5)
        reference=original(q,k,v,None,0.0,is_causal,scale=scale,enable_gqa=enable_gqa) if s['enabled'] else None
        if s['enabled']:
            torch.cuda.synchronize()
            if name not in s['captured']:
                target=Path(c['output_directory'])/('actual_attention_inputs_'+name.replace('.','_')+'.pt')
                torch.save(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),kwargs=kwargs),target)
                s['captured'].add(name)
            with torch.cuda.stream(torch.cuda.default_stream(q.device)):
                legacy_output=legacy_sage(q,k,v,**kwargs)
            torch.cuda.synchronize()
            assert torch.isfinite(legacy_output).all(),(name,'OriginalDefaultStreamSageNonfinite')
        output=sageattn_qk_int8_pv_fp8_cuda(q,k,v,**kwargs)
        if s['enabled']:
            torch.cuda.synchronize()
            from omni_a8_scope_ffdown_runtime_r1 import exact
            assert exact(output,legacy_output),(name,'StreamRepairChangedSageOutputBytes')
            s['stream_oracle_checks'][name]+=1
        assert output.shape==q.shape and output.dtype==q.dtype
        if s['enabled']:
            assert torch.isfinite(output).all(),(name,'nonfinite_attention')
            delta=output.float()-reference.float()
            s['checks'][name].append(dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm()/reference.float().norm().clamp_min(1e-12)).item()))
            s['shapes'][name].add((tuple(q.shape),tuple(k.shape),tuple(v.shape),bool(is_causal),kwargs['sm_scale']))
            s['call_counts'][name]+=1
        return output

    F.scaled_dot_product_attention=sdpa
    s['original_sdpa']=original
    policy.model._attention_A8_expansion_state=s
    return dict(modules=list(spec['modules']),source_commit=spec['source_commit'],
                precision='INT8QK_FP8PV',api_arguments=dict(spec['api_arguments']),
                original_mask_causal_scale_preserved=True,softmax_not8bit=True,
                raw_pytorch_SDPA_globally_preserved_for_unselected_modules=True,
                persistent_weight_cache=False,accuracy_acceptance=False,adoption=False)


def finish_validation(policy,c):
    s=policy.model._attention_A8_expansion_state
    assert not s['active'] and set(s['checks'])==set(c['attention_A8_expansion']['modules'])
    expected=c['attention_A8_expansion']['calls_per_module_for_25queries']
    assert all(len(s['checks'][n])==expected[n] for n in s['checks']),s['call_counts']
    assert s['stream_oracle_checks']==expected,s['stream_oracle_checks']
    s['enabled']=False
    return dict(status='ALL_SELECTED_ATTENTION_OWN_INPUT_ERROR_MEASURED_NO_QUALITY_ACCEPTANCE',
                checks=dict(s['checks']),actual_shapes={k:sorted(v) for k,v in s['shapes'].items()},
                call_counts=dict(s['call_counts']),stream_repair_vs_original_default_stream_byte_checks=dict(s['stream_oracle_checks']),validation_disabled_before_timing=True,
                accuracy_acceptance=False,adoption=False)


def begin_projection_revalidation(policy,c):
    """Original implementation gates also cover the new quantized trajectory."""
    groups=[
        ('_self_qkv_fusion_r1_state','enabled',['checks','output_checks'],[]),
        ('_cross_kv_fusion_r2_state','enabled',['checks','output_checks'],
         ['compact_storage_checks','activation_checks','weight_byte_checks']),
        ('_language_joint_fusion_state','enabled',['checks'],['activation_checks','joint_weight_checks']),
        ('_vision_joint_fusion_state','enabled',['checks'],['operand_checks']),
        ('_omni_a8_scope_extra_state','validation_enabled',['checks'],[]),
        ('_omni_a8_modulation_state','validation_enabled',['checks'],[]),
        ('_omni_fp8_norm_producer_state','validation_enabled',
         ['checks','MMA_output_checks','residual_checks'],[]),
    ]
    optional=[('rms_tail_fusion','_rms_tail_fusion_state',['norm_checks','operand_checks','output_checks']),
              ('mask_bias_fusion','_mask_bias_fusion_state',['checks']),
              ('swiglu_fusion','_swiglu_fusion_state',['checks','weight_checks'])]
    for feature,attr,keys in optional:
        if feature=='rms_tail_fusion' and c['rms_square_fusion']['enabled_by_arm'][c['_arm']]:keys=keys+['square_checks']
        if c[feature]['enabled_by_arm'][c['_arm']]:groups.append((attr,'enabled',keys,[]))
    for attr,flag,keys,scalars in groups:
        state=getattr(policy.model,attr);assert not state[flag]
        for key in keys:state[key]={n:0 for n in state[key]}
        for key in scalars:state[key]=0
        state[flag]=True
    policy.model._omni_a8_modulation_state['shapes']={n:{} for n in policy.model._omni_a8_modulation_state['shapes']}
    # Forty-eight active attention projections still use the unfused native
    # wrapper. QKV and cross-KV wrappers have been superseded and are checked by
    # their current group gates above. Keep original-stage receipts separate.
    attention=policy.model._omni_a8_attention_state
    assert not attention['validation_enabled']
    policy.model._attention8_prior_native_checks=attention['checks']
    attention['checks']={n:0 for n in attention['checks']}
    expected={f'action_head.model.transformer_blocks.{i}.attn1.to_out.0' for i in range(32)}
    expected|={f'action_head.model.transformer_blocks.{i}.attn1.to_q' for i in range(0,32,2)}
    holders=[]
    for name in sorted(expected):
        module=policy.model.get_submodule(name);original=module.forward
        def forward(x,_original=original):
            assert not attention['validation_enabled']
            attention['validation_enabled']=True
            try:return _original(x)
            finally:attention['validation_enabled']=False
        module.forward=forward;holders.append((module,original))
    policy.model._attention8_native_remaining_holders=holders


def finish_projection_revalidation(policy,c):
    from a8_self_qkv_fusion_runtime_r3 import finish_validation as selfqkv
    from a8_cross_kv_native_joint_runtime_r4 import finish_validation as crosskv
    from a8_language_joint_runtime_r1 import finish_validation as language
    from a8_vision_joint_runtime_r1b import finish_validation as vision
    from omni_a8_scope_ffdown_runtime_r1 import finish_validation as ffdown
    from omni_a8_scope_modulation_runtime_r1b import finish_validation as modulation
    funcs=[('Modulation33',modulation),('SelfQKV',selfqkv),('CrossKV',crosskv),('Language',language),('Vision',vision),
           ('FFdown',ffdown)]
    # The actual R14 producer module is fixed in the parent's contract; its
    # adapter exposes finish_validation(policy,c), as used by the main worker.
    import importlib
    producer=importlib.import_module(c['lowbit_runtime_modules']['FP8']['producer'])
    funcs.append(('FFup',producer.finish_validation))
    out={name:fn(policy,c) for name,fn in funcs}
    attention=policy.model._omni_a8_attention_state
    expected={f'action_head.model.transformer_blocks.{i}.attn1.to_out.0' for i in range(32)}
    expected|={f'action_head.model.transformer_blocks.{i}.attn1.to_q' for i in range(0,32,2)}
    active={n:v for n,v in attention['checks'].items() if v}
    assert set(active)==expected and set(active.values())=={100},active
    out['remaining_Attention48']=dict(status='ALL4800_CURRENT_NATIVE_ATTENTION_PROJECTION_CHECKS_ON_EXPANDED_INPUTS_EXACT',checks=active)
    attention['validation_enabled']=False
    attention['checks']=policy.model._attention8_prior_native_checks
    for module,original in policy.model._attention8_native_remaining_holders:module.forward=original
    del policy.model._attention8_native_remaining_holders
    del policy.model._attention8_prior_native_checks
    rms_module='a8_rms_tail_square_runtime_r2' if c['rms_square_fusion']['enabled_by_arm'][c['_arm']] else 'a8_rms_tail_joint_runtime_r1'
    for feature,mod in [('rms_tail_fusion',rms_module),
                        ('mask_bias_fusion','a8_attention_mask_bias_runtime_r1'),
                        ('swiglu_fusion','a8_swiglu_joint_runtime_r1b')]:
        if c[feature]['enabled_by_arm'][c['_arm']]:
            out[feature]=importlib.import_module(mod).finish_validation(policy,c)
    return dict(status='ALL_CURRENT_PROJECTION_AND_PRODUCER_GATES_ON_EXPANDED_TRAJECTORY_PASS',gates=out)
