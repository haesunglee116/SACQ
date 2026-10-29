"""Install the final opt-in kernel recipe before the compatible INT8 bootstrap.

This hooks the original installer; it does not build/checkpoint a policy, select
samples, launch a job, or certify a relocated GPU implementation.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .prepare import ROOT, verify_prepared

_CONFIGURED = False
BRIDGE_SHA256 = 'f0dfab43a002f6e8d36d5f729092e44bdf4ca4014b10af12a344b4986df8f19e'


def configure(workspace, bootstrap_contract, trace_directory, bridge):
    """Return kernel validation counters; bootstrap must subsequently run its gates.

Use in a fresh process, under torch.inference_mode(), before bridge.build().
The bridge must be the pinned per-row INT8 r5 bridge loaded as rollout_bridge_r1.
"""
    global _CONFIGURED
    if _CONFIGURED:
        raise RuntimeError('One overlay per process; start a fresh process after failure')
    if not __debug__:
        raise RuntimeError('Python -O disables numerical guards and is unsupported')
    workspace = Path(workspace).resolve()
    receipt = verify_prepared(workspace)
    bridge_path = Path(bridge.__file__).resolve()
    if hashlib.sha256(bridge_path.read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError('Compatible pinned INT8 r5 bridge required')
    if sys.modules.get('rollout_bridge_r1') is not bridge:
        raise ValueError('Load bridge as rollout_bridge_r1 before configuring kernels')
    preset = json.loads((ROOT / 'presets/final_w1p5a8.json').read_text())
    if bootstrap_contract.get('perrow_method') != preset['projection']:
        raise ValueError('Projection configuration differs from the measured recipe')
    if bootstrap_contract.get('accuracy_recovery', {}).get('method') != preset['accuracy_recovery_method']:
        raise ValueError('Attention/numerical configuration differs from the measured recipe')
    import torch
    if torch.is_grad_enabled():
        raise RuntimeError('Inference-only; enter torch.inference_mode() first')
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (8, 9):
        raise RuntimeError('This frozen kernel recipe targets SM89')
    # Do not silently combine old loaded modules with the staged implementation.
    modules = {Path(row['file']).stem: workspace / row['file']
               for row in receipt['sources'] if row['file'].endswith('.py')}
    for name, path in modules.items():
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path.resolve():
            raise RuntimeError(f'Conflicting loaded module: {name}; use a fresh process')
    runtime = workspace / 'libero_pro_eval/inference_acceleration_2026_10_11'
    sys.path[:0] = [str(runtime / 'a8_accuracy_recovery_2026_10_20_r1'),
                   str(runtime), str(workspace / 'gr00t_final_seed17_20260922'),
                   str(workspace / 'libero_pro_eval')]
    _CONFIGURED = True
    root = Path(trace_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    a = SimpleNamespace(order='integration', condition='int8_fused')
    c = receipt['selections']
    bc = bootstrap_contract
    import int8_ordered_epilogue_20261028_r1 as ordered
    import int8_torch_epilogue_runtime_r2 as runtime
    runtime._epilogue=ordered._epilogue
    import codex_gemm_stage6_20261029_r1 as stage6
    stage6_summary=stage6.enable()
    import int8_small_m_frozen_20261028_r2 as small_m
    small_m.enable(c['tile_selection'])
    import int8_large_m_frozen_20261028_r1 as large_m
    large_m.enable(c['large_tile_selection'])
    import int8_splitk_frozen_20261028_r1 as splitk
    splitk.enable(c['splitk_selection'])
    import perrow_int8_fast_policy_r3 as fast
    import int8_refined_policy_20261028_r1 as refined
    import int8_direct_integer_round_20261028_r1 as direct_round
    (root/(a.order+'_'+a.condition+'_DIRECT_ROUND_EDGE_GATE.json')).write_text(json.dumps(direct_round.validate_edges(),indent=2))
    import int8_aligned_index_decoder_20261028_r1 as packed_decoder
    import int8_aligned_index_unclamped_20261028_r1 as unclamped
    import int8_static_zero_affine_20261028_r1 as zero_shift
    packed_decoder._affine_bf16x4=zero_shift._affine_bf16x4
    unclamped._affine_bf16x4=zero_shift._affine_bf16x4
    packed_decoder._decode_codebook_perrow_int8=unclamped._decode_codebook_perrow_int8
    packed_decoder._decode_affine_perrow_int8=unclamped._decode_affine_perrow_int8
    import codex_affine_vec8rcpftz_integration_20261029_r1 as affine_vec8
    affine_vec8_summary=affine_vec8.enable(packed_decoder)
    import codex_codebook_vec4rcpftz_integration_20261029_r2 as generic_vec4
    generic_vec4_summary=generic_vec4.enable(packed_decoder)
    refined.decoder=packed_decoder
    import int8_aligned_index_frozen_20261028_r1 as frozen_decoder
    frozen_decoder.install(c['decode_selection'])
    import int8_aligned_index_retuned_frozen_20261028_r1 as retuned_decoder
    retuned_decoder.install(c['direct_decode_selection'])
    import int8_aligned_index_retuned_frozen_20261028_r1 as aligned_tuner
    aligned_tuner.install(c['aligned_retimed_selection'])
    fast.install=refined.install
    import int8_activation_frozen_20261028_r1 as frozen_activation
    import int8_direct_round_producer_20261028_r1 as direct_producer
    import perrow_int8_padded_producer_r1 as original_producer
    original_producer._activation=direct_producer._activation
    frozen_activation.enable(c['activation_selection'])
    import int8_shared_padded_activation_20261028_r1 as shared
    sharing=shared.enable()
    import int8_grouped_qkv_aligned_frozen_20261028_r1 as grouped_qkv
    import int8_grouped_aligned_decode_frozen_20261028_r1 as grouped_decoder
    import int8_grouped_aligned_unclamped_20261028_r1 as grouped_unclamped
    grouped_decoder._decode_grouped_qkv_weights=grouped_unclamped._decode_grouped_qkv_weights
    grouped_decoder.install(c['grouped_decode_selection'])
    import codex_qkv_vec4rcpftz_integration_20261029_r1 as qkv_vec4
    qkv_vec4_summary=qkv_vec4.enable(grouped_decoder)
    grouping=grouped_qkv.enable()
    import int8_grouped_gateup_20261028_r1 as grouped_gateup
    gateup_grouping=grouped_gateup.enable()
    import codex_gelu_lookup_w32_20261028_r1 as gelu_fusion
    gelu_summary=gelu_fusion.enable(c['activation_selection'])
    import int8_swiglu_lookup_fusion_20261028_r1 as swiglu_fusion
    swiglu_summary=swiglu_fusion.enable(c['activation_selection'])
    import codex_norm3_regpack_fusion_20261029_r1 as norm3_producer
    norm3_summary=norm3_producer.enable()
    import codex_adaln_sharedmax_fusion_20261029_r2 as adaln_producer
    adaln_summary=adaln_producer.enable()
    import int8_ffdown_residual_epilogue_20261028_r2 as ffdown_fusion
    ffdown_summary=ffdown_fusion.enable()
    import int8_sage_mean_parallel_preparation_20261028_r1 as unpadded_mean
    unpadded_mean.enable()
    import codex_aligned_mask_bias_20261028_r2 as mask_bias
    mask_summary=mask_bias.enable(bc)
    import codex_ffdecode_vec4cta128_integration_20261029_r1 as grouped_ffdecode
    ffdecode_summary=grouped_ffdecode.enable()
    import codex_language_rms_20261028_r1 as native_rms_tail
    rms_summary=native_rms_tail.enable(bc)
    import int8_vision_norm_producer_20261028_r1 as vision_norm
    vision_norm_summary=vision_norm.enable(bc)
    import claude_int8_attention_path_20261028_r1 as attn_path
    attn_path_summary=attn_path.enable(c1=True,c2=False,c3=True,tile_selection=c['tile_selection'],activation_selection=c['activation_selection'])
    import codex_qknorm_rope_20261028_r1 as qk_rope
    qknorm_summary,rope_summary=qk_rope.enable()
    import codex_vision_sdpa_view_20261028_r1 as vision_view
    vision_view_summary=vision_view.enable(bc)
    import codex_vision_segment_producer_20261029_r3 as segment_producer
    segment_summary=segment_producer.enable(bc)
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
    previous_zero_guard_install=fast.install
    zero_shift_rows={}
    def guarded_zero_install(policy,names,contract):
     for name in names:
      m=policy.model.get_submodule(name)
      if type(m) is Packed:
       assert m.shift.numel()==1,('NonScalarShift',name)
       value=float(m.shift.item())
       assert value==0.0,('NonzeroShiftForbidden',name,value)
       zero_shift_rows[name]=value
     assert zero_shift_rows,'NoPackedZeroShiftModules'
     return previous_zero_guard_install(policy,names,contract)
    fast.install=guarded_zero_install

    return {
        'stage6_summary': stage6_summary,
        'affine_vec8_summary': affine_vec8_summary,
        'generic_vec4_summary': generic_vec4_summary,
        'qkv_vec4_summary': qkv_vec4_summary,
        'sharing': sharing,
        'grouping': grouping,
        'gateup_grouping': gateup_grouping,
        'gelu_summary': gelu_summary,
        'swiglu_summary': swiglu_summary,
        'norm3_summary': norm3_summary,
        'adaln_summary': adaln_summary,
        'ffdown_summary': ffdown_summary,
        'mask_summary': mask_summary,
        'ffdecode_summary': ffdecode_summary,
        'rms_summary': rms_summary,
        'vision_norm_summary': vision_norm_summary,
        'attn_path_summary': attn_path_summary,
        'qknorm_summary': qknorm_summary,
        'rope_summary': rope_summary,
        'vision_view_summary': vision_view_summary,
        'segment_summary': segment_summary,
        'zero_shift_rows': zero_shift_rows,
    }
