"""Candidate p2_rescale_sagecrop: control runtime plus exact attention-path eliminations c1=True c2=False c3=True (see claude_int8_attention_path_20261028_r1)."""
import argparse,hashlib,importlib.util,json,pickle,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent;W=R.parents[1];OLD=W/'gr00t_final_seed17_20260922';B=W/'libero_pro_eval'
sys.path[:0]=[str(D),str(R),str(OLD),str(B),str(B/'Isaac-GR00T-N1.7'),str(B/'Isaac-GR00T-N1.7/scripts/deployment')]
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--order',required=True);ap.add_argument('--condition',required=True);a=ap.parse_args();c=json.loads(Path(a.protocol).read_text());spec=c['condition_bootstraps'][a.order][a.condition];bc=json.loads(Path(spec['path']).read_text());assert ref(spec['path'])==spec
 root=Path(c['output_directory']);fixture=json.loads((root/'fixture.json').read_text());assert fixture['observation']==ref(fixture['observation']['path'])
 observation=pickle.loads(Path(fixture['observation']['path']).read_bytes())
 import numpy as np
 import torch
 from fixed_episode_noise_2026_09_09 import read_episode
 from omni_a8_all_rng_snapshot_r1 import snapshot
 import benchmark_inference as official
 bridge_spec=importlib.util.spec_from_file_location('rollout_bridge_r1',D/('rollout_bridge_perrow_int8_r5.py' if a.condition in ('int8_fused','int8_reference') else 'rollout_bridge_perrow_int8_r2.py'));bridge=importlib.util.module_from_spec(bridge_spec);sys.modules['rollout_bridge_r1']=bridge;bridge_spec.loader.exec_module(bridge)
 noise=read_episode(Path(c['canonical_noise_directory']),0)[0]
 batch={key:(value,) if isinstance(value,str) else np.stack([value]) for key,value in observation.items()}
 torch.set_num_threads(2);stream=torch.cuda.Stream()
 with torch.cuda.stream(stream),torch.inference_mode():
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
  bundle=bridge.build(spec['path'],batch,noise);policy=bundle['policy'];bridge.noise_proxy.select(noise)
  (root/'orders'/a.order/(a.condition+'_static_zero_shift_coverage.json')).write_text(json.dumps(zero_shift_rows,indent=2))
  obs=bridge.nested(batch,policy);inp=official.prepare_model_inputs(policy,obs);bi,ai=policy.model.prepare_input(inp)
  mask_counts=mask_summary['state']['checks']
  assert mask_summary['state']['validated_cache_hits']==1400 and mask_summary['state']['validated_cache_misses']==200,mask_summary['state']
  (root/'orders'/a.order/(a.condition+'_shared_mask_cache.json')).write_text(json.dumps({k:v for k,v in mask_summary['state'].items() if k not in ('checks','shapes')},indent=2))
  assert len(mask_counts)==16 and set(mask_counts.values())=={100},mask_counts
  (root/'orders'/a.order/(a.condition+'_cross_mask_bias_checks.json')).write_text(json.dumps(mask_counts,indent=2))
  assert len(ffdecode_summary['reciprocal_checks'])==64 and all(x['reciprocal_bits_exact'] for x in ffdecode_summary['reciprocal_checks']),ffdecode_summary
  assert ffdecode_summary['weight_checks']>=6400,ffdecode_summary
  (root/'orders'/a.order/(a.condition+'_grouped_ffdecode.json')).write_text(json.dumps(ffdecode_summary,indent=2))
  assert len(rms_summary['checks'])==32 and set(rms_summary['checks'].values())=={25},rms_summary
  assert rms_summary['checks']==rms_summary['squares']
  (root/'orders'/a.order/(a.condition+'_native_rms_tail_checks.json')).write_text(json.dumps(rms_summary,indent=2))
  assert vision_norm_summary['native_norm_checks']==1200 and vision_norm_summary['validated_hits']==1200,vision_norm_summary
  (root/'orders'/a.order/(a.condition+'_vision_norm_checks.json')).write_text(json.dumps(vision_norm_summary,indent=2))
  assert norm3_summary['validated_hits']>0
  (root/'orders'/a.order/(a.condition+'_norm3_int8_validation.json')).write_text(json.dumps(norm3_summary,indent=2))
  (root/'orders'/a.order/(a.condition+'_adaln_int8_validation.json')).write_text(json.dumps(adaln_summary,indent=2))
  assert adaln_summary['validated_hits']==3200,adaln_summary
  assert stage6_summary['byte_checks']>0,stage6_summary
  (root/'orders'/a.order/(a.condition+'_stage6_checks.json')).write_text(json.dumps(stage6_summary,indent=2))
  assert sharing['validated_hits']>0
  assert gelu_summary['actual_output_checks']>0
  assert swiglu_summary['actual_output_checks']>0
  (root/(a.order+'_'+a.condition+'_SWIGLU_LOOKUP_VALIDATION.json')).write_text(json.dumps(swiglu_summary,indent=2))
  (root/(a.order+'_'+a.condition+'_GELU_LOOKUP_VALIDATION.json')).write_text(json.dumps(gelu_summary,indent=2))
  (root/'orders'/a.order/(a.condition+'_shared_activation.json')).write_text(json.dumps(sharing,indent=2))
  assert affine_vec8_summary['checks']>=104,affine_vec8_summary
  (root/'orders'/a.order/(a.condition+'_affine_vec8_checks.json')).write_text(json.dumps(affine_vec8_summary,indent=2))
  assert generic_vec4_summary['checks']>=288,generic_vec4_summary
  (root/'orders'/a.order/(a.condition+'_generic_vec4_checks.json')).write_text(json.dumps(generic_vec4_summary,indent=2))
  assert qkv_vec4_summary['group_checks']>=80,qkv_vec4_summary
  (root/'orders'/a.order/(a.condition+'_qkv_vec4_checks.json')).write_text(json.dumps(qkv_vec4_summary,indent=2))
  rawref=bundle['bootstrap']['whole_model_graph_validation']['actual_actions'];references=np.load(rawref['path']);assert rawref==ref(rawref['path']);assert len(references)==25 and np.array_equal(references,np.repeat(references[:1],25,axis=0))
  original=c['reference_raw_int8'];assert ref(original['path'])==original
  original_values=np.load(original['path']);assert np.array_equal(references.view(np.uint32),original_values.view(np.uint32)),'Fused producer/decode changedwhole-model INT8referencebytes'
  expected=references[0];owner=policy.model._whole_model_graph_owner;before_calls=owner['calls'];official.set_seed(bc['seed']);rng=snapshot();wall=[];gpu=[]
  from gpu_clock_sampler_r1 import ClockSampler
  sampler=ClockSampler();sampler.start()
  for query in range(bc['warmup']+bc['num_iterations']):
   torch.cuda.synchronize();start=time.perf_counter();y=owner['call'](bi,ai);torch.cuda.synchronize();duration=(time.perf_counter()-start)*1000
   assert np.array_equal(y.float().cpu().numpy().view(np.uint32),expected.view(np.uint32));assert snapshot()==rng
   if query>=bc['warmup']:wall.append(duration)
  e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
  for query in range(bc['warmup']+bc['num_iterations']):
   e0.record(stream);y=owner['call'](bi,ai);e1.record(stream);e1.synchronize();duration=e0.elapsed_time(e1)
   assert np.array_equal(y.float().cpu().numpy().view(np.uint32),expected.view(np.uint32));assert snapshot()==rng
   if query>=bc['warmup']:gpu.append(duration)
  clocks=sampler.stop();assert len(wall)==len(gpu)==20 and owner['calls']-before_calls==50
  from torch.profiler import profile,ProfilerActivity
  from collections import Counter
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
   y=owner['call'](bi,ai);torch.cuda.synchronize()
  assert np.array_equal(y.float().cpu().numpy().view(np.uint32),expected.view(np.uint32));assert snapshot()==rng
  trace=root/'orders'/a.order/(a.condition+'_dispatch.json');trace.parent.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];hist=Counter(e['name'] for e in events if e.get('cat')=='kernel' and e.get('ph')=='X')
  native=2*hist.get('_int8_grouped_gateup',0)+3*hist.get('_int8_grouped_qkv',0)+sum(v for n,v in hist.items() if ('cutlass::Kernel2' in n and 'cutlass_80_tensorop_i16832gemm_s8_' in n) or n in ('_int8_scaled_gemm','_int8_splitk_partial'));sage=sum(v for n,v in hist.items() if 'qk_int_sv_f8_attn' in n);expected_native=888;expected_sage=sum(bc['attention_A8_expansion']['calls_per_module_for_25queries'].values())//25
  assert hist.get('_int8_grouped_qkv',0)>0
  assert hist.get('_decode_grouped_qkv_weights',0)==0 and sum(v for k,v in hist.items() if 'decode_grouped_qkv_vec4rcpftz_cuda_r1' in k)==80,hist
  assert sum(v for n,v in hist.items() if 'omni_vision_norm_int8::kernel' in n)==48,hist
  assert hist.get('_bf16_rms_tail_only',0)==0 and hist.get('_bf16_square_f32',0)==0 and hist.get('_codex_native_rms2048',0)==32 and hist.get('_codex_qk_rms_rope128',0)==32,hist
  assert hist.get('_decode_codebook_perrow_int8',0)+sum(v for k,v in hist.items() if 'decode_codebook_vec4rcpftz_cuda_r1' in k)==288,hist
  assert hist.get('_decode_affine_perrow_int8',0)+sum(v for k,v in hist.items() if 'decode_affine_vec8rcpftz_cuda_r1' in k)==104,hist
  assert sum(v for k,v in hist.items() if 'decode_affine_vec8rcpftz_cuda_r1' in k)>0,hist
  assert hist.get('_decode_grouped_ff_weights',0)==0 and sum(v for k,v in hist.items() if 'decode_grouped_ff_vec4cta128_cuda_r1' in k)==128,hist
  assert hist.get('_mean41_native_order',0)==64 and hist.get('_mean_native_parallel_y',0)==48,hist
  assert hist.get('_bool_mask_to_bf16_bias',0)==8,hist
  assert grouping['validated_outputs']>0
  assert gateup_grouping['validated_outputs']>0 and hist.get('_int8_grouped_gateup',0)>0
  assert norm3_summary['residual_checks']==3200
  assert sum(v for k,v in hist.items() if 'codex_norm3_regpack::kernel' in k)==128
  assert ffdown_summary['validation_reference_calls']==3200
  assert ffdown_summary['checks']==3200 and hist.get('_int8_splitk_reduce_residual',0)==128
  (root/'orders'/a.order/(a.condition+'_ffdown_residual.json')).write_text(json.dumps(ffdown_summary,indent=2))
  (root/'orders'/a.order/(a.condition+'_grouped_gateup.json')).write_text(json.dumps(gateup_grouping,indent=2))
  (root/'orders'/a.order/(a.condition+'_grouped_qkv.json')).write_text(json.dumps(grouping,indent=2))
  assert (native,sage)==(expected_native,expected_sage)
  graph=sum(e.get('cat')=='cuda_runtime' and e.get('ph')=='X' and e.get('name')=='cudaGraphLaunch' for e in events);assert graph==1
  attn_trace=attn_path.trace_counts(events);assert attn_path_summary['processor_source_sha256'] is not None and attn_path_summary['unit_rescale_skipped']>0 and attn_trace['action_unit_scale_mul_kernels']<=4 and attn_path_summary['hnd_producer_checks']>=1600 and attn_path_summary['hnd_fast_calls']>0 and attn_trace['sage_followed_by_copy']==0,(attn_path_summary,attn_trace)
  (root/'orders'/a.order/(a.condition+'_claude_attention_path.json')).write_text(json.dumps(dict(summary=attn_path_summary,trace=attn_trace),indent=2))
  assert rope_summary['source_guard'] and rope_summary['checks']>=400 and hist.get('_rope_one',0)==0,rope_summary
  assert mask_summary['alignment']['checks']>=1600,mask_summary['alignment']
  (root/'orders'/a.order/(a.condition+'_codex_rms_rope_bias.json')).write_text(json.dumps(dict(rope=rope_summary,bias=mask_summary['alignment']),indent=2))
  assert len(qknorm_summary['checks'])==32 and set(qknorm_summary['checks'].values())=={25},qknorm_summary
  (root/'orders'/a.order/(a.condition+'_codex_qknorm.json')).write_text(json.dumps(qknorm_summary,indent=2))
  assert vision_view_summary['validation_layout_checks']>=1200,vision_view_summary
  (root/'orders'/a.order/(a.condition+'_vision_sdpa_view.json')).write_text(json.dumps(vision_view_summary,indent=2))
  assert segment_summary['checks']==600 and hist.get('_vision_segment_int8',0)==24,segment_summary
  (root/'orders'/a.order/(a.condition+'_vision_segment.json')).write_text(json.dumps(segment_summary,indent=2))
  resident_with_prepared_inputs=torch.cuda.memory_allocated()/1e9
  # Match rollout/post-measurement residency: prepared caller inputs are
  # temporary and excluded from the original packed-model resident bound.
  del bi,ai,obs,inp,y,prof
  import gc
  gc.collect();torch.cuda.synchronize()
  resident=torch.cuda.memory_allocated()/1e9;reserved=torch.cuda.memory_reserved()/1e9;mstats={k:int(v) for k,v in torch.cuda.memory_stats().items() if k in ('allocated_bytes.all.peak','allocated_bytes.all.current','reserved_bytes.all.current','reserved_bytes.all.peak','active_bytes.all.current','inactive_split_bytes.all.current','num_alloc_retries')}
  if bc['candidate_details']['memory_bound_GB'] is not None:assert resident<=bc['candidate_details']['memory_bound_GB'],('post_release_resident',resident,bc['candidate_details']['memory_bound_GB'])
  result=dict(status='MATCHED_PREPARED_MODEL_WALL_EVENT_RAW_ACTION_RNG_DISPATCH_PASS',order=a.order,condition=a.condition,bootstrap_contract=spec,bootstrap_native_ref=ref(bundle['bootstrap_path']),fixture=ref(root/'fixture.json'),language_rows=bridge.rows,warmup=bc['warmup'],measured=bc['num_iterations'],seed=bc['seed'],denoising_steps=4,batch=1,canonical_noise_episode=0,canonical_noise_call=0,prepared_wall_ms=wall,prepared_wall_median_ms=float(np.median(wall)),prepared_wall_p95_ms=float(np.percentile(wall,95)),GPU_event_ms=gpu,GPU_event_median_ms=float(np.median(gpu)),all50_timed_raw_actions_exact=True,whole_model_reference_vs_original_INT8_R1_bytes_exact=True,original_INT8_reference=original,all4_RNG_exact=True,whole_model_dispatch_count_timing=50,projection_INT8_GEMMs=native,fused_INT8_GEMMs=hist.get('_int8_scaled_gemm',0),Sage_attention_calls=sage,whole_model_Graph_launches_per_query=1,actual_trace=ref(trace),resident_allocated_GB=resident,resident_reserved_GB=reserved,memory_stats_subset=mstats,sidecar_naming='condition_qualified',resident_including_caller_prepared_inputs_GB=resident_with_prepared_inputs,resident_measurement_point='After releasing caller prepared inputs andprofiler; staticgraph inputs/output/fixedcaches/noise/packedpayload retained.',peak_setup_and_measurement_GB=torch.cuda.max_memory_allocated()/1e9,clock_samples=clocks,hoisted_ff_decode=False,claude_attention_path=dict(variant='p2_rescale_sagecrop',c1=True,c2=False,c3=True,summary=attn_path_summary,trace=attn_trace),wall_scope='Prepared inputs: original whole-model metadata checks/inputcopy+Backbone+Action(4steps); synchronize before/after; output validation/profiling/preprocessing/decode/physics outside timed window.',no_success_rollout=True,no_training_or_calibration=True,held_out=False,diagnostic_only=True,source=ref(__file__))
  path=root/'orders'/a.order/(a.condition+'.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path);print(json.dumps({k:result[k] for k in ['order','condition','prepared_wall_median_ms','GPU_event_median_ms','resident_allocated_GB']}))
if __name__=='__main__':main()
