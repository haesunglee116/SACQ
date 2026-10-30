"""Same fixed-demo timing: official Omni BF16 vs unchanged final SACQ A16.

The direct official pipeline excludes server/scheduler/WebSocket overhead.
Fresh fully optimized controls or common FA2 diagnostic; unchanged SACQ A16 payload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path('@WORKSPACE@/libero_pro_eval/inference_acceleration_2026_10_11')
ROOT=P.parents[1]

def atomic(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();c=json.loads(Path(args.protocol).read_text())
    isolation=c["isolation"]
    c['_protocol_path']=str(Path(args.protocol).resolve())
    c['_arm']=args.arm
    omni=True
    if omni and Path(sys.executable).resolve()!=Path(c['omni_runtime']['python']).resolve():
        os.execve(c['omni_runtime']['python'],[c['omni_runtime']['python'],__file__,*sys.argv[1:]],dict(os.environ,OMNI_BENCH_CHILD='1'))
    # Both virtualenv interpreters may resolve to the same base executable.
    if omni and os.environ.get('OMNI_BENCH_CHILD')!='1':
        os.execve(c['omni_runtime']['python'],[c['omni_runtime']['python'],__file__,*sys.argv[1:]],dict(os.environ,OMNI_BENCH_CHILD='1'))
    out=Path(c['output_directory'])/(args.arm+'.json')
    if out.exists():raise FileExistsError(out)
    if args.arm in c.get('reused_control_arms',[]) and 'reused_controls' in c:
        source=c['reused_controls']
        path=Path(source['directory'])/(args.arm+'.json')
        d=json.loads(path.read_text())
        original_sha=hashlib.sha256(Path(source['protocol']).read_bytes()).hexdigest()
        assert d['status']=='TIMING_COMPLETE' and d['rng_exact'] and d['protocol_sha256']==original_sha
        d['reused_from']={'path':str(path),'original_protocol_sha256':original_sha}
        d['arm']=args.arm
        d['protocol_sha256']=hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()
        atomic(out,d)
        if args.arm=='bf16_basic':
            blob=(Path(source['directory'])/'nvidia_bf16_actions.npy').read_bytes()
            dst=out.with_name('nvidia_bf16_actions.npy');tmp=dst.with_suffix('.tmp')
            tmp.write_bytes(blob);os.replace(tmp,dst)
            assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(blob).digest()
        if args.arm=='sacq_basic':
            from benchmark_a8_joint_pair_confirm_r1 import durable_numpy_save
            import numpy as np
            durable_numpy_save(out.with_name('previous_pipeline_sacq_actions.npy'),np.load(Path(source['directory'])/'previous_pipeline_sacq_actions.npy'))
        return
    workspace=c['memory_runtime']['cublas_workspace']
    if workspace is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else:os.environ['CUBLAS_WORKSPACE_CONFIG']=workspace
    sys.path[:0]=[str(Path(__file__).resolve().parent/'shape_modules'),str(P),c['cwd'],str(Path(c['cwd'])/'scripts/deployment'),str(ROOT/'libero_pro_eval'),str(ROOT/'libero_pro_eval/claude_packed_runtime_2026_08_31')]
    # Imported kernel installers prepend legacy paths. Resolve all approved shape
    # adapters explicitly, so those installers cannot restore fixed156 modules.
    import importlib.abc, importlib.util
    _shape_root=Path(__file__).resolve().parent/'shape_modules'
    _shape_names={p.stem:p for p in _shape_root.glob('*.py')}
    class ShapeModuleFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in _shape_names:
                return importlib.util.spec_from_file_location(fullname,_shape_names[fullname])
            return None
    for _name,_path in _shape_names.items():
        if _name in sys.modules:
            assert Path(sys.modules[_name].__file__).resolve()==_path.resolve(),('Wrong shape module already loaded',_name)
    sys.meta_path.insert(0,ShapeModuleFinder())
    import numpy as np
    import torch
    from omni_a8_all_rng_snapshot_r1 import snapshot as all_rng_snapshot
    import rollout_bridge_r1 as rollout_bridge
    import final_model
    import benchmark_inference as official
    plan=json.loads(Path(c['plan']['path']).read_text())
    if omni:
        from transformers import AutoConfig,AutoProcessor,Qwen3VLProcessor
        for cls in [AutoConfig,AutoProcessor,Qwen3VLProcessor]:
            orig=cls.from_pretrained
            def local(cls,name,*a,_orig=orig,**kw):
                if str(name) in ('nvidia/Cosmos-Reason2-2B','Qwen/Qwen3-VL-2B-Instruct'):name=plan['cosmos']
                return _orig(name,*a,**kw)
            cls.from_pretrained=classmethod(local)
        from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline
        from vllm_omni.diffusion.models.gr00t.dataio.types import MessageType,VLAStepData
        official.MessageType=MessageType;official.VLAStepData=VLAStepData
        from types import SimpleNamespace
        pipeline=Gr00tN1d7Pipeline(od_config=SimpleNamespace(model=plan['models'][c['suite']],
              model_config={'embodiment_tag':'LIBERO_PANDA','strict':True}))
        policy=pipeline.policy
    else:policy=final_model.build_policy(plan,c['suite'],'cuda:0')
    policy.model.eval().requires_grad_(False)
    packed=args.arm.startswith('sacq')
    precision=c['precision_by_arm'].get(args.arm,'BF16')
    stage_id=c['a8_scope_expansion']['stage_by_arm'].get(args.arm,0)
    assert stage_id in (0,5)
    extra_enabled=packed and precision=='FP8' and stage_id>=1
    c['a8_scope_expansion']['stage']=stage_id
    result_scope=dict(stage=stage_id,FP8_projection_count=(441 if stage_id==5 else 408 if stage_id>=4 else 304 if stage_id>=3 else 192 if stage_id>=2 else 64 if extra_enabled else 32) if packed else 0)
    if precision=='A16':result_scope['FP8_projection_count']=0
    packing=None
    if packed:
        sys.path.insert(0,str(ROOT/'gr00t_final_seed17_20260922'))
        from omni_sacq_adapter_r2 import install
        packing=install(policy,plan,c['suite'])
        from omni_sacq_adapter_r2 import install_kernels
        kernel_receipt=install_kernels(policy,c)
        print('PACKING_COMPLETE',packing['installed_codebook_layers'],flush=True)
        if c.get('native_reference_selection'):
            from omni_decode_simd_r3 import install as install_reference_decoder
            reference_selection=json.loads(Path(c['native_reference_selection']['path']).read_text())['layers']
            reference_decoder_receipt=install_reference_decoder(policy,reference_selection)
            print('REFERENCE_DECODER_READY',reference_decoder_receipt['selected_layers'],flush=True)
    if packed and c.get('register_reference_selection'):
        from omni_decode_register_lut_r1 import install as install_register_reference
        reference_register_selection=json.loads(Path(c['register_reference_selection']['path']).read_text())['layers']
        with torch.inference_mode():
            reference_register_receipt=install_register_reference(policy,reference_register_selection)
        print('REGISTER_REFERENCE_READY',reference_register_receipt['selected_layers'],flush=True)
    if packed and c.get('dictionary_reference_selection'):
        from omni_dictionary_decode_r1 import install as install_dictionary_reference
        reference_dictionary_selection=json.loads(Path(c['dictionary_reference_selection']['path']).read_text())['layers']
        with torch.inference_mode():
            reference_dictionary_receipt=install_dictionary_reference(policy,reference_dictionary_selection)
        print('DICTIONARY_REFERENCE_READY',reference_dictionary_receipt['selected_scalar_layers'],flush=True)
    print('POLICY_READY',args.arm,torch.__version__,flush=True)
    modalities=policy.modality_configs if omni else policy.get_modality_config()
    obs=rollout_bridge.nested(rollout_bridge.first_observation,policy)
    from types import SimpleNamespace
    step=SimpleNamespace(text=obs['language'][policy.language_key][0][0])
    rollout_bridge.install_noise(policy)
    h=hashlib.sha256()
    for group in ['video','state']:
        for k,v in sorted(obs[group].items()):h.update(group.encode()+k.encode()+str(v.shape).encode()+str(v.dtype).encode()+v.tobytes())
    h.update(step.text.encode())
    result=dict(arm=args.arm,status='RUNNING',diagnostic_only=True,input_sha256=h.hexdigest(),
        protocol_sha256=final_model.sha(args.protocol),precision='SACQ final 1.5 nominal W / A16' if packed else 'BF16',hardware=torch.cuda.get_device_name(),
        torch_version=torch.__version__,runtime='official_vllm_omni_gr00t_pipeline' if omni else 'nvidia_native',
        measurement_scope='official component-derived model E2E; excludes server/scheduler/WebSocket',
        accuracy_acceptance='NOT_ASSESSED; raw output differences retained')
    result['packing']=packing
    result['a8_scope']=result_scope
    if packed and c.get('native_reference_selection'):result['reference_decoder']=reference_decoder_receipt
    if packed and c.get('register_reference_selection'):result['register_reference_decoder']=reference_register_receipt
    assert '4090' in result['hardware']
    with torch.inference_mode():
        inputs=official.prepare_model_inputs(policy,obs);bi,ai=policy.model.prepare_input(inputs)
        def describe(value,prefix=''):
            if isinstance(value,torch.Tensor):return {prefix:{'shape':list(value.shape),'dtype':str(value.dtype)}}
            if isinstance(value,dict) or hasattr(value,'items'):
                acc={}
                for k,v in value.items():acc.update(describe(v,prefix+'/'+str(k)))
                return acc
            return {}
        result['prepared_input_shapes']={'backbone':describe(bi),'action':describe(ai)}
        result['denoising_steps']=policy.model.action_head.num_inference_timesteps
        result['action_horizon']=policy.model.action_head.action_horizon
        assert result['denoising_steps']==4
        rollout_bridge.set_rows(c,int(bi['input_ids'].shape[-1]))
        assert packed and args.arm=='sacq_basic'
        from lossless_scalar_affine_runtime_r1 import install as install_affine
        selection=json.loads(Path(c['lossless_affine_selection']['path']).read_text())['layers']
        result['lossless_affine_trial']=install_affine(policy,selection)
        official.set_seed(c['seed']);refs=[]
        for _ in range(c['warmup']+c['num_iterations']):
            bo=policy.model.backbone(bi);refs.append(policy.model.action_head.get_action(bo,ai).action_pred.clone())
        expected_rng=torch.cuda.get_rng_state().clone()
        expected_all_rng=all_rng_snapshot()
        result['expected_all_rng']=expected_all_rng
        result['BF16_baseline_reused_not_recomputed']=True

        from benchmark_a8_joint_pair_confirm_r1 import durable_numpy_save
        from perrow_int8_fast_policy_r3 import install as install_row,finish_validation as finish_row
        from compact_storage_a8_r5 import compact
        names=sorted(c['perrow_method']['projection_shapes'])
        parents={'.'.join(name.split('.')[:i]) for name in names for i in range(1,len(name.split('.')))}
        for name in parents:
            assert 'forward' not in policy.model.get_submodule(name).__dict__,('UngroupedParentRequired',name)
        policy.model._row_candidate_parent_fusions_disabled=True
        durable_numpy_save(out.with_name(args.arm+'_lean_A16_reference_actions.npy'),torch.stack(refs).float().cpu().numpy())
        result['perrow_install']=install_row(policy,names,c['perrow_method'])
        result['a8_scope']=dict(actual_8bit_projection_count=441,FP8_projection_count=441 if c['perrow_method']['projection_format']=='fp8_e4m3fn' else 0,projection_format=c['perrow_method']['projection_format'])
        runtime_ref=c['attention_A8_expansion']['runtime_receipt'];assert hashlib.sha256(Path(runtime_ref['path']).read_bytes()).hexdigest()==runtime_ref['sha256']
        sys.path.insert(0,json.loads(Path(runtime_ref['path']).read_text())['pythonpath'])
        if c['sage_single_preparation']['enabled_by_arm'][args.arm]:
            from recovery_sage_single_preparation_fp32_r2 import install as install_prep,finish_validation as finish_prep
            result['sage_single_preparation']=install_prep(policy,c)
        from recovery_attention_adapter_fp32_r2 import install as install_attention,finish_validation as finish_attention
        result['attention_A8_expansion']=install_attention(policy,c)
        equivalence_refs=[]
        official.set_seed(c['seed'])
        for _ in range(c['warmup']+c['num_iterations']):
            bo=policy.model.backbone(bi);equivalence_refs.append(policy.model.action_head.get_action(bo,ai).action_pred.clone())
        assert all(torch.isfinite(v).all() for v in equivalence_refs) and all_rng_snapshot()==expected_all_rng
        result['perrow_validation']=finish_row(policy)
        counts=result['perrow_validation']['counts']
        assert len(counts)==441 and all(v==(100 if name.startswith('action_head.') else 25) for name,v in counts.items()),counts
        result['attention_A8_expansion_validation']=finish_attention(policy,c)
        if c['sage_single_preparation']['enabled_by_arm'][args.arm]:result['sage_single_preparation_validation']=finish_prep(policy,c)
        durable_numpy_save(out.with_name(args.arm+'_perrow8_native_reference_actions.npy'),torch.stack(equivalence_refs).float().cpu().numpy())
        parent_ref=c['continuation_parent_bootstrap_reference'];assert hashlib.sha256(Path(parent_ref['path']).read_bytes()).hexdigest()==parent_ref['sha256']
        assert np.array_equal(torch.stack(equivalence_refs).float().cpu().numpy().view(np.uint32),np.load(parent_ref['path']).view(np.uint32)), 'Initial25ParentINT8ActionBytesChanged'
        result['continuation_initial25_parent_action_bytes_exact']=True
        result['perrow_vs_lean_A16_errors']=[dict(max_abs=(v.float()-r.float()).abs().max().item(),relative_l2=((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12)).item()) for v,r in zip(equivalence_refs,refs)]
        del refs
        result['storage_compaction']=compact(policy.model)
        result['stages']=[]
        def stage(name,receipt):
            official.set_seed(c['seed']);checks=[]
            for r in equivalence_refs:
                bo=policy.model.backbone(bi);v=policy.model.action_head.get_action(bo,ai).action_pred
                assert torch.equal(v,r),('Row8ExactImplementationMismatch',name)
                checks.append(dict(exact=True,max_abs=0.,relative_l2=0.))
            assert all_rng_snapshot()==expected_all_rng
            result['stages'].append(dict(stage=name,receipt=receipt,action_checks=checks,rng_exact=True,all_rng_exact=True))
            atomic(out.with_name(args.arm+'_stages.json'),result['stages']);print('PERROW_STAGE_PASS',name,flush=True)
        from common_vision_rotary_fusion_r1 import install as install_rotary,finish_validation as finish_rotary
        result['common_vision_rotary']=install_rotary(policy,c);stage('vision_rotary_exact',result['common_vision_rotary']);result['common_vision_rotary_validation']=finish_rotary(policy,c)
        from common_language_qk_norm_fusion_r2 import install as install_qknorm,finish_validation as finish_qknorm
        result['common_language_qk_norm']=install_qknorm(policy,c);stage('language_qk_norm_exact',result['common_language_qk_norm']);result['common_language_qk_norm_validation']=finish_qknorm(policy,c)
        from adaln_norm_modulate_runtime_r2 import install as install_adaln,finish_validation as finish_adaln
        result['adaln_norm_modulate']=install_adaln(policy,c)
        from omni_action_stages_r3 import ActionStages
        state=ActionStages(policy,bi,ai);stage('fixed_outputs',state.fixed_outputs());result['adaln_norm_modulate_validation']=finish_adaln(policy,c)
        stage('fixed_outputs_plus_kv',state.cross_kv());stage('action_graph',state.graph())
        from omni_backbone_graph_r1 import install as backbone_graph
        stage('backbone_graph',backbone_graph(policy,bi))
        from whole_model_graph_runtime_r2 import install as whole_graph
        result['whole_model_graph_validation']=whole_graph(policy,bi,ai,equivalence_refs,c,args.arm,all_rng_snapshot,expected_all_rng,None)
        return rollout_bridge.complete(policy,c,obs,result,bi,ai,expected_all_rng)
if __name__=='__main__':main()
