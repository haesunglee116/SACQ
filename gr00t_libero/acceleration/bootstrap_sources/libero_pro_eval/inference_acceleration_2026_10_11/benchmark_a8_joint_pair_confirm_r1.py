"""Four isolated processes, NVIDIA's own component timing functions and demo.

Timing is diagnostic until optimized-versus-reference action checks are reviewed.
This is not a rollout, accuracy evaluation, or new calibration experiment.
"""
import argparse
import gc
import io
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / 'libero_pro_eval/Isaac-GR00T-N1.7'
sys.path[:0] = [str(HERE), str(REPO), str(REPO / 'scripts/deployment'),
               str(ROOT / 'libero_pro_eval'),
               str(ROOT / 'libero_pro_eval/claude_packed_runtime_2026_08_31')]
import numpy as np
import torch
import final_model
import benchmark_inference as official


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def durable_numpy_save(path, array):
    # numpy ndarray.tofile can leave a short direct write on the shared filesystem.
    # Keep exactly the existing NPY format/data; validate before committing.
    buffer=io.BytesIO()
    np.save(buffer,array,allow_pickle=False)
    payload=buffer.getvalue()
    path=Path(path);temp=path.with_suffix('.npy.tmp')
    with temp.open('wb') as f:
        view=memoryview(payload)
        while view:
            count=f.write(view)
            if not count:raise OSError('Short reference write')
            view=view[count:]
        f.flush();os.fsync(f.fileno())
    actual=np.load(temp,allow_pickle=False)
    if not np.array_equal(actual,array):raise RuntimeError('Saved reference bytes mismatch')
    if temp.stat().st_size!=len(payload):raise RuntimeError('Reference file size mismatch')
    os.replace(temp,path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True)
    p.add_argument('--arm', choices=['bf16_basic', 'bf16_optimized', 'sacq_basic', 'sacq_optimized'], required=True)
    args = p.parse_args()
    c = json.loads(Path(args.protocol).read_text())
    out = Path(c['output_directory']) / (args.arm+'.json')
    if args.arm.endswith('optimized'):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=c['memory_runtime']['cublas_workspace']
    if args.arm=='sacq_optimized':
        control=out.with_name('a8_r22_control.json')
        if control.exists() or out.exists():raise FileExistsError(control)
        # Same immutable model/input/noise and same allocated GPU; fresh process
        # measures the prior A8 optimum before this candidate. Its frozenA16
        # reference is also retained; no references are silently overwritten.
        subprocess.run([sys.executable,str(HERE/'benchmark_a8_fused_r22.py'),'--protocol',args.protocol,'--arm',args.arm],check=True)
        os.replace(out,control)
    if out.exists():
        raise FileExistsError(out)
    result = dict(arm=args.arm, status='RUNNING', diagnostic_only=True,
                  protocol_sha256=final_model.sha(args.protocol),
                  hardware=torch.cuda.get_device_name(), torch_version=torch.__version__)
    if '4090' not in result['hardware']:
        raise RuntimeError('Approved hardware is RTX 4090')
    official.set_seed(c['seed'])
    # The launch gate already checked every immutable payload.
    plan = json.loads(Path(c['plan']['path']).read_text())
    policy = final_model.build_policy(plan, c['suite'], 'cuda:0')
    packed = args.arm.startswith('sacq')
    print('POLICY_LOADED', args.arm, flush=True)
    if packed:
        result['packing'] = final_model.install(policy, plan, c['suite'], 'dense_reconstruct')
        print('PACKING_COMPLETE', flush=True)
    modalities = policy.get_modality_config()
    dataset = official.LeRobotEpisodeLoader(dataset_path=c['dataset_path'],
                modality_configs=modalities, video_backend='torchcodec')
    step = official.extract_step_data(dataset[c['episode_index']], step_index=c['step_index'],
                modality_configs=modalities, embodiment_tag=policy.embodiment_tag, allow_padding=False)
    obs = dict(video={k: np.stack(v)[None] for k, v in step.images.items()},
               state={k: v[None] for k, v in step.states.items()},
               language={modalities['language'].modality_keys[0]: [[step.text]]})
    digest = hashlib.sha256()
    for group in ['video', 'state']:
        for k, v in sorted(obs[group].items()):
            digest.update(group.encode()+k.encode()+str(v.shape).encode()+str(v.dtype).encode()+v.tobytes())
    digest.update(step.text.encode())
    result.update(input_sha256=digest.hexdigest(), cameras=list(obs['video']),
                  video_shapes={k: list(v.shape) for k, v in obs['video'].items()},
                  denoising_steps=policy.model.action_head.num_inference_timesteps,
                  action_horizon=policy.model.action_head.action_horizon, batch=1)
    shared_dp_path = Path(c['output_directory']) / 'shared_data_processing.json'
    if args.arm == 'bf16_basic':
        shared_dp = official.benchmark_data_processing(policy, obs, c['num_iterations'], warmup=10)
        atomic(shared_dp_path, dict(input_sha256=result['input_sha256'], times_ms=shared_dp.tolist()))
    else:
        shared = json.loads(shared_dp_path.read_text())
        assert shared['input_sha256'] == result['input_sha256']
        shared_dp = np.array(shared['times_ms'])
    with torch.inference_mode():
        inputs = official.prepare_model_inputs(policy, obs)
        bi, ai = policy.model.prepare_input(inputs)
        from fused_action_measure_r1 import capture, measure
        samples, handles = ({}, [])
        if packed and args.arm.endswith('optimized'):
            if not c.get('freeze_only',False):
                from priority_measure_r3 import capture
                samples,handles=capture(policy.model,c['priority_stage'])
        # Verify every seed-42 draw used by the approved warmup+timing sequence.
        official.set_seed(c['seed'])
        refs = []
        for _ in range(c['warmup'] + c['num_iterations']):
            bo = policy.model.backbone(bi)
            refs.append(policy.model.action_head.get_action(bo, ai).action_pred.clone())
        reference_rng = torch.cuda.get_rng_state().clone()
        reference_path=Path(c['output_directory'])/('sacq_default_actions.npy' if packed else 'bf16_default_actions.npy')
        if args.arm.endswith('basic'):
            durable_numpy_save(reference_path,torch.stack(refs).float().cpu().numpy())
        else:
            reference_default=torch.from_numpy(np.load(reference_path)).to(device='cuda',dtype=refs[0].dtype)
            refs=list(reference_default.unbind(0))
            del reference_default
        for handle in handles: handle.remove()
        if packed and args.arm.endswith('optimized'):
            from decoder_tuning import SELECTED
            tuning = json.loads(Path(c['decoder_tuning']['path']).read_text())
            for row in tuning['groups']:
                SELECTED[tuple(row['geometry'])] = tuple(row['selected'])
        if packed and args.arm.endswith('optimized'):
            import decoder_candidates_r1 as candidate
            selected = json.loads(Path(c['candidate_selection']['path']).read_text())
            assert final_model.sha(c['candidate_selection']['path']) == c['candidate_selection']['sha256']
            for row in selected['rows']:
                candidate.SELECTED[tuple(row['geometry'])] = tuple(row['selected'])
            candidate.activate()
            import dictionary_candidates_r1 as dictionary
            ds = json.loads(Path(c['dictionary_selection']['path']).read_text())
            assert final_model.sha(c['dictionary_selection']['path']) == c['dictionary_selection']['sha256']
            for row in ds['rows']:
                dictionary.SELECTED[tuple(row['geometry'])] = tuple(row['selected'])
            dictionary.activate()
        if args.arm.endswith('optimized'):
            if packed:
                import a8_joint_pair_r1 as a8
                settings=c['a8']
                if settings['approved'] is not True:
                    raise RuntimeError('A8 settings are not approved')
                assert settings['activation_scale']=='dynamic_per_tensor_absmax'
                assert settings['weight_scale']=='per_tensor_absmax'
                scope_ref=settings['scope_receipt']
                assert final_model.sha(scope_ref['path'])==scope_ref['sha256']
                names=json.loads(Path(scope_ref['path']).read_text())['packing']['already_packed_layers']
                assert len(names)==len(set(names))==337
                assert len([n for n in names if n.startswith('backbone.')])==112
                vision_names=[r['layer'] for r in json.loads(Path(scope_ref['path']).read_text())['packing']['scalar']['layers'] if '.visual.' in r['layer']]
                assert len(vision_names)==104
                result['a8_installation'],_restore=a8.install(policy.model,names,vision_names,settings['format'],'real_gemm',c['a8_frozen_decode_selection'])
                scale_checks=[]
                def checked_activation(a,aq,actual):
                    maximum=a.abs().amax().float()
                    reference=torch.where(maximum>0,maximum/448.0,torch.ones_like(maximum))
                    if not torch.equal(actual.view(torch.int32),reference.view(torch.int32)):
                        raise RuntimeError('Fused dynamicFP8 scale differs from approved TorchFP32 formula')
                    expected=(a.float()/reference).clamp(-448.0,448.0).to(torch.float8_e4m3fn)
                    rows=a.shape[0]
                    if not torch.equal(aq[:rows].view(torch.uint8),expected.view(torch.uint8)):
                        raise RuntimeError('Fused activation FP8 bytes differ from approved RTN')
                    if rows<aq.shape[0] and not torch.equal(aq[rows:].view(torch.uint8),torch.zeros_like(aq[rows:].view(torch.uint8))):
                        raise RuntimeError('FP8 padding is not exact zero')
                    scale_checks.append(tuple(a.shape))
                a8.ACTIVATION_OBSERVER=checked_activation
                names=names+vision_names
                # Verify actual low-precision dispatch before optimizer graph construction.
                invoked=set()
                handles=[policy.model.get_submodule(n).register_forward_pre_hook(
                    lambda m,args,n=n:invoked.add(n)) for n in names]
                official.set_seed(c['seed'])
                preopt_checks=[]
                for ref in refs:
                    bo=policy.model.backbone(bi)
                    action=policy.model.action_head.get_action(bo,ai).action_pred
                    if not torch.isfinite(action).all():raise RuntimeError('Nonfinite real A8 action')
                    delta=action.float()-ref.float()
                    preopt_checks.append(dict(max_abs=delta.abs().max().item(),
                        relative_l2=(delta.norm()/ref.float().norm().clamp_min(1e-12)).item()))
                a8.ACTIVATION_OBSERVER=None
                result['dynamic_scale_checks']={'calls':len(scale_checks),
                    'shapes':sorted(set(scale_checks)), 'exact_fp32':True, 'exact_fp8_activation_operands':True,
                    'scope':'Every A8 call in all25 approved preoptimization draws'}
                if not scale_checks:raise RuntimeError('No dynamic scale checks executed')
                result['a8_before_common_optimization_checks']=preopt_checks
                result['a8_before_common_optimization_rng_exact']=torch.equal(reference_rng,torch.cuda.get_rng_state())
                assert result['a8_before_common_optimization_rng_exact']
                for handle in handles:handle.remove()
                result['a8_invoked_before_caching']=sorted(invoked)
                assert invoked==set(names), 'Requested A8 scope was not fully invoked'
                torch.cuda.synchronize()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
            from memory_bounded_a8_shared_r8 import install
            result['optimizations'] = install(policy, bi, ai, packed)
        official.set_seed(c['seed'])
        comparisons = []
        for ref in refs:
            bo = policy.model.backbone(bi)
            action = policy.model.action_head.get_action(bo, ai).action_pred
            assert torch.isfinite(action).all(), 'Non-finite action output'
            delta = action.float() - ref.float()
            comparisons.append(dict(exact=torch.equal(action.view(torch.int16), ref.view(torch.int16)),
                   max_abs=delta.abs().max().item(),
                   relative_l2=(delta.norm()/ref.float().norm().clamp_min(1e-12)).item()))
        result['action_checks'] = comparisons
        result['correctness_scope'] = 'All 5 warmup + 20 measured seed42 noise draws; BF16 bit-pattern comparison'
        result['rng_exact'] = torch.equal(reference_rng, torch.cuda.get_rng_state())
        if not result['rng_exact']:
            raise RuntimeError('Optimization changed the policy RNG stream')
    del refs, ref, action, inputs, bi, ai, bo
    gc.collect()
    torch.cuda.empty_cache()
    # Numerical differences are reported, never silently accepted as equivalent.
    result['action_equivalence'] = 'EXACT' if all(x['exact'] for x in comparisons) else 'REVIEW_REQUIRED'
    result['accuracy_acceptance'] = 'NOT_ASSESSED; user waived bitwise equality; numeric error reported only'
    if args.arm.endswith('basic') and result['action_equivalence'] != 'EXACT':
        raise RuntimeError('Unmodified control changed output')
    result['resident_GB'] = torch.cuda.memory_allocated()/1e9
    torch.cuda.reset_peak_memory_stats()
    official.set_seed(c['seed'])
    print('TIMING_BEGIN', args.arm, result['action_equivalence'], flush=True)
    components = official.benchmark_components(policy, obs, c['num_iterations'], c['warmup'])
    components['data_processing'] = shared_dp
    components['e2e'] = official.compute_e2e_from_components(components)
    result['times_ms'] = {k: v.tolist() for k, v in components.items()}
    result['median_ms'] = {k: float(np.median(v)) for k, v in components.items()}
    result['resident_after_measurement_GB'] = torch.cuda.memory_allocated()/1e9
    result['reserved_after_measurement_GB']=torch.cuda.memory_reserved()/1e9
    result['peak_allocated_GB']=torch.cuda.max_memory_allocated()/1e9
    result['peak_reserved_GB']=torch.cuda.max_memory_reserved()/1e9
    if args.arm.endswith('optimized'):
        baseline=json.loads(out.with_name(('sacq_basic' if packed else 'bf16_basic')+'.json').read_text())
        result['memory_no_increase']={k:result[k]<=baseline[k] for k in ['resident_after_measurement_GB','reserved_after_measurement_GB','peak_allocated_GB','peak_reserved_GB']}
    result['status'] = 'TIMING_COMPLETE'
    atomic(out, result)
    print('RESULT', args.arm, result['median_ms'], flush=True)
    if args.arm=='sacq_optimized' and c.get('profile_candidate',False):
        from no_counter_diagnostics_r1 import profile
        profile(policy,obs,official,c,reference_rng,out.with_name('candidate_profile.json'))


if __name__ == '__main__':
    main()
