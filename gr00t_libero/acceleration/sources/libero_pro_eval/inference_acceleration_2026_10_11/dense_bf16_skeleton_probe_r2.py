"""Dense-GEMM admission check on the unchanged Omni actual FF-up operand.

No numerical kernel is installed into the model. All candidate errors are raw
diagnostics; finite output is not an accuracy tolerance or acceptance.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

EXT=None
CONFIGS=[
    dict(id=0,tile=[64,64,32],warp=[32,32,32],stages=6),
    dict(id=1,tile=[64,64,32],warp=[32,32,32],stages=3),
    dict(id=2,tile=[64,128,32],warp=[32,64,32],stages=3),
    dict(id=3,tile=[64,32,32],warp=[32,16,32],stages=3,output_elements=4),
    dict(id=4,tile=[64,64,64],warp=[32,32,64],stages=3),
    dict(id=5,tile=[32,128,32],warp=[16,64,32],stages=3),
    dict(id=6,tile=[32,64,64],warp=[16,32,64],stages=3),
    dict(id=7,selection='CUTLASS BF16 Sm80 device::Gemm default template')]


def extension():
    global EXT
    if EXT is None:
        import subprocess
        assert subprocess.check_output(['git','-C','@LOCAL_RUNTIME@/cutlass','rev-parse','HEAD'],text=True).strip()=='7107b05535f8977f5ecb9d01ee203205b1fd9bc4'
        subprocess.run(['git','-C','@LOCAL_RUNTIME@/cutlass','diff','--quiet','--','include'],check=True)
        for name,expected in [('gemm/device/gemm.h','5561969760c09e2d1d385f60f20195e387fef66c0a95cc212e12dc184ecae07d'),
                              ('version.h','cf39f90908935b48162aa63ed1fd0d9383d886b567b2288c195a4b2a2f30989d')]:
            assert hashlib.sha256((Path('@LOCAL_RUNTIME@/cutlass/include/cutlass')/name).read_bytes()).hexdigest()==expected
        from torch.utils.cpp_extension import load
        p=Path(__file__).parent/'dense_bf16_skeleton_r1b'
        build=Path('@LOCAL_RUNTIME@/dense_bf16_skeleton_r1b_sm89');build.mkdir(parents=True,exist_ok=True)
        EXT=load(name='sacq_dense_bf16_skeleton_r1b',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],
            build_directory=str(build),extra_include_paths=['@LOCAL_RUNTIME@/cutlass/include'],
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v',
                    '-gencode=arch=compute_89,code=sm_89'],verbose=True)
    return EXT


def bitsha(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def delta(y,reference):
    difference=y.double()-reference.double()
    return dict(finite=bool(torch.isfinite(y).all()),exact=bool(torch.equal(y,reference)),
        max_abs=difference.abs().max().item(),relative_l2=(difference.norm()/reference.double().norm().clamp_min(1e-30)).item())


def run(policy,obs,contract,arm,out):
    from omni_remaining_cost_profile_r2 import save,eager_call,prepared,decoder_for,telemetry
    from omni_graph_microtime_r1 import measure
    import benchmark_inference as official
    assert arm=='sacq_optimized'
    p=Path(out);cfg=contract['dense_skeleton_probe'];name=cfg['layer']
    model=policy.model;module=model.get_submodule(name)
    with torch.inference_mode():
        values=official.prepare_model_inputs(policy,obs);bi,ai=model.prepare_input(values)
        official.set_seed(contract['seed'])
        refs=[prepared(policy,bi,ai).clone() for _ in range(contract['warmup']+contract['num_iterations'])]
        expected_rng=torch.cuda.get_rng_state().clone()
        captured=[];calls=[]
        def hook(m,args):
            calls.append(tuple(args[0].shape))
            if not captured:captured.append(args[0].detach().clone())
        handle=module.register_forward_pre_hook(hook)
        official.set_seed(contract['seed'])
        try:
            for ref in refs:
                y=eager_call(policy,bi,ai)
                assert torch.equal(y,ref),'FF-up operand observer changed action'
            assert torch.equal(expected_rng,torch.cuda.get_rng_state()),'Observer RNG mismatch'
        finally:handle.remove()
        assert len(captured)==1 and len(calls)==len(refs)*model.action_head.num_inference_timesteps
        x=captured[0];assert list(x.shape)==cfg['input_shape']
        assert bitsha(x)==cfg['input_sha256'],'Actual first operand differs from immutable profiling receipt'
        kind,decode,owner=decoder_for(name,module,contract,{})
        weight=decode();original_weight=module.dequantize_weight()
        assert list(weight.shape)==cfg['weight_shape'] and torch.equal(weight.view(torch.int16),original_weight.view(torch.int16))
        bias=module.bias
        baseline=F.linear(x,weight,bias)
        original_output=module(x)
        assert torch.equal(baseline,original_output),'Selected decoder/native operand path changed'
        del original_output,original_weight,refs,ref,y,values
        xflat=x.reshape(-1,x.shape[-1]);b=bias if bias is not None else torch.empty((0,),device=x.device,dtype=x.dtype)
        # Compile and initialize the public-library kernels before any timer.
        ext=extension()
        assert hashlib.sha256(Path('@LOCAL_RUNTIME@/cutlass/include/cutlass/gemm/device/default_gemm_configuration.h').read_bytes()).hexdigest()=='834d77aea9a59d9f6305a25ca7914322f0c3f910d91d2d1219f8f5b6c7f2502f'
        max_shared=torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
        # CUTLASS pinned Sm80 default is128x256x64,3stages. Its mainloop
        # alone needs147456B, independently of union/epilogue padding.
        unavailable=[];supported=[]
        for c in CONFIGS:
            if c['id']==7 and (128+256)*64*2*3>max_shared:
                unavailable.append(dict(config=c,status='UNSUPPORTED_DEVICE_RESOURCE',
                    minimum_mainloop_shared_bytes=(128+256)*64*2*3,device_optin_shared_limit_bytes=max_shared,
                    source='pinnedCUTLASS default_gemm_configuration.h Sm80 tensorop default'))
                print('DENSE_CONFIG_UNSUPPORTED',c['id'],max_shared,flush=True)
                continue
            print('DENSE_CONFIG_INITIALIZE',c['id'],flush=True)
            z=ext.dense(xflat,weight,b,c['id']);assert torch.isfinite(z).all()
            supported.append(c)
        assert supported
        del z
        reference64=xflat.double()@weight.double().t()
        if bias is not None:reference64+=bias.double()
        reference64=reference64.reshape_as(baseline)
        old_rng=torch.cuda.get_rng_state().clone()
        initial_telemetry=telemetry()
        def native():return F.linear(x,weight,bias)
        def split():return F.linear(x,decode(),bias)
        native_before,yn=measure(native,contract)
        split_before,ys=measure(split,contract)
        assert torch.equal(yn,ys),'Fresh control split/native output mismatch'
        trials=[]
        for c in supported:
            fn=lambda:ext.dense(xflat,weight,b,c['id']).reshape_as(baseline)
            timing,yc=measure(fn,contract)
            error=delta(yc,baseline);assert error['finite']
            row=dict(config=c,timing=timing,vs_native=error,vs_fp64=delta(yc,reference64),
                faster_than_native_before=timing['median_us']<native_before['median_us'],
                faster_than_existing_split_before=timing['median_us']<split_before['median_us'])
            trials.append(row)
            save(p/'dense_skeleton_probe.json',dict(status='DENSE_PROBE_RUNNING',trials=trials))
            print('DENSE_BF16_SKELETON',c['id'],timing['median_us'],error,flush=True)
            del yc
        native_after,_=measure(native,contract);split_after,_=measure(split,contract)
        best=min(trials,key=lambda r:r['timing']['median_us'])
        # Confirm the winning skeleton independently, no expanded sweep.
        fn=lambda:ext.dense(xflat,weight,b,best['config']['id']).reshape_as(baseline)
        confirmation,yc=measure(fn,contract)
        assert torch.isfinite(yc).all() and torch.equal(old_rng,torch.cuda.get_rng_state())
        assert bitsha(x)==cfg['input_sha256']
        weight_sha=bitsha(weight);assert torch.equal(weight.view(torch.int16),module.dequantize_weight().view(torch.int16))
        native_range=[native_before['median_us'],native_after['median_us']]
        split_range=[split_before['median_us'],split_after['median_us']]
        result=dict(status='DENSE_SKELETON_CHECK_COMPLETE',diagnostic_only=True,model_dispatch_changed=False,
            layer=name,input_shape=list(x.shape),weight_shape=list(weight.shape),input_sha256=bitsha(x),
            weight_sha256=weight_sha,bias_present=bias is not None,active_decoder=kind,
            source_geometry=cfg['source_geometry'],observed_layer_calls=len(calls),
            current25actions_rng_exact=True,weights_unchanged=True,micro_rng_unchanged=True,
            native_before=native_before,native_after=native_after,existing_split_before=split_before,existing_split_after=split_after,
            native_vs_fp64=delta(baseline,reference64),trials=trials,unavailable_configs=unavailable,best_config=best['config'],
            best_initial_us=best['timing']['median_us'],best_confirmation=confirmation,
            confirmation_vs_native=delta(yc,baseline),confirmation_vs_fp64=delta(yc,reference64),
            confirmation_over_native_range=[confirmation['median_us']/v for v in native_range],
            remaining_vs_existing_split_us=[v-confirmation['median_us'] for v in split_range],
            passes_minimum_dense_admission=confirmation['median_us']<min(split_range),
            beats_both_native_controls=confirmation['median_us']<min(native_range),
            telemetry_before=initial_telemetry,telemetry_after=telemetry(),
            accuracy_acceptance=False,permanent_weight_cache_installed=False,
            interpretation='Dense-only CUTLASS BF16 skeleton on actual current SACQ operands. Hot graph micro5+20; two fresh native/split controls and winner confirmation. Candidate output errors retained, no numeric tolerance/rollout/model speed claim. Remaining time is a necessary local integration budget, not guaranteed achievable whole-model gain; fused decode will change resources and data paths.')
        save(p/'dense_skeleton_probe.json',result)
        print('DENSE_SKELETON_COMPLETE',json.dumps({k:result[k] for k in ['best_config','best_confirmation','confirmation_over_native_range','remaining_vs_existing_split_us','passes_minimum_dense_admission','beats_both_native_controls']}),flush=True)
    return result
