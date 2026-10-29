"""One actual SACQ layer; no model dispatch installation or accuracy acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F
from dense_bf16_skeleton_probe_r2 import delta

def bitsha(x):
 return hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

EXT=None
CONFIGS=[
 dict(id=0,kind='dense_synchronous_shared_supply_control',tile=[64,64,32],stages=6),
 dict(id=1,kind='packed_global_lut',tile=[64,64,32],stages=6),
 dict(id=2,kind='packed_warp_register_lut',tile=[64,64,32],stages=6),
 dict(id=3,kind='packed_global_lut',tile=[64,64,32],stages=3),
 dict(id=4,kind='packed_warp_register_lut',tile=[64,64,32],stages=3),
 dict(id=5,kind='packed_warp_register_lut',tile=[64,64,64],stages=3),
 dict(id=6,kind='packed_global_lut',tile=[64,32,64],stages=3),
 dict(id=7,kind='packed_warp_register_lut',tile=[64,32,64],stages=3),
]

def extension():
 global EXT
 if EXT is None:
  assert subprocess.check_output(['git','-C','@LOCAL_RUNTIME@/cutlass','rev-parse','HEAD'],text=True).strip()=='7107b05535f8977f5ecb9d01ee203205b1fd9bc4'
  subprocess.run(['git','-C','@LOCAL_RUNTIME@/cutlass','diff','--quiet','--','include'],check=True)
  header=Path('@LOCAL_RUNTIME@/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h')
  assert hashlib.sha256(header.read_bytes()).hexdigest()=='f762ee08aeb8201b8f2c1cd537de8f766dc6674a57a80641c258e08946543400'
  from torch.utils.cpp_extension import load
  p=Path(__file__).parent/'fused_ffup_cutlass_r1'
  build=Path('@LOCAL_RUNTIME@/fused_ffup_cutlass_r1_sm89');build.mkdir(parents=True,exist_ok=True)
  EXT=load(name='sacq_fused_ffup_cutlass_r1',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],
   build_directory=str(build),extra_include_paths=['@LOCAL_RUNTIME@/cutlass/include'],
   extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v',
    '-gencode=arch=compute_89,code=sm_89'],verbose=True)
 return EXT

def run(policy,obs,contract,arm,out):
 from omni_remaining_cost_profile_r2 import save,eager_call,prepared,decoder_for,telemetry
 from omni_graph_microtime_r1 import measure
 from dense_bf16_skeleton_probe_r2 import extension as dense_extension
 import benchmark_inference as official
 assert arm=='sacq_optimized'
 out=Path(out);cfg=contract['fused_ffup_probe'];name=cfg['layer']
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
    assert torch.equal(y,ref),'Operand observer changed current action'
   assert torch.equal(expected_rng,torch.cuda.get_rng_state()),'Observer RNG mismatch'
  finally:handle.remove()
  assert len(captured)==1 and len(calls)==len(refs)*model.action_head.num_inference_timesteps
  x=captured[0];assert list(x.shape)==cfg['input_shape'] and bitsha(x)==cfg['input_sha256']
  kind,decode,_=decoder_for(name,module,contract,{})
  weight=decode();original=module.dequantize_weight()
  assert list(weight.shape)==cfg['weight_shape'] and torch.equal(weight.view(torch.int16),original.view(torch.int16))
  assert module.vector_dim==4 and module.group_size==64 and module.shift.numel()==1
  assert int(module.shift.view(torch.int16).item())==0,'Positive-zero specialization gate'
  bits=module.index_bits
  assert bits>0 and bits<=16
  baseline=F.linear(x,weight,module.bias)
  assert torch.equal(baseline,module(x))
  del original,refs,ref,y,values
  p=module.packed_indices;cb=module.codebook;s=module.scales;h=module.shift;b=module.bias
  assert b is not None
  operand_receipt=dict(layer=name,input_shape=list(x.shape),weight_shape=list(weight.shape),input_sha256=bitsha(x),
    weight_sha256=bitsha(weight),packed_sha256=bitsha(p),codebook_sha256=bitsha(cb),scales_sha256=bitsha(s),
    shift_sha256=bitsha(h),bias_sha256=bitsha(b),index_bits=bits,vector_dim=module.vector_dim,
    group_size=module.group_size,positive_zero_shift=True,active_decoder=kind,observed_layer_calls=len(calls),
    packed_buffer_bytes=p.numel()*p.element_size(),codebook_buffer_bytes=cb.numel()*cb.element_size(),
    scales_buffer_bytes=s.numel()*s.element_size(),no_requantization=True)
  save(out/'fused_ffup_operand_receipt.json',operand_receipt)
  ext=extension();dense_ext=dense_extension()
  checked=ext.decode_check(p,cb,s,h,bits)
  assert torch.equal(checked.view(torch.int16),weight.view(torch.int16)),'Fused codec changed any BF16 weight bit'
  del checked
  empty=torch.empty((0,),device=x.device,dtype=x.dtype);xf=x.reshape(41,1536)
  def candidate(c):return ext.fused(xf,p,cb,s,h,b,weight if c['id']==0 else empty,bits,c['id']).reshape_as(baseline)
  supported=[];unavailable=[]
  for c in CONFIGS:
   if 'register' in c['kind'] and bits not in (5,6):
    unavailable.append(dict(config=c,reason='Register LUT explicitly supports32/64entries'));continue
   z=candidate(c);assert torch.isfinite(z).all()
   supported.append(c)
  del z
  fp64=(xf.double()@weight.double().t()+b.double()).reshape_as(baseline)
  rng_before=torch.cuda.get_rng_state().clone();before=telemetry()
  def native():return F.linear(x,weight,b)
  def split():return F.linear(x,decode(),b)
  def dense():return dense_ext.dense(xf,weight,b,0).reshape_as(baseline)
  controls={}
  for label,fn in [('native_before',native),('cutlass_dense_before',dense),('split_before',split)]:
   t,z=measure(fn,contract);assert torch.equal(z,baseline);controls[label]=t
  rows=[]
  for c in supported:
   t,z=measure(lambda:candidate(c),contract)
   err=delta(z,baseline);assert err['finite']
   row=dict(config=c,timing=t,vs_native=err,vs_fp64=delta(z,fp64))
   rows.append(row)
   save(out/'fused_ffup_probe.json',dict(status='FUSED_PROBE_RUNNING',operand=operand_receipt,controls=controls,trials=rows,unavailable=unavailable))
   print('FUSED_FFUP_TRIAL',c,t['median_us'],err,flush=True)
  for label,fn in [('split_after',split),('cutlass_dense_after',dense),('native_after',native)]:
   t,z=measure(fn,contract);assert torch.equal(z,baseline);controls[label]=t
  best=min((r for r in rows if r['config']['id']!=0),key=lambda r:r['timing']['median_us'])
  confirmations=[]
  for order in [('split','fused'),('fused','split')]:
   block={}
   for label in order:
    t,z=measure(split if label=='split' else lambda:candidate(best['config']),contract)
    err=delta(z,baseline);assert err['finite']
    block[label]=dict(timing=t,vs_native=err)
   block['order']=list(order)
   block['speedup']=block['split']['timing']['median_us']/block['fused']['timing']['median_us']
   block['saved_us']=block['split']['timing']['median_us']-block['fused']['timing']['median_us']
   confirmations.append(block)
  assert torch.equal(rng_before,torch.cuda.get_rng_state())
  assert bitsha(x)==cfg['input_sha256'] and bitsha(weight)==operand_receipt['weight_sha256']
  for t,k in [(p,'packed'),(cb,'codebook'),(s,'scales'),(h,'shift'),(b,'bias')]:assert bitsha(t)==operand_receipt[k+'_sha256']
  result=dict(status='FUSED_FFUP_CHECK_COMPLETE',diagnostic_only=True,model_dispatch_changed=False,
    permanent_expanded_weight_cache_installed=False,accuracy_acceptance=False,
    current25actions_rng_exact=True,weights_exact=True,micro_rng_unchanged=True,
    operand=operand_receipt,controls=controls,trials=rows,unavailable=unavailable,
    best_config=best['config'],confirmations=confirmations,confirmed_faster_in_both_blocks=all(r['speedup']>1 for r in confirmations),
    telemetry_before=before,telemetry_after=telemetry(),native_vs_fp64=delta(baseline,fp64),
    fused_global_scratch_bytes=0,
    interpretation='Actual first FF-up only. Original SACQ compressed payload decoded to original CUTLASS shared stages; BF16 weights exact, FP32 MMA accumulation/BF16 output. Dense sync-supply control separates shared-store integration overhead. Original CUDAgraph5+20 and two reversed independent winner/control confirmation blocks. Output errors recorded without tolerance or policy adoption. No whole-model speedup claim.')
  save(out/'fused_ffup_probe.json',result)
  print('FUSED_FFUP_COMPLETE',json.dumps(dict(best=result['best_config'],confirmations=confirmations)),flush=True)
 return result
