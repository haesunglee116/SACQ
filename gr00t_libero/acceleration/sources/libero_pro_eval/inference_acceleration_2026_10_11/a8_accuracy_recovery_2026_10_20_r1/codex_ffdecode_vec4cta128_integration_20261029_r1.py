import torch,triton
import codex_ffdecode_vec4cta128_cuda_20261029_r1 as cuda
class Kernel:
 def __getitem__(self,grid):
  def launch(*args,**kwargs):
   assert len(args)==24,len(args)
   assert tuple(args[12:16])==(6144,1536,1536,6144),args[12:16]
   assert tuple(args[22:24])==(8,128)
   cuda.ext().decode_pair_vec4cta128_r1(list(args[:12]),args[16],args[18],args[19],args[21])
  return launch
_decode_grouped_ff_weights=Kernel()
def enable():
 import perrow_int8_fast_policy_r3 as fast
 import int8_refined_policy_20261028_r1 as refined
 from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear as Packed
 previous_install=fast.install
 summary=dict(reciprocal_checks=[],groups=[],weight_checks=0,pair_decodes=0,no_persistent_weight_cache=True)
 def install(policy,names,contract):
  cuda.ext()
  result=previous_install(policy,names,contract)
  state=policy.model._perrow_a8_candidate_state
  previous_decode=refined.decoder.decode
  frames=[]
  def decode(module,mode):
   if not frames or module not in frames[-1]['modules']:return previous_decode(module,mode)
   frame=frames[-1];mods=frame['modules'];modes=frame['modes'];i=next(i for i,m in enumerate(mods) if m is module)
   assert modes[i]==mode
   if 'outputs' not in frame:
    outputs=[torch.empty((m.out_features,m.in_features),dtype=torch.int8,device=m.codebook.device) for m in mods]
    args=[]
    for m,o in zip(mods,outputs):args.extend([m.packed_indices,m.codebook,m.scales,m.shift,o,m._a8_perrow_weight_scale])
    bn,bv=8,128
    grid=sum(triton.cdiv(m.out_features,bn)*triton.cdiv(m.in_features//4,bv) for m in mods)
    _decode_grouped_ff_weights[(grid,)](*args,mods[0].out_features,mods[0].in_features,mods[1].out_features,mods[1].in_features,mods[0].index_bits,mods[0].index_window_bytes,modes[0],mods[1].index_bits,mods[1].index_window_bytes,modes[1],bn,bv,num_warps=4,enable_fp_fusion=False)
    if state['validation_enabled']:
     for m,mo,out in zip(mods,modes,outputs):
      assert torch.equal(out,previous_decode(m,mo)),'GroupedFFActualWeightBytes'
      summary['weight_checks']+=1
    frame['outputs']=outputs;summary['pair_decodes']+=1
   return frame['outputs'][i]
  refined.decoder.decode=decode
  selected=set(names)
  for i in range(32):
   base=f'action_head.model.transformer_blocks.{i}.ff'
   full=[base+'.net.0.proj',base+'.net.2'];assert all(n in selected for n in full)
   mods=[policy.model.get_submodule(n) for n in full]
   assert all(type(m) is Packed and m.vector_dim==4 and m.group_size==64 and m.shift.numel()==1 and m.packed_indices.data_ptr()%4==0 for m in mods)
   assert [(m.out_features,m.in_features) for m in mods]==[(6144,1536),(1536,6144)]
   for m in mods:summary['reciprocal_checks'].append(cuda.validate_reciprocal(m._a8_perrow_weight_scale))
   modes=[state['weight_division_modes'][n] for n in full]
   def enter(module,args,_mods=mods,_modes=modes):frames.append(dict(modules=_mods,modes=_modes))
   def leave(module,args,output):frames.pop()
   ff=policy.model.get_submodule(base);ff.register_forward_pre_hook(enter);ff.register_forward_hook(leave,always_call=True)
   summary['groups'].append(full)
  return result
 fast.install=install
 return summary
