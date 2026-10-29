import torch
import weakref
from codex_ffdecode_vec4rcpftz_cuda_20261029_r1 import validate_reciprocal
import codex_codebook_vec4rcpftz_cuda_20261029_r2 as cuda

def enable(decoder):
 original=decoder._decode_codebook_perrow_int8
 validated={}
 summary=dict(reciprocal_checks=0,checks=0,shapes={},fallback_shapes={},no_persistent_cache=True)
 class Kernel:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    assert len(args)==16,len(args)
    n,k,groups,gs,vd,bits,iw,mode,bn,bv=args[6:]
    shape=str((n,k,gs,bits,mode))
    if not (k in (1536,2048,6144) and gs==64 and vd==4 and 5<=bits<=10 and mode in (0,6)):
     summary['fallback_shapes'][shape]=summary['fallback_shapes'].get(shape,0)+1
     return original[grid](*args,**kwargs)
    assert groups==k//64 and n*k%16==0
    if not torch.cuda.is_current_stream_capturing():
     if mode==6:
      scale=args[5];key=id(scale);old=validated.get(key);version=None if torch.is_inference(scale) else scale._version
      if version is None or old is None or old[0]() is not scale or old[1]!=version:
       check=validate_reciprocal(scale);assert check['reciprocal_bits_exact']
       validated[key]=(weakref.ref(scale),version);summary['reciprocal_checks']+=1
     ref=torch.empty_like(args[4]);a=list(args);a[4]=ref;original[grid](*a,**kwargs)
     cuda.ext().decode_single_vec4rcpftz_r1(list(args[:6]),n,k,bits,mode)
     assert torch.equal(args[4],ref),'GenericVec4WeightBytes'
     summary['checks']+=1;summary['shapes'][shape]=summary['shapes'].get(shape,0)+1
    else:cuda.ext().decode_single_vec4rcpftz_r1(list(args[:6]),n,k,bits,mode)
   return launch
 decoder._decode_codebook_perrow_int8=Kernel()
 return summary
