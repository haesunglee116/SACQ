import torch
from codex_ffdecode_vec4rcpftz_cuda_20261029_r1 import validate_reciprocal
import codex_affine_vec8rcpftz_cuda_20261029_r1 as cuda

def enable(decoder):
 original=decoder._decode_affine_perrow_int8
 summary=dict(reciprocal_checks=0,checks=0,shapes={},fallback_shapes={},no_persistent_cache=True)
 class Kernel:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    assert len(args)==10,len(args)
    count,k,bits,mode,block=args[5:]
    shape=str((count,k,bits,mode))
    if not (count%64==0 and k%64==0 and bits in (2,3,4) and mode in (0,6)):
     summary['fallback_shapes'][shape]=summary['fallback_shapes'].get(shape,0)+1
     return original[grid](*args,**kwargs)
    if not torch.cuda.is_current_stream_capturing():
     if mode==6:
      assert validate_reciprocal(args[3])['reciprocal_bits_exact'];summary['reciprocal_checks']+=1
     ref=torch.empty_like(args[4]);a=list(args);a[4]=ref;original[grid](*a,**kwargs)
     cuda.ext().decode_affine_vec8rcpftz_r1(list(args[:5]),count,k,bits,mode)
     assert torch.equal(args[4],ref),'AffineVec8WeightBytes'
     summary['checks']+=1;summary['shapes'][shape]=summary['shapes'].get(shape,0)+1
    else:cuda.ext().decode_affine_vec8rcpftz_r1(list(args[:5]),count,k,bits,mode)
   return launch
 decoder._decode_affine_perrow_int8=Kernel()
 return summary
