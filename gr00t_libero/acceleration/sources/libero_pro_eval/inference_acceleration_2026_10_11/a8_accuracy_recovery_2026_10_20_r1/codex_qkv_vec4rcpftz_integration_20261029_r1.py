import torch
from codex_ffdecode_vec4rcpftz_cuda_20261029_r1 import validate_reciprocal
import codex_qkv_vec4rcpftz_cuda_20261029_r1 as cuda
CHOICES=[(2048, 1024, 1024, 2048, 5, 6, 8, 2, 2, 1, 6, 6, 6), (2048, 1024, 1024, 2048, 5, 7, 5, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 6, 6, 6, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 5, 5, 6, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 6, 7, 5, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 6, 7, 6, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 7, 5, 8, 2, 2, 1, 6, 6, 6), (2048, 1024, 1024, 2048, 7, 6, 5, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 6, 5, 6, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 5, 5, 5, 2, 2, 2, 6, 6, 6), (2048, 1024, 1024, 2048, 5, 9, 5, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 5, 6, 6, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 5, 5, 6, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 5, 5, 5, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 6, 5, 6, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 5, 6, 5, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 6, 6, 5, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 7, 6, 6, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 7, 5, 7, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 8, 5, 6, 1, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 7, 7, 6, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 7, 9, 5, 2, 2, 2, 6, 6, 6), (1536, 1536, 1536, 1536, 7, 10, 10, 2, 2, 2, 6, 6, 6)]

def enable(decoder):
 original=decoder._decode_grouped_qkv_weights
 summary=dict(reciprocal_checks=0,group_checks=0,weight_checks=0,shapes={},fallback={},no_persistent_cache=True)
 class Kernel:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    assert len(args)==33,len(args)
    key=tuple(args[18:31]);shape=str(key)
    if key not in CHOICES:
     summary['fallback'][shape]=summary['fallback'].get(shape,0)+1
     return original[grid](*args,**kwargs)
    choice=CHOICES.index(key)
    if not torch.cuda.is_current_stream_capturing():
     for j in [5,11,17]:
      assert validate_reciprocal(args[j])['reciprocal_bits_exact'];summary['reciprocal_checks']+=1
     a=list(args);refs=[]
     for j in [4,10,16]:
      ref=torch.empty_like(args[j]);refs.append(ref);a[j]=ref
     original[grid](*a,**kwargs)
     cuda.ext().decode_qkv_vec4rcpftz_r1(list(args[:18]),choice)
     assert all(torch.equal(args[j],v) for j,v in zip([4,10,16],refs)),'QKVVec4WeightBytes'
     summary['group_checks']+=1;summary['weight_checks']+=3;summary['shapes'][shape]=summary['shapes'].get(shape,0)+1
    else:cuda.ext().decode_qkv_vec4rcpftz_r1(list(args[:18]),choice)
   return launch
 decoder._decode_grouped_qkv_weights=Kernel()
 return summary
