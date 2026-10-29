import torch

def enable():
 import int8_small_m_frozen_20261028_r2 as small
 import int8_large_m_frozen_20261028_r1 as large
 summary=dict(changed=0,byte_checks=0,shapes={},extra_persistent_weight_bytes=0)
 class Kernel:
  def __init__(self,original):self.original=original
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    assert len(args)==13,len(args)
    if tuple(args[10:13])!=(64,64,128):return self.original[grid](*args,**kwargs)
    assert args[8]*127*127<=torch.iinfo(torch.int32).max
    newkw=dict(kwargs);newkw['num_stages']=6
    self.original[grid](*args,**newkw)
    if not torch.cuda.is_current_stream_capturing():
     ref=torch.empty_like(args[5]);oldargs=list(args);oldargs[5]=ref
     self.original[grid](*oldargs,**kwargs)
     assert torch.equal(ref.view(torch.int16),args[5].view(torch.int16)),'GEMMStage6OutputBits'
     summary['byte_checks']+=1
     key=str(tuple(args[6:13]));summary['shapes'][key]=summary['shapes'].get(key,0)+1
    summary['changed']+=1
   return launch
 small._int8_scaled_gemm=Kernel(small._int8_scaled_gemm)
 large._int8_scaled_gemm=Kernel(large._int8_scaled_gemm)
 return summary
