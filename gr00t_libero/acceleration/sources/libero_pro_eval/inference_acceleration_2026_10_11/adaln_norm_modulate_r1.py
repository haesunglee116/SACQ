"""Isolated native-Welford and exact BF16-modulation extension."""
from pathlib import Path
EXT=None
def extension():
 global EXT
 if EXT is None:
  from torch.utils.cpp_extension import load
  p=Path(__file__).resolve().parent/'adaln_norm_modulate_r1'
  b=Path('@LOCAL_RUNTIME@/adaln_norm_modulate_r1_sm89');b.mkdir(exist_ok=True)
  EXT=load(name='adaln_norm_modulate_r1',sources=[str(p/'binding.cpp'),str(p/'kernel.cu')],build_directory=str(b),
   extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v','-gencode=arch=compute_89,code=sm_89'],verbose=True)
 return EXT
