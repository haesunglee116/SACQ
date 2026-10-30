"""Register-resident warp LUT and persistent-grid exact BF16 reconstruction."""
from pathlib import Path
import os
import torch

NAME='omni_decode_register_lut_r1_sm89'
BUILD='@LOCAL_RUNTIME@/omni_decode_register_lut_r1_sm89'
CONFIGS=[(vpt,blocks,lut) for vpt in (2,4,8) for blocks in (0,4,8) for lut in (False,True)]
_module=None


def extension():
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load
        Path(BUILD).mkdir(parents=True,exist_ok=True)
        _module=load(name=NAME,sources=[str(Path(__file__).with_suffix('.cu'))],
            build_directory=BUILD,with_cuda=True,extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v','-gencode=arch=compute_89,code=sm_89'],verbose=True)
    return _module


def dequantize(m,config=(4,8,True)):
    assert m.vector_dim==4 and m.group_size==64 and m.shift.numel()==1
    vpt,blocks,lut=config
    return extension().decompress(m.packed_indices,m.codebook,m.scales,m.shift,
        m.out_features,m.in_features,m.index_bits,vpt,blocks,lut)


def linear(x,m,config=(4,8,True)):
    return torch.nn.functional.linear(x,dequantize(m,config),m.bias)


def install(policy,selection):
    import decoder_tuning as decoder
    names={id(m):n for n,m in policy.model.named_modules()};original=decoder.linear
    extension();checked=0
    for m in policy.model.modules():
        cfg=selection.get(names[id(m)])
        if cfg is None:continue
        expected=m.dequantize_weight();actual=dequantize(m,tuple(cfg['config']))
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16)),names[id(m)]
        del expected,actual
        checked+=1
    assert checked==len(selection)>0
    def selected(x,m):
        cfg=selection.get(names[id(m)])
        if cfg is None:return original(x,m)
        return linear(x,m,tuple(cfg['config']))
    decoder.linear=selected
    return dict(selected_layers=checked,weights_exact=True,temporary_weight_shape_unchanged=True,
        persistent_expanded_weights=False,register_cache_lifetime='one kernel execution only',
        register_lut_effective_bits=[5,6],other_bits='same payload, global LUT lookup')
