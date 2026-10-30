"""Lossless custom level-dictionary packing for already-installed GPTQ slots.

Not the official GPTQ kernel/layout. No fitting, nearest-level mapping, changing
bits, or persistent dense cache. Caller must separately validate full actions.
"""
import torch
from exact_packed_linear import ExactPackedLinear

def pack_installed_gptq(model, rows):
    assert len({r['layer'] for r in rows})==len(rows)
    report=[]
    for row in rows:
        name=row['layer'];bits=int(row['bits'])
        old=model.get_submodule(name)
        assert type(old) is torch.nn.Linear and old.weight.dtype==torch.bfloat16
        device=old.weight.device
        # Constructor verifies every reconstructed BF16 bit against old weight.
        packed=ExactPackedLinear(old.weight,old.bias,bits,group_size=64).to(device)
        assert torch.equal(packed.reconstruct().view(torch.int16),old.weight.view(torch.int16))
        parent_name,leaf=name.rsplit('.',1)
        setattr(model.get_submodule(parent_name),leaf,packed)
        report.append(dict(layer=name,bits=bits,weight_bit_pattern_exact=True,
            representation='custom_exact_level_dictionary',
            buffers_bytes=sum(x.numel()*x.element_size() for x in packed.buffers())))
        del old,packed
    return dict(layers=report,requantization=False,full_action_equivalence=False,
        official_gptq_kernel=False,persistent_dense_cache=False)
