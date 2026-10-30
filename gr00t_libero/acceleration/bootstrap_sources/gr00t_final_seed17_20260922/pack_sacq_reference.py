"""Lossless SACQ index serialization for the existing native reference decoder.

No index reassignment, calibration, codebook/scale change or dense cache.
Whole-policy action equivalence remains a separate required measurement gate.
"""
import math
import torch

def pack_payload(payload):
    assert 'indices' in payload and 'packed_indices' not in payload
    source=torch.as_tensor(payload['indices']).detach().cpu()
    assert source.dtype in (torch.uint8,torch.int16,torch.int32,torch.int64)
    k=int(payload['codebook'].shape[0]);bits=max(1,(k-1).bit_length())
    assert 1<=bits<=16 and source.numel()>0
    flat=source.reshape(-1).long()
    assert int(flat.min())>=0 and int(flat.max())<k
    positions=torch.arange(flat.numel(),dtype=torch.int64)*bits
    packed=torch.zeros(math.ceil(flat.numel()*bits/8),dtype=torch.int64)
    for offset in range(bits):
        bitpos=positions+offset
        packed.scatter_add_(0,bitpos//8,((flat>>offset)&1)<<(bitpos%8))
    assert bool((packed<=255).all())
    recovered=torch.zeros_like(flat)
    for offset in range(bits):
        bitpos=positions+offset
        recovered|=((packed[bitpos//8]>>(bitpos%8))&1)<<offset
    assert torch.equal(recovered,flat)
    result={key:value for key,value in payload.items() if key!='indices'}
    result.update(packed_indices=packed.to(torch.uint8),index_bits=bits,index_shape=tuple(source.shape))
    return result

def pack_installed_sacq(model):
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear
    names=[name for name,layer in model.named_modules()
           if type(layer) is PrecomputedPackedVectorCodebookLinear and layer.packed_indices is None]
    report=[]
    for name in names:
        old=model.get_submodule(name)
        assert old.runtime_backend=='dense_reconstruct','no cached/graph runtime allowed'
        payload={k:getattr(old,k) for k in ['codebook','indices','scales','shift',
                 'in_features','out_features','group_size','vector_dim']}
        new=PrecomputedPackedVectorCodebookLinear(pack_payload(payload),old.bias,str(old.codebook.device))
        new.runtime_backend='dense_reconstruct'
        new.runtime_scope=old.runtime_scope
        new.runtime_accelerate_vlm=False
        assert torch.equal(new.dequantize_weight().view(torch.int16),old.dequantize_weight().view(torch.int16))
        parent,leaf=name.rsplit('.',1)
        setattr(model.get_submodule(parent),leaf,new)
        report.append(dict(layer=name,index_bits=new.index_bits,weight_bit_pattern_exact=True,
            buffers_bytes=sum(t.numel()*t.element_size() for t in new.buffers())))
    return dict(layers=report,index_reassigned=False,full_action_equivalence=False)
