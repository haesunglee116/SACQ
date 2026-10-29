"""Lossless reference packing of already-quantized BF16 weights for memory tests.

This is NOT the original GPTQ packed format/kernel. It stores the exact observed
BF16 level dictionary per row/group plus packed indices. No scale inference or
requantization. Report this custom representation separately and include every
buffer in measured residency. It never retains a dense reconstruction cache.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ExactPackedLinear(nn.Module):
    def __init__(self, weight, bias, bits, group_size=64):
        super().__init__()
        assert bits in (2,3,4) and weight.dtype == torch.bfloat16
        assert weight.ndim == 2 and weight.shape[1] % group_size == 0
        self.out_features,self.in_features=weight.shape
        self.bits,self.group_size=bits,group_size
        self.groups=self.in_features//group_size
        source=weight.detach().cpu().contiguous().reshape(-1,group_size)
        levels=1<<bits
        ordered=source.sort(dim=1).values
        first=torch.ones_like(ordered,dtype=torch.bool)
        first[:,1:]=ordered[:,1:]!=ordered[:,:-1]
        counts=first.sum(dim=1)
        assert bool((counts<=levels).all()),'not representable at requested bits without requantization'
        indices_in_order=first.long().cumsum(dim=1)-1
        dictionary=torch.zeros((source.shape[0],levels),dtype=source.dtype)
        row=torch.arange(source.shape[0]).unsqueeze(1).expand_as(source)
        dictionary[row[first],indices_in_order[first]]=ordered[first]
        # Find exact existing levels, not nearest-level quantization.
        ids=torch.searchsorted(dictionary.sort(dim=1).values,source)
        dictionary=dictionary.sort(dim=1).values.contiguous()
        assert torch.equal(dictionary.gather(1,ids),source)
        flat=ids.flatten().to(torch.int64)
        nbytes=math.ceil(flat.numel()*bits/8)
        shifts=torch.arange(flat.numel(),dtype=torch.int64)*bits
        packed=torch.zeros(nbytes+1,dtype=torch.int64)
        packed.scatter_add_(0,shifts//8,(flat<<(shifts%8))&255)
        packed.scatter_add_(0,shifts//8+1,flat>>(8-shifts%8))
        assert bool((packed<=255).all())
        self.register_buffer('packed',packed.to(torch.uint8))
        self.register_buffer('levels',dictionary)
        self.register_buffer('bias',None if bias is None else bias.detach().cpu().clone())
        assert torch.equal(self.reconstruct().view(torch.int16),weight.detach().cpu().view(torch.int16)), 'BF16 bit pattern mismatch'

    def reconstruct(self):
        count=self.out_features*self.in_features
        pos=torch.arange(count,device=self.packed.device,dtype=torch.int64)*self.bits
        pair=self.packed[pos//8].to(torch.int64) | (self.packed[pos//8+1].to(torch.int64)<<8)
        ids=((pair>>(pos%8))&((1<<self.bits)-1)).reshape(-1,self.group_size)
        return self.levels.gather(1,ids).reshape(self.out_features,self.in_features)

    def forward(self,x):
        return F.linear(x,self.reconstruct(),self.bias)
