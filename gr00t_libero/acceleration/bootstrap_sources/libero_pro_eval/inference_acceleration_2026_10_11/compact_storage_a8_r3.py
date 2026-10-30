"""Exact storage relocation of model tensors; no repacking or value changes."""
import torch,gc,collections

def compact(model):
    before=torch.cuda.memory_allocated()
    tensors=list(model.parameters())+list(model.buffers())
    groups={};seen=set()
    for t in tensors:
        if not t.is_cuda or id(t) in seen:continue
        seen.add(id(t));st=t.untyped_storage();key=(t.device,st.data_ptr())
        groups.setdefault(key,[]).append(t)
    by_dtype=collections.defaultdict(list);skipped=0
    for ts in groups.values():
        if len({t.dtype for t in ts})!=1:skipped+=1;continue
        t=ts[0];n=t.untyped_storage().nbytes()
        if n%t.element_size():skipped+=1;continue
        by_dtype[(t.device,t.dtype)].append((ts,n))
    arenas=[];padding=[];total_bytes=0;checked=0
    with torch.no_grad():
        for (device,dtype),items in by_dtype.items():
            e=items[0][0][0].element_size()
            total=sum((n+255)//256*256 for _,n in items)
            arena=torch.empty(total//e,dtype=dtype,device=device);arenas.append(arena);offset=0
            for ts,n in items:
                first=ts[0]
                old=torch.as_strided(first,(n//e,),(1,),storage_offset=0)
                target=arena.narrow(0,offset//e,n//e)
                target.copy_(old)
                assert torch.equal(target.view(torch.uint8),old.view(torch.uint8))
                for t in ts:
                    view=arena.as_strided(t.size(),t.stride(),offset//e+t.storage_offset())
                    assert torch.equal(t.contiguous().reshape(-1).view(torch.uint8),view.contiguous().reshape(-1).view(torch.uint8))
                    t.data=view;checked+=1
                del old,target,view
                padded=(n+255)//256*256
                if padded>n:padding.append((arena,offset+n,padded-n))
                offset+=padded
            total_bytes+=total
    model._compact_storage_arenas=arenas
    model._compact_storage_padding=padding
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    return dict(before_bytes=before,after_bytes=torch.cuda.memory_allocated(),arena_bytes=total_bytes,tensors_exact=checked,skipped_mixed_dtype_storages=skipped,alignment_bytes=256)


def relocate_a8_scales(model):
    scales=getattr(model,'_a8_weight_scales_ref',None)
    if scales is None:return {'applied':False}
    ids=list(scales)
    cursor=0;used=[]
    for arena,offset,bytes_available in model._compact_storage_padding:
        # The original payload is chiefly BF16. Only reinterpret unused bytes;
        # preserve every full original storage span and align FP32 pointers.
        aligned=(offset+3)//4*4
        available=bytes_available-(aligned-offset)
        count=min(available//4,len(ids)-cursor)
        if count<=0:continue
        assert aligned>=offset and aligned+count*4<=offset+bytes_available
        # This interval is strictly outside every original tensor's entire
        # storage span, not a guessed gap between logical views.
        target=arena.view(torch.uint8).narrow(0,aligned,count*4).view(torch.float32)
        for slot in range(count):
            key=ids[cursor+slot]
            source=scales[key]
            destination=target[slot]
            destination.copy_(source)
            assert torch.equal(destination.view(torch.int32),source.view(torch.int32))
            scales[key]=destination
        used.append({'offset_bytes':aligned,'bytes':count*4,'available_bytes':available,'original_dtype':str(arena.dtype)})
        cursor+=count
        if cursor==len(ids):break
    if cursor!=len(ids):raise RuntimeError('Not enough proven arena padding for A8 scales')
    return {'applied':True,'scale_count':cursor,'logical_bytes':cursor*4,
            'new_gpu_allocations':0,'exact_fp32':True,'padding_intervals':used}
