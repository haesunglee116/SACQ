import torch
from compact_storage_a8_r3 import compact, relocate_a8_scales
from exact_optimizations_a8_shared_r6 import install as original_install

def install(policy,bi,ai,packed):
    import gr00t.eval.cosmos_quant_shim as shim
    shim.materialize_single_embodiment_action_head(policy.model,int(ai.embodiment_id.item()),str(next(policy.model.parameters()).device))
    policy.model.backbone.model.lm_head=torch.nn.Identity()
    storage=compact(policy.model)
    storage['a8_scales_in_padding']=relocate_a8_scales(policy.model)
    print('COMPACT',storage,flush=True)
    pool=torch.cuda.graph_pool_handle()
    stream=torch.cuda.Stream()
    result=original_install(policy,bi,ai,packed,graph_pool=pool,graph_stream=stream)
    result['storage_compaction']=storage
    return result
