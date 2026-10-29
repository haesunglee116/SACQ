"""Exact bool mask -> stride-aligned BF16 bias, no key/value padding.

Preserve diffusers' original mask shape/repeat logic and the existing SDPA
backend. Only replace SDPA's repeated fill/where construction of its bias.
"""
import torch
import triton
import triton.language as tl
from omni_a8_scope_ffdown_runtime_r1 import exact


@triton.jit
def _bool_mask_to_bf16_bias(MASK, OUT, ROWS:tl.constexpr, COLS:tl.constexpr, PAD:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//PAD;col=i%PAD
    keep=tl.load(MASK+row*COLS+col,(row<ROWS)&(col<COLS),False)
    value=tl.where(col<COLS,tl.where(keep,0.0,float('-inf')),0.0).to(tl.bfloat16)
    tl.store(OUT+i,value,row<ROWS)


def install(policy,c):
    s=dict(enabled=True,checks={},shapes={},cache_hits=0,cache_misses=0,validated_cache_hits=0,validated_cache_misses=0)
    frames=[]
    parent=policy.model.action_head.model
    def enter(module,args):frames.append({})
    def leave(module,args,output):frames.pop()
    parent.register_forward_pre_hook(enter)
    parent.register_forward_hook(leave,always_call=True)
    for index,b in enumerate(parent.transformer_blocks):
        if index%2:continue
        attn=b.attn1
        assert attn.processor.__class__.__name__=='AttnProcessor2_0'
        old=attn.prepare_attention_mask
        name=f'action_head.model.transformer_blocks.{index}.attn1'
        s['checks'][name]=0;s['shapes'][name]=set()
        def prepare(mask,target_length,batch_size,out_dim=3,_old=old,_name=name,_heads=attn.heads):
            assert frames,'CrossMaskOutsideDiTForward'
            assert mask.dtype==torch.bool
            key=(id(mask),mask.data_ptr(),tuple(mask.shape),tuple(mask.stride()),target_length,batch_size,out_dim,_heads)
            frame=frames[-1];hit=key in frame
            if hit:
                source,bias=frame[key]
                assert source is mask
                s['cache_hits']+=1
            else:
                prepared=_old(mask,target_length,batch_size,out_dim=out_dim)
                assert prepared.dtype==torch.bool and prepared.is_contiguous()
                cols=prepared.shape[-1];pad=triton.cdiv(cols,8)*8;rows=prepared.numel()//cols
                storage=torch.empty((*prepared.shape[:-1],pad),dtype=torch.bfloat16,device=prepared.device)
                _bool_mask_to_bf16_bias[(triton.cdiv(rows*pad,1024),)](prepared,storage,rows,cols,pad,1024,num_warps=4)
                bias=storage[...,:cols]
                assert all(v%8==0 for v in bias.stride()[:-1]) and bias.stride(-1)==1
                frame[key]=(mask,bias);s['cache_misses']+=1
            if policy.model._perrow_a8_candidate_state['validation_enabled']:
                prepared=_old(mask,target_length,batch_size,out_dim=out_dim)
                reference=torch.where(prepared,0.0,float('-inf')).to(torch.bfloat16)
                assert exact(bias,reference),(_name,'SharedDiTMaskBias')
                s['checks'][_name]+=1;s['shapes'][_name].add(tuple(prepared.shape))
                s['validated_cache_hits' if hit else 'validated_cache_misses']+=1
            return bias
        attn.prepare_attention_mask=prepare
    assert len(s['checks'])==c['mask_bias_fusion']['module_count']==16
    policy.model._mask_bias_fusion_state=s
    return dict(module_count=16,mask_values='True->BF16+0;False->BF16-inf',cache_lifetime='one DiT forward only',original_SDPA_backend=True,persistent_cache=False)


def finish_validation(policy,c):
    s=policy.model._mask_bias_fusion_state
    assert len(s['checks'])==16 and set(s['checks'].values())=={100},s['checks']
    s['enabled']=False
    return dict(status='ALL1600_CROSS_MASK_BIASES_EXACT',checks=dict(s['checks']),
                actual_shapes={k:sorted(v) for k,v in s['shapes'].items()},
                validation_disabled_before_timing=True)


def enable(full_contract):
 import perrow_int8_fast_policy_r3 as fast
 previous=fast.install
 summary={}
 def wrapped(policy,names,contract):
  result=previous(policy,names,contract)
  install(policy,full_contract)
  summary['state']=policy.model._mask_bias_fusion_state
  import torch.nn.functional as F
  old_sdpa=F.scaled_dot_product_attention
  row_state=policy.model._perrow_a8_candidate_state
  summary['alignment']=dict(calls=0,checks=0,layouts=[])
  def sdpa(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
   before=attn_mask;changed=False
   if attn_mask is not None and attn_mask.dtype==torch.bfloat16 and attn_mask.ndim==4 and attn_mask.stride(-1)==1:
    sizes=attn_mask.shape;strides=attn_mask.stride()
    if all(size==1 or stride%8==0 for size,stride in zip(sizes[:-1],strides[:-1])):
     new_strides=tuple(0 if size==1 else stride for size,stride in zip(sizes[:-1],strides[:-1]))+(1,)
     if new_strides!=strides:
      attn_mask=attn_mask.as_strided(sizes,new_strides);changed=True
      summary['alignment']['calls']+=1
      layout=[list(sizes),list(strides),list(new_strides)]
      if layout not in summary['alignment']['layouts']:summary['alignment']['layouts'].append(layout)
   output=old_sdpa(q,k,v,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
   if changed and row_state['validation_enabled']:
    assert dropout_p==0.0
    reference=old_sdpa(q,k,v,before,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
    assert torch.equal(output.view(torch.int16),reference.view(torch.int16)),'AlignedBiasSDPABytes'
    summary['alignment']['checks']+=1
   return output
  F.scaled_dot_product_attention=sdpa
  return result
 fast.install=wrapped
 return summary
