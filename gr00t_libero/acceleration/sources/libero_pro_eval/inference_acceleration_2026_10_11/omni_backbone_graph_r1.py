"""Exact-gated Transformers 5 fixed-layout backbone CUDA Graph.

Only token/grid metadata is fixed. Vision features and all model projections
are recomputed. Every replay copies pixels; tokens/masks/geometry are checked.
"""
import torch
from transformers.feature_extraction_utils import BatchFeature


def install(policy, bi):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    bb = policy.model.backbone
    hf = bb.model.model
    visual = hf.visual
    text = hf.language_model
    original = bb.forward
    keys = ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids']
    static = {k: bi[k].clone() for k in keys if k in bi}
    frozen = {k: v.clone() for k, v in static.items() if k != 'pixel_values'}
    reference = {k: v.clone() for k, v in original(bi).items()}
    holders = []

    def freeze_function(owner, name):
        old = getattr(owner, name)
        cache = []
        def call(*args, **kwargs):
            if not cache:
                cache.append(old(*args, **kwargs))
            return cache[0]
        setattr(owner, name, call)
        holders.append((owner, name, old, cache))

    for name in ['get_vision_interpolation_indices_and_weights', 'get_vision_position_ids',
                 'get_vision_attention_seqlens', 'create_causal_mask']:
        freeze_function(qwen, name)
    freeze_function(hf, 'compute_3d_position_ids')
    freeze_function(hf, 'get_placeholder_mask')
    split_sizes = (static['image_grid_thw'].prod(-1) // visual.spatial_merge_size**2).tolist()

    def get_images(pixel_values, image_grid_thw=None, **kwargs):
        opts = dict(kwargs); opts['return_dict'] = True
        y = visual(pixel_values.type(visual.dtype), grid_thw=image_grid_thw, **opts)
        y.pooler_output = torch.split(y.pooler_output, split_sizes)
        y.deepstack_features = [torch.split(x, split_sizes) for x in y.deepstack_features]
        return y
    hf.get_image_features = get_images

    # SDPA vision chunks have fixed lengths, but stock calls lengths.tolist().
    # Cache the lengths only; preserve the exact stock QKV/attention arithmetic.
    for block in visual.blocks:
        attn = block.attn
        assert attn.config._attn_implementation == 'sdpa', attn.config._attn_implementation
        lengths = []
        def attention(hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None,
                      _m=attn, _lengths=lengths, **kwargs):
            if not _lengths:
                _lengths.extend((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
            seq = hidden_states.shape[0]
            q, k, v = _m.qkv(hidden_states).reshape(seq, 3, _m.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            cos, sin = position_embeddings
            q, k = qwen.apply_rotary_pos_emb_vision(q, k, cos, sin)
            q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
            fn = qwen.ALL_ATTENTION_FUNCTIONS.get_interface(_m.config._attn_implementation, qwen.eager_attention_forward)
            splits = [torch.split(x, _lengths, dim=2) for x in (q, k, v)]
            out = [fn(_m, q, k, v, attention_mask=None, scaling=_m.scaling,
                      dropout=0.0 if not _m.training else _m.attention_dropout,
                      is_causal=False, **kwargs)[0] for q, k, v in zip(*splits)]
            return _m.proj(torch.cat(out, dim=1).reshape(seq, -1).contiguous())
        attn.forward = attention

    deep_indices = []
    def deepstack(hidden_states, mask, visual_embeds):
        assert hidden_states.shape[0] == 1
        if not deep_indices:
            deep_indices.append(mask.reshape(-1).nonzero(as_tuple=False).reshape(-1))
        indices = deep_indices[0]
        y = hidden_states.clone()
        flat = y.reshape(-1, y.shape[-1])
        value = flat.index_select(0, indices) + visual_embeds.to(y.device, y.dtype)
        flat.index_copy_(0, indices, value)
        return y
    text._deepstack_process = deepstack
    prepared = original(BatchFeature(data=dict(static)))
    assert all(torch.equal(prepared[k], v) for k, v in reference.items()), 'Backbone metadata preparation changed output'
    assert all(cache for _, _, _, cache in holders)
    owner = policy.model.action_head._omni_action_graph_owner
    stream = owner['stream']
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            original(BatchFeature(data=dict(static)))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream, pool=owner['graph'].pool()):
        output = original(BatchFeature(data=dict(static)))
    captured = dict(output)
    g.replay(); torch.cuda.synchronize()
    assert all(torch.equal(captured[k], v) for k, v in reference.items()), 'Backbone graph output mismatch'

    def forward(inputs):
        for k, v in static.items():
            x = inputs[k]
            if x.shape != v.shape or x.dtype != v.dtype:
                raise ValueError('Backbone graph layout changed: ' + k)
        for k, v in frozen.items():
            if not torch.equal(inputs[k], v):
                raise ValueError('Fixed-demo tokens/mask/grid changed: ' + k)
        static['pixel_values'].copy_(inputs['pixel_values'])
        g.replay()
        return BatchFeature(data=dict(captured))
    bb.forward = forward
    bb._omni_backbone_graph_owner = dict(graph=g, static=static, frozen=frozen, captured=captured,
        holders=holders, deep_indices=deep_indices, stream=stream, original=original)
    return dict(exact_preparation=True, exact_graph=True, fixed_prompt_and_grid=True,
                pixels_recomputed=True, all_features_recomputed=True, shared_action_graph_pool=True,
                attention_implementation_unchanged=True, decoded_weight_cache=False)
