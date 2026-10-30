"""Full backbone graph for the approved fixed prompt/layout; pixels remain inputs.

Only layout-dependent metadata is cached. Different tokens, masks or geometry
are explicitly rejected. This diagnostic adapter is not a general rollout API.
"""
import torch
from transformers.feature_extraction_utils import BatchFeature


def install(backbone, bi, graph_pool=None, graph_stream=None):
    hf = backbone.model
    model = hf.model
    visual = model.visual
    keys = ['input_ids','attention_mask','pixel_values','image_grid_thw']
    static = {k:bi[k].clone() for k in keys}
    fixed_keys = ['input_ids','attention_mask','image_grid_thw']
    frozen = {k:bi[k].clone() for k in fixed_keys}
    original = backbone.forward
    reference = original(bi).backbone_features.clone()
    grid = static['image_grid_thw']
    split_sizes = (grid.prod(-1)//visual.spatial_merge_size**2).tolist()
    positions = model.get_rope_index(static['input_ids'],grid,None,attention_mask=static['attention_mask'])
    pos_embeds = visual.fast_pos_embed_interpolate(grid)
    rotary = visual.rot_pos_emb(grid)
    emb = torch.cat((rotary,rotary),dim=-1)
    position_embeddings = (emb.cos(),emb.sin())
    cu_seqlens = torch.repeat_interleave(grid[:,1]*grid[:,2],grid[:,0]).cumsum(dim=0,dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens,(1,0),value=0)

    def vision_forward(hidden_states, grid_thw, **kwargs):
        hidden_states = visual.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.reshape(hidden_states.size(0),-1)
        deep = []
        for i,block in enumerate(visual.blocks):
            hidden_states = block(hidden_states,cu_seqlens=cu_seqlens,position_embeddings=position_embeddings,**kwargs)
            if i in visual.deepstack_visual_indexes:
                deep.append(visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(i)](hidden_states))
        return visual.merger(hidden_states),deep
    visual.forward = vision_forward

    def get_image_features(pixel_values, image_grid_thw=None):
        images,deep = visual(pixel_values.type(visual.dtype),grid_thw=image_grid_thw)
        return torch.split(images,split_sizes),deep
    model.get_image_features = get_image_features
    model.get_rope_index = lambda *args,**kwargs:positions
    stock_masks = model.get_placeholder_mask
    cache = {}
    def get_masks(*args,**kwargs):
        if torch.cuda.is_current_stream_capturing():
            if 'masks' not in cache:
                raise RuntimeError('Placeholder masks were not primed')
            return cache['masks']
        value = stock_masks(*args,**kwargs)
        cache['masks'] = tuple(t.clone() for t in value)
        return value
    model.get_placeholder_mask = get_masks
    prepared = original(static).backbone_features.clone()
    if not torch.equal(prepared,reference):
        raise RuntimeError('Fixed-layout backbone metadata preparation changed output')
    stream = graph_stream if graph_stream is not None else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            original(static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=graph_pool,stream=stream):
        output = original(static)
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(output.backbone_features,reference):
        raise RuntimeError('Whole-backbone graph output mismatch')

    # Action-head preprocessing replaces values in its input BatchFeature.
    # Preserve graph-owned tensors separately and return a fresh container.
    captured_tensors = dict(output)

    def forward(vl_input):
        for k in keys:
            if vl_input[k].shape != static[k].shape or vl_input[k].dtype != static[k].dtype:
                raise ValueError('Fixed-demo layout changed: '+k)
        for k in fixed_keys:
            if not torch.equal(vl_input[k],frozen[k]):
                raise ValueError('Fixed-demo tokens/mask/grid changed: '+k)
        static['pixel_values'].copy_(vl_input['pixel_values'])
        graph.replay()
        return BatchFeature(data=dict(captured_tensors))
    backbone.forward = forward
    backbone._whole_graph_owner = dict(graph=graph,static=static,frozen=frozen,output=output,
        positions=positions,pos_embeds=pos_embeds,position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,mask_cache=cache,original=original)
    return dict(captured=True,exact=True,fixed_prompt_and_grid=True,pixels_recomputed_each_query=True)
