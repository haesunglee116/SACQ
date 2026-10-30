"""Exact-output candidate: stock GEMM, fused decode and fixed AdaLN modulations."""
import torch
from types import SimpleNamespace
from transformers.feature_extraction_utils import BatchFeature
import hf_capture_safe_2026_08_31 as hffix
import hqq_fused_dequant_2026_08_31 as hqqf
from fixed_adaln_r2 import FixedAdaLN
from exact_pointwise import euler_update_
import qkv_shared_aux_r3 as qkvc
import kv_hoist_a8_logical_r2 as kvh
from graph_wrap_2026_08_31 import graph_wrap_forward


def install(policy, bi, ai, packed, graph_pool=None, graph_stream=None):
    device = next(policy.model.parameters()).device
    args = SimpleNamespace(graph_vlm=True, vlm_fa2=True, graph_vision=True)
    result = {}
    ah = policy.model.action_head
    # Apply the same inference-only structural simplifications to both arms.
    # SACQ's approved installer already performs these transformations.
    import gr00t.eval.cosmos_quant_shim as shim
    result['single_embodiment_newly_materialized'] = shim.materialize_single_embodiment_action_head(
        policy.model, int(ai.embodiment_id.item()), str(device))
    policy.model.backbone.model.lm_head = torch.nn.Identity()
    result['unused_lm_head_removed'] = True
    adaln = FixedAdaLN([b.norm1 for b in ah.model.transformer_blocks])
    original_bo = policy.model.backbone(bi)
    from fixed_schedule_outputs import FixedScheduleOutputs
    schedule = FixedScheduleOutputs([
        ('dit_timestep_encoder',ah.model.timestep_encoder),
        ('dit_output_modulation',ah.model.proj_out_1),
        ('action_sinusoidal_time',ah.action_encoder.pos_encoding),
        ('action_position_embedding',ah.position_embedding)])
    schedule.prepare(lambda: adaln.prepare(lambda: ah.get_action(original_bo, ai)))
    assert schedule.steps == ah.num_inference_timesteps
    result['fixed_schedule_outputs'] = schedule.receipt()
    assert adaln.steps == ah.num_inference_timesteps
    result['adaln'] = dict(mode='stock_output_cache', bytes=adaln.bytes, steps=adaln.steps)
    if packed:
        from decoder_tuning import (
            dequantize as dequantize_packed_codebook_weight, linear as packed_fused_dequant_codebook_linear)
        from exact_dictionary_decode import install as fuse_dictionary
        checked = []
        for name, layer in policy.model.named_modules():
            if type(layer) is shim.PrecomputedPackedVectorCodebookLinear:
                original_weight = layer.dequantize_weight()
                decoded = dequantize_packed_codebook_weight(layer)
                if not torch.equal(original_weight.view(torch.int16), decoded.view(torch.int16)):
                    raise RuntimeError('Codebook weight bit pattern mismatch: ' + name)
                layer.forward = lambda x, _layer=layer: packed_fused_dequant_codebook_linear(x, _layer)
                checked.append(name)
                del original_weight, decoded
        result['weight_exact_layers'] = checked
        result['dictionary_layers'] = fuse_dictionary(policy.model)
        state = hqqf.install(policy.model)
        if state['mismatch']:
            raise RuntimeError('HQQ reconstruction mismatch')
        result['hqq_patched'] = state['patched']
    if packed:
        from language_a8_input_sharing_r1 import install as share_language_inputs
        result['language_a8_input_sharing']=share_language_inputs(policy)
    kv = kvh.install(policy)
    result['qkv'] = qkvc.install(policy)
    if args.graph_vlm:
        # The backbone consumes hidden_states[-1], measured to be the pre-final-norm
        # residual. That tuple is gathered by hooks during the real forward, so a
        # replay cannot produce it; clone the norm's input instead, which the graph
        # then rewrites on every replay.
        hf = policy.model.backbone.model
        textm = hf.model.language_model
        keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        stash = {}
        textm.norm.register_forward_pre_hook(
            lambda m, inp: stash.__setitem__("v", inp[0].clone()))
        from transformers.feature_extraction_utils import BatchFeature
        bb = policy.model.backbone

        def patched_backbone(vl_input):
            bb.set_frozen_modules_to_eval_mode()
            vin = {k: vl_input[k] for k in keys}
            hf(**vin)
            return BatchFeature(data={
                "backbone_features": stash["v"],
                "backbone_attention_mask": vin["attention_mask"] == 1,
                "image_mask": vin["input_ids"] == hf.config.image_token_id})

        with torch.no_grad():
            stock_ref = policy.model.backbone(bi).backbone_features.clone()
        bb.forward = patched_backbone
        with torch.no_grad():
            hook_ref = policy.model.backbone(bi).backbone_features.clone()
        result["vlm_hook_bitwise"] = bool(torch.equal(hook_ref, stock_ref))
        print(f"[hg] VLM hook substitution bitwise: {result['vlm_hook_bitwise']}",
              flush=True)
        hffix.install_all(textm, use_sdpa=not args.vlm_fa2,
                          configs=(hf.config, getattr(hf.config, "text_config", None),
                                   textm.config),
                          vision_model=hf.model.visual if args.vlm_fa2 else None)
        with torch.no_grad():
            for _ in range(4):
                sd = policy.model.backbone(bi).backbone_features
        d = sd.float() - stock_ref.float()
        result["vlm_attn_bitwise"] = bool(torch.equal(sd, stock_ref))
        result["vlm_sdpa_deviation"] = {
            "max_abs": float(d.abs().max()),
            "rel_l2": float(d.norm() / stock_ref.float().norm().clamp_min(1e-12))}
        print(f"[hg] VLM SDPA deviation vs FlashAttention: "
              f"{result['vlm_sdpa_deviation']}", flush=True)
        if not result['vlm_hook_bitwise'] or not result['vlm_attn_bitwise']:
            raise RuntimeError('VLM graph preparation changed backbone output')
        from full_backbone_graph_a8_shared_r2 import install as graph_backbone
        result['whole_backbone_graph'] = graph_backbone(policy.model.backbone,bi,graph_pool=graph_pool,graph_stream=graph_stream)
    bo = policy.model.backbone(bi)
    raw_bo = {k:v.clone() for k,v in bo.items()}
    raw_ai = {k:ai[k].clone() for k in ('state','embodiment_id')}
    feat = ah._encode_features(bo, ai)
    vl = feat.backbone_features.clone()
    state = feat.state_features.clone()
    emb = ai.embodiment_id.clone()
    image_mask = bo.image_mask.clone()
    attention_mask = bo.backbone_attention_mask.clone()
    batch = vl.shape[0]
    assert batch == 1 and 'action' not in ai
    actions = torch.zeros((batch, ah.config.action_horizon, ah.action_dim), device=device, dtype=vl.dtype)
    strength = torch.ones_like(actions)
    ts = [torch.full((batch,), int(s / ah.num_inference_timesteps * ah.num_timestep_buckets), device=device)
          for s in range(ah.num_inference_timesteps)]
    pos = torch.arange(ah.config.action_horizon, device=device)
    dt = 1.0 / ah.num_inference_timesteps

    adaln.enable()
    schedule.enable()

    def body():
        # Fresh mapping: the stock encoder mutates backbone_features in-place.
        encoded = ah._encode_features(BatchFeature(data=dict(raw_bo)), BatchFeature(data=raw_ai))
        vl.copy_(encoded.backbone_features)
        state.copy_(encoded.state_features)
        kv.begin_action()
        for step in range(ah.num_inference_timesteps):
            adaln.select_step(step)
            schedule.select_step(step)
            af = ah.action_encoder(actions, ts[step], emb)
            if ah.config.add_pos_embed:
                af = af + ah.position_embedding(pos[:af.shape[1]]).unsqueeze(0)
            kwargs = dict(hidden_states=torch.cat((state, af), dim=1), encoder_hidden_states=vl, timestep=ts[step])
            if ah.config.use_alternate_vl_dit:
                kwargs.update(image_mask=image_mask, backbone_attention_mask=attention_mask)
            pred = ah.action_decoder(ah.model(**kwargs), emb)
            euler_update_(actions, pred[:, -ah.action_horizon:], strength, dt)
            kv.advance()

    stream = graph_stream if graph_stream is not None else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actions.zero_()
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    actions.zero_()
    body()
    reference = actions.clone()
    actions.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=graph_pool,stream=stream):
        body()
    actions.zero_()
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(actions, reference):
        raise RuntimeError('DiT graph differs from identical eager loop')

    def get_action(backbone_output, action_input, options=None):
        if ah.num_inference_timesteps != adaln.steps:
            raise ValueError('Denoising schedule changed; rebuild cache')
        if 'action' in action_input or options:
            raise ValueError('This fixed-input benchmark adapter does not support RTC')
        for k,v in raw_bo.items():
            incoming = backbone_output[k]
            if incoming.shape != v.shape or incoming.dtype != v.dtype:
                raise ValueError('Backbone output layout changed: '+k)
            v.copy_(incoming)
        for k,v in raw_ai.items():
            incoming = action_input[k]
            if incoming.shape != v.shape or incoming.dtype != v.dtype:
                raise ValueError('Action input layout changed: '+k)
            v.copy_(incoming)
        emb.copy_(action_input.embodiment_id)
        image_mask.copy_(backbone_output.image_mask)
        attention_mask.copy_(backbone_output.backbone_attention_mask)
        actions.copy_(torch.randn(size=actions.shape, dtype=vl.dtype, device=device))
        graph.replay()
        backbone_output['backbone_features'] = vl
        return BatchFeature(data=dict(action_pred=actions, backbone_features=vl, state_features=state))

    ah.get_action = get_action
    # CUDA Graph stores addresses, not Python tensor ownership. Keep every
    # captured constant alive after install() returns (in particular timestep
    # and positional indices used by embedding/index_select kernels).
    ah._benchmark_graph_state = dict(graph=graph, body=body, timesteps=ts,
                                    positions=pos, strength=strength, kv=kv, raw_bo=raw_bo, raw_ai=raw_ai, schedule=schedule)
    result['shared_graph_memory_pool'] = graph_pool is not None
    result['dit_graph_exact'] = True
    result['feature_encoder_in_graph'] = True
    result['pointwise_fusion'] = 'AdaLN modulation + Euler; explicit BF16 rounding; FMA disabled'
    result['fixed_input_only'] = True
    return result
