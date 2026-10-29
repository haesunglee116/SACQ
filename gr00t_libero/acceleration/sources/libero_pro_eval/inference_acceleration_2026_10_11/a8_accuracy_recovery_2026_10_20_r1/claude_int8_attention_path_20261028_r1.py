"""Exact eliminations on the DiT attention path. No arithmetic change, no extra memory.

C1  rescale: diffusers' AttnProcessor2_0 ends every attention call with `hidden_states / rescale_output_factor`.
    With factor 1.0 PyTorch runs a multiply-by-1.0 kernel whose output is bit-identical to its input.
    The processor is replaced by an audited copy (source hash of the installed diffusers 0.35.1 `__call__`
    is checked against the installed diffusers 0.38.0 text, fail closed) that omits that call when the factor is exactly 1.0.
C3  sage_crop: the Sage self-attention output is a [1,H,N,48] crop of a 64-padded HND buffer; the processor's
    transpose/reshape copies it before `to_out`. A strided INT8 producer reads the crop directly with the
    same per-row max/scale/divide/round arithmetic (max is order independent), removing the copy.
C2  bias_pad: cross-attention keys have length 149; PyTorch's memory-efficient SDPA pads the BF16 additive
    bias to a multiple of 8 on every call (fill + copy). Padding the backbone states/masks once per DiT
    forward to 152 rows (zero features, masked False -> -inf bias, exp(-inf)=0) makes the bias already aligned.
    The padded cross K/V projections (M=152) replay the frozen tile configuration selected for M=149.
Every variant is validated by the existing whole-model action/RNG byte gates plus the producer byte checks here.
"""
import hashlib
import inspect
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import perrow_int8_fast_policy_r3 as fast
import int8_refined_policy_20261028_r1 as refined
import int8_torch_epilogue_runtime_r2 as integer_mm
import perrow_int8_padded_producer_r1 as padded_producer
from int8_direct_round_producer_20261028_r1 import _divide, round_int8

EXPECTED_CALL_SOURCE = Path(__file__).with_name('claude_attnprocessor2_0_call_diffusers0380.txt')  # installed 0.38.0; body identical to 0.35.1 except 3 type hints
_base_activation = padded_producer.activation


@triton.jit
def _activation_hnd_crop(X, Q, S, M: tl.constexpr, H: tl.constexpr, DO: tl.constexpr, SH: tl.constexpr, SN: tl.constexpr,
                         B: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    K: tl.constexpr = H * DO
    h = col // DO
    d = col % DO
    x = tl.load(X + h * SH + row * SN + d, (row < M) & (col < K), 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    inverse = tl.full((), 1.0 / 127.0, tl.float32)
    scale = tl.where(maximum > 0, maximum * inverse, 1.0)
    normalized = tl.minimum(tl.maximum(_divide(x, scale, MODE), -127.0), 127.0)
    encoded = round_int8(normalized)
    tl.store(Q + row * K + col, encoded, col < K)
    tl.store(S + row, scale)


def hnd_activation(x):
    assert x.ndim == 4 and x.shape[0] == 1 and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    _, H, N, DO = x.shape
    K = H * DO
    mp = max(32, triton.cdiv(N, 16) * 16)
    q = torch.empty((mp, K), device=x.device, dtype=torch.int8)
    s = torch.empty((mp, 1), device=x.device, dtype=torch.float32)
    _activation_hnd_crop[(mp,)](x, q, s, N, H, DO, x.stride(1), x.stride(2), triton.next_power_of_2(K), 0, num_warps=4, enable_fp_fusion=False)
    return q, s


def _normalized_source(text):
    return '\n'.join(line.rstrip() for line in text.rstrip('\n').split('\n')) + '\n'


class ClaudeExactAttnProcessor:
    """Audited copy of diffusers 0.35.1 AttnProcessor2_0.__call__ with the two exact eliminations."""

    def __init__(self, skip_unit_rescale, sage_crop_direct, summary):
        self.skip_unit_rescale = skip_unit_rescale
        self.sage_crop_direct = sage_crop_direct
        self.summary = summary

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            raise RuntimeError('deprecated scale argument not supported in audited processor copy')
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # Sage returns a [1,H,N,48] crop of a 64-padded HND buffer: last stride 1, row stride 64 > 48,
        # head stride = N*64 (head-major). The memory-efficient cross path returns a transposed [1,N,H,48]
        # view (head stride 48 < row stride) and keeps the original reshape path.
        direct = (self.sage_crop_direct and hidden_states.ndim == 4 and hidden_states.shape[0] == 1
                  and hidden_states.shape[-1] == head_dim and hidden_states.stride(-1) == 1
                  and hidden_states.stride(-2) > head_dim and hidden_states.stride(1) == hidden_states.shape[2] * hidden_states.stride(-2)
                  and hidden_states.dtype == query.dtype and getattr(attn.to_out[0], '_claude_hnd_direct', False))
        if direct:
            self.summary['sage_crop_direct_calls'] += 1
        else:
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        if self.skip_unit_rescale and attn.rescale_output_factor == 1.0:
            self.summary['unit_rescale_skipped'] += 1
        else:
            hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def enable(c1=True, c2=False, c3=True, padded_rows_multiple=8, tile_selection=None, activation_selection=None):
    previous_install = fast.install
    selection_rows = None
    activation_rows = None
    if c2:
        assert tile_selection is not None and activation_selection is not None, 'c2 requires the frozen tile and activation selection references'
        raw = Path(tile_selection['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == tile_selection['sha256']
        selection_rows = json.loads(raw)
        raw = Path(activation_selection['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == activation_selection['sha256']
        activation_rows = json.loads(raw)
    summary = dict(c1_skip_unit_rescale=c1, c2_bias_pad=c2, c3_sage_crop_direct=c3, unit_rescale_skipped=0, sage_crop_direct_calls=0,
                   hnd_producer_checks=0, hnd_fast_calls=0, padded_dit_calls=0, padded_linear_calls=0, processor_source_sha256=None,
                   original_rows=None, padded_rows=None, padded_gemm_configs={})

    def install(policy, names, contract):
        result = previous_install(policy, names, contract)
        st = policy.model._perrow_a8_candidate_state
        import diffusers.models.attention_processor as ap
        src = _normalized_source(inspect.getsource(ap.AttnProcessor2_0.__call__))
        expected = _normalized_source(EXPECTED_CALL_SOURCE.read_text())
        summary['processor_source_sha256'] = hashlib.sha256(src.encode()).hexdigest()
        if src != expected:
            raise RuntimeError('Installed diffusers AttnProcessor2_0.__call__ differs from the audited copy; fail closed')
        blocks = policy.model.action_head.model.transformer_blocks
        assert len(blocks) == 32
        if c1 or c3:
            for idx, block in enumerate(blocks):
                attn = block.attn1
                assert type(attn.processor) is ap.AttnProcessor2_0, type(attn.processor)
                assert float(attn.rescale_output_factor) == 1.0 and attn.spatial_norm is None and attn.group_norm is None
                assert not attn.residual_connection and attn.norm_q is None and attn.norm_k is None
                attn.processor = ClaudeExactAttnProcessor(c1, c3, summary)
        if c3:
            for idx, block in enumerate(blocks):
                if idx % 2 == 0:
                    continue
                name = f'action_head.model.transformer_blocks.{idx}.attn1.to_out.0'
                mod = policy.model.get_submodule(name)
                assert name in st['weight_division_modes'], name
                mode = st['weight_division_modes'][name]
                original_forward = mod.forward
                heads = block.attn1.heads
                head_dim = mod.in_features // heads

                def forward_hnd(x, _mod=mod, _orig=original_forward, _mode=mode, _heads=heads, _hd=head_dim):
                    if not (x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == _heads and x.shape[-1] == _hd and x.stride(-1) == 1 and x.stride(-2) > _hd):
                        assert x.ndim != 4, ('UnexpectedFourDimInput', tuple(x.shape), tuple(x.stride()))
                        return _orig(x)
                    n = x.shape[2]
                    if st['validation_enabled']:
                        xc = x.transpose(1, 2).reshape(1, n, _heads * _hd).contiguous()
                        q, s = hnd_activation(x)
                        qr, sr = _base_activation(xc.reshape(n, _heads * _hd), 0)
                        assert torch.equal(q, qr) and torch.equal(s.view(torch.int32), sr.view(torch.int32)), 'HNDCropProducerBytes'
                        summary['hnd_producer_checks'] += 1
                        return _orig(xc)
                    aq, sa = hnd_activation(x)
                    wq = refined.decoder.decode(_mod, _mode)
                    y = integer_mm.linear(aq, wq, sa, _mod._a8_perrow_weight_scale, _mod.bias, True, actual_rows=n)
                    summary['hnd_fast_calls'] += 1
                    return y.reshape(1, n, _mod.out_features)

                mod.forward = forward_hnd
                mod._claude_hnd_direct = True
        if c2:
            dit = policy.model.action_head.model
            original_dit_forward = dit.forward
            signature = inspect.signature(original_dit_forward)

            def padded(value, pad, fill):
                if value is None:
                    return None
                if value.dtype == torch.bool:
                    extra = torch.zeros(value.shape[:-1] + (pad,), dtype=torch.bool, device=value.device)
                    return torch.cat([value, extra], dim=-1)
                return F.pad(value, (0, 0, 0, pad)) if value.ndim == 3 else F.pad(value, (0, pad), value=fill)

            def dit_forward(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                ehs = bound.arguments['encoder_hidden_states']
                s = ehs.shape[1]
                pad = (-s) % padded_rows_multiple
                summary['original_rows'] = s
                summary['padded_rows'] = s + pad
                if pad:
                    bound.arguments['encoder_hidden_states'] = F.pad(ehs, (0, 0, 0, pad))
                    for key in ('image_mask', 'backbone_attention_mask', 'encoder_attention_mask'):
                        bound.arguments[key] = padded(bound.arguments.get(key), pad, False)
                    summary['padded_dit_calls'] += 1
                return original_dit_forward(*bound.args, **bound.kwargs)

            dit.forward = dit_forward
            # Padded cross K/V projections (M=152) replay the frozen small-M tile choice of M=149.
            from int8_small_m_gemm_20261028_r1 import _int8_scaled_gemm
            rows = selection_rows
            summary['padded_gemm_configs'] = {}
            previous_linear = integer_mm.linear
            table = {}
            if rows is not None:
                for row in rows:
                    m, n, k, b = row['shape']
                    if row['selected'] is not None:
                        table[(m, n, k, b)] = tuple(row['selected'])

            def linear(a, w, sa, sw, bias, fma, actual_rows=None):
                m = a.shape[0] if actual_rows is None else actual_rows
                key = (m, w.shape[0], w.shape[1], bias is not None)
                if key in table or m > 160:
                    return previous_linear(a, w, sa, sw, bias, fma, actual_rows=actual_rows)
                base = (summary['original_rows'], w.shape[0], w.shape[1], bias is not None)
                if summary['original_rows'] is None or base not in table or m != summary['padded_rows']:
                    return previous_linear(a, w, sa, sw, bias, fma, actual_rows=actual_rows)
                cfg = table[base]
                bm, bn, bk, warps, stages = cfg
                assert fma and a.dtype == w.dtype == torch.int8 and a.is_contiguous() and w.is_contiguous()
                n, k = w.shape
                y = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
                _int8_scaled_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](a, w, sa, sw, bias if bias is not None else y, y, m, n, k, bias is not None, bm, bn, bk,
                                                                            num_warps=warps, num_stages=stages, enable_fp_fusion=False)
                summary['padded_linear_calls'] += 1
                summary['padded_gemm_configs'][str(key)] = list(cfg)
                return y

            integer_mm.linear = linear
            # Padded cross K/V activation (152 rows) replays the frozen warp choice of the 149-row shape,
            # shared between to_k/to_v within one cross-attention forward exactly like the sibling cache.
            act_table = {tuple(r['shape']): r['selected_warps'] for r in activation_rows}
            shared_activation = padded_producer.activation
            frames = []
            def frame_enter(module, args):
                frames.append({})

            def frame_leave(module, args, output):
                frames.pop()   # must return None: a forward hook's return value would replace the output

            for idx, block in enumerate(blocks):
                if idx % 2:
                    continue
                block.attn1.register_forward_pre_hook(frame_enter)
                block.attn1.register_forward_hook(frame_leave, always_call=True)

            def my_activation(x, mode):
                key = tuple(x.shape)
                orig = summary['original_rows']
                if not (frames and orig is not None and key[0] == summary['padded_rows'] and key not in act_table and (orig, key[1]) in act_table):
                    return shared_activation(x, mode)
                assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous() and mode == 0
                ckey = (x.data_ptr(), key, tuple(x.stride()), mode)
                frame = frames[-1]
                if ckey in frame:
                    source, (q, s) = frame[ckey]
                    assert source is x or source.data_ptr() == x.data_ptr()
                    summary['padded_activation_hits'] = summary.get('padded_activation_hits', 0) + 1
                    return q, s
                m, k = key
                mp = max(32, triton.cdiv(m, 16) * 16)
                q = torch.empty((mp, k), device=x.device, dtype=torch.int8)
                s = torch.empty((mp, 1), device=x.device, dtype=torch.float32)
                padded_producer._activation[(mp,)](x, q, s, m, k, triton.next_power_of_2(k), mode, num_warps=act_table[(orig, k)], enable_fp_fusion=False)
                frame[ckey] = (x, (q, s))
                summary['padded_activation_calls'] = summary.get('padded_activation_calls', 0) + 1
                summary['padded_activation_warps'] = act_table[(orig, k)]
                return q, s

            padded_producer.activation = my_activation
        return result

    fast.install = install
    return summary


def trace_counts(events):
    ks = sorted([e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'], key=lambda e: e['ts'])
    first_adaln = next((i for i, e in enumerate(ks) if 'omni_adaln_int8' in e['name']), len(ks))
    action = ks[first_adaln:]
    unit_mul = sum(1 for e in action if 'BUnaryFunctor' in e['name'] and 'MulFunctor' in e['name'])
    fmha_fill = sum(1 for i, e in enumerate(action) if 'fmha_cutlassF' in e['name'] and i >= 2 and 'FillFunctor' in action[i - 2]['name'])
    sage_copy = sum(1 for i, e in enumerate(action) if 'qk_int_sv_f8' in e['name'] and i + 1 < len(action) and 'elementwise_kernel' in action[i + 1]['name'] and 'direct_copy' in action[i + 1]['name'])
    fmha = sum('fmha_cutlassF' in e['name'] for e in action)
    sage = sum('qk_int_sv_f8' in e['name'] for e in action)
    return dict(action_kernels=len(action), action_unit_scale_mul_kernels=unit_mul, fmha_calls=fmha, fmha_preceded_by_fill=fmha_fill,
                sage_calls=sage, sage_followed_by_copy=sage_copy)
