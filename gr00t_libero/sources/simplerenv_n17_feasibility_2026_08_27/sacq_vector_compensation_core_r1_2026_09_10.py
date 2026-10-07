"""Fixed-book, fixed-scale vector GPTQ reassignment (no model/data defaults).

For U = chol(inv(H + damping * mean(diag(H)) I), upper=True),
the next vector J minimizes ||(w_J - q_J) inv(U_JJ)||^2. Its error
coefficient E_J=(w_J-q_J) inv(U_JJ) updates remaining columns by
W_R -= E_J U_JR. Vectors stay intact; preceding choices are frozen.
Within a lazy block EVERY vector is selected from the updated weight.
The 128-column lazy update is only a scheduling optimization.

Books, group scales, shifts, bias, partition geometry, and nominal bits
are preserved exactly. Checkpoints contain updated remaining weights and
completed indices so an interruption need not restart a layer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import torch

REVISION = 'sacq_vector_compensation_r1'
ROLES = ('q', 'k', 'v')


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), tuple(t.shape))).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def assert_only_indices_changed(before, after, path='packed'):
    if isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError(f'{path}: payload keys changed')
        for key in before:
            if key == 'indices':
                a, b = before[key], after[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f'{path}: index shape/dtype changed')
            else:
                assert_only_indices_changed(before[key], after[key], path + '.' + key)
    elif torch.is_tensor(before):
        if before.dtype != after.dtype or not torch.equal(before.cpu(), after.cpu()):
            raise ValueError(f'{path}: fixed tensor changed')
    elif before != after:
        raise ValueError(f'{path}: fixed metadata changed')


def inverse_factor(H, damping):
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('Full square input second moment required')
    if not torch.isfinite(H).all():
        raise ValueError('Nonfinite statistics')
    h = H.float()
    # Symmetrize accumulation roundoff, not a diagonal approximation.
    h = (h + h.T) * 0.5
    mean = h.diag().mean()
    if mean <= 0 or not 0 < damping:
        raise ValueError('Nonpositive input energy or damping')
    h = h.clone()
    h.diagonal().add_(mean * damping)
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    return torch.linalg.cholesky(inverse, upper=True)


def _representable_candidates(book, scales, shift):
    # Match rolep1_allocation_adapter.decode_part: BF16 multiplication then add.
    q = book.to(torch.bfloat16)[None] * scales.to(torch.bfloat16)[:, None, None]
    q = q + shift.to(torch.bfloat16)
    return q.float()


def _choose_vector(w, candidates, inv_u):
    """All output rows are independent; lowest index deterministically wins ties."""
    wt = w @ inv_u
    qt = candidates @ inv_u
    distances = (qt - wt[:, None, :]).square().sum(-1)
    index = distances.argmin(dim=1)
    rows = torch.arange(w.shape[0], device=w.device)
    return index, candidates[rows, index]


def decode_fixed(p, device):
    if p.get('format') == 'packed_separate_qkv_codebooks_v1':
        return torch.cat([decode_fixed(p['roles'][r], device) for r in ROLES])
    book = p['codebook'].to(device=device, dtype=torch.bfloat16)
    ix = p['indices'].to(device=device, dtype=torch.long)
    scales = p['scales'].to(device=device, dtype=torch.bfloat16)
    w = book[ix].reshape(int(p['out_features']), -1, int(p['group_size']))
    w.mul_(scales.unsqueeze(-1))
    w = w.reshape(int(p['out_features']), int(p['in_features']))
    w.add_(torch.as_tensor(p['shift'], device=device, dtype=torch.bfloat16))
    return w.float()


def reconstruction_error(p, weight, H):
    # Unregularized layer-output MSE per output channel on captured inputs.
    # Diagnostic only: never accept/reject/revert a layer based on this number.
    error = weight.float() - decode_fixed(p, weight.device)
    return float(((error @ H.float()) * error).sum(dtype=torch.float64) / weight.shape[0])


@torch.no_grad()
def _part(p, weight, U, *, checkpoint_path, provenance, block_columns,
          row_chunk=512, stop_after_blocks=None, progress=None):
    if p.get('format') != 'packed_vector_codebook_v1':
        raise ValueError('Unsupported packed format')
    o, n, g, d = (int(p[k]) for k in
                  ('out_features', 'in_features', 'group_size', 'vector_dim'))
    if (g, d) != (64, 4) or n % g or block_columns % g:
        raise ValueError('Must preserve existing group64/vector4 geometry')
    if tuple(weight.shape) != (o, n) or tuple(U.shape) != (n, n):
        raise ValueError('Weight/statistic geometry mismatch')
    book, scales, shift, old_indices = (p[k] for k in ('codebook','scales','shift','indices'))
    shift = torch.as_tensor(shift)
    if book.ndim != 2 or book.shape[1] != d or shift.numel() != 1:
        raise ValueError('Only one fixed codebook and scalar shift per role')
    if tuple(scales.shape) != (o, n//g) or tuple(old_indices.shape) != (o, n//g, g//d):
        raise ValueError('Scale/index shape mismatch')
    if old_indices.long().min() < 0 or old_indices.long().max() >= len(book):
        raise ValueError('Invalid original indices')
    if not all(torch.isfinite(x).all() for x in (weight, book, scales, shift, U)):
        raise ValueError('Nonfinite weight/book/scale/shift/factor')
    signature = dict(revision=REVISION, provenance=provenance,
        weight=tensor_sha(weight), factor=tensor_sha(U), book=tensor_sha(book),
        scales=tensor_sha(scales), shift=tensor_sha(shift), indices=tensor_sha(old_indices),
        block_columns=block_columns, geometry=[o,n,g,d])
    device = weight.device
    W = weight.float().clone()
    indices = old_indices.to(device=device, dtype=torch.long).clone().reshape(o, n//d)
    start = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Resume provenance/inputs mismatch')
        start, W, indices = state['next_column'], state['remaining_weight'], state['indices']
        if start not in range(0, n+1, block_columns) and start != n:
            raise ValueError('Bad resume boundary')
        if tuple(W.shape) != (o,n) or tuple(indices.shape) != (o,n//d):
            raise ValueError('Bad resume tensor geometry')
    book, scales, shift = (x.to(device) for x in (book, scales, shift))
    blocks = 0
    for left in range(start, n, block_columns):
        right = min(left + block_columns, n)
        W1 = W[:, left:right].clone()
        E = torch.zeros_like(W1)
        for group in range(left, right, g):
            for row0 in range(0, o, row_chunk):
                row1 = min(row0 + row_chunk, o)
                candidates = _representable_candidates(book, scales[row0:row1,group//g], shift)
                for col in range(group, min(group+g, right), d):
                    local = col - left
                    ujj = U[col:col+d,col:col+d]
                    inv_u = torch.linalg.solve_triangular(ujj, torch.eye(d, device=device), upper=True)
                    w = W1[row0:row1,local:local+d]
                    index, q = _choose_vector(w, candidates, inv_u)
                    error = (w - q) @ inv_u
                    indices[row0:row1,col//d] = index
                    E[row0:row1,local:local+d] = error
                    W1[row0:row1,local:] -= error @ U[col:col+d,col:right]
                    W1[row0:row1,local:local+d] = q
        W[:, left:right] = W1
        if right < n:
            W[:, right:] -= E @ U[left:right,right:]
        if not torch.isfinite(W).all():
            raise ValueError('Nonfinite compensated weights; no silent fallback')
        if checkpoint_path is not None:
            atomic_save(dict(signature=signature, next_column=right,
                remaining_weight=W.cpu(), indices=indices.cpu()), checkpoint_path)
        blocks += 1
        if progress:
            progress(dict(next_column=right,total_columns=n))
        if stop_after_blocks is not None and blocks >= stop_after_blocks and right < n:
            raise InterruptedError('Requested test interruption after durable block')
    out = copy.deepcopy(p)
    out['indices'] = indices.reshape(old_indices.shape).to(device=old_indices.device, dtype=old_indices.dtype)
    assert_only_indices_changed(p,out)
    changed = int((out['indices'].cpu() != old_indices.cpu()).sum())
    return out, dict(changed_indices=changed,total_indices=old_indices.numel(),
                    restored_from_column=start,complete=True)


@torch.no_grad()
def compensate_packed(packed, weight, H, *, checkpoint_dir=None, provenance=None,
                     block_columns=128, damping=0.1, row_chunk=512,
                     stop_after_blocks=None, progress=None):
    if block_columns % 64 or block_columns <= 0:
        raise ValueError('Lazy block must respect existing scale groups')
    if row_chunk <= 0:
        raise ValueError('row_chunk must be positive')
    if H.device != weight.device:
        raise ValueError('Weight and H must be on the same device')
    U = inverse_factor(H, damping)
    provenance = dict(provenance or {}, damping=damping)
    fused = packed.get('format') == 'packed_separate_qkv_codebooks_v1'
    out, reports = copy.deepcopy(packed), {}
    if fused:
        if set(packed['roles']) != set(ROLES):
            raise ValueError('Fused Q/K/V must retain separate books')
        expected = sum(int(packed['roles'][r]['out_features']) for r in ROLES)
        if weight.shape[0] != expected:
            raise ValueError('Fused weight row count mismatch')
    offset = 0
    for role in ROLES if fused else ('whole',):
        part = packed['roles'][role] if fused else packed
        o = int(part['out_features'])
        result, report = _part(part, weight[offset:offset+o], U,
            checkpoint_path=Path(checkpoint_dir)/f'{role}.pt' if checkpoint_dir else None,
            provenance=dict(provenance,role=role), block_columns=block_columns,
            row_chunk=row_chunk,stop_after_blocks=stop_after_blocks,progress=progress)
        if fused:
            out['roles'][role] = result
        else:
            out = result
        reports[role] = report
        offset += o
    if offset != weight.shape[0]:
        raise ValueError('Weight rows not fully covered')
    assert_only_indices_changed(packed, out)
    return out, dict(revision=REVISION,status='PASS',roles=reports,
        damping=damping,block_columns=block_columns,
        layer_output_mse_before=reconstruction_error(packed, weight, H),
        layer_output_mse_after=reconstruction_error(out, weight, H),
        method='conditional_input_H_vector_GPTQ_fixed_books_scales')
