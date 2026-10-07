#!/usr/bin/env python3
"""P1: end-to-end codebook centroid fine-tuning (AQLM-style, VLA variant).

Frozen : assignment indices, all non-quantized weights (embed/vision/vlsa/bias/norm)
Trained: codebook entries C (k x 4) and per-(row,group) scales of every packed
         layer -- VLM 101 (shared) + DiT 226 (per suite), ~1-2M floats total.
Loss   : || final_decoded_action_q(x) - final_decoded_action_bf16(x) ||^2
         with fixed denoising seeds (same protocol as the proxy).
Data   : captured requests (hard raw captures of this suite), refs computed
         in-process with the bf16 model before quantization, cached to disk.

Gate   : val relative error printed every eval; best checkpoint saved as
         overlay-compatible dense dict for direct rollout evaluation.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import os as _os

PROJ = Path("/data/workspace/libero_pro_eval")
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "Isaac-GR00T-N1.7"))

from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from measure_dit_single_layer_downgrade_sensitivity import patched_env, replace_embedding, replace_module  # noqa: E402
from measure_dit_scalar_vs_vector_2026_07_17 import load_batches  # noqa: E402
from measure_embedding_vision_quant_memory_v1 import quantize_visual_linears  # noqa: E402
from measure_gr00t_dit_fullstack_proxy_2026_07_21 import VLM_ENV  # noqa: E402
from measure_hotspot_protection_proxy import move_to_device  # noqa: E402
from proxy_fast_2026_07_24 import local_ckpt  # noqa: E402
from train_spinquant_r1r2_vla_objective import decode_action_torch_libero  # noqa: E402


def run_action_grad(policy, backbone_inputs, action_inputs, seed):
    """Same computation as measure_hotspot_protection_proxy.run_outputs, but the
    returned action keeps its autograd graph and stays on device (that helper
    detaches to CPU, which silently kills training)."""
    ah = policy.model.action_head
    backbone_output = policy.model.backbone(backbone_inputs)
    features = ah._encode_features(backbone_output, action_inputs)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_inputs.embodiment_id
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        actions = torch.randn(size=(batch_size, ah.config.action_horizon, ah.action_dim),
                              dtype=vl_embeds.dtype, device=device)
    dt = 1.0 / ah.num_inference_timesteps
    for step_idx in range(ah.num_inference_timesteps):
        t_cont = step_idx / float(ah.num_inference_timesteps)
        t_discretized = int(t_cont * ah.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = ah.action_encoder(actions, timesteps_tensor, embodiment_id)
        if ah.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + ah.position_embedding(pos_ids).unsqueeze(0)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        if ah.config.use_alternate_vl_dit:
            model_output = ah.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                                    timestep=timesteps_tensor,
                                    image_mask=backbone_output.image_mask,
                                    backbone_attention_mask=backbone_output.backbone_attention_mask)
        else:
            model_output = ah.model(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                                    timestep=timesteps_tensor)
        pred = ah.action_decoder(model_output, embodiment_id)
        actions = actions + dt * pred[:, -ah.action_horizon:]
    return decode_action_torch_libero(policy, actions)

QA = PROJ / "quantization_analysis"
ROT = QA / "vlm_rotgrid_2026-07-25"
FS2 = QA / "gr00t_dit_fs2_2026-07-23"
RAW = PROJ / "request_replay_runs/calib_hard_raw_2026-08-04"
import csv as _csv
import json as _json

# proxy(val_hard)로 쓰는 요청은 FT 학습에서 절대 뽑히면 안 된다 -- val_hard가
# calib_hard_raw에서 shutil.copy2로 "복사"만 됐을 뿐 원본이 raw에 그대로 남아있어서,
# 예전엔 collect_requests()가 이 파일들까지 셔플 풀에 포함시켰다(실측: object/goal
# suite는 val 4개 전부, spatial은 12개 중 7개가 FT 384개 학습셋과 그대로 겹침,
# 2026-08-22 발견). 이 경로들을 기본적으로 항상 제외한다.
_VAL_HARD_MANIFEST = PROJ / "request_replay_runs/calib_hard_2026-08-04/manifest.json"
try:
    VAL_HARD_EXCLUDE = {str(Path(x["src"]).resolve()) for x in _json.loads(_VAL_HARD_MANIFEST.read_text())}
except FileNotFoundError:
    VAL_HARD_EXCLUDE = set()

MODULE_CSV = QA / "gr00t_dit_scalar_vs_vector_2026-07-17/dit_modules.csv"
roles = {r["module_name"]: r["module_type"] for r in _csv.DictReader(open(MODULE_CSV))}
for i in range(32):
    roles[f"action_head.model.transformer_blocks.{i}.norm1.linear"] = "adaln"
dit_core = [n for n in sorted(roles) if "vl_self_attention" not in n]
dit_vlsa = [n for n in sorted(roles) if "vl_self_attention" in n]


WCACHE = True      # --no-wcache 로 끈다 (A/B 검증용)
FP32_CODEBOOK_GRAD = False  # opt-in: deployment-identical forward, FP32 gather backward


class _FP32CodebookGather(torch.autograd.Function):
    """BF16 codebook gather whose codebook gradient is accumulated in FP32.

    ``self.codebook.to(bfloat16)[idx]`` makes the stock gather backward scatter
    into a temporary BF16 tensor before casting that gradient back to the FP32
    AdamW master parameter.  For small codebooks, many assignments collide on
    each entry and BF16 atomics become both slow and unnecessarily lossy.  The
    forward below is value-identical; only the linear gradient reduction is
    accumulated directly into FP32.  It is mathematically the same derivative,
    but (like any parallel reduction change) not bitwise identical.
    """

    @staticmethod
    def forward(ctx, codebook, idx):
        ctx.save_for_backward(idx)
        ctx.codebook_shape = codebook.shape
        return codebook.to(torch.bfloat16)[idx]

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        grad_codebook = torch.zeros(
            ctx.codebook_shape,
            device=grad_output.device,
            dtype=torch.float32,
        )
        grad_codebook.index_add_(
            0,
            idx.reshape(-1),
            grad_output.reshape(-1, grad_output.shape[-1]).float(),
        )
        return grad_codebook, None


def atomic_torch_save(payload, path: Path) -> None:
    """Write a checkpoint on the same filesystem, then atomically publish it."""
    tmp = path.with_name(f".{path.name}.tmp.{_os.getpid()}")
    try:
        torch.save(payload, tmp)
        _os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def pack_indices(indices: torch.Tensor, index_bits: int) -> torch.Tensor:
    """gr00t.eval.cosmos_quant_shim.PrecomputedPackedVectorCodebookLinear의
    packed_indices 언패킹과 정확히 맞물리는 역함수 (1--16bit CPU/GPU
    round-trip 검증됨, 2026-08-22).
    byte 정렬 "indices"(uint8 고정)는 k>256 층에서 값이 잘리는 버그가 있어
    항상 이 비트팩킹 경로로 저장한다."""
    if index_bits < 1 or index_bits > 56:
        raise ValueError(f"index_bits must be in [1, 56], got {index_bits}")
    flat = indices.reshape(-1).long()
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= (1 << index_bits)):
        raise ValueError(f"indices do not fit in {index_bits} bits")
    count = flat.numel()
    n_bytes = (count * index_bits + 7) // 8
    # An index can start at bit offset 7, so arbitrary widths may span more
    # than two bytes (11-bit indices need three). Keep enough temporary guard
    # bytes for the widest possible alignment and trim them before saving.
    max_offset = 8 - math.gcd(index_bits, 8)
    window_bytes = (index_bits + max_offset + 7) // 8
    out = torch.zeros(n_bytes + window_bytes - 1, dtype=torch.int64)
    positions = torch.arange(count, dtype=torch.long) * index_bits
    byte_ids = torch.div(positions, 8, rounding_mode="floor")
    offsets = positions.remainder(8)
    shifted = flat << offsets
    for byte_offset in range(window_bytes):
        out.index_put_(
            (byte_ids + byte_offset,),
            (shifted >> (8 * byte_offset)) & 0xFF,
            accumulate=True,
        )
    assert (out <= 255).all(), "byte overflow -- 인덱스 겹침"
    return out[:n_bytes].to(torch.uint8)


class QuantLinear(nn.Module):
    """W reconstructed on the fly from trainable codebook + scales, frozen idx."""

    def __init__(self, packed: dict, bias, device):
        super().__init__()
        self.out_features = packed["out_features"]
        self.in_features = packed["in_features"]
        self.gs = packed["group_size"]
        self.vd = packed["vector_dim"]
        self.codebook = nn.Parameter(packed["codebook"].to(device, torch.float32))
        self.scales = nn.Parameter(packed["scales"].to(device, torch.float32))
        self.register_buffer(
            "shift",
            torch.as_tensor(packed.get("shift", 0.0), device=device, dtype=torch.bfloat16),
        )
        idx = packed["indices"].to(torch.long)
        self.register_buffer("idx", idx.to(device))
        self._wc = None          # 스텝 내 W 캐시 (매 샘플 시작 시 무효화)
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device, torch.bfloat16))
        else:
            self.bias = None

    def weight_(self):
        # Match packed deployment arithmetic exactly while retaining FP32 master
        # parameters for AdamW.  Casting remains differentiable, so gradients
        # flow to the master codebook/scales through the BF16 storage simulation.
        # Runtime order: BF16 gather -> reshape -> BF16 scale -> reshape -> shift.
        scales = self.scales.to(torch.bfloat16)
        if FP32_CODEBOOK_GRAD and torch.is_grad_enabled():
            vec = _FP32CodebookGather.apply(self.codebook, self.idx)
        else:
            codebook = self.codebook.to(torch.bfloat16)
            vec = codebook[self.idx]                        # (out, g, gs//vd, vd)
        Wn = vec.reshape(self.out_features, -1, self.gs)    # (out, g, gs)
        W = Wn * scales.unsqueeze(-1)
        return W.reshape(self.out_features, self.in_features) + self.shift

    def forward(self, x):
        if getattr(self, "_leaf", None) is not None:
            return nn.functional.linear(x, self._leaf.to(torch.bfloat16), self.bias)
        # 한 샘플의 forward 안에서 코드북/스케일은 불변인데, DiT 층은 디노이징
        # 4스텝 때문에 같은 W 를 4번 재구성한다. 첫 호출 결과를 캐시해 재사용한다.
        # 같은 그래프 노드를 재사용하므로 backward 는 두 경로의 기울기를 그대로
        # 누적한다 — 수치적으로 재계산과 동일(무손실). 샘플 시작 시 _wc 를 비운다.
        if not WCACHE:
            return nn.functional.linear(x, self.weight_().to(torch.bfloat16), self.bias)
        W = self._wc
        if W is None:
            W = self.weight_().to(torch.bfloat16)
            self._wc = W
        return nn.functional.linear(x, W, self.bias)


def collect_requests(suite: str, limit: int, root=None, exclude=(), oversample=None,
                     exclude_files=None):
    """exclude_files: 절대경로 문자열 집합. None이면 기본적으로 VAL_HARD_EXCLUDE(proxy로
    쓰는 val_hard 32개)를 제외한다 -- FT 학습이 proxy 평가셋을 그대로 학습해버리는 걸
    막기 위함(2026-08-22). 정말로 제외를 끄고 싶으면 exclude_files=set()을 명시한다."""
    if exclude_files is None:
        exclude_files = VAL_HARD_EXCLUDE
    roots = [Path(r) for r in str(root or RAW).split(":")]
    skip = {f"{suite}_t{i}" for i in exclude}
    dirs = []
    for rt in roots:
        dirs += sorted(d / "requests" for d in rt.iterdir()
                       if d.name.startswith(suite) and d.name not in skip)
    files = []
    n_excluded = 0
    for d in dirs:
        fs = sorted(d.glob("*.msgpack"))
        if exclude_files:
            kept = [p for p in fs if str(p.resolve()) not in exclude_files]
            n_excluded += len(fs) - len(kept)
            fs = kept
        w = 1
        if oversample:
            ti = d.parent.name.rsplit("_t", 1)[-1]
            w = oversample.get(ti, 1)
        files.extend(fs * w)
    if n_excluded:
        print(f"[ft] collect_requests({suite}): proxy(val_hard)와 겹치는 {n_excluded}개 파일 "
              f"학습 풀에서 제외", flush=True)
    random.Random(0).shuffle(files)
    return files[:limit]




def pv_step(policy, trainables, items, run_loss_fn, batch_n, max_switch, device):
    """손실 인지 배정 갱신 (PV-Tuning V-step, 선형항 + 대각 Fisher 2차항).
    ΔL(c) ≈ g·Δ + ½·f·Δ²  (f = 요청별 grad² 누적 = 경험적 Fisher 대각).
    2차항이 신뢰영역 역할을 해 '기울기만 좋고 멀리 있는' 코드워드로의 폭주를 막는다."""
    for _, m in trainables:
        m._leaf = m.weight_().detach().to(torch.bfloat16).requires_grad_(True)
        m._gsum = torch.zeros_like(m._leaf)
        m._fsum = torch.zeros_like(m._leaf)
    n_done = 0
    for i, (rid, bb, ai) in items[:batch_n]:
        loss, _ = run_loss_fn(i, rid, bb, ai, grad=True)
        loss.backward()
        for _, m in trainables:
            if m._leaf.grad is not None:
                m._gsum += m._leaf.grad
                m._fsum += m._leaf.grad.square()
                m._leaf.grad = None
        n_done += 1
    switched = total = 0
    for _, m in trainables:
        out, G = m.out_features, m.scales.shape[1]
        gs, vd = m.gs, m.vd
        gv = m._gsum.float().reshape(out, G, gs // vd, vd).reshape(-1, vd)
        fv = m._fsum.float().reshape(out, G, gs // vd, vd).reshape(-1, vd).clamp_min(1e-12)
        sv = m.scales.detach().to(torch.float32).reshape(out, G, 1).expand(out, G, gs // vd).reshape(-1)
        C = m.codebook.detach().to(torch.float32)
        C2 = C.square()
        idx_flat = m.idx.reshape(-1)
        N = gv.shape[0]
        best_gain = torch.zeros(N, device=device)
        best_c = idx_flat.clone()
        CH = 131072
        for s0 in range(0, N, CH):
            e0 = min(s0 + CH, N)
            g_, f_, s_ = gv[s0:e0], fv[s0:e0], sv[s0:e0, None]
            old_c = idx_flat[s0:e0]
            Cold = C[old_c]
            lin = (g_ @ C.T - (g_ * Cold).sum(1, keepdim=True)) * s_
            quad = 0.5 * s_.square() * (f_ @ C2.T
                    - 2.0 * ((f_ * Cold) @ C.T)
                    + (f_ * Cold.square()).sum(1, keepdim=True))
            dL = lin + quad
            mn, am = dL.min(dim=1)
            best_gain[s0:e0] = -mn                       # 양수 = 손실 감소 예측
            best_c[s0:e0] = am
        cand = (best_gain > 0) & (best_c != idx_flat)
        n_allow = int(N * max_switch)
        if cand.sum() > n_allow:
            thr = best_gain[cand].topk(n_allow).values.min()
            cand &= best_gain >= thr
        idx_flat[cand] = best_c[cand]
        m.idx.copy_(idx_flat.reshape_as(m.idx))
        switched += int(cand.sum()); total += N
        m._leaf = None; m._gsum = None; m._fsum = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[pv] V-step: {n_done}요청, 재배정 {switched}/{total} ({100*switched/total:.2f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--limit", type=int, default=384)
    ap.add_argument("--val-every", type=int, default=10, help="every Nth request -> val")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-scale", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=0, help="smoke: stop after N steps")
    ap.add_argument("--refs-only", action="store_true",
                    help="strict request selection과 BF16 action reference cache만 생성하고 종료")
    ap.add_argument("--verify-refs", type=int, default=0,
                    help="캐시 적중 후 앞 N개 BF16 action을 재계산해 bitwise 검증")
    ap.add_argument("--no-wcache", action="store_true",
                    help="W 캐시 비활성 (A/B 검증용 — 캐시 유무가 수치를 바꾸는지 판별)")
    ap.add_argument(
        "--fp32-codebook-grad",
        action="store_true",
        help=(
            "BF16 deployment forward는 유지하고 codebook gather backward만 "
            "FP32 index_add로 누적한다 (수학적으로 동일, reduction order는 다름)"
        ),
    )
    ap.add_argument("--out-tag", default="ftv1")
    ap.add_argument("--vlm-fits", default="vlm_packed_a230")
    ap.add_argument("--dit-fits", default="fits_packed_a230")
    ap.add_argument("--capture-root", default="")
    ap.add_argument("--exclude-tasks", default="",
                    help="캘리브에서 뺄 태스크 인덱스, 예: 1,4,5,9 (held-out 일반화 측정용)")
    ap.add_argument("--tune-norms", action="store_true",
                    help="AQLM식: norm 가중치와 bias도 학습 (lr-scale 그룹)")
    ap.add_argument("--oversample", default="",
                    help="태스크 오버샘플, 예: 3:3,6:2 (t3 3배, t6 2배)")
    ap.add_argument("--pv", action="store_true", help="에폭 종료마다 손실 인지 배정 갱신(V-step)")
    ap.add_argument("--pv-batch", type=int, default=48, help="V-step 그래디언트 수집 요청 수")
    ap.add_argument("--pv-max-switch", type=float, default=0.02, help="라운드당 재배정 허용 비율")
    ap.add_argument("--vlm-alloc", default="", help="VLM 배분 CSV (혼합 k)")
    ap.add_argument("--dit-alloc", default="", help="DiT 배분 CSV (혼합 k)")
    ap.add_argument(
        "--alloc",
        default="",
        help=(
            "최종 통합 할당 CSV(layer,params,bit,k,scope,payload_bytes). "
            "HQQ sentinel 90000/80000과 VC k를 함께 엄격 검증한다."
        ),
    )
    ap.add_argument(
        "--global-batch",
        type=int,
        default=0,
        help=(
            "DDP 전체 optimizer batch. 0은 기존 world-size batch와 동일. "
            "world size의 배수여야 하며, 동일 global batch면 GPU 수와 무관하게 "
            "같은 shuffled request 묶음을 사용한다."
        ),
    )
    ap.add_argument("--vanilla-fits", action="store_true", help="vanilla 피팅 사용")
    ap.add_argument("--resume", action="store_true", help="codebook_state.pt 가 있으면 이어서 학습")
    args = ap.parse_args()
    if args.alloc and (args.vlm_alloc or args.dit_alloc):
        ap.error("--alloc cannot be combined with --vlm-alloc/--dit-alloc")
    global WCACHE, FP32_CODEBOOK_GRAD
    WCACHE = not args.no_wcache
    FP32_CODEBOOK_GRAD = args.fp32_codebook_grad
    ddp = "RANK" in _os.environ
    rank = int(_os.environ.get("RANK", 0))
    world = int(_os.environ.get("WORLD_SIZE", 1))
    global_batch = args.global_batch or world
    if global_batch < world or global_batch % world:
        ap.error(f"--global-batch ({global_batch}) must be a positive multiple of world size ({world})")
    if ddp:
        torch.cuda.set_device(rank)
        # Cold-node request preprocessing can be highly asymmetric when
        # several two-rank calibrations share a node. A rank may reach the
        # post-preprocessing barrier >45 minutes before its peer; the former
        # timeout then aborted a healthy job with SIGABRT. This controls only
        # failure detection and does not change any numerical operation.
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3),
                                device_id=torch.device(f"cuda:{rank}"))
    device = torch.device(f"cuda:{rank}" if ddp else "cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    def log0(*a):
        if rank == 0: print(*a, flush=True)
    log0(
        f"[ft] codebook gather backward: "
        f"{'FP32 index_add' if FP32_CODEBOOK_GRAD else 'stock BF16 scatter'}"
    )

    out_dir = QA / f"centroid_ft_2026-08-05/{args.out_tag}_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_alloc_rows = None
    final_vlm_kmap = None
    final_dit_kmap = None
    final_hqq_vlm = None
    final_hqq_dit = None
    alloc_sha256 = None
    if args.alloc:
        import hashlib

        alloc_path = Path(args.alloc).resolve()
        final_alloc_rows = list(_csv.DictReader(open(alloc_path)))
        required = {"layer", "params", "bit", "k", "scope", "payload_bytes"}
        if not final_alloc_rows or not required.issubset(final_alloc_rows[0]):
            raise ValueError(f"invalid final allocation schema: {alloc_path}")
        names = [row["layer"] for row in final_alloc_rows]
        if len(names) != 338 or len(set(names)) != 338:
            raise ValueError(f"final allocation must contain 338 unique layers, got {len(names)}/{len(set(names))}")
        if any(row["scope"] not in ("vlm", "dit") for row in final_alloc_rows):
            raise ValueError("final allocation contains an invalid scope")
        if any((row["scope"] == "dit") != row["layer"].startswith("action_head")
               for row in final_alloc_rows):
            raise ValueError("final allocation scope/name mismatch")
        valid_k = {90000, 80000, 2048, 1024, 512, 256, 128, 64, 32, 16, 8}
        invalid_k = sorted({int(row["k"]) for row in final_alloc_rows}.difference(valid_k))
        if invalid_k:
            raise ValueError(f"invalid k values in final allocation: {invalid_k}")
        final_vlm_kmap = {
            row["layer"]: int(row["k"]) for row in final_alloc_rows if row["scope"] == "vlm"
        }
        final_dit_kmap = {
            row["layer"]: int(row["k"]) for row in final_alloc_rows if row["scope"] == "dit"
        }
        if len(final_vlm_kmap) != 112 or len(final_dit_kmap) != 226:
            raise ValueError(
                f"final allocation scope counts must be VLM112/DiT226, got "
                f"{len(final_vlm_kmap)}/{len(final_dit_kmap)}"
            )
        final_hqq_vlm = {name: 4 if k == 90000 else 3
                         for name, k in final_vlm_kmap.items() if k in (90000, 80000)}
        final_hqq_dit = {name: 4 if k == 90000 else 3
                         for name, k in final_dit_kmap.items() if k in (90000, 80000)}
        alloc_sha256 = hashlib.sha256(alloc_path.read_bytes()).hexdigest()
        log0(
            f"[ft] strict final allocation: {alloc_path} sha256={alloc_sha256} "
            f"VLM VC/HQQ={112-len(final_hqq_vlm)}/{len(final_hqq_vlm)} "
            f"DiT VC/HQQ={226-len(final_hqq_dit)}/{len(final_hqq_dit)}"
        )

    # ---- policy + requests + bf16 refs (before quantization) -----------------
    policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), local_ckpt(args.suite),
                         device=str(device), strict=True)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    excl = [int(x) for x in args.exclude_tasks.split(",") if x.strip()]
    osmp = {k: int(v) for k, v in (x.split(":") for x in args.oversample.split(",") if x)} or None
    req_files = collect_requests(args.suite, args.limit,
                                 args.capture_root if args.capture_root else None,
                                 exclude=excl, oversample=osmp)
    if excl:
        print(f"[ft] 캘리브 제외 태스크: {excl}", flush=True)
    print(f"[ft] {args.suite}: {len(req_files)} requests", flush=True)
    # load_batches expects a directory -- stage symlinks
    stage = out_dir / "req_stage"
    if rank == 0:
        stage.mkdir(exist_ok=True)
        for f in stage.glob("*.msgpack"):
            f.unlink()
        for i, f in enumerate(req_files):
            (stage / f"{i:04d}__{f.name}").symlink_to(f)
    if ddp:
        dist.barrier()
    batches = load_batches(policy, stage, len(req_files))

    import hashlib
    req_manifest = []
    for f in req_files:
        stat = f.stat()
        req_manifest.append(f"{f.resolve()}\t{stat.st_size}\t{stat.st_mtime_ns}")
    req_sig = hashlib.sha256("\n".join(req_manifest).encode()).hexdigest()[:16]
    shared_refs = QA / "centroid_ft_refs_cache" / f"{args.suite}_{req_sig}.pt"
    shared_refs.parent.mkdir(exist_ok=True)
    refs_path = out_dir / "refs.pt"
    if rank == 0 and not refs_path.exists() and shared_refs.exists():
        import shutil as _sh
        _sh.copy2(shared_refs, refs_path)
        print(f"[ft] 공유 refs 캐시 적중: {shared_refs.name}", flush=True)
    if ddp:
        dist.barrier()
    if refs_path.exists():
        refs = torch.load(refs_path, map_location="cpu")
        log0(f"[ft] refs cache hit ({len(refs)})")
    elif ddp and rank != 0:
        dist.barrier()
        refs = torch.load(refs_path, map_location="cpu")
    else:
        refs = {}
        with torch.no_grad():
            for i, (rid, bb, ai) in enumerate(batches):
                a = run_action_grad(policy, move_to_device(bb, device, torch.bfloat16),
                                    move_to_device(ai, device, torch.bfloat16), seed=7000 + i)
                refs[rid] = a.detach().to("cpu", torch.float32)
                if (i + 1) % 50 == 0:
                    print(f"[ft] refs {i+1}/{len(batches)}", flush=True)
        refs_tmp = refs_path.with_suffix(f".tmp.{_os.getpid()}")
        torch.save(refs, refs_tmp)
        _os.replace(refs_tmp, refs_path)
        if not shared_refs.exists():
            shared_tmp = shared_refs.with_suffix(f".tmp.{_os.getpid()}")
            torch.save(refs, shared_tmp)
            try:
                _os.replace(shared_tmp, shared_refs)
            finally:
                if shared_tmp.exists():
                    shared_tmp.unlink()
        print(f"[ft] refs cached -> {refs_path} (+공유 캐시)", flush=True)
        if ddp:
            dist.barrier()
    batch_rids = [rid for rid, _, _ in batches]
    if len(set(batch_rids)) != len(batch_rids) or set(refs) != set(batch_rids):
        raise RuntimeError(
            f"BF16 refs/request-id mismatch refs={len(refs)} "
            f"batches={len(batches)} unique_batch_ids={len(set(batch_rids))}"
        )
    if args.verify_refs:
        n_verify = min(args.verify_refs, len(batches))
        with torch.no_grad():
            for i, (rid, bb, ai) in enumerate(batches[:n_verify]):
                current = run_action_grad(
                    policy,
                    move_to_device(bb, device, torch.bfloat16),
                    move_to_device(ai, device, torch.bfloat16),
                    seed=7000 + i,
                ).detach().to("cpu", torch.float32)
                cached = refs[rid].to("cpu", torch.float32)
                if not torch.equal(current, cached):
                    max_abs = float((current - cached).abs().max())
                    raise RuntimeError(
                        f"BF16 ref value mismatch suite={args.suite} index={i} "
                        f"request_id={rid} max_abs={max_abs}"
                    )
        log0(f"[ft] BF16 refs bitwise verified: {n_verify}/{len(batches)}")
    if args.refs_only:
        log0(f"[ft] refs-only complete: suite={args.suite} requests={len(batches)} sig={req_sig}")
        if ddp:
            dist.barrier()
        return

    # ---- frozen quant background (embed/vision/vlsa) -------------------------
    named = dict(policy.model.named_modules())
    dit_bias_all = {n: (None if named[n].bias is None else named[n].bias.detach().clone())
                    for n in dit_core + dit_vlsa}
    if args.alloc:
        # The unified final ladder mixes frozen HQQ3/4 and trainable vector
        # codebooks.  Install exactly that HQQ background first.  VC layers are
        # forced to BF16 temporarily and replaced by QuantLinear below, avoiding
        # an unrelated on-the-fly HQQ2 pass over their weights.
        vlsa_hqq4 = sorted(dit_vlsa)
        # The module CSV intentionally lists only the 24 quantized VLSA
        # projections.  VLSA also has eight ordinary LayerNorm modules (not
        # AdaLN linears); their BF16 affine vectors stay outside quantization.
        vlsa_norm_bf16 = [
            f"action_head.vl_self_attention.transformer_blocks.{block}.{norm}"
            for block in range(4) for norm in ("norm1", "norm3")
        ]
        if len(vlsa_hqq4) != 24 or len(vlsa_norm_bf16) != 8:
            raise RuntimeError(
                f"VLSA split must be HQQ4x24/BF16x8, got {len(vlsa_hqq4)}/{len(vlsa_norm_bf16)}"
            )
        vlm_force = {
            name: final_hqq_vlm.get(name, 16) for name in final_vlm_kmap
        }
        # DIT_BITS=16 short-circuits the compiler before force specs are
        # applied, so use a quantizing base and spell out every exception.
        # This makes the installed background exactly equal to the allocation:
        # VC candidates BF16 for later QuantLinear replacement, HQQ sentinels
        # 3/4-bit, fixed VLSA projections HQQ4, and VLSA AdaLN linears BF16.
        dit_force = {
            **{name: 16 for name in final_dit_kmap},
            **final_hqq_dit,
            **{name: 4 for name in vlsa_hqq4},
        }
        final_env = dict(VLM_ENV)
        final_env.update({
            "VLM_BITS": "2",
            "DIT_BITS": "4",
            "VLM_QUANT_MODE": "uniform",
            "DIT_QUANT_MODE": "uniform",
            "VLM_ROTATION_MODE": "none",
            "VLM_FORCE_BITS_SPECS": ";".join(
                f"{name}={bits}" for name, bits in sorted(vlm_force.items())
            ),
            "VLM_FORCE_BITS_SPECS_FILE": "",
            "DIT_FORCE_BITS_SPECS": ";".join(
                f"{name}={bits}" for name, bits in sorted(dit_force.items())
            ),
            "DIT_FORCE_BITS_SPECS_FILE": "",
            "DIT_INCLUDE_ADALN": "1",
            "DIT_SKIP_VL_SELF_ATTENTION": "0",
            "VLM_CODEBOOK_OVERLAY_PATH": "",
            "DIT_CODEBOOK_OVERLAY_PATH": "",
            "GR00T_STRICT_QUANT_LAYOUT": "1",
        })
        with patched_env(final_env):
            quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device=str(device))
        named_bg = dict(policy.model.named_modules())
        expected_hqq = {**final_hqq_vlm, **final_hqq_dit,
                        **{name: 4 for name in vlsa_hqq4}}
        bad_hqq = [
            name for name in expected_hqq
            if name not in named_bg or "HQQ" not in type(named_bg[name]).__name__
        ]
        if bad_hqq:
            raise RuntimeError(f"strict final background HQQ mismatch: {bad_hqq[:8]}")
        expected_bf16_linears = {
            name for name, k in {**final_vlm_kmap, **final_dit_kmap}.items()
            if k not in (90000, 80000)
        }
        bad_bf16 = [
            name for name in expected_bf16_linears
            if name not in named_bg
            or not isinstance(named_bg[name], nn.Linear)
            or named_bg[name].weight.dtype != torch.bfloat16
        ]
        if bad_bf16:
            raise RuntimeError(f"strict final background BF16 mismatch: {bad_bf16[:8]}")
        bad_vlsa_norm = []
        for name in vlsa_norm_bf16:
            module = named_bg.get(name)
            if not isinstance(module, nn.LayerNorm):
                bad_vlsa_norm.append((name, type(module).__name__))
                continue
            tensors = [p for p in (module.weight, module.bias) if p is not None]
            if any(p.dtype != torch.bfloat16 for p in tensors):
                bad_vlsa_norm.append((name, [str(p.dtype) for p in tensors]))
        if bad_vlsa_norm:
            raise RuntimeError(f"strict VLSA BF16 LayerNorm mismatch: {bad_vlsa_norm[:8]}")
        log0(
            f"[ft] strict background: allocation HQQ={len(final_hqq_vlm)+len(final_hqq_dit)} "
            f"VLSA HQQ4={len(vlsa_hqq4)} VLSA norms BF16={len(vlsa_norm_bf16)}"
        )
    else:
        dit_snaps = {
            n: named[n].weight.detach().to(device, torch.float32).clone() for n in dit_vlsa
        }
        with patched_env(VLM_ENV):
            quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device=str(device))
    visual_candidates = {
        name for name, module in policy.model.named_modules()
        if name.startswith("backbone.model.model.visual.") and isinstance(module, nn.Linear)
    }
    replace_embedding(policy.model, 4, 64, True)
    n_visual = quantize_visual_linears(
        policy.model, nbits=4, group_size=64, device=str(device)
    )
    if args.alloc:
        from gr00t.eval.cosmos_quant_shim import PackedAffineEmbedding

        named_fixed = dict(policy.model.named_modules())
        embed = named_fixed.get("backbone.model.model.language_model.embed_tokens")
        if not isinstance(embed, PackedAffineEmbedding) or int(embed.nbits) != 4:
            raise RuntimeError(
                f"strict embedding background mismatch: {type(embed).__name__}"
            )
        if not isinstance(named_fixed.get("backbone.model.lm_head"), nn.Identity):
            raise RuntimeError("strict background expected dropped lm_head")
        if n_visual != len(visual_candidates) or not visual_candidates:
            raise RuntimeError(
                f"strict vision count mismatch: applied={n_visual} "
                f"expected={len(visual_candidates)}"
            )
        bad_visual = []
        for name in visual_candidates:
            module = named_fixed.get(name)
            cfg = getattr(module, "quant_config", None)
            weight_cfg = cfg.get("weight_quant_params", {}) if isinstance(cfg, dict) else {}
            if int(weight_cfg.get("nbits", -1)) != 4:
                bad_visual.append((name, type(module).__name__, weight_cfg.get("nbits")))
        if bad_visual:
            raise RuntimeError(f"strict vision HQQ4 mismatch: {bad_visual[:4]}")
        log0(
            f"[ft] strict fixed background: embedding4 + vision HQQ4x{n_visual} "
            f"+ lm_head dropped"
        )

    def put_frozen(name, W):
        b = dit_bias_all[name]
        lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None,
                        device=str(device), dtype=torch.bfloat16)
        lin.weight.data.copy_(W.to(torch.bfloat16))
        if b is not None:
            lin.bias.data.copy_(b.to(device=str(device), dtype=torch.bfloat16))
        replace_module(policy.model, name, lin)

    if not args.alloc:
        for n in dit_vlsa:
            put_frozen(n, quantize_weight_hqq(dit_snaps[n], nbits=4, group_size=64,
                                              device=str(device)).to(device, torch.float32))
        del dit_snaps
    gc.collect(); torch.cuda.empty_cache()

    # ---- install trainable QuantLinear from packed fits ----------------------
    named_q = dict(policy.model.named_modules())
    trainables = []
    KB = {2048: "275", 1024: "250", 512: "225", 256: "200",
          128: "175", 64: "150", 32: "125", 16: "100", 8: "075"}
    pre_v = "vlm_packed_van_u" if args.vanilla_fits else "vlm_packed_u"
    pre_d = "fits_packed_van_" if args.vanilla_fits else "fits_packed_u"
    def find_vlm(name, k):
        """(레이어, k) packed 를 여러 폴더에서 탐색."""
        cands = [ROT / f"{pre_v}{KB[k]}" / f"{name}.pt"]
        cands += ([ROT / "vlm_packed_vanilla_a230" / f"{name}.pt"] if args.vanilla_fits
                  else [ROT / d / f"{name}.pt" for d in ("vlm_packed_a230", "vlm_packed_a210")])
        for c in cands:
            if not c.exists():
                continue
            o = torch.load(c, map_location="cpu")
            if isinstance(o, dict) and "packed" in o and o["packed"]["codebook"].shape[0] == k:
                return c
        raise FileNotFoundError(f"packed 없음: {name} k={k}")

    if args.alloc:
        from artifact_registry import find_packed
        artifact_init = "vanilla" if args.vanilla_fits else "aware"
        vfiles = [
            (name, find_packed(
                "vlm", name, k, suite=args.suite, init=artifact_init, exact=True
            ))
            for name, k in sorted(final_vlm_kmap.items()) if k not in (90000, 80000)
        ]
    elif args.vlm_alloc:
        import csv as _c
        vfiles = [(r["layer"], find_vlm(r["layer"], int(r["k"])))
                  for r in _c.DictReader(open(args.vlm_alloc))]
    else:
        vfiles = [(f.stem, f) for f in sorted((ROT / args.vlm_fits).glob("*.pt"))]
    n_vlm = 0
    for name, f in vfiles:
        payload = torch.load(f, map_location="cpu", mmap=True)
        parent, attr = name.rsplit(".", 1)
        old = getattr(named_q[parent], attr)
        ql = QuantLinear(payload["packed"], getattr(old, "bias", None), device)
        setattr(named_q[parent], attr, ql)
        trainables.append((name, ql)); n_vlm += 1
    def find_dit(name, k):
        cands = [FS2 / f"{pre_d}{KB[k]}_{args.suite}" / f"{name}__ck.pt"]
        cands += ([FS2 / f"fits_packed_vanilla_a230_{args.suite}" / f"{name}__ck.pt"] if args.vanilla_fits
                  else [FS2 / f"fits_packed_{t}_{args.suite}" / f"{name}__ck.pt" for t in ("a230", "a210")])
        for c in cands:
            if not c.exists():
                continue
            o = torch.load(c, map_location="cpu")
            if isinstance(o, dict) and "packed" in o and o["packed"]["codebook"].shape[0] == k:
                return c
        raise FileNotFoundError(f"packed 없음: {name} k={k}")

    if args.alloc:
        from artifact_registry import find_packed
        artifact_init = "vanilla" if args.vanilla_fits else "aware"
        dfiles = [
            (name, find_packed(
                "dit", name, k, suite=args.suite, init=artifact_init, exact=True
            ))
            for name, k in sorted(final_dit_kmap.items()) if k not in (90000, 80000)
        ]
    elif args.dit_alloc:
        import csv as _c2
        kmap = {r["layer"]: int(r["k"]) for r in _c2.DictReader(open(args.dit_alloc))}
        dfiles = [(n, find_dit(n, kmap[n])) for n in dit_core if n in kmap]
    else:
        dfiles = [(n, FS2 / f"{args.dit_fits}_{args.suite}" / f"{n}__ck.pt") for n in dit_core]
    n_dit = 0
    for name, f in dfiles:
        payload = torch.load(f, map_location="cpu", mmap=True)
        ql = QuantLinear(payload["packed"], dit_bias_all[name], device)
        replace_module(policy.model, name, ql)
        trainables.append((name, ql)); n_dit += 1
    print(f"[ft] trainable layers: VLM {n_vlm} + DiT {n_dit}", flush=True)
    if args.alloc:
        expected_vlm_vc = 112 - len(final_hqq_vlm)
        expected_dit_vc = 226 - len(final_hqq_dit)
        if (n_vlm, n_dit) != (expected_vlm_vc, expected_dit_vc):
            raise RuntimeError(
                f"strict final VC count mismatch: got {n_vlm}/{n_dit}, "
                f"expected {expected_vlm_vc}/{expected_dit_vc}"
            )
        installed = {name for name, _ in trainables}
        expected = {
            name for name, k in {**final_vlm_kmap, **final_dit_kmap}.items()
            if k not in (90000, 80000)
        }
        if installed != expected:
            raise RuntimeError(
                f"strict final VC layer mismatch missing={sorted(expected-installed)[:4]} "
                f"extra={sorted(installed-expected)[:4]}"
            )
    else:
        # 101 = 기존(7월 핫스팟 11층 제외), 112 = 핫스팟 포함 전체
        # 111 = down_proj 1층만 제외(단일층 보호 실험, 2026-08-18)
        assert n_vlm in (101, 111, 112) and n_dit >= 225, (n_vlm, n_dit)
    gc.collect(); torch.cuda.empty_cache()

    step_base = 0          # 이전 실행들에서 이미 소비한 스텝
    resume_best = None     # 이전 실행까지의 best val (best 덮어쓰기 방지)
    resume_payload = None
    if args.resume:
        # latest(100스텝마다 저장) 를 우선 사용한다. codebook_state 는 'best' 시점이라
        # 학습 후반처럼 개선이 드문 구간에서 선점되면 수백 스텝을 재작업하게 된다.
        # (27회 선점 × 회당 수십~수백 스텝이 실제로 낭비됐다)
        st = None
        st_p = None
        for candidate in (out_dir / "latest_state.pt", out_dir / "codebook_state.pt"):
            if not candidate.exists():
                continue
            try:
                st = torch.load(candidate, map_location="cpu")
                st_p = candidate
                break
            except Exception as exc:
                log0(f"[ft] WARNING: unusable resume checkpoint {candidate.name}: {exc}")
        if st is not None:
            resume_payload = st
            resume_meta = st.get("__meta__", {})
            step_base = int(resume_meta.get("steps_done", 0))
            resume_best = resume_meta.get("best_val")
            if resume_meta.get("allocation_sha256") not in (None, alloc_sha256):
                raise RuntimeError("resume allocation SHA256 does not match --alloc")
            if resume_meta.get("global_batch") not in (None, global_batch):
                raise RuntimeError("resume global batch does not match --global-batch")
            if resume_meta.get("fp32_codebook_grad") not in (None, FP32_CODEBOOK_GRAD):
                raise RuntimeError(
                    "resume codebook-gradient mode does not match --fp32-codebook-grad"
                )
            n_res = 0
            for name, m in trainables:
                if name in st:
                    m.codebook.data.copy_(st[name]["codebook"].to(device))
                    m.scales.data.copy_(st[name]["scales"].to(device))
                    n_res += 1
            log0(f"[ft] resume: {n_res}개 레이어 상태 복원, 소비 스텝 {step_base}, "
                 f"best {resume_best} ({st_p.name})")

    cb_params = [m.codebook for _, m in trainables]
    sc_params = [m.scales for _, m in trainables]
    if args.tune_norms:
        extra = []
        for pname, prm in policy.model.named_parameters():
            if ("norm" in pname.lower() and pname.endswith("weight")) or pname.endswith(".bias"):
                prm.requires_grad_(True)
                extra.append(prm)
        sc_params = sc_params + extra
        print(f"[ft] tune-norms: norm/bias {sum(p.numel() for p in extra)/1e6:.2f}M 추가", flush=True)
    opt = torch.optim.AdamW([
        {"params": cb_params, "lr": args.lr},
        {"params": sc_params, "lr": args.lr_scale},
    ], weight_decay=0.0)
    if resume_payload is not None:
        if "__optimizer__" in resume_payload:
            opt.load_state_dict(resume_payload["__optimizer__"])
            log0("[ft] resume: AdamW optimizer state restored")
        elif step_base:
            log0(
                "[ft] WARNING: legacy resume state has no AdamW state; "
                "optimizer restart is not bitwise-equivalent"
            )
    n_train_p = sum(p.numel() for p in cb_params + sc_params)
    print(f"[ft] trainable params: {n_train_p/1e6:.2f}M", flush=True)

    val_ids = {i for i in range(len(batches)) if i % args.val_every == 0}
    train_items = [(i, b) for i, b in enumerate(batches) if i not in val_ids]
    val_items = [(i, b) for i, b in enumerate(batches) if i in val_ids]

    def run_loss(i, rid, bb, ai, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            a = run_action_grad(policy, move_to_device(bb, device, torch.bfloat16),
                                move_to_device(ai, device, torch.bfloat16), seed=7000 + i).to(torch.float32)
            r = refs[rid].to(a.device, torch.float32)
            rel = (a - r).norm() / r.norm().clamp_min(1e-9)
            loss = (a - r).square().mean()
        return loss, rel.item()

    def clear_wc():
        for _, _m in trainables:
            _m._wc = None

    def validate():
        # 검증 동안 파라미터가 불변이므로, 시작 시 한 번만 캐시를 비우고
        # 64개 샘플 전체에서 같은 W 를 재사용한다 (역시 무손실).
        clear_wc()
        rels = []
        for i, (rid, bb, ai) in val_items:
            _, rel = run_loss(i, rid, bb, ai, grad=False)
            rels.append(rel)
        clear_wc()                      # no_grad 캐시를 학습으로 가져가지 않는다
        return sum(rels) / len(rels)

    def save_final_overlays():
        """best 코드북 상태를 복원해, codebook/idx/scales를 그대로 비트팩킹
        오버레이로 저장한다(dense 복원 안 함).

        인덱스는 FT 내내 동결이라 상태 = 코드북+스케일이 전부다. 예전엔 여기서
        m.weight_()로 dense BF16(3.7GB)을 복원해 저장했는데, 그러면 배포 경로
        (cosmos_quant_shim.py)가 그냥 dense nn.Linear로 갈아끼워서 압축이 전혀
        실현이 안 됐다(2026-08-20 발견). codebook/idx/scales를 packed_indices로
        비트팩킹해서 저장하면 값은 bf16 1-ULP 이내로 동일하면서 실제로 압축된다
        (실측: DiT 2.18GB->0.20GB, VLM 1.49GB->0.14GB, spatial/cg150 기준)."""
        if rank != 0:
            return
        st_p = out_dir / "codebook_state.pt"
        if st_p.exists():
            st = torch.load(st_p, map_location="cpu")
            for name, m in trainables:
                if name in st:
                    m.codebook.data.copy_(st[name]["codebook"].to(device))
                    m.scales.data.copy_(st[name]["scales"].to(device))
        packed = {}
        for name, m in trainables:
            k = m.codebook.shape[0]
            index_bits = max(1, math.ceil(math.log2(k)))
            idx_cpu = m.idx.detach().to("cpu")
            packed[name] = {
                "format": "packed_vector_codebook_v1",
                "codebook": m.codebook.detach().to("cpu", torch.bfloat16),
                "packed_indices": pack_indices(idx_cpu, index_bits),
                "index_bits": index_bits,
                "index_shape": tuple(idx_cpu.shape),
                "scales": m.scales.detach().to("cpu", torch.bfloat16),
                "shift": m.shift.detach().to("cpu", torch.bfloat16),
                "in_features": m.in_features,
                "out_features": m.out_features,
                "group_size": m.gs,
                "vector_dim": m.vd,
            }
        vlm_w = {n: w for n, w in packed.items() if not n.startswith("action_head")}
        dit_w = {n: w for n, w in packed.items() if n.startswith("action_head")}
        overlay_meta = {
            "suite": args.suite,
            "source_allocation": str(Path(args.alloc).resolve()) if args.alloc else "",
            "source_allocation_sha256": alloc_sha256,
            "global_batch": global_batch,
            "epochs": args.epochs,
            "optimizer_steps_budget": budget,
            "fp32_codebook_grad": FP32_CODEBOOK_GRAD,
            "codebook_initialization": "vanilla" if args.vanilla_fits else "structure_aware",
            "assignment_frozen": True,
            "deployment_arithmetic": "BF16 gather -> BF16 scale -> BF16 shift",
        }
        atomic_torch_save(
            {"weights": vlm_w, "expected_applied": len(vlm_w),
             "expected_packed_applied": len(vlm_w), "meta": overlay_meta},
            out_dir / "vlm_ft_overlay.pt",
        )
        atomic_torch_save(
            {"weights": dit_w, "expected_applied": len(dit_w),
             "expected_packed_applied": len(dit_w), "meta": overlay_meta},
            out_dir / f"dit_ft_overlay_{args.suite}.pt",
        )
        print(f"[ft] final overlays written (best 상태 기준, packed+비트팩킹)", flush=True)

    if ddp:
        # 랭크마다 모델 구성(체크포인트/코드북 로딩) 완료 시점 편차가 크면
        # 첫 forward/backward(첫 NCCL collective)에서 아직 구성 중인 랭크와
        # 충돌할 수 있다 -- 완전히 동기화한 뒤 검증/학습을 시작한다.
        dist.barrier()
    v0 = validate()
    print(f"[ft] initial val rel-err: {v0:.6f}", flush=True)
    # 재개 시: 과거 best 가 지금(latest 복원 직후) val 보다 좋을 수 있다.
    # best 를 과거 값으로 이어받아, 더 나빠진 상태가 best 오버레이를 덮지 않게 한다.
    best = v0 if resume_best is None else min(v0, float(resume_best))
    # A fresh run must have a materialized step-0 best checkpoint.  Previously
    # codebook_state.pt was written only after a strict validation improvement;
    # if calibration never beat its initialization, final export silently used
    # the last (worse) parameters while labelling them as the best state.
    save_initial_best = resume_payload is None
    step = 0
    log = [{"step": 0, "val": v0}]
    rng = random.Random(1)
    import time as _t
    main._t20 = _t.time()
    train_params = cb_params + sc_params
    if len(train_items) % global_batch:
        raise RuntimeError(
            f"train set ({len(train_items)}) must be divisible by global batch ({global_batch})"
        )
    steps_per_epoch = len(train_items) // global_batch
    # `step` is an optimizer step, not a per-rank sample count.  The old DDP
    # code compared rank-local steps with epochs*all_samples, so a resumed job
    # could train up to world_size times too long.  This budget is hardware
    # independent when global_batch is fixed.
    budget = args.epochs * steps_per_epoch

    def sync_gradients(divisor):
        if not ddp:
            for prm in train_params:
                if prm.grad is not None:
                    prm.grad.div_(divisor)
            return
        grads = []
        for prm in train_params:
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)
            grads.append(prm.grad)
        # Hundreds of tiny NCCL collectives dominated the original 10.4s/step
        # calibration.  One flattened all-reduce is mathematically the same
        # sum/average and does not alter the optimizer or sample batches.
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(divisor)
        for grad, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(synced)

    def state_meta(total_steps):
        return {
            "steps_done": total_steps,
            "best_val": best,
            "global_batch": global_batch,
            "steps_per_epoch": steps_per_epoch,
            "allocation_sha256": alloc_sha256,
            "fp32_codebook_grad": FP32_CODEBOOK_GRAD,
            "codebook_initialization": "vanilla" if args.vanilla_fits else "structure_aware",
        }

    if rank == 0 and save_initial_best:
        initial = {
            name: {
                "codebook": m.codebook.detach().cpu(),
                "scales": m.scales.detach().cpu(),
            }
            for name, m in trainables
        }
        initial["__meta__"] = state_meta(0)
        atomic_torch_save(initial, out_dir / "codebook_state.pt")
        log0(f"[ft] initial best state saved (val {best:.6f})")
    if ddp:
        dist.barrier()

    if step_base:
        log0(f"[ft] 예산 {budget}스텝 중 {step_base} 소비됨 -> 남은 {max(0, budget - step_base)}스텝")
        if step_base >= budget:
            log0(f"[ft] done. best val {best:.6f} (init {v0:.6f})  [예산 소진, 추가 학습 없음]")
            save_final_overlays()
            return
    if args.pv and global_batch != world:
        raise RuntimeError("--pv with gradient accumulation is unsupported; use --global-batch=world")
    resume_epoch = step_base // steps_per_epoch
    resume_batch = step_base % steps_per_epoch
    for ep in range(args.epochs):
        order = train_items[:]
        rng.shuffle(order)
        if ep < resume_epoch:
            continue
        if args.pv and ep > 0:
            rank_order = order[rank::world] if ddp else order
            idx_snap = {name: m.idx.clone() for name, m in trainables}
            v_before = validate()
            pv_step(policy, trainables, rank_order, run_loss,
                    args.pv_batch, args.pv_max_switch, device)
            opt.zero_grad(set_to_none=True)
            v_after = validate()
            if v_after > v_before:
                for name, m in trainables:
                    m.idx.copy_(idx_snap[name])
                print(f"[pv] V-step 기각 (val {v_before:.4f} -> {v_after:.4f}), 배정 복원", flush=True)
            else:
                print(f"[pv] V-step 채택 (val {v_before:.4f} -> {v_after:.4f})", flush=True)
            del idx_snap
        first_batch = resume_batch if ep == resume_epoch else 0
        for batch_index in range(first_batch, steps_per_epoch):
            chunk = order[batch_index * global_batch:(batch_index + 1) * global_batch]
            local_chunk = chunk[rank::world] if ddp else chunk
            opt.zero_grad(set_to_none=True)
            local_losses = []
            local_rels = []
            for i, (rid, bb, ai) in local_chunk:
                clear_wc()              # each request gets a fresh reconstructed W graph
                loss, rel = run_loss(i, rid, bb, ai, grad=True)
                loss.backward()
                local_losses.append(loss.detach())
                local_rels.append(rel)
            sync_gradients(len(chunk))
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 and rank == 0:
                now = _t.time()
                dt20 = now - main._t20
                main._t20 = now
                mean_loss = torch.stack(local_losses).mean().item()
                mean_rel = sum(local_rels) / len(local_rels)
                print(f"[ft] ep{ep} step{step} loss={mean_loss:.6f} rel={mean_rel:.4f} "
                      f"({dt20/20:.2f}s/step)", flush=True)
            total_step = step_base + step
            if total_step % 100 == 0 or (args.steps and step >= args.steps):
                if ddp:
                    dist.barrier()
                if rank == 0:
                    v = validate()
                    log.append({"step": step_base + step, "val": v})
                    print(f"[ft] step{step_base + step} VAL rel-err: {v:.6f} "
                          f"(best {best:.6f})", flush=True)
                    if v < best:
                        best = v
                        state = {name: {"codebook": m.codebook.detach().cpu(),
                                        "scales": m.scales.detach().cpu()}
                                 for name, m in trainables}
                        state["__meta__"] = state_meta(step_base + step)
                        atomic_torch_save(state, out_dir / "codebook_state.pt")
                        print(f"[ft] best state saved (val {v:.6f})", flush=True)
                    latest = {name: {"codebook": m.codebook.detach().cpu(),
                                     "scales": m.scales.detach().cpu()}
                              for name, m in trainables}
                    latest["__meta__"] = state_meta(step_base + step)
                    latest["__optimizer__"] = opt.state_dict()
                    atomic_torch_save(latest, out_dir / "latest_state.pt")
                    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                if ddp:
                    dist.barrier()
            if args.steps and step >= args.steps:
                print("[ft] smoke stop", flush=True)
                save_final_overlays()
                return
            if step_base + step >= budget:      # 재개분 포함 총 예산 소진
                # 1152 is not divisible by the periodic 100-step gate.  The
                # old path therefore trained steps 1101--1152 but neither
                # validated nor checkpointed them, making the true endpoint
                # ineligible for best-state selection and leaving latest_state
                # at 1100.  Always materialize and evaluate the exact budget
                # endpoint before exporting the best overlay.
                if ddp:
                    dist.barrier()
                if rank == 0:
                    total_step = step_base + step
                    v = validate()
                    log.append({"step": total_step, "val": v})
                    print(f"[ft] endpoint step{total_step} VAL rel-err: {v:.6f} "
                          f"(best {best:.6f})", flush=True)
                    if v < best:
                        best = v
                        state = {
                            name: {
                                "codebook": m.codebook.detach().cpu(),
                                "scales": m.scales.detach().cpu(),
                            }
                            for name, m in trainables
                        }
                        state["__meta__"] = state_meta(total_step)
                        atomic_torch_save(state, out_dir / "codebook_state.pt")
                        print(f"[ft] endpoint best state saved (val {v:.6f})", flush=True)
                    latest = {
                        name: {
                            "codebook": m.codebook.detach().cpu(),
                            "scales": m.scales.detach().cpu(),
                        }
                        for name, m in trainables
                    }
                    latest["__meta__"] = state_meta(total_step)
                    latest["__optimizer__"] = opt.state_dict()
                    atomic_torch_save(latest, out_dir / "latest_state.pt")
                    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                if ddp:
                    dist.barrier()
                print(f"[ft] done. best val {best:.6f} (init {v0:.6f})  "
                      f"[총 {step_base + step}스텝 = 예산 {budget}]", flush=True)
                save_final_overlays()
                return
        resume_batch = 0
    if ddp:
        dist.barrier()
    save_final_overlays()
    print(f"[ft] done. best val {best:.6f} (init {v0:.6f})", flush=True)


if __name__ == "__main__":
    main()

