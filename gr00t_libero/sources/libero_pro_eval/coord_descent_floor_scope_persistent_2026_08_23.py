#!/usr/bin/env python3
"""Fixed-val Setting1 floor × scope ablation with one persistent process per GPU.

This reuses only the already validated persistent-worker transport from
coord_descent_alloc_setting1_improved_2026_08_23.py.  It intentionally does
not use that experiment's calibration rotation or rollback logic.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "@WORKSPACE@/libero_pro_eval")
import coord_descent_alloc_setting1_improved_2026_08_23 as infra  # noqa: E402
from train_spinquant_r1r2_vla_objective import decode_action_torch_libero  # noqa: E402

PROJ = Path("@WORKSPACE@/libero_pro_eval")
QA = PROJ / "quantization_analysis"
ALLOC = QA / "alloc_floor_2026-08-06"
SETDIR = PROJ / "request_replay_runs/calib_hard_2026-08-04/val/requests"
SUITE = "libero_spatial"

HQQ_SENTINEL_K = {90000: 4, 80000: 3}
BIT_OF_K = {
    90000: 4.00,
    80000: 3.00,
    **infra.BIT_OF_K,
}
LADDER = sorted(BIT_OF_K, reverse=True)
VC_START_K = 2048
VECTOR_DIM = 4
GROUP_SIZE = 64
HOT_NAMES = list(infra.HOT)
UNIFIED_PARAMS_CSV = QA / "coord_descent_setting3_2026_08_22_ckpt1.50.csv"
D_REFERENCE_CSV = QA / "coord_descent_2x2_D_floor_scope_persistent_2026_08_23.csv"


def packed_payload_bits(params: int, k: int) -> int:
    """Serialized payload for HQQ3/4 or packed d4 vector-codebook states."""
    if k in HQQ_SENTINEL_K:
        assert params % GROUP_SIZE == 0, (params, GROUP_SIZE)
        groups = params // GROUP_SIZE
        # BaseQuantizeConfig stores unquantized scale and zero metadata in the
        # BF16 compute dtype: two BF16 scalars per group.
        metadata_bytes = groups * 2 * 2
        if HQQ_SENTINEL_K[k] == 4:
            packed_weight_bytes = params // 2
        else:
            # HQQ's 3bit_32 packs ten 3-bit values into one int32.  Grouping
            # reshapes W to (params/64, 64), then pads the first dimension to
            # a multiple of ten.
            packed_weight_bytes = math.ceil(groups / 10) * GROUP_SIZE * 4
        return 8 * (packed_weight_bytes + metadata_bytes)

    assert params % VECTOR_DIM == 0, (params, VECTOR_DIM)
    assert params % GROUP_SIZE == 0, (params, GROUP_SIZE)
    index_bits = math.ceil(math.log2(k))
    n_vectors = params // VECTOR_DIM
    index_bytes = (n_vectors * index_bits + 7) // 8
    codebook_bytes = k * VECTOR_DIM * 2
    scale_bytes = (params // GROUP_SIZE) * 2
    return 8 * (index_bytes + codebook_bytes + scale_bytes)


def mixed_dense_cached(kind, name, k, cache, orig_weights, device):
    key = (kind, name, k)
    if key in cache:
        return cache[key].float()
    if k in HQQ_SENTINEL_K:
        weight = infra.quantize_weight_hqq(
            orig_weights[name],
            nbits=HQQ_SENTINEL_K[k],
            group_size=GROUP_SIZE,
            device=str(device),
        ).to(torch.float32)
    else:
        weight = infra.dense_from_packed_cached(kind, name, k, cache)
        return weight
    cache[key] = weight.to(torch.bfloat16)
    return weight


@torch.no_grad()
def decoded_action_only(
    policy,
    backbone_inputs,
    action_inputs,
    seed: int | list[int] | tuple[int, ...],
    cached_vlm=None,
):
    """Exact proxy forward returning only the tensor consumed by the search."""
    action_head = policy.model.action_head
    if cached_vlm is None:
        backbone_output = policy.model.backbone(backbone_inputs)
        features = action_head._encode_features(backbone_output, action_inputs)
        vl_embeds = features.backbone_features
        state_features = features.state_features
        image_mask = backbone_output.image_mask
        backbone_attention_mask = backbone_output.backbone_attention_mask
    else:
        vl_embeds, image_mask, backbone_attention_mask = cached_vlm
        embodiment_id = action_inputs.embodiment_id
        state = action_inputs.state
        assert state.shape[1] == action_head.config.state_history_length
        state = state.view(state.shape[0], 1, -1)
        state_features = action_head.state_encoder(state, embodiment_id)

    embodiment_id = action_inputs.embodiment_id
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    seeds = [seed] if isinstance(seed, int) else list(seed)
    if len(seeds) not in (1, batch_size):
        raise ValueError(
            f"expected one seed or one seed per batch item, got "
            f"{len(seeds)} seeds for batch size {batch_size}"
        )
    if len(seeds) == 1:
        seeds *= batch_size
    # Generate every item's initial diffusion noise independently.  This is
    # exactly the same RNG contract as evaluating batch-size-one requests in
    # sequence and lets the allocator batch deterministic proxy forwards.
    noises = []
    for item_seed in seeds:
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(item_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(item_seed)
            noises.append(
                torch.randn(
                    size=(
                        1,
                        action_head.config.action_horizon,
                        action_head.action_dim,
                    ),
                    dtype=vl_embeds.dtype,
                    device=device,
                )
            )
    actions = torch.cat(noises, dim=0)

    dt = 1.0 / action_head.num_inference_timesteps
    vel_strength = torch.ones_like(actions)
    for step_idx in range(action_head.num_inference_timesteps):
        t_cont = step_idx / float(action_head.num_inference_timesteps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = action_head.action_encoder(
            actions, timesteps_tensor, embodiment_id
        )
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)

        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity * vel_strength

    return decode_action_torch_libero(policy, actions).detach().cpu().float()


def prepare_fast_round(replica, subset_idx):
    """Move requests once and cache the VLM side invariant to DiT trials."""
    prepared = {}
    action_head = replica.policy.model.action_head
    with torch.no_grad():
        for index in subset_idx:
            request_id, batch, action_input = replica.batches[index]
            batch_device = infra.move_to_device(
                batch, replica.device, torch.bfloat16
            )
            action_device = infra.move_to_device(
                action_input, replica.device, torch.bfloat16
            )
            backbone_output = replica.policy.model.backbone(batch_device)
            features = action_head._encode_features(backbone_output, action_device)
            cached_vlm = (
                features.backbone_features,
                backbone_output.image_mask,
                backbone_output.backbone_attention_mask,
            )
            prepared[index] = (
                request_id,
                batch_device,
                action_device,
                cached_vlm,
            )
    return prepared


def measure_fast_exact(replica, subset_idx, prepared, use_cached_vlm: bool):
    """Keep the old CPU float64 proxy arithmetic exactly unchanged."""
    errors = []
    for index in subset_idx:
        request_id, batch, action_input, cached_vlm = prepared[index]
        action = decoded_action_only(
            replica.policy,
            batch,
            action_input,
            seed=7000 + index,
            cached_vlm=cached_vlm if use_cached_vlm else None,
        ).to(torch.float64)
        reference = replica.refs[request_id]
        errors.append(
            ((action - reference).norm() / reference.norm().clamp_min(1e-9)).item()
        )
    return sum(errors) / len(errors)


class FixedReplica(infra.Replica):
    """Replica using the canonical fixed val_hard requests instead of rotation."""

    def __init__(self, device_id: int):
        limit = int(os.environ.get("CD_FIXED_LIMIT", "40"))
        self.device = torch.device(f"cuda:{device_id}")
        self.cache: dict = {}
        self.refs = infra.load_refs("val_hard", limit=40)
        assert self.refs is not None, "val_hard limit40 BF16 reference cache is missing"
        self.policy = infra.Gr00tPolicy(
            infra.EmbodimentTag.resolve("LIBERO_PANDA"),
            infra.local_ckpt(SUITE),
            device=str(self.device),
            strict=True,
        )
        self.policy.model.eval()
        for parameter in self.policy.model.parameters():
            parameter.requires_grad_(False)
        self.batches = infra.load_batches(self.policy, SETDIR, limit)
        assert len(self.batches) == limit, (len(self.batches), limit)

        named = dict(self.policy.model.named_modules())
        orig_hot = {
            name: named[name].weight.detach().to(self.device, torch.float32).clone()
            for name in infra.HOT
        }
        with infra.patched_env(infra.VLM_ENV):
            infra.quantize_gr00t_vlm27_dit4(
                self.policy.model, group_size=64, device=str(self.device)
            )
        infra.replace_embedding(self.policy.model, 4, 64, True)
        infra.quantize_visual_linears(
            self.policy.model, nbits=4, group_size=64, device=str(self.device)
        )
        self.named_q = dict(self.policy.model.named_modules())
        for name, nbits in infra.HOT.items():
            restored = infra.quantize_weight_hqq(
                orig_hot[name], nbits=nbits, group_size=64, device=str(self.device)
            ).to(torch.float32)
            self.put(name, restored)
        del orig_hot
        gc.collect()
        torch.cuda.empty_cache()

    def measure_shard_deltas(self, shard, state, kind_of, subset_idx):
        """Same delta measurement as Claude's worker, skipping illegal floor moves."""
        min_bit = float(os.environ.get("CD_MIN_BIT", "1.0"))
        fast_exact = os.environ.get("CD_FAST_EXACT", "0") == "1"
        prepared = prepare_fast_round(self, subset_idx) if fast_exact else None
        base_err = (
            measure_fast_exact(self, subset_idx, prepared, False)
            if fast_exact else self.measure(subset_idx)
        )
        out = {}
        for name in shard:
            idx = LADDER.index(state[name])
            if idx + 1 >= len(LADDER):
                continue
            k_next = LADDER[idx + 1]
            if BIT_OF_K[k_next] < min_bit - 1e-9:
                continue
            current = infra.dense_from_packed_cached(
                kind_of(name), name, state[name], self.cache
            )
            downgraded = infra.dense_from_packed_cached(
                kind_of(name), name, k_next, self.cache
            )
            self.put(name, downgraded)
            err = (
                measure_fast_exact(
                    self,
                    subset_idx,
                    prepared,
                    use_cached_vlm=kind_of(name) == "dit",
                )
                if fast_exact else self.measure(subset_idx)
            )
            out[name] = err - base_err
            self.put(name, current)
        return base_err, out


class UnifiedReplica:
    """Fixed-val replica with all 338 candidates captured before quantization."""

    def __init__(self, device_id: int):
        limit = int(os.environ.get("CD_FIXED_LIMIT", "40"))
        self.device = torch.device(f"cuda:{device_id}")
        self.cache: dict = {}
        self.refs = infra.load_refs("val_hard", limit=40)
        assert self.refs is not None, "val_hard limit40 BF16 reference cache is missing"
        self.policy = infra.Gr00tPolicy(
            infra.EmbodimentTag.resolve("LIBERO_PANDA"),
            infra.local_ckpt(SUITE),
            device=str(self.device),
            strict=True,
        )
        self.policy.model.eval()
        for parameter in self.policy.model.parameters():
            parameter.requires_grad_(False)
        self.batches = infra.load_batches(self.policy, SETDIR, limit)
        assert len(self.batches) == limit, (len(self.batches), limit)

        vlm_rows = list(csv.DictReader(open(ALLOC / "vlm_new11_cg150.csv")))
        dit_rows = list(csv.DictReader(open(ALLOC / "dit_cg150.csv")))
        candidates = [row["layer"] for row in vlm_rows + dit_rows] + HOT_NAMES
        assert len(candidates) == 338 and len(set(candidates)) == 338
        named = dict(self.policy.model.named_modules())
        # HQQ3/4 must always be generated from the untouched BF16 weights.
        self.orig_weights = {
            name: named[name].weight.detach().to(self.device, torch.float32).clone()
            for name in candidates
        }

        with infra.patched_env(infra.VLM_ENV):
            infra.quantize_gr00t_vlm27_dit4(
                self.policy.model, group_size=64, device=str(self.device)
            )
        infra.replace_embedding(self.policy.model, 4, 64, True)
        infra.quantize_visual_linears(
            self.policy.model, nbits=4, group_size=64, device=str(self.device)
        )
        self.named_q = dict(self.policy.model.named_modules())
        gc.collect()
        torch.cuda.empty_cache()

    def put(self, name, weight):
        module = self.named_q[name]
        bias = getattr(module, "bias", None)
        linear = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=bias is not None,
            device=self.device,
            dtype=torch.bfloat16,
        )
        linear.weight.data.copy_(weight.to(self.device, torch.bfloat16))
        if bias is not None:
            linear.bias.data.copy_(bias.detach().to(self.device, torch.bfloat16))
        parent_name, attr = name.rsplit(".", 1)
        setattr(self.named_q[parent_name], attr, linear)

    def measure(self, subset_idx):
        errors = []
        for index in subset_idx:
            request_id, batch, action_input = self.batches[index]
            output = infra.run_outputs(
                self.policy,
                infra.move_to_device(batch, self.device, torch.bfloat16),
                infra.move_to_device(action_input, self.device, torch.bfloat16),
                seed=7000 + index,
            )
            action = output["final_decoded_action"].detach().to("cpu", torch.float64)
            reference = self.refs[request_id]
            errors.append(
                ((action - reference).norm() / reference.norm().clamp_min(1e-9)).item()
            )
        return sum(errors) / len(errors)

    def install_full_state(self, state, kind_of):
        for name, k in state.items():
            self.put(
                name,
                mixed_dense_cached(
                    kind_of(name),
                    name,
                    k,
                    self.cache,
                    self.orig_weights,
                    self.device,
                ),
            )

    def measure_shard_deltas(self, shard, state, kind_of, subset_idx):
        min_bit = float(os.environ.get("CD_MIN_BIT", "1.0"))
        fast_exact = os.environ.get("CD_FAST_EXACT", "0") == "1"
        prepared = prepare_fast_round(self, subset_idx) if fast_exact else None
        base_err = (
            measure_fast_exact(self, subset_idx, prepared, False)
            if fast_exact else self.measure(subset_idx)
        )
        output = {}
        for name in shard:
            idx = LADDER.index(state[name])
            if idx + 1 >= len(LADDER):
                continue
            k_next = LADDER[idx + 1]
            if BIT_OF_K[k_next] < min_bit - 1e-9:
                continue
            current = mixed_dense_cached(
                kind_of(name), name, state[name], self.cache,
                self.orig_weights, self.device,
            )
            downgraded = mixed_dense_cached(
                kind_of(name), name, k_next, self.cache,
                self.orig_weights, self.device,
            )
            self.put(name, downgraded)
            error = (
                measure_fast_exact(
                    self,
                    subset_idx,
                    prepared,
                    use_cached_vlm=kind_of(name) == "dit",
                )
                if fast_exact else self.measure(subset_idx)
            )
            output[name] = error - base_err
            self.put(name, current)
        return base_err, output


def run_worker_persistent(args) -> None:
    shard = [line.strip() for line in open(args.shard_file) if line.strip()]
    work_dir = Path(args.shard_file).parent

    def kind_of(name):
        return "dit" if name.startswith("action_head") else "vlm"

    replica = (
        UnifiedReplica(args.device)
        if os.environ.get("CD_UNIFIED_HQQ", "0") == "1"
        else FixedReplica(args.device)
    )
    round_idx = 0
    while True:
        if (work_dir / "stop.flag").exists():
            break
        state_file = work_dir / f"round_{round_idx}_state.csv"
        subset_file = work_dir / f"round_{round_idx}_subset.txt"
        if not state_file.exists() or not subset_file.exists():
            time.sleep(0.5)
            continue

        state = {
            row["layer"]: int(row["k"])
            for row in csv.DictReader(open(state_file))
        }
        subset = [int(value) for value in subset_file.read_text().split()]

        # Keep only one round's reconstructed CPU weights.  Retaining every
        # historical (layer, k) tensor makes each persistent worker grow by
        # several GiB over a long descent and can hit the Slurm host-RAM
        # cgroup.  Clearing this cache does not change the reconstructed
        # values or GPU model state; the current round is simply reloaded from
        # the same packed artifacts below.
        replica.cache.clear()
        gc.collect()
        replica.install_full_state(state, kind_of)
        base_err, deltas = replica.measure_shard_deltas(
            shard, state, kind_of, subset
        )
        output = work_dir / f"round_{round_idx}_result_{args.device}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(
            infra.json.dumps({"base_err": base_err, "deltas": deltas})
        )
        temporary.rename(output)
        round_idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-avg-bits", type=float, default=1.50)
    parser.add_argument("--refresh-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--min-bit", type=float, default=1.0,
                        choices=sorted(set(BIT_OF_K.values())))
    parser.add_argument("--scope-balanced", action="store_true")
    parser.add_argument("--moving-floor", action="store_true")
    parser.add_argument("--unified-hqq", action="store_true")
    parser.add_argument(
        "--fast-exact",
        action="store_true",
        help="Reuse invariant VLM features for DiT trials and omit unused outputs.",
    )
    parser.add_argument(
        "--match-d-total-budget",
        action="store_true",
        help="Stop at D candidate payload plus its fixed Legacy-11 HQQ payload.",
    )
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--checkpoint-step", type=float, default=0.1)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        assert args.persistent, "This script supports persistent workers only"
        run_worker_persistent(args)
        return

    assert args.out, "--out is required for the coordinator"
    assert 0 < args.limit <= 40
    os.environ["CD_FIXED_LIMIT"] = str(args.limit)
    os.environ["CD_MIN_BIT"] = str(args.min_bit)
    os.environ["CD_UNIFIED_HQQ"] = "1" if args.unified_hqq else "0"
    os.environ["CD_FAST_EXACT"] = "1" if args.fast_exact else "0"
    started = time.time()

    vlm_rows = list(csv.DictReader(open(ALLOC / "vlm_new11_cg150.csv")))
    dit_rows = list(csv.DictReader(open(ALLOC / "dit_cg150.csv")))
    vlm_all = [row["layer"] for row in vlm_rows]
    dit_all = [row["layer"] for row in dit_rows]
    params_by_name = {
        row["layer"]: int(row["params"]) for row in vlm_rows + dit_rows
    }
    unified_param_rows = {
        row["layer"]: int(row["params"])
        for row in csv.DictReader(open(UNIFIED_PARAMS_CSV))
    }
    hot_params = {name: unified_param_rows[name] for name in HOT_NAMES}
    assert len(hot_params) == 11
    if args.unified_hqq:
        vlm_all = vlm_all + HOT_NAMES
        params_by_name.update(hot_params)
    all_candidates = vlm_all + dit_all
    assert len(all_candidates) == (338 if args.unified_hqq else 327)
    assert len(set(all_candidates)) == len(all_candidates)

    search_candidates = list(all_candidates)
    if args.subset_size > 0:
        import random
        rng = random.Random(20260820)
        n_vlm = min(len(vlm_all), args.subset_size // 3)
        n_dit = min(len(dit_all), args.subset_size - n_vlm)
        search_candidates = (
            rng.sample(vlm_all, n_vlm) + rng.sample(dit_all, n_dit)
        )

    vlm_set = set(vlm_all)
    dit_set = set(dit_all)
    scopes = {
        "vlm": [name for name in search_candidates if name in vlm_set],
        "dit": [name for name in search_candidates if name in dit_set],
    }
    scope_params = {
        scope: sum(params_by_name[name] for name in names)
        for scope, names in scopes.items()
    }
    total_params = sum(params_by_name[name] for name in search_candidates)
    comparison_total_params = (
        sum(int(row["params"]) for row in vlm_rows + dit_rows)
        + sum(hot_params.values())
    )
    fixed_hot_payload_bits = sum(
        packed_payload_bits(
            hot_params[name], 90000 if nbits == 4 else 80000
        )
        for name, nbits in infra.HOT.items()
    )
    d_reference_rows = list(csv.DictReader(open(D_REFERENCE_CSV)))
    d_target_payload_bits = (
        sum(
            packed_payload_bits(int(row["params"]), int(row["k"]))
            for row in d_reference_rows
        )
        + fixed_hot_payload_bits
    )

    if args.resume_from:
        resumed = {
            row["layer"]: int(row["k"])
            for row in csv.DictReader(open(args.resume_from))
        }
        missing = [name for name in search_candidates if name not in resumed]
        assert not missing, f"resume CSV missing {len(missing)} layers: {missing[:3]}"
        start_k = LADDER[0] if args.unified_hqq else VC_START_K
        state = {name: resumed.get(name, start_k) for name in all_candidates}
    else:
        start_k = LADDER[0] if args.unified_hqq else VC_START_K
        state = {name: start_k for name in all_candidates}

    def kind_of(name):
        return "dit" if name.startswith("action_head") else "vlm"

    def nominal_avg():
        return sum(
            params_by_name[name] * BIT_OF_K[state[name]]
            for name in search_candidates
        ) / total_params

    def payload_avg(names, denominator):
        return sum(
            packed_payload_bits(params_by_name[name], state[name])
            for name in names
        ) / denominator

    def actual_avg():
        return payload_avg(search_candidates, total_params)

    def total_payload_bits():
        search_payload = sum(
            packed_payload_bits(params_by_name[name], state[name])
            for name in search_candidates
        )
        return search_payload + (0 if args.unified_hqq else fixed_hot_payload_bits)

    def total_actual_avg():
        return total_payload_bits() / comparison_total_params

    def scope_actual_avgs():
        return {
            scope: payload_avg(names, scope_params[scope])
            for scope, names in scopes.items()
        }

    def save_alloc(path):
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "params", "bit", "k"])
            for name in sorted(search_candidates):
                writer.writerow([
                    name,
                    params_by_name[name],
                    f"{BIT_OF_K[state[name]]:.2f}",
                    state[name],
                ])

    start_avg = nominal_avg()
    checkpoints = []
    if args.checkpoint_step > 0:
        point = round(
            int(start_avg / args.checkpoint_step + 1e-9) * args.checkpoint_step,
            2,
        )
        if point >= start_avg - 1e-9:
            point = round(point - args.checkpoint_step, 2)
        while point > args.target_avg_bits + 1e-9:
            checkpoints.append(point)
            point = round(point - args.checkpoint_step, 2)

    def floor_for_target(target):
        lower = [
            bit for bit in set(BIT_OF_K.values())
            if args.min_bit - 1e-9 <= bit < target - 1e-9
        ]
        return max(lower) if lower else args.min_bit

    def next_stage_target():
        return checkpoints[0] if checkpoints else args.target_avg_bits

    active_floor = (
        floor_for_target(next_stage_target())
        if args.moving_floor else args.min_bit
    )

    def above_target():
        if args.match_d_total_budget:
            return total_payload_bits() > d_target_payload_bits
        return nominal_avg() > args.target_avg_bits

    if args.dry_run:
        print(
            f"[cdfs-p] DRY candidates={len(search_candidates)} "
            f"unified={args.unified_hqq} start_nominal={start_avg:.6f} "
            f"start_actual={actual_avg():.6f} total_actual={total_actual_avg():.6f} "
            f"floor={active_floor:.2f} target_total_bytes={d_target_payload_bits // 8} "
            f"fixed_hot_bytes={fixed_hot_payload_bits // 8} "
            f"comparison_params={comparison_total_params}",
            flush=True,
        )
        return

    shards = [
        search_candidates[index::args.n_gpus] for index in range(args.n_gpus)
    ]
    work_dir = QA / f"coord_descent_proc_work_{Path(args.out).stem}"
    pool = infra.PersistentPool(
        args.n_gpus,
        work_dir,
        shards,
        str(Path(__file__).resolve()),
        sys.executable,
    )
    subset = list(range(args.limit))
    print(
        f"[cdfs-p] candidates={len(search_candidates)}, GPUs={args.n_gpus}, "
        f"fixed_val_hard={args.limit}, refresh={args.refresh_every}, "
        f"min_bit={args.min_bit:.2f}, moving_floor={args.moving_floor}, "
        f"active_floor={active_floor:.2f}, scope_balanced={args.scope_balanced}, "
        f"unified_hqq={args.unified_hqq}, start_nominal={start_avg:.4f}, "
        f"target_total_bytes={d_target_payload_bits // 8 if args.match_d_total_budget else 0}",
        flush=True,
    )
    print(f"[cdfs-p] persistent work_dir={work_dir}", flush=True)

    try:
        base_err, deltas = pool.measure_round(state, subset)
        current_scope = scope_actual_avgs()
        print(
            f"[cdfs-p] initial round: base_err={base_err:.6f}, "
            f"nominal={nominal_avg():.4f}, actual={actual_avg():.4f}, "
            f"total_actual={total_actual_avg():.4f}, active_floor={active_floor:.2f}, "
            f"scope=VLM:{current_scope['vlm']:.4f}/DiT:{current_scope['dit']:.4f}, "
            f"deltas={len(deltas)}, elapsed={time.time()-started:.0f}s",
            flush=True,
        )

        step = 0
        steps_since_refresh = 0
        while above_target():
            eligible = {"vlm": [], "dit": []}
            for name, delta in deltas.items():
                idx = LADDER.index(state[name])
                if idx + 1 >= len(LADDER):
                    continue
                k_next = LADDER[idx + 1]
                if BIT_OF_K[k_next] < active_floor - 1e-9:
                    continue
                eligible[kind_of(name)].append((name, delta, k_next))

            available = [scope for scope, rows in eligible.items() if rows]
            if not available:
                print("[cdfs-p] no legal downgrade remains", flush=True)
                break
            if args.scope_balanced and len(available) == 2:
                scope_now = scope_actual_avgs()
                selected_scope = max(available, key=lambda scope: scope_now[scope])
                rows = eligible[selected_scope]
            else:
                selected_scope = available[0] if len(available) == 1 else "global"
                rows = [row for scope in available for row in eligible[scope]]

            best_name = None
            best_cost = float("inf")
            for name, delta, k_next in rows:
                saved = (
                    packed_payload_bits(params_by_name[name], state[name])
                    - packed_payload_bits(params_by_name[name], k_next)
                )
                assert saved > 0
                cost = delta / saved
                if cost < best_cost:
                    best_name, best_cost = name, cost

            idx = LADDER.index(state[best_name])
            state[best_name] = LADDER[idx + 1]
            del deltas[best_name]
            step += 1
            steps_since_refresh += 1

            current_avg = nominal_avg()
            floor_changed = False
            while checkpoints and current_avg <= checkpoints[0]:
                point = checkpoints.pop(0)
                checkpoint = (
                    str(Path(args.out).with_suffix("")) + f"_ckpt{point:.2f}.csv"
                )
                save_alloc(checkpoint)
                print(
                    f"[cdfs-p] checkpoint {point:.2f}: nominal={current_avg:.4f}, "
                    f"actual={actual_avg():.4f}, total_actual={total_actual_avg():.4f}, "
                    f"floor={active_floor:.2f}, step={step} -> {checkpoint}",
                    flush=True,
                )

            if args.moving_floor:
                updated_floor = floor_for_target(next_stage_target())
                if updated_floor < active_floor - 1e-9:
                    old_floor = active_floor
                    active_floor = updated_floor
                    floor_changed = True
                    print(
                        f"[cdfs-p] moving floor {old_floor:.2f}->{active_floor:.2f} "
                        f"for target {next_stage_target():.2f}; forcing refresh",
                        flush=True,
                    )

            if step % 50 == 0:
                scope_now = scope_actual_avgs()
                print(
                    f"[cdfs-p] step{step}: nominal={current_avg:.4f}, "
                    f"actual={actual_avg():.4f}, total_actual={total_actual_avg():.4f}, "
                    f"floor={active_floor:.2f}, selected_scope={selected_scope}, "
                    f"scope=VLM:{scope_now['vlm']:.4f}/DiT:{scope_now['dit']:.4f}, "
                    f"elapsed={time.time()-started:.0f}s",
                    flush=True,
                )

            if floor_changed or steps_since_refresh >= args.refresh_every or not deltas:
                base_err, deltas = pool.measure_round(state, subset)
                steps_since_refresh = 0
                scope_now = scope_actual_avgs()
                print(
                    f"[cdfs-p] refresh step{step}: base_err={base_err:.6f}, "
                    f"nominal={nominal_avg():.4f}, actual={actual_avg():.4f}, "
                    f"total_actual={total_actual_avg():.4f}, floor={active_floor:.2f}, "
                    f"scope=VLM:{scope_now['vlm']:.4f}/DiT:{scope_now['dit']:.4f}, "
                    f"deltas={len(deltas)}, elapsed={time.time()-started:.0f}s",
                    flush=True,
                )

        final_err, _ = pool.measure_round(state, subset)
        final_scope = scope_actual_avgs()
        save_alloc(args.out)
        print(
            f"[cdfs-p] FINAL nominal={nominal_avg():.4f}, actual={actual_avg():.4f}, "
            f"total_actual={total_actual_avg():.4f}, floor={active_floor:.2f}, "
            f"scope=VLM:{final_scope['vlm']:.4f}/DiT:{final_scope['dit']:.4f}, "
            f"proxy={final_err:.6f}, steps={step}, elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        print(f"[cdfs-p] saved {args.out}", flush=True)
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
