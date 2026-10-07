#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tree

from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.policy.server_client import MsgSerializer
from train_spinquant_r1r2_vla_objective import decode_action_torch_libero


def normalize_flat_observation(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(observation)
    expected_language_keys = tuple(policy.modality_configs["language"].modality_keys)
    if not any(key in normalized for key in expected_language_keys):
        for fallback in (
            "annotation.human.action.task_description",
            "annotation.human.coarse_action",
            "task",
        ):
            if fallback in normalized:
                normalized[expected_language_keys[0]] = normalized[fallback]
                break
    return normalized


def collate_request(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_flat_observation(policy, observation)
    collated_inputs = policy.processor.process_observation(normalized, policy.embodiment_tag)
    return _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)


def move_to_device(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    def _to(v):
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                return v.to(device=device, dtype=dtype)
            return v.to(device=device)
        return v

    return tree.map_structure(_to, x)


class DequantHQQLinear(nn.Module):
    def __init__(self, orig: nn.Linear, nbits: int, group_size: int, device: torch.device):
        super().__init__()
        weight = orig.weight.detach().to(device=device, dtype=torch.float32)
        q_weight = quantize_weight_hqq(weight, nbits=nbits, group_size=group_size, device=device)
        self.register_buffer("q_weight", q_weight.to(torch.float32), persistent=False)
        if orig.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", orig.bias.detach().to(device=device, dtype=torch.float32), persistent=False)
        self.in_features = orig.in_features
        self.out_features = orig.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x.to(torch.float32), self.q_weight, self.bias)
        return y.to(x.dtype)


def replace_for_setting(
    policy: Gr00tPolicy,
    all_modules: list[str],
    protected_bits: dict[str, int | None],
    base_bits: int,
    group_size: int,
    device: torch.device,
) -> list[tuple[str, nn.Module]]:
    originals: list[tuple[str, nn.Module]] = []
    for module_name in all_modules:
        nbits = protected_bits.get(module_name, base_bits)
        if nbits is None:
            continue
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = policy.model.get_submodule(parent_name)
        orig = getattr(parent, child_name)
        if not isinstance(orig, nn.Linear):
            raise TypeError(f"{module_name} is not nn.Linear: {type(orig)}")
        proxy = DequantHQQLinear(orig, nbits=nbits, group_size=group_size, device=device)
        setattr(parent, child_name, proxy)
        originals.append((module_name, orig))
    return originals


def restore(policy: Gr00tPolicy, originals: list[tuple[str, nn.Module]]) -> None:
    for module_name, orig in reversed(originals):
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = policy.model.get_submodule(parent_name)
        setattr(parent, child_name, orig)


def rel_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    ref_f = ref.detach().reshape(-1).to(torch.float64)
    cand_f = cand.detach().reshape(-1).to(torch.float64)
    diff = cand_f - ref_f
    ref_rms = torch.sqrt(torch.mean(ref_f.square())).item()
    diff_rms = torch.sqrt(torch.mean(diff.square())).item()
    cosine = F.cosine_similarity(ref_f.float().unsqueeze(0), cand_f.float().unsqueeze(0)).item()
    return {
        "ref_rms": ref_rms,
        "cand_rms": torch.sqrt(torch.mean(cand_f.square())).item(),
        "diff_rms": diff_rms,
        "rel_rms_error": diff_rms / max(ref_rms, 1e-12),
        "cosine_distance": 1.0 - cosine,
        "max_abs_error": diff.abs().max().item(),
    }


@torch.no_grad()
def run_outputs(policy: Gr00tPolicy, backbone_inputs: Any, action_inputs: Any, seed: int) -> dict[str, torch.Tensor]:
    action_head = policy.model.action_head
    backbone_output = policy.model.backbone(backbone_inputs)
    features = action_head._encode_features(backbone_output, action_inputs)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_inputs.embodiment_id
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device

    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

    dt = 1.0 / action_head.num_inference_timesteps
    vel_strength = torch.ones_like(actions)
    final_dit_output = None

    for step_idx in range(action_head.num_inference_timesteps):
        t_cont = step_idx / float(action_head.num_inference_timesteps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)

        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
        final_dit_output = model_output
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity * vel_strength

    assert final_dit_output is not None
    decoded = decode_action_torch_libero(policy, actions)
    return {
        "dit_input": vl_embeds.detach().cpu().float(),
        "final_dit_output": final_dit_output.detach().cpu().float(),
        "final_action_pred": actions.detach().cpu().float(),
        "final_decoded_action": decoded.detach().cpu().float(),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in rows[0]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_top_modules(path: Path, top_k: int) -> list[str]:
    with path.open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["dit_input_global_rel_rms_error"]), reverse=True)
    return [r["module_name"] for r in rows[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="@WORKSPACE@/scratch/models-gr00t-n17-libero/libero_spatial")
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--module-file", required=True)
    parser.add_argument("--sensitivity-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_modules = [line.strip() for line in Path(args.module_file).read_text().splitlines() if line.strip()]
    l2_down = "backbone.model.model.language_model.layers.2.mlp.down_proj"
    top3 = read_top_modules(Path(args.sensitivity_csv), 3)
    top5 = read_top_modules(Path(args.sensitivity_csv), 5)

    settings: list[tuple[str, dict[str, int | None]]] = [
        ("vlm3_all", {}),
        ("vlm3_l2down_4bit", {l2_down: 4}),
        ("vlm3_l2down_8bit", {l2_down: 8}),
        ("vlm3_l2down_bf16", {l2_down: None}),
        ("vlm3_top3_4bit", {m: 4 for m in top3}),
        ("vlm3_top5_4bit", {m: 4 for m in top5}),
    ]

    policy = Gr00tPolicy("LIBERO_PANDA", args.model_path, device=args.device)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)

    request_paths = sorted(Path(args.request_dir).glob("*.msgpack"))[: args.limit]
    batches: list[tuple[str, Any, Any]] = []
    for path in request_paths:
        payload = MsgSerializer.from_bytes(path.read_bytes())
        collated = collate_request(policy, payload["observation"])
        backbone_inputs, action_inputs = policy.model.prepare_input(collated)
        batches.append(
            (
                payload["request_id"],
                tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, backbone_inputs),
                tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, action_inputs),
            )
        )

    print(f"[hotspot-proxy] building bf16 references for {len(batches)} requests", flush=True)
    references: dict[str, dict[str, torch.Tensor]] = {}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        references[req_id] = run_outputs(policy, backbone_inputs, action_inputs, seed=1000 + req_idx)

    per_request: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    positions = ("dit_input", "final_dit_output", "final_action_pred", "final_decoded_action")

    for setting_name, protected_bits in settings:
        print(f"[hotspot-proxy] setting={setting_name}", flush=True)
        originals = replace_for_setting(
            policy,
            all_modules,
            protected_bits=protected_bits,
            base_bits=3,
            group_size=args.group_size,
            device=device,
        )
        try:
            buckets: dict[str, list[dict[str, float]]] = {p: [] for p in positions}
            for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
                backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
                action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
                cand = run_outputs(policy, backbone_inputs, action_inputs, seed=1000 + req_idx)
                ref = references[req_id]
                for position in positions:
                    metrics = rel_metrics(ref[position], cand[position])
                    buckets[position].append(metrics)
                    per_request.append(
                        {
                            "setting": setting_name,
                            "request_id": req_id,
                            "position": position,
                            **metrics,
                        }
                    )
            for position in positions:
                summary.append({"setting": setting_name, "position": position, **aggregate(buckets[position])})
        finally:
            restore(policy, originals)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(out_dir / "per_request_metrics.csv", per_request)
    write_csv(out_dir / "summary_metrics.csv", summary)
    (out_dir / "settings.json").write_text(
        json.dumps(
            {
                "settings": [
                    {"name": name, "protected_bits": {k: ("bf16" if v is None else v) for k, v in protected.items()}}
                    for name, protected in settings
                ],
                "top3": top3,
                "top5": top5,
            },
            indent=2,
        )
    )
    print(f"[hotspot-proxy] wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
