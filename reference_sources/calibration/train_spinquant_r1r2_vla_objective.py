#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tree

from spinquant_stiefel_optimizer import SGDG
from gr00t.policy.server_client import MsgSerializer
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


QK_INPUT_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
)
MLP_INPUT_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
)


def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard init requires power-of-two size, got {size}")
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < size:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    signs = torch.randint(0, 2, (size,), device=device, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1)
    return (h * signs.view(1, -1) / (size**0.5)).to(torch.float32)


def fake_quantize_weight_ste(weight: torch.Tensor, nbits: int, group_size: int) -> torch.Tensor:
    if nbits >= 16:
        return weight
    maxq = 2 ** (nbits - 1) - 1
    reshaped = weight.view(weight.shape[0], -1, group_size)
    scale = reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / maxq
    q = torch.clamp(torch.round(reshaped / scale), -(maxq + 1), maxq)
    dq = (q * scale).view_as(weight)
    return weight + (dq - weight).detach()


def apply_head_rotation(weight: torch.Tensor, r2: torch.Tensor, transpose: bool) -> torch.Tensor:
    had_dim = r2.shape[0]
    if transpose:
        init_shape = weight.shape
        temp = weight.reshape(-1, init_shape[-1] // had_dim, had_dim)
        temp = temp.to(torch.float64) @ r2.to(torch.float64)
        return temp.reshape(init_shape).to(weight.dtype)
    weight_t = weight.t()
    transposed_shape = weight_t.shape
    temp = weight_t.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
    temp = temp.to(torch.float64) @ r2.to(torch.float64)
    return temp.reshape(transposed_shape).t().to(weight.dtype)


def iter_request_paths(request_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(request_dir.glob("*.msgpack"))
    if limit is not None:
        files = files[:limit]
    return files


def normalize_flat_observation(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    """Massage captured flat observations into the exact keys expected by the processor."""
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
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
    return collated_inputs


class SpinQuantBackboneLinearProxy(nn.Module):
    def __init__(
        self,
        orig_linear: nn.Linear,
        module_name: str,
        get_r1,
        get_r2,
        nbits: int,
        group_size: int,
    ):
        super().__init__()
        self.orig_linear = orig_linear
        self.module_name = module_name
        self._get_r1 = get_r1
        self._get_r2 = get_r2
        self.nbits = nbits
        self.group_size = group_size

        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = self._get_r1()
        parts = self.module_name.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        r2 = self._get_r2(layer_idx)

        weight = self.orig_linear.weight.to(torch.float32)
        bias = self.orig_linear.bias.to(torch.float32) if self.orig_linear.bias is not None else None

        if self.module_name.endswith(QK_INPUT_SUFFIXES) or self.module_name.endswith(MLP_INPUT_SUFFIXES):
            x_rot = torch.matmul(x.to(torch.float32), r1)
            w_rot = torch.matmul(weight, r1)
            w_q = fake_quantize_weight_ste(w_rot, self.nbits, self.group_size)
            return torch.nn.functional.linear(x_rot, w_q, bias).to(x.dtype)

        if self.module_name.endswith("self_attn.v_proj"):
            x_rot = torch.matmul(x.to(torch.float32), r1)
            w_rot = torch.matmul(weight, r1)
            w_rot = apply_head_rotation(w_rot, r2, transpose=False)
            w_q = fake_quantize_weight_ste(w_rot, self.nbits, self.group_size)
            b_q = apply_head_rotation(bias.unsqueeze(1), r2, transpose=False).squeeze(1) if bias is not None else None
            return torch.nn.functional.linear(x_rot, w_q, b_q).to(x.dtype)

        if self.module_name.endswith("self_attn.o_proj"):
            x_rot = apply_head_rotation(x.to(torch.float32), r2, transpose=True)
            w_rot = torch.matmul(r1.t(), weight)
            w_rot = apply_head_rotation(w_rot, r2, transpose=True)
            w_q = fake_quantize_weight_ste(w_rot, self.nbits, self.group_size)
            y_rot = torch.nn.functional.linear(x_rot, w_q, bias)
            y = torch.matmul(y_rot, r1.t())
            return y.to(x.dtype)

        if self.module_name.endswith("mlp.down_proj"):
            w_rot = torch.matmul(r1.t(), weight)
            w_q = fake_quantize_weight_ste(w_rot, self.nbits, self.group_size)
            y_rot = torch.nn.functional.linear(x.to(torch.float32), w_q, bias)
            y = torch.matmul(y_rot, r1.t())
            return y.to(x.dtype)

        raise ValueError(f"Unhandled module name: {self.module_name}")


def replace_backbone_linears(backbone_model: nn.Module, get_r1, get_r2, nbits: int, group_size: int) -> None:
    linear_types = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    for layer_idx in range(16):
        for suffix in linear_types:
            rel_name = f"language_model.layers.{layer_idx}.{suffix}"
            parent_name, child_name = rel_name.rsplit(".", 1)
            parent = backbone_model.get_submodule(parent_name)
            orig = getattr(parent, child_name)
            full_name = f"backbone.model.model.{rel_name}"
            proxy = SpinQuantBackboneLinearProxy(
                orig_linear=orig,
                module_name=full_name,
                get_r1=get_r1,
                get_r2=get_r2,
                nbits=nbits,
                group_size=group_size,
            )
            setattr(parent, child_name, proxy)


def move_to_device(x: Any, device: torch.device, dtype: torch.dtype) -> Any:
    def _to(v):
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                return v.to(device=device, dtype=dtype)
            return v.to(device=device)
        return v
    return tree.map_structure(_to, x)


def decode_action_torch_libero(
    policy: Gr00tPolicy,
    action_pred: torch.Tensor,
) -> torch.Tensor:
    """Torch version of decode_action for LIBERO.

    LIBERO uses absolute non-EEF actions with min/max unnormalization only, so we can
    reproduce decode_action exactly enough for training without the numpy processor path.
    """
    embodiment_tag = policy.embodiment_tag.value
    action_cfg = policy.processor.modality_configs[embodiment_tag]["action"]
    norm_params = policy.processor.state_action_processor.norm_params[embodiment_tag]["action"]
    mean_std_keys = set(action_cfg.mean_std_embedding_keys or [])
    action_horizon = len(action_cfg.delta_indices)

    decoded_groups = []
    start_idx = 0
    for key in action_cfg.modality_keys:
        params = norm_params[key]
        joint_dim = int(params["dim"].item())
        group = action_pred[..., :action_horizon, start_idx : start_idx + joint_dim].to(torch.float32)
        start_idx += joint_dim

        if key in mean_std_keys:
            mean = torch.as_tensor(params["mean"], device=group.device, dtype=group.dtype)
            std = torch.as_tensor(params["std"], device=group.device, dtype=group.dtype)
            decoded = group * std + mean
        else:
            min_vals = torch.as_tensor(params["min"], device=group.device, dtype=group.dtype)
            max_vals = torch.as_tensor(params["max"], device=group.device, dtype=group.dtype)
            decoded = (torch.clamp(group, -1.0, 1.0) + 1.0) / 2.0 * (max_vals - min_vals) + min_vals

        decoded_groups.append(decoded)

    return torch.cat(decoded_groups, dim=-1)


def differentiable_action_pred(
    action_head: nn.Module,
    backbone_output,
    action_input,
    seed: int,
) -> torch.Tensor:
    features = action_head._encode_features(backbone_output, action_input)
    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_input.embodiment_id

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

    for t in range(action_head.num_inference_timesteps):
        t_cont = t / float(action_head.num_inference_timesteps)
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
        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        actions = actions + dt * pred_velocity * vel_strength

    return actions


def compute_objective_target(
    policy: Gr00tPolicy,
    objective: str,
    backbone_inputs,
    action_inputs,
    seed: int,
) -> torch.Tensor:
    if objective == "final_hidden":
        return policy.model.backbone(backbone_inputs)["backbone_features"].detach().cpu().float()

    backbone_outputs = policy.model.backbone(backbone_inputs)
    if objective == "vl_embeds":
        processed = policy.model.action_head.process_backbone_output(backbone_outputs)
        return processed["backbone_features"].detach().cpu().float()
    if objective == "action_pred":
        pred = differentiable_action_pred(policy.model.action_head, backbone_outputs, action_inputs, seed=seed)
        return pred.detach().cpu().float()
    if objective == "decoded_action":
        pred = differentiable_action_pred(policy.model.action_head, backbone_outputs, action_inputs, seed=seed)
        decoded = decode_action_torch_libero(policy, pred)
        return decoded.detach().cpu().float()
    raise ValueError(f"Unsupported objective: {objective}")


def compute_candidate_objective(
    policy: Gr00tPolicy,
    objective: str,
    backbone_inputs,
    action_inputs,
    seed: int,
) -> torch.Tensor:
    if objective == "final_hidden":
        return policy.model.backbone(backbone_inputs)["backbone_features"].to(torch.float32)

    backbone_outputs = policy.model.backbone(backbone_inputs)
    if objective == "vl_embeds":
        processed = policy.model.action_head.process_backbone_output(backbone_outputs)
        return processed["backbone_features"].to(torch.float32)
    if objective == "action_pred":
        return differentiable_action_pred(policy.model.action_head, backbone_outputs, action_inputs, seed=seed).to(torch.float32)
    if objective == "decoded_action":
        pred = differentiable_action_pred(policy.model.action_head, backbone_outputs, action_inputs, seed=seed)
        return decode_action_torch_libero(policy, pred).to(torch.float32)
    raise ValueError(f"Unsupported objective: {objective}")


def train_objective(
    model_path: str,
    request_dir: Path,
    out_path: Path,
    objective: str,
    hidden_size: int,
    head_dim: int,
    num_layers: int,
    nbits: int,
    group_size: int,
    lr: float,
    steps: int,
    limit: int | None,
    device: str,
) -> None:
    dev = torch.device(device)
    policy = Gr00tPolicy("LIBERO_PANDA", model_path, device=device)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)

    request_paths = iter_request_paths(request_dir, limit=limit)
    if not request_paths:
        raise RuntimeError(f"No request msgpack files under {request_dir}")

    backbone_batches: list[dict[str, Any]] = []
    action_batches: list[dict[str, Any]] = []
    reference_outputs: list[torch.Tensor] = []

    with torch.no_grad():
        for req_idx, path in enumerate(request_paths):
            payload = MsgSerializer.from_bytes(path.read_bytes())
            observation = payload["observation"]
            collated = collate_request(policy, observation)
            backbone_inputs, action_inputs = policy.model.prepare_input(collated)
            backbone_batches.append(tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, backbone_inputs))
            action_batches.append(tree.map_structure(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, action_inputs))
            ref = compute_objective_target(
                policy,
                objective=objective,
                backbone_inputs=backbone_inputs,
                action_inputs=action_inputs,
                seed=1000 + req_idx,
            )
            reference_outputs.append(ref)

    r1 = nn.Parameter(hadamard_matrix(hidden_size, dev))
    r2_modules = nn.ParameterDict(
        {str(layer_idx): nn.Parameter(hadamard_matrix(head_dim, dev)) for layer_idx in range(num_layers)}
    )

    def get_r1():
        return r1

    def get_r2(idx: int):
        return r2_modules[str(idx)]

    replace_backbone_linears(policy.model.backbone.model, get_r1, get_r2, nbits, group_size)

    optimizer = SGDG([r1] + list(r2_modules.parameters()), lr=lr, stiefel=True)
    history: list[dict[str, float]] = []
    num_requests = max(1, len(backbone_batches))

    for step in range(steps):
        optimizer.zero_grad()
        total_loss_value = 0.0

        for req_idx, (backbone_inputs_cpu, action_inputs_cpu, ref_cpu) in enumerate(
            zip(backbone_batches, action_batches, reference_outputs)
        ):
            backbone_inputs = move_to_device(backbone_inputs_cpu, dev, torch.bfloat16)
            action_inputs = move_to_device(action_inputs_cpu, dev, torch.bfloat16)
            ref = ref_cpu.to(device=dev, dtype=torch.float32)
            cand = compute_candidate_objective(
                policy,
                objective=objective,
                backbone_inputs=backbone_inputs,
                action_inputs=action_inputs,
                seed=1000 + req_idx,
            )
            loss_i = torch.mean((cand - ref) ** 2) / num_requests
            total_loss_value += float(loss_i.detach().item())
            loss_i.backward()

        optimizer.step()

        with torch.no_grad():
            r1_ortho_err = torch.linalg.norm(r1.t() @ r1 - torch.eye(hidden_size, device=dev))
            r2_errs = []
            for param in r2_modules.values():
                eye = torch.eye(head_dim, device=dev)
                r2_errs.append(torch.linalg.norm(param.t() @ param - eye))
            r2_ortho_err = torch.stack(r2_errs).mean() if r2_errs else torch.zeros((), device=dev)
        history.append(
            {
                "step": step,
                "loss": total_loss_value,
                "r1_ortho_err": float(r1_ortho_err.item()),
                "r2_ortho_err": float(r2_ortho_err.item()),
            }
        )
        print(
            f"[spinquant-{objective}] step={step} loss={total_loss_value:.6f} "
            f"r1_ortho_err={r1_ortho_err.item():.6f} "
            f"r2_ortho_err={r2_ortho_err.item():.6f}",
            flush=True,
        )

    payload = {"linear_layers": {}}
    for layer_idx in range(num_layers):
        r2 = r2_modules[str(layer_idx)].detach().cpu()
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            name = f"backbone.model.model.language_model.layers.{layer_idx}.{suffix}"
            entry: dict[str, object] = {"weight_only": False}
            if suffix in QK_INPUT_SUFFIXES or suffix in MLP_INPUT_SUFFIXES:
                entry["input_rotation"] = r1.detach().cpu()
            elif suffix == "self_attn.v_proj":
                entry["input_rotation"] = r1.detach().cpu()
                entry["head_rotation"] = r2
                entry["head_rotation_transpose"] = False
            elif suffix == "self_attn.o_proj":
                entry["output_rotation"] = r1.detach().cpu()
                entry["head_rotation"] = r2
                entry["head_rotation_transpose"] = True
            elif suffix == "mlp.down_proj":
                entry["output_rotation"] = r1.detach().cpu()
            payload["linear_layers"][name] = entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps({"objective": objective, "history": history}, indent=2)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--request-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--objective",
        choices=("final_hidden", "vl_embeds", "action_pred", "decoded_action"),
        default="vl_embeds",
    )
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--nbits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_objective(
        model_path=args.model_path,
        request_dir=Path(args.request_dir),
        out_path=Path(args.out),
        objective=args.objective,
        hidden_size=args.hidden_size,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        nbits=args.nbits,
        group_size=args.group_size,
        lr=args.lr,
        steps=args.steps,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
