#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import tree
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

PROJ = Path("@WORKSPACE@/libero_pro_eval")
REPO = PROJ / "Isaac-GR00T-N1.7"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(REPO))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.eval.cosmos_quant_shim import (  # noqa: E402
    PackedAffineEmbedding,
    quantize_gr00t_vlm27_dit4,
)
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from gr00t.policy.server_client import MsgSerializer  # noqa: E402
from measure_embedding_vision_quant_memory_v1 import quantize_visual_linears  # noqa: E402
from measure_hotspot_protection_proxy import collate_request, move_to_device, rel_metrics, run_outputs  # noqa: E402


@contextmanager
def patched_env(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_setting_env(settings_json: Path, setting_name: str) -> dict[str, str]:
    payload = json.loads(settings_json.read_text())
    setting = next(s for s in payload["settings"] if s["name"] == setting_name)
    return {str(k): str(v) for k, v in setting["env"].items()}


def replace_module(root: torch.nn.Module, full_name: str, new_module: torch.nn.Module) -> torch.nn.Module:
    parent = root
    parts = full_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    old = getattr(parent, parts[-1])
    setattr(parent, parts[-1], new_module)
    return old


def replace_embedding(model: torch.nn.Module, nbits: int, group_size: int, drop_lm_head: bool) -> None:
    parent = model.get_submodule("backbone.model.model.language_model")
    emb = parent.embed_tokens
    if not isinstance(emb, torch.nn.Embedding):
        raise TypeError(f"embed_tokens is not nn.Embedding: {type(emb)}")
    parent.embed_tokens = PackedAffineEmbedding.from_embedding(emb, nbits=nbits, group_size=group_size)
    if drop_lm_head and hasattr(model.backbone.model, "lm_head"):
        model.backbone.model.lm_head = torch.nn.Identity()
    del emb


def is_dit_linear_name(name: str, module: torch.nn.Module, skip_vl_self_attention: bool) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if any(pattern in name for pattern in ("timestep", "Embed", "embedder")):
        return False
    if min(module.in_features, module.out_features) < 64:
        return False
    is_dit_model = "action_head.model" in name
    is_vl_self_attention = "action_head.vl_self_attention" in name
    if skip_vl_self_attention and is_vl_self_attention:
        return False
    return (is_dit_model or is_vl_self_attention) and ".norm" not in name


def collect_dit_bf16_snapshots(
    model: torch.nn.Module,
    skip_vl_self_attention: bool,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not is_dit_linear_name(name, module, skip_vl_self_attention):
            continue
        snapshots[name] = {
            "weight": module.weight.detach().cpu().to(torch.bfloat16).clone(),
            "bias": None if module.bias is None else module.bias.detach().cpu().to(torch.bfloat16).clone(),
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "params": int(module.weight.numel() + (0 if module.bias is None else module.bias.numel())),
        }
    return snapshots


def make_hqq_from_snapshot(snapshot: dict[str, Any], nbits: int, group_size: int, device: str) -> HQQLinear:
    linear = torch.nn.Linear(
        snapshot["in_features"],
        snapshot["out_features"],
        bias=snapshot["bias"] is not None,
        device=device,
        dtype=torch.bfloat16,
    )
    linear.weight.data.copy_(snapshot["weight"].to(device=device, dtype=torch.bfloat16))
    if snapshot["bias"] is not None:
        linear.bias.data.copy_(snapshot["bias"].to(device=device, dtype=torch.bfloat16))
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
    return HQQLinear(linear, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)


def load_batches(policy: Gr00tPolicy, request_dir: Path, limit: int) -> list[tuple[str, Any, Any]]:
    batches = []
    for path in sorted(request_dir.glob("*.msgpack"))[:limit]:
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
    return batches


def average_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(sum(row[key] for row in metric_rows) / len(metric_rows)) for key in metric_rows[0]}


@torch.no_grad()
def eval_policy(
    policy: Gr00tPolicy,
    batches: list[tuple[str, Any, Any]],
    references: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    positions: tuple[str, ...],
    seed_base: int,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = {position: [] for position in positions}
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        cand = run_outputs(policy, backbone_inputs, action_inputs, seed=seed_base + req_idx)
        ref = references[req_id]
        for position in positions:
            buckets[position].append(rel_metrics(ref[position], cand[position]))
    return {position: average_metrics(rows) for position, rows in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(PROJ / "models-gr00t-n17-libero/libero_spatial"))
    parser.add_argument("--request-dir", default=str(PROJ / "request_replay_runs/bf16_capture/requests"))
    parser.add_argument("--settings-json", default=str(PROJ / "quantization_analysis/act_vector_codebook_seed_sweep_2p5_v1/settings.json"))
    parser.add_argument("--setting-name", default="gain23_2.5_top5to4_act_d4_k256_seed20260601")
    parser.add_argument("--adapter-path", default=str(PROJ / "quantization_analysis/vla_aware_stage9_action_weighted_adapter_2026-06-13/adapters/r64_g2_uniform.pt"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--embedding-bits", type=int, default=4)
    parser.add_argument("--embedding-group-size", type=int, default=64)
    parser.add_argument("--drop-lm-head", action="store_true")
    parser.add_argument("--vision-bits", type=int, default=4)
    parser.add_argument("--single-embodiment-id", type=int, default=2)
    parser.add_argument("--skip-vl-self-attention", action="store_true")
    parser.add_argument("--max-layers", type=int, default=0, help="debug: only evaluate first N DiT layers")
    parser.add_argument("--layer-start", type=int, default=0, help="inclusive index into filtered DiT layer list")
    parser.add_argument("--layer-end", type=int, default=0, help="exclusive index into filtered DiT layer list; 0 means all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    positions = ("final_dit_output", "final_action_pred", "final_decoded_action")

    print("[dit-sensitivity] loading bf16 reference policy", flush=True)
    ref_policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), args.model_path, device=args.device, strict=True)
    ref_policy.model.eval()
    batches = load_batches(ref_policy, Path(args.request_dir), args.limit)
    references: dict[str, dict[str, torch.Tensor]] = {}
    print(f"[dit-sensitivity] building bf16 references for {len(batches)} requests", flush=True)
    for req_idx, (req_id, backbone_cpu, action_cpu) in enumerate(batches):
        backbone_inputs = move_to_device(backbone_cpu, device, torch.bfloat16)
        action_inputs = move_to_device(action_cpu, device, torch.bfloat16)
        references[req_id] = run_outputs(ref_policy, backbone_inputs, action_inputs, seed=7000 + req_idx)
    del ref_policy
    gc.collect()
    torch.cuda.empty_cache()

    print("[dit-sensitivity] loading quantized base policy", flush=True)
    policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), args.model_path, device=args.device, strict=True)
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad_(False)

    snapshots = collect_dit_bf16_snapshots(policy.model, args.skip_vl_self_attention)
    target_names = list(snapshots)
    if args.layer_start or args.layer_end:
        target_names = target_names[args.layer_start : args.layer_end or None]
    if args.max_layers > 0:
        target_names = target_names[: args.max_layers]
    print(f"[dit-sensitivity] captured {len(snapshots)} DiT bf16 linear snapshots; evaluating {len(target_names)}", flush=True)

    env = load_setting_env(Path(args.settings_json), args.setting_name)
    env["VLM_2BIT_CODEBOOK_PACKED"] = "1"
    env["DIT_BITS"] = "4"
    env["DIT_QUANT_MODE"] = "uniform"
    if args.adapter_path:
        env["VLM_TO_DIT_ADAPTER_PATH"] = args.adapter_path
    if args.single_embodiment_id >= 0:
        env["ACTION_HEAD_SINGLE_EMBODIMENT_ID"] = str(args.single_embodiment_id)
    with patched_env(env):
        n_quant = quantize_gr00t_vlm27_dit4(policy.model, group_size=args.group_size, device=args.device)
    if args.embedding_bits < 16:
        replace_embedding(policy.model, args.embedding_bits, args.embedding_group_size, args.drop_lm_head)
    if args.vision_bits > 0:
        quantize_visual_linears(policy.model, nbits=args.vision_bits, group_size=args.group_size, device=args.device)
    gc.collect()
    torch.cuda.empty_cache()

    print("[dit-sensitivity] evaluating all-DiT-4bit base", flush=True)
    base_metrics = eval_policy(policy, batches, references, device, positions, seed_base=7000)
    base_action_err = base_metrics["final_decoded_action"]["rel_rms_error"]

    summary_rows: list[dict[str, Any]] = []
    base_row = {
        "rank": 0,
        "module_name": "__base_all_dit4__",
        "params": 0,
        "final_decoded_action_rel_rms_error": base_action_err,
        "delta_vs_base": 0.0,
        "final_action_pred_rel_rms_error": base_metrics["final_action_pred"]["rel_rms_error"],
        "final_dit_output_rel_rms_error": base_metrics["final_dit_output"]["rel_rms_error"],
    }
    (out_dir / "base_metrics.json").write_text(json.dumps(base_metrics, indent=2) + "\n")

    raw_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(target_names, start=1):
        print(f"[dit-sensitivity] {idx}/{len(target_names)} downgrade {name}: 4bit -> 3bit", flush=True)
        hqq3 = make_hqq_from_snapshot(snapshots[name], nbits=3, group_size=args.group_size, device=args.device)
        old = replace_module(policy.model, name, hqq3)
        metrics = eval_policy(policy, batches, references, device, positions, seed_base=7000)
        replace_module(policy.model, name, old)
        gc.collect()
        torch.cuda.empty_cache()
        action_err = metrics["final_decoded_action"]["rel_rms_error"]
        row = {
            "module_name": name,
            "params": snapshots[name]["params"],
            "final_decoded_action_rel_rms_error": action_err,
            "delta_vs_base": action_err - base_action_err,
            "final_action_pred_rel_rms_error": metrics["final_action_pred"]["rel_rms_error"],
            "final_dit_output_rel_rms_error": metrics["final_dit_output"]["rel_rms_error"],
        }
        raw_rows.append(row)
        write_csv(out_dir / "raw_sensitivity_progress.csv", raw_rows)

    ranked = sorted(raw_rows, key=lambda r: (r["delta_vs_base"], r["final_decoded_action_rel_rms_error"]))
    for rank, row in enumerate(ranked, start=1):
        summary_rows.append({"rank": rank, **row})
    write_csv(out_dir / "dit_single_layer_4to3_sensitivity_ranked.csv", [base_row] + summary_rows)
    metadata = {
        "setting_name": args.setting_name,
        "quantized_layers": n_quant,
        "num_requests": len(batches),
        "num_dit_layers_total": len(snapshots),
        "num_dit_layers_evaluated": len(target_names),
        "layer_start": args.layer_start,
        "layer_end": args.layer_end or None,
        "base_final_decoded_action_rel_rms_error": base_action_err,
        "embedding_bits": args.embedding_bits,
        "vision_bits": args.vision_bits,
        "single_embodiment_id": args.single_embodiment_id,
        "adapter_path": args.adapter_path,
        "ranking": "lower delta_vs_base means safer to downgrade from 4bit to 3bit",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [
        "# DiT single-layer 4bit -> 3bit sensitivity ranking",
        "",
        f"- Base setting: `{args.setting_name}` + DiT all 4bit + embed{args.embedding_bits} + vision{args.vision_bits} + single embodiment {args.single_embodiment_id}",
        f"- Requests: {len(batches)} cached bf16 replay requests",
        "- Metric: relative RMS error vs full bf16 reference",
        f"- Base final decoded action rel RMS err: `{base_action_err:.6f}`",
        "",
        "Lower `delta_vs_base` means the DiT layer is safer to quantize from 4bit to 3bit.",
        "",
        "| rank | DiT layer | params | final action err | delta vs base | action_pred err | final DiT output err |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:30]:
        lines.append(
            f"| {row['rank']} | `{row['module_name']}` | {row['params']} | "
            f"{row['final_decoded_action_rel_rms_error']:.6f} | {row['delta_vs_base']:.6f} | "
            f"{row['final_action_pred_rel_rms_error']:.6f} | {row['final_dit_output_rel_rms_error']:.6f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[dit-sensitivity] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
