#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from measure_embedding_quant_memory_v1 import (
    EmbodimentTag,
    Gr00tPolicy,
    bytes_params_buffers,
    cuda_snapshot,
    load_setting_env,
    patched_env,
    quantize_gr00t_vlm27_dit4,
    replace_embedding,
    summarize,
    tensor_rows,
)


PROJ = Path("@WORKSPACE@/libero_pro_eval")


def replace_module(root: torch.nn.Module, full_name: str, new_module: torch.nn.Module) -> None:
    parent = root
    parts = full_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_visual_linears(model: torch.nn.Module, *, nbits: int, group_size: int, device: str) -> int:
    candidates: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if name.startswith("backbone.model.model.visual.") and isinstance(module, torch.nn.Linear):
            candidates.append((name, module))
    for name, module in candidates:
        cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
        hqq = HQQLinear(module, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
        replace_module(model, name, hqq)
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", default=str(PROJ / "models-gr00t-n17-libero/libero_spatial"))
    parser.add_argument("--settings-json", default=str(PROJ / "quantization_analysis/act_vector_codebook_seed_sweep_2p5_v1/settings.json"))
    parser.add_argument("--setting-name", default="gain23_2.5_top5to4_act_d4_k256_seed20260601")
    parser.add_argument("--adapter-path", default=str(PROJ / "quantization_analysis/vla_aware_stage9_action_weighted_adapter_2026-06-13/adapters/r64_g2_uniform.pt"))
    parser.add_argument("--embedding-bits", type=int, default=4)
    parser.add_argument("--embedding-group-size", type=int, default=64)
    parser.add_argument("--drop-lm-head", action="store_true")
    parser.add_argument("--vision-bits", type=int, default=0, help="0 means bf16 vision/projector")
    parser.add_argument("--vision-group-size", type=int, default=64)
    parser.add_argument("--single-embodiment-id", type=int, default=-1, help=">=0 materializes only this action-head embodiment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    env = load_setting_env(Path(args.settings_json), args.setting_name)
    env["VLM_2BIT_CODEBOOK_PACKED"] = "1"
    if args.adapter_path:
        env["VLM_TO_DIT_ADAPTER_PATH"] = args.adapter_path
    if args.single_embodiment_id >= 0:
        env["ACTION_HEAD_SINGLE_EMBODIMENT_ID"] = str(args.single_embodiment_id)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve("LIBERO_PANDA"),
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    after_load = cuda_snapshot(device)

    with patched_env(env):
        n_quant = quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device=args.device)
    if args.embedding_bits < 16:
        replace_embedding(policy.model, args.embedding_bits, args.embedding_group_size, args.drop_lm_head)
    n_vision = 0
    if args.vision_bits:
        n_vision = quantize_visual_linears(
            policy.model,
            nbits=args.vision_bits,
            group_size=args.vision_group_size,
            device=args.device,
        )
    gc.collect()
    torch.cuda.empty_cache()

    rows = tensor_rows(policy.model)
    summary = summarize(rows)
    result = {
        "label": args.label,
        "embedding_bits": args.embedding_bits,
        "embedding_group_size": args.embedding_group_size,
        "drop_lm_head": args.drop_lm_head,
        "vision_bits": "bf16" if args.vision_bits == 0 else args.vision_bits,
        "vision_group_size": args.vision_group_size,
        "single_embodiment_id": None if args.single_embodiment_id < 0 else args.single_embodiment_id,
        "quantized_layers": n_quant,
        "vision_quantized_linears": n_vision,
        "after_load_cuda": after_load,
        "final_cuda": cuda_snapshot(device),
        "final_param_buffer_gb": bytes_params_buffers(policy.model) / (1024**3),
        "summary_bytes": summary,
        "summary_gb": {key: value / (1024**3) for key, value in summary.items()},
    }
    (out_dir / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out_dir / f"{args.label}_tensors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
