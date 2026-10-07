#!/usr/bin/env python3
"""DiT HQQ vs codebook under the FULL deployed stack (proxy level).

Every DiT method comparison so far isolated the DiT (VLM/vision/embedding at
bf16) so the 0.04-0.11 differences would not sit on the deployed VLM's ~0.15
error floor. This run asks whether the verdict survives composition: same DiT
weights as the isolated runs (identical fitting code and seeds), but applied on
top of the real stack -- VLM 2.0729 packed full-7 codebook, vision HQQ4,
embedding 4-bit, lm_head dropped, single-embodiment, no adapter.

Variants:
    dit_bf16       DiT untouched -> the VLM-floor anchor
    dit_deployed   218 @HQQ3 + adaLN @HQQ2   (deployed DiT 3.0; rollout 96.00%)
    respec_hqq     new 3->2-ranking 2.5-bit spec, 2-bit as HQQ2   (isolated: 0.0393)
    respec_cb      same-budget cb-ranking spec, 2-bit as codebook (isolated: 0.0481)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import zlib
from pathlib import Path

import torch

PROJ = Path("@WORKSPACE@/libero_pro_eval")
REPO = PROJ / "Isaac-GR00T-N1.7"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(REPO))

from analyze_hqq_hadamard_mlp_outliers import quantize_weight_hqq  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from measure_dit_single_layer_downgrade_sensitivity import (  # noqa: E402
    eval_policy, patched_env, replace_embedding, replace_module,
)
from measure_dit_scalar_vs_vector_2026_07_17 import load_batches  # noqa: E402
from measure_embedding_vision_quant_memory_v1 import quantize_visual_linears  # noqa: E402
from measure_gr00t_qk_diag_codebook_2026_06_27 import quantize_vector_codebook  # noqa: E402
from measure_hotspot_protection_proxy import move_to_device, run_outputs  # noqa: E402

POSITIONS = ("final_dit_output", "final_action_pred", "final_decoded_action")
QA = PROJ / "quantization_analysis"
MODULE_CSV = QA / "gr00t_dit_scalar_vs_vector_2026-07-17/dit_modules.csv"
PLANS = QA / "gr00t_dit_2p5_respec_proxy_2026-07-21/plans.json"
CB = dict(group_size=64, vector_dim=4, codebook_size=256, sample_size=200000, iters=12)

VLM_ENV = {
    # full deployed stack for the LLM; DiT deliberately untouched by the shim
    "VLM_BITS": "2", "DIT_BITS": "16",
    "VLM_QUANT_MODE": "uniform", "DIT_QUANT_MODE": "uniform", "VLM_ROTATION_MODE": "none",
    "VLM_FORCE_BITS_SPECS_FILE": os.environ.get("VLM_FORCE_BITS_SPECS_FILE_OVERRIDE")
    or str(QA / "gr00t_full7_6bit_rollout_2026-07-12/env/bits_2p1_vlm_force_bits_specs.txt"),
    "VLM_2BIT_CODEBOOK_PACKED": "1", "VLM_2BIT_CODEBOOK_MODE": "act_vector_kmeans",
    "VLM_2BIT_CODEBOOK_SAMPLE": "200000", "VLM_2BIT_CODEBOOK_ITERS": "12",
    "VLM_2BIT_CODEBOOK_VECTOR_DIM": "4", "VLM_2BIT_CODEBOOK_SIZE": "256",
    "VLM_2BIT_CODEBOOK_SEED": "20260601",
    "VLM_CODEBOOK_OVERLAY_PATH": str(QA / "gr00t_full7_2p1_packed_2026-07-13/full7_2p1_packed_refined.pt"),
    "ACTION_HEAD_SINGLE_EMBODIMENT_ID": "2",
    "DIT_INCLUDE_ADALN": "0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(PROJ / "models-gr00t-n17-libero/libero_spatial"))
    ap.add_argument("--request-dir", default=str(PROJ / "request_replay_runs/calib_4suite_2026-07-02/val/requests"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    roles = {r["module_name"]: r["module_type"] for r in csv.DictReader(open(MODULE_CSV))}
    adaln = [f"action_head.model.transformer_blocks.{i}.norm1.linear" for i in range(32)]
    for n in adaln:
        roles[n] = "adaln"
    targets = sorted(roles)
    plans = json.loads(PLANS.read_text())

    print("[fullstack] bf16 references", flush=True)
    ref = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), args.model_path, device=args.device, strict=True)
    ref.model.eval()
    batches = load_batches(ref, Path(args.request_dir), args.limit)
    references = {}
    for i, (rid, bb, ai) in enumerate(batches):
        references[rid] = run_outputs(ref, move_to_device(bb, device, torch.bfloat16),
                                      move_to_device(ai, device, torch.bfloat16), seed=7000 + i)
    del ref
    gc.collect(); torch.cuda.empty_cache()

    print("[fullstack] loading policy + applying full VLM stack (DiT untouched)", flush=True)
    policy = Gr00tPolicy(EmbodimentTag.resolve("LIBERO_PANDA"), args.model_path, device=args.device, strict=True)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)
    named = dict(policy.model.named_modules())
    snaps = {n: named[n].weight.detach().to(device, torch.float32).clone() for n in targets}
    biases = {n: (None if named[n].bias is None else named[n].bias.detach().clone()) for n in targets}
    with patched_env(VLM_ENV):
        quantize_gr00t_vlm27_dit4(policy.model, group_size=64, device=args.device)
    replace_embedding(policy.model, 4, 64, True)
    quantize_visual_linears(policy.model, nbits=4, group_size=64, device=args.device)
    gc.collect(); torch.cuda.empty_cache()

    def put(name, W):
        lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=biases[name] is not None,
                              device=args.device, dtype=torch.bfloat16)
        lin.weight.data.copy_(W.to(torch.bfloat16))
        if biases[name] is not None:
            lin.bias.data.copy_(biases[name].to(device=args.device, dtype=torch.bfloat16))
        replace_module(policy.model, name, lin)

    def assignment(variant):
        # name -> ("hqq", bits) | ("cb", 2)
        if variant == "dit_bf16":
            return {}
        if variant == "dit_deployed":
            return {n: ("hqq", 2 if roles[n] == "adaln" else 3) for n in targets}
        three = set(plans[variant])
        out = {}
        for n in targets:
            if n in three:
                out[n] = ("hqq", 3)
            elif variant == "respec_hqq":
                out[n] = ("hqq", 2)
            else:
                out[n] = ("cb", 2)
        return out

    rows = []
    for variant in ("dit_bf16", "dit_deployed", "respec_hqq", "respec_cb"):
        print(f"[fullstack] ===== {variant} =====", flush=True)
        plan = assignment(variant)
        for i, n in enumerate(targets, 1):
            if n not in plan:
                put(n, snaps[n])  # restore bf16
                continue
            kind, bits = plan[n]
            if kind == "hqq":
                Wq = quantize_weight_hqq(snaps[n], nbits=bits, group_size=64, device=args.device).to(torch.float32)
            else:
                seed = args.seed + zlib.crc32(n.encode()) % 9973
                Wq = quantize_vector_codebook(snaps[n], **CB, seed=seed, metric="euclidean", scalar_importance=None)
            put(n, Wq)
            if i % 60 == 0:
                print(f"    {i}/{len(targets)}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        m = eval_policy(policy, batches, references, device, POSITIONS, seed_base=7000)
        err = m["final_decoded_action"]["rel_rms_error"]
        print(f"[fullstack] {variant:<13} final_decoded_action = {err:.6f}", flush=True)
        rows.append({"variant": variant, **{p: m[p]["rel_rms_error"] for p in POSITIONS}})

    with (out_dir / "dit_fullstack_proxy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[fullstack] wrote {out_dir}/dit_fullstack_proxy.csv", flush=True)


if __name__ == "__main__":
    main()
