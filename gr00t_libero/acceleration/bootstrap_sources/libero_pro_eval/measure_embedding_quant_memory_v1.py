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

import torch
import torch.nn.functional as F

PROJ = Path("@WORKSPACE@/libero_pro_eval")
REPO = PROJ / "Isaac-GR00T-N1.7"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(REPO))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


@contextmanager
def patched_env(env: dict[str, str]):
    old = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PackedAffineEmbedding(torch.nn.Module):
    """Group-wise affine quantized embedding with compact uint8/nibble storage."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        nbits: int,
        embedding_dim: int,
        padding_idx: int | None,
    ):
        super().__init__()
        self.nbits = int(nbits)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("zeros", zeros.contiguous())

    @classmethod
    @torch.no_grad()
    def from_embedding(cls, emb: torch.nn.Embedding, nbits: int, group_size: int) -> "PackedAffineEmbedding":
        if nbits not in (4, 8):
            raise ValueError(f"Only 4/8bit embedding quantization is supported, got {nbits}")
        W = emb.weight.detach().to(torch.float32)
        vocab, dim = W.shape
        pad = (group_size - (dim % group_size)) % group_size
        if pad:
            Wp = F.pad(W, (0, pad))
        else:
            Wp = W
        padded_dim = Wp.shape[1]
        groups = padded_dim // group_size
        Wg = Wp.reshape(vocab, groups, group_size)
        minv = Wg.amin(dim=2)
        maxv = Wg.amax(dim=2)
        qmax = (1 << nbits) - 1
        scales = ((maxv - minv) / max(qmax, 1)).clamp_min(1e-8)
        zeros = minv
        q = torch.round((Wg - zeros.unsqueeze(-1)) / scales.unsqueeze(-1)).clamp(0, qmax).to(torch.uint8)
        q = q.reshape(vocab, padded_dim)
        if pad:
            q = q[:, :dim]
        if nbits == 4:
            if dim % 2 != 0:
                q = F.pad(q, (0, 1))
            lo = q[:, 0::2]
            hi = q[:, 1::2]
            q_store = (lo | (hi << 4)).to(torch.uint8)
        else:
            q_store = q.to(torch.uint8)
        return cls(
            qweight=q_store.to(device=emb.weight.device),
            scales=scales.to(device=emb.weight.device, dtype=torch.bfloat16),
            zeros=zeros.to(device=emb.weight.device, dtype=torch.bfloat16),
            nbits=nbits,
            embedding_dim=dim,
            padding_idx=emb.padding_idx,
        )

    def dequantize_weight(self) -> torch.Tensor:
        if self.nbits == 4:
            packed = self.qweight
            lo = packed & 0x0F
            hi = packed >> 4
            q = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.uint8)
            q[:, 0::2] = lo
            q[:, 1::2] = hi
            q = q[:, : self.embedding_dim]
        else:
            q = self.qweight
        vocab, dim = q.shape
        group_size = self.scales.shape[1]
        # Here group_size variable is number of groups; recover true group width.
        num_groups = self.scales.shape[1]
        true_group = (dim + num_groups - 1) // num_groups
        pad = num_groups * true_group - dim
        qf = q.to(torch.float32)
        if pad:
            qf = F.pad(qf, (0, pad))
        qg = qf.reshape(vocab, num_groups, true_group)
        W = qg * self.scales.to(torch.float32).unsqueeze(-1) + self.zeros.to(torch.float32).unsqueeze(-1)
        return W.reshape(vocab, num_groups * true_group)[:, :dim].to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.dequantize_weight(),
            padding_idx=self.padding_idx,
        )


def bytes_params_buffers(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for iterator in (module.parameters(), module.buffers()):
        for tensor in iterator:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return int(total)


def tensor_rows(model: torch.nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for kind, iterator in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in iterator:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "bucket": bucket_name(name),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                    "elem_size": int(tensor.element_size()),
                    "bytes": int(tensor.numel() * tensor.element_size()),
                }
            )
    return rows


def bucket_name(name: str) -> str:
    if "_vlm_to_dit_adapter" in name:
        return "vlm_to_dit_adapter"
    if "language_model.embed_tokens" in name:
        return "vlm_embed_tokens"
    if name.startswith("backbone.model.model.language_model."):
        return "vlm_language_model_other"
    if name.startswith("backbone.model.model.visual.") or ".vision" in name:
        return "vlm_vision_or_projector"
    if name.startswith("backbone."):
        return "vlm_other"
    if name.startswith("action_head.model.") or name.startswith("action_head.vl_self_attention."):
        return "dit_action_head"
    if name.startswith("action_head."):
        return "action_head_other"
    return "other"


def summarize(rows: list[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row["bucket"])
        out[key] = out.get(key, 0) + int(row["bytes"])
    return out


def cuda_snapshot(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def load_setting_env(settings_json: Path, setting_name: str) -> dict[str, str]:
    payload = json.loads(settings_json.read_text())
    setting = next(s for s in payload["settings"] if s["name"] == setting_name)
    return {str(k): str(v) for k, v in setting["env"].items()}


def replace_embedding(model: torch.nn.Module, nbits: int, group_size: int, drop_lm_head: bool) -> None:
    parent = model.get_submodule("backbone.model.model.language_model")
    emb = parent.embed_tokens
    if not isinstance(emb, torch.nn.Embedding):
        raise TypeError(f"embed_tokens is not nn.Embedding: {type(emb)}")
    parent.embed_tokens = PackedAffineEmbedding.from_embedding(emb, nbits=nbits, group_size=group_size)
    if drop_lm_head and hasattr(model.backbone.model, "lm_head"):
        model.backbone.model.lm_head = torch.nn.Identity()
    del emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", default=str(PROJ / "models-gr00t-n17-libero/libero_spatial"))
    parser.add_argument("--settings-json", default=str(PROJ / "quantization_analysis/act_vector_codebook_seed_sweep_2p5_v1/settings.json"))
    parser.add_argument("--setting-name", default="gain23_2.5_top5to4_act_d4_k256_seed20260601")
    parser.add_argument("--adapter-path", default=str(PROJ / "quantization_analysis/vla_aware_stage9_action_weighted_adapter_2026-06-13/adapters/r64_g2_uniform.pt"))
    parser.add_argument("--embedding-bits", type=int, required=True)
    parser.add_argument("--embedding-group-size", type=int, default=64)
    parser.add_argument("--drop-lm-head", action="store_true")
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
    gc.collect()
    torch.cuda.empty_cache()

    rows = tensor_rows(policy.model)
    summary = summarize(rows)
    result = {
        "label": args.label,
        "embedding_bits": args.embedding_bits,
        "embedding_group_size": args.embedding_group_size,
        "drop_lm_head": args.drop_lm_head,
        "quantized_layers": n_quant,
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
