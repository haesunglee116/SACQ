"""Quantization helpers shared with the smoothness study.

Provides post-load weight quantization of nn.Linear children:
- bnb NF4 (4-bit) via bitsandbytes
- HQQLinear at 1/2/3/4/8 bits via the hqq library

Diffusers' `Timesteps`/`TimestepEmbedding` modules have a quirky forward path
that bypasses standard nn.Linear semantics and breaks when their internal
linears are quantized (dtype mismatch in matmul). We skip Linears whose
ancestor module name contains "timestep" / "embed" or whose tensors are too
small to be worth quantizing.
"""

from __future__ import annotations
import math
import os
import re
import types
import torch
import torch.nn.functional as F


SKIP_NAME_PATTERNS = ("timestep", "Embed", "embedder")
MIN_DIM_TO_QUANTIZE = 64  # skip tiny Linears (Timesteps' linear_1/2 are 256→1024)

# Lazy runtime imports keep Triton entirely out of the stock quantization path.
# Cache the callables after the first opt-in invocation; importing them in every
# DiT linear call is measurable because one action invokes thousands of layers.
_TRITON_VECTOR_LINEAR = None
_TRITON_INTERLEAVED_LINEAR = None
_TRITON_INTERLEAVED_SPLITK_LINEAR = None
_TRITON_FUSED_DEQUANT_LINEAR = None
_CUTLASS_INTERLEAVED_LINEAR = None


def _parse_pattern_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]


def _parse_group_force_env(name: str) -> dict[str, list[int]]:
    raw = os.getenv(name, "")
    if not raw:
        return {}
    out: dict[str, list[int]] = {}
    # Inline specs historically use semicolons, while the generated
    # *_FORCE_BITS_SPECS_FILE artifacts contain one assignment per line.
    # Treat both as equivalent delimiters.  Splitting only on semicolons made
    # a non-empty newline-delimited paper spec silently parse as zero entries.
    for entry in [p.strip() for p in re.split(r"[;\r\n]+", raw) if p.strip()]:
        if "@" not in entry:
            continue
        pattern, groups_raw = entry.split("@", 1)
        group_ids = []
        for tok in re.split(r"[|,]", groups_raw):
            tok = tok.strip()
            if not tok:
                continue
            try:
                group_ids.append(int(tok))
            except ValueError:
                continue
        if pattern.strip() and group_ids:
            out[pattern.strip()] = sorted(set(group_ids))
    return out


def _parse_force_bits_env(name: str) -> dict[str, int]:
    raw = os.getenv(name, "")
    file_path = os.getenv(f"{name}_FILE", "").strip()
    if file_path:
        try:
            raw = open(file_path, "r").read().strip()
        except OSError as exc:
            print(f"[quant-compiler] WARNING: failed to read {name}_FILE={file_path}: {exc}", flush=True)
    if not raw:
        return {}
    out: dict[str, int] = {}
    # Inline specs historically use semicolons, while generated allocation
    # files contain one assignment per line.  They are the same logical
    # format, so accept both delimiters.  Otherwise a valid multi-line file
    # is consumed as one malformed entry and silently installs no overrides.
    for entry in [p.strip() for p in re.split(r"[;\r\n]+", raw) if p.strip()]:
        if "=" not in entry:
            continue
        pattern, bits_raw = entry.split("=", 1)
        pattern = pattern.strip()
        bits_raw = bits_raw.strip()
        if not pattern or not bits_raw:
            continue
        try:
            bits = int(bits_raw)
        except ValueError:
            continue
        out[pattern] = bits
    return out


def _parse_float_spec_env(name: str) -> dict[str, float]:
    raw = os.getenv(name, "")
    if not raw:
        return {}
    out: dict[str, float] = {}
    for entry in [p.strip() for p in raw.split(";") if p.strip()]:
        if "=" not in entry:
            continue
        pattern, value_raw = entry.split("=", 1)
        pattern = pattern.strip()
        value_raw = value_raw.strip()
        if not pattern or not value_raw:
            continue
        try:
            out[pattern] = float(value_raw)
        except ValueError:
            continue
    return out


def _parse_codebook_transform_env(name: str) -> dict[str, tuple[str, float]]:
    raw = os.getenv(name, "")
    if not raw:
        return {}
    out: dict[str, tuple[str, float]] = {}
    for entry in [p.strip() for p in raw.split(";") if p.strip()]:
        if "=" not in entry or ":" not in entry:
            continue
        pattern, spec = entry.split("=", 1)
        mode, value_raw = spec.split(":", 1)
        pattern = pattern.strip()
        mode = mode.strip().lower()
        value_raw = value_raw.strip()
        if not pattern or mode not in ("offset", "spread"):
            continue
        try:
            out[pattern] = (mode, float(value_raw))
        except ValueError:
            continue
    return out


def _should_skip(name: str, child: torch.nn.Linear) -> bool:
    if any(p in name for p in SKIP_NAME_PATTERNS):
        return True
    if min(child.in_features, child.out_features) < MIN_DIM_TO_QUANTIZE:
        return True
    return False


def _matched_float_spec(name: str, specs: dict[str, float], default: float = 1.0) -> float:
    value = default
    for pattern, candidate in specs.items():
        if pattern in name:
            value = candidate
    return value


def _matched_codebook_transform(
    name: str,
    specs: dict[str, tuple[str, float]],
) -> tuple[str, float] | None:
    value = None
    for pattern, candidate in specs.items():
        if pattern in name:
            value = candidate
    return value


def _normalize_post_calib_layer_map(
    raw_layers: dict[str, dict[str, torch.Tensor | list[float]]],
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    calib_map: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, layer_payload in raw_layers.items():
        if not isinstance(layer_payload, dict):
            continue
        scale = layer_payload.get("scale")
        if scale is None:
            continue
        scale_tensor = torch.as_tensor(scale, dtype=torch.bfloat16, device=device)
        entry = {"scale": scale_tensor}
        bias = layer_payload.get("bias")
        if bias is not None:
            entry["bias"] = torch.as_tensor(bias, dtype=torch.bfloat16, device=device)
        calib_map[layer_name] = entry
    return calib_map


def _load_post_calibration_payload(path: str, device: str) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    if not path:
        return {"linear_layers": {}, "mlp_product_layers": {}, "block_delta_layers": {}}
    if not os.path.exists(path):
        print(f"[quant-compiler] WARNING: VLM post-calibration file not found: {path}", flush=True)
        return {"linear_layers": {}, "mlp_product_layers": {}, "block_delta_layers": {}}
    try:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to load VLM post-calibration file {path}: {exc}", flush=True)
        return {"linear_layers": {}, "mlp_product_layers": {}, "block_delta_layers": {}}
    if not isinstance(payload, dict):
        print(f"[quant-compiler] WARNING: invalid VLM post-calibration payload type: {type(payload)}", flush=True)
        return {"linear_layers": {}, "mlp_product_layers": {}, "block_delta_layers": {}}

    raw_linear_layers = payload.get("linear_layers")
    if raw_linear_layers is None:
        raw_linear_layers = payload.get("layers", {})
    if not isinstance(raw_linear_layers, dict):
        raw_linear_layers = {}

    raw_mlp_product_layers = payload.get("mlp_product_layers", {})
    if not isinstance(raw_mlp_product_layers, dict):
        raw_mlp_product_layers = {}
    raw_block_delta_layers = payload.get("block_delta_layers", {})
    if not isinstance(raw_block_delta_layers, dict):
        raw_block_delta_layers = {}

    linear_layers = _normalize_post_calib_layer_map(raw_linear_layers, device)
    mlp_product_layers = _normalize_post_calib_layer_map(raw_mlp_product_layers, device)
    block_delta_layers = _normalize_post_calib_layer_map(raw_block_delta_layers, device)
    print(
        f"[quant-compiler] loaded post-calibration from {path}: "
        f"{len(linear_layers)} linear layers, "
        f"{len(mlp_product_layers)} mlp product layers, "
        f"{len(block_delta_layers)} block delta layers",
        flush=True,
    )
    return {
        "linear_layers": linear_layers,
        "mlp_product_layers": mlp_product_layers,
        "block_delta_layers": block_delta_layers,
    }


class VLMToDiTLowRankAdapter(torch.nn.Module):
    def __init__(self, down: torch.Tensor, up: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("down", down.to(torch.float32).contiguous())
        self.register_buffer("up", up.to(torch.float32).contiguous())
        self.register_buffer("bias", bias.to(torch.float32).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y + (y @ self.down) @ self.up + self.bias
        return y.to(dtype=x.dtype)


def attach_vlm_to_dit_adapter(model: torch.nn.Module, path: str, device: str = "cuda:0") -> bool:
    if not path:
        return False
    if not os.path.exists(path):
        print(f"[quant-compiler] WARNING: VLM_TO_DIT_ADAPTER_PATH not found: {path}", flush=True)
        return False
    if not hasattr(model, "action_head"):
        print("[quant-compiler] WARNING: model has no action_head; cannot attach VLM-to-DiT adapter", flush=True)
        return False
    try:
        payload = torch.load(path, map_location=device)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        down = state["down"]
        up = state["up"]
        bias = state["bias"]
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to load VLM-to-DiT adapter {path}: {exc}", flush=True)
        return False

    adapter = VLMToDiTLowRankAdapter(down, up, bias).to(device).eval()
    for param in adapter.parameters():
        param.requires_grad_(False)

    action_head = model.action_head
    if getattr(action_head, "_vlm_to_dit_adapter_attached", False):
        print("[quant-compiler] VLM-to-DiT adapter already attached; skipping duplicate attach", flush=True)
        return True

    orig_encode = action_head._encode_features

    def _encode_features_with_adapter(self, backbone_output, action_input):
        features = orig_encode(backbone_output, action_input)
        corrected = adapter(features.backbone_features)
        features["backbone_features"] = corrected
        return features

    action_head._encode_features = types.MethodType(_encode_features_with_adapter, action_head)
    action_head._vlm_to_dit_adapter_attached = True
    action_head._vlm_to_dit_adapter_path = path
    action_head._vlm_to_dit_adapter = adapter
    print(
        f"[quant-compiler] attached VLM-to-DiT low-rank adapter from {path} "
        f"(down={tuple(down.shape)}, up={tuple(up.shape)})",
        flush=True,
    )
    return True


class FixedQKDiagLinear(torch.nn.Module):
    """Dense inference wrapper for precomputed q/k qk_diag codebook weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        self.register_buffer("weight", weight.detach().to(torch.bfloat16).contiguous())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(torch.bfloat16).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), None if self.bias is None else self.bias.to(dtype=x.dtype))


class PrecomputedPackedVectorCodebookLinear(torch.nn.Module):
    """Packed d4/k256 codebook reconstructed from a precomputed overlay payload."""

    def __init__(self, payload: dict, bias: torch.Tensor | None, device: str):
        super().__init__()
        required = {"codebook", "scales", "in_features", "out_features", "group_size", "vector_dim"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Packed codebook payload is missing: {sorted(missing)}")
        if "indices" not in payload and "packed_indices" not in payload:
            raise ValueError("Packed codebook payload requires indices or packed_indices")
        device_obj = torch.device(device)
        self.in_features = int(payload["in_features"])
        self.out_features = int(payload["out_features"])
        self.group_size = int(payload["group_size"])
        self.vector_dim = int(payload["vector_dim"])
        self.register_buffer("codebook", torch.as_tensor(payload["codebook"], device=device_obj, dtype=torch.bfloat16))
        if "packed_indices" in payload:
            self.indices = None
            self.index_bits = int(payload["index_bits"])
            self.index_shape = tuple(int(x) for x in payload["index_shape"])
            packed_source = torch.as_tensor(payload["packed_indices"], dtype=torch.uint8)
            if packed_source.ndim != 1:
                raise ValueError(
                    f"packed_indices must be flat, got shape={tuple(packed_source.shape)}"
                )
            if self.index_bits < 1 or self.index_bits > 56:
                raise ValueError(f"index_bits must be in [1, 56], got {self.index_bits}")
            expected_bytes = (math.prod(self.index_shape) * self.index_bits + 7) // 8
            if packed_source.numel() != expected_bytes:
                raise ValueError(
                    f"packed_indices has {packed_source.numel()} bytes, "
                    f"expected {expected_bytes} for shape={self.index_shape}, "
                    f"index_bits={self.index_bits}"
                )
            # An index can start at bit offset 7. Arbitrary widths may therefore
            # span more than two bytes (11-bit indices need three). Keep only
            # the minimum zero guard needed for aligned window reads.
            max_offset = 8 - math.gcd(self.index_bits, 8)
            self.index_window_bytes = (self.index_bits + max_offset + 7) // 8
            guard_bytes = self.index_window_bytes - 1
            packed_with_guard = torch.empty(
                packed_source.numel() + guard_bytes, device=device_obj, dtype=torch.uint8
            )
            packed_with_guard[: packed_source.numel()].copy_(packed_source)
            if guard_bytes:
                packed_with_guard[packed_source.numel() :].zero_()
            self.register_buffer("packed_indices", packed_with_guard)
        else:
            self.index_bits = 8
            self.index_window_bytes = 1
            self.index_shape = tuple(int(x) for x in torch.as_tensor(payload["indices"]).shape)
            # dtype 강제 캐스팅 금지: k<=256이면 uint8, k>256(예: 512/1024)이면 int32로
            # 저장돼 있다. uint8로 강제하면 256 이상 값이 잘려나간다(2026-08-20 발견,
            # 실롤아웃 0/10 붕괴로 재현됨). 원본 dtype 그대로 유지.
            self.register_buffer("indices", torch.as_tensor(payload["indices"], device=device_obj))
            self.packed_indices = None
        self.register_buffer("scales", torch.as_tensor(payload["scales"], device=device_obj, dtype=torch.bfloat16))
        self.register_buffer("shift", torch.as_tensor(payload.get("shift", 0.0), device=device_obj, dtype=torch.bfloat16))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(device=device_obj, dtype=torch.bfloat16).contiguous())
        # Experimental opt-in only.  The default deployment path stays
        # byte-for-byte unchanged while the direct packed GEMM is validated.
        self.runtime_backend = os.environ.get("GR00T_CODEBOOK_LINEAR_BACKEND", "dense_reconstruct")
        # Overlay application fills this with either ``vlm`` or ``dit``.  Keep
        # unknown modules conservative: the VLM's longer-token projections are
        # more sensitive to the different BF16 accumulation order of the
        # direct packed kernel than the DiT path was in deterministic replay.
        self.runtime_scope = "unknown"
        self.runtime_accelerate_vlm = os.environ.get(
            "GR00T_CODEBOOK_ACCELERATE_VLM", "0"
        ) == "1"
        expected_index_shape = None
        if (
            self.vector_dim > 0
            and self.group_size % self.vector_dim == 0
            and self.in_features % self.group_size == 0
        ):
            expected_index_shape = (
                self.out_features,
                self.in_features // self.group_size,
                self.group_size // self.vector_dim,
            )
        # The current Triton decoder uses int32 windows. Unsupported future
        # artifact layouts must fall back to exact dense reconstruction instead
        # of silently truncating a wider packed index.
        self.runtime_kernel_eligible = (
            self.vector_dim == 4
            and self.index_bits <= 24
            and self.index_window_bytes <= 4
            and self.index_shape == expected_index_shape
            and self.shift.numel() == 1
        )

    def dequantize_weight(self) -> torch.Tensor:
        num_groups = self.scales.shape[1]
        if self.packed_indices is None:
            indices = self.indices.long()
        else:
            count = math.prod(self.index_shape)
            # Reuse intermediates in-place and release them as soon as their
            # last consumer has been launched.  A large VLM projection has
            # millions of vector indices, so keeping every int64 temporary
            # alive until function return materially increases CUDA peak.
            max_bit_position = count * self.index_bits
            position_dtype = (
                torch.int32
                if max_bit_position <= torch.iinfo(torch.int32).max
                else torch.long
            )
            positions = torch.arange(
                count, device=self.packed_indices.device, dtype=position_dtype
            )
            positions.mul_(self.index_bits)
            byte_ids = torch.div(positions, 8, rounding_mode="floor")
            positions.remainder_(8)  # positions now stores only bit offsets.
            word_dtype = (
                torch.int32 if self.index_bits + 7 <= 31 else torch.long
            )
            words = self.packed_indices[byte_ids].to(word_dtype)
            for byte_offset in range(1, self.index_window_bytes):
                upper = self.packed_indices[byte_ids + byte_offset].to(word_dtype)
                upper.bitwise_left_shift_(8 * byte_offset)
                words.bitwise_or_(upper)
                del upper
            del byte_ids
            words.bitwise_right_shift_(positions)
            words.bitwise_and_((1 << self.index_bits) - 1)
            del positions
            indices = words.reshape(self.index_shape)
        q_vectors = self.codebook[indices]
        if self.packed_indices is not None:
            del indices, words
        # q_vectors is a fresh gather result, so scaling and shifting it in
        # place cannot mutate any registered quantization state.
        W = q_vectors.reshape(self.out_features, num_groups, self.group_size)
        W.mul_(self.scales.unsqueeze(-1))
        W = W.reshape(self.out_features, self.in_features)
        W.add_(self.shift)
        return W.to(dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _TRITON_VECTOR_LINEAR, _TRITON_INTERLEAVED_LINEAR
        global _TRITON_INTERLEAVED_SPLITK_LINEAR
        global _TRITON_FUSED_DEQUANT_LINEAR
        global _CUTLASS_INTERLEAVED_LINEAR
        if self.runtime_backend in (
            "triton_interleaved",
            "triton_interleaved_splitk",
            "triton_interleaved_fused_dequant",
            "cutlass_interleaved_fused_dequant",
        ):
            runtime_max_m = 192 if (
                self.runtime_scope == "dit" or self.runtime_accelerate_vlm
            ) else 64
        else:
            runtime_max_m = 64
        runtime_m = x.numel() // x.shape[-1]
        if (
            self.runtime_backend in (
                "triton_interleaved_fused_dequant",
                "cutlass_interleaved_fused_dequant",
            )
            and self.packed_indices is not None
            and self.runtime_kernel_eligible
            and x.dtype == torch.bfloat16
            and runtime_m > runtime_max_m
        ):
            if _TRITON_FUSED_DEQUANT_LINEAR is None:
                from codebook_linear_triton_2026_08_23 import (
                    packed_fused_dequant_codebook_linear,
                )

                _TRITON_FUSED_DEQUANT_LINEAR = packed_fused_dequant_codebook_linear
            return _TRITON_FUSED_DEQUANT_LINEAR(x, self)
        if (
            self.runtime_backend in (
                "triton_vector",
                "triton_interleaved",
                "triton_interleaved_splitk",
                "triton_interleaved_fused_dequant",
                "cutlass_interleaved_fused_dequant",
            )
            and self.packed_indices is not None
            and self.runtime_kernel_eligible
            and x.dtype == torch.bfloat16
            and runtime_m <= runtime_max_m
        ):
            if (
                self.runtime_backend == "cutlass_interleaved_fused_dequant"
                and runtime_m > 16
            ):
                if _CUTLASS_INTERLEAVED_LINEAR is None:
                    from cuda_codebook_cutlass_2026_08_23 import (
                        packed_cutlass_codebook_linear,
                    )

                    _CUTLASS_INTERLEAVED_LINEAR = packed_cutlass_codebook_linear
                return _CUTLASS_INTERLEAVED_LINEAR(x, self)
            if self.runtime_backend == "triton_interleaved_splitk":
                if _TRITON_INTERLEAVED_SPLITK_LINEAR is None:
                    from codebook_linear_triton_2026_08_23 import (
                        packed_interleaved_splitk_dispatch,
                    )

                    _TRITON_INTERLEAVED_SPLITK_LINEAR = packed_interleaved_splitk_dispatch
                return _TRITON_INTERLEAVED_SPLITK_LINEAR(x, self)
            if self.runtime_backend in (
                "triton_interleaved",
                "triton_interleaved_fused_dequant",
                "cutlass_interleaved_fused_dequant",
            ):
                if _TRITON_INTERLEAVED_LINEAR is None:
                    from codebook_linear_triton_2026_08_23 import (
                        packed_interleaved_codebook_linear,
                    )

                    _TRITON_INTERLEAVED_LINEAR = packed_interleaved_codebook_linear
                return _TRITON_INTERLEAVED_LINEAR(x, self)
            if _TRITON_VECTOR_LINEAR is None:
                from codebook_linear_triton_2026_08_23 import packed_vector_codebook_linear

                _TRITON_VECTOR_LINEAR = packed_vector_codebook_linear
            return _TRITON_VECTOR_LINEAR(x, self)
        weight = self.dequantize_weight().to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


def apply_qkdiag_overlay_from_env(
    model: torch.nn.Module,
    assignments: dict[str, int],
    device: str = "cuda:0",
) -> int:
    path = os.environ.get("VLM_QKDIAG_OVERLAY_PATH", "").strip()
    if not path:
        return 0
    if not os.path.exists(path):
        print(f"[quant-compiler] WARNING: VLM_QKDIAG_OVERLAY_PATH not found: {path}", flush=True)
        return 0
    try:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to load qk_diag overlay {path}: {exc}", flush=True)
        return 0
    weights = payload.get("weights", payload) if isinstance(payload, dict) else {}
    if not isinstance(weights, dict):
        print(f"[quant-compiler] WARNING: invalid qk_diag overlay payload in {path}", flush=True)
        return 0

    named_modules = dict(model.named_modules())
    applied = 0
    for full, weight in weights.items():
        if assignments.get(full) != 2:
            continue
        if not (full.endswith(".q_proj") or full.endswith(".k_proj")):
            continue
        if "." not in full:
            continue
        parent_name, attr = full.rsplit(".", 1)
        parent = named_modules.get(parent_name)
        if parent is None or not hasattr(parent, attr):
            print(f"[quant-compiler] WARNING: qk_diag target not found: {full}", flush=True)
            continue
        old_layer = getattr(parent, attr)
        bias = getattr(old_layer, "bias", None)
        setattr(parent, attr, FixedQKDiagLinear(torch.as_tensor(weight, device=device), bias=bias))
        applied += 1
        print(f"[quant-compiler] applied qk_diag overlay to {full}", flush=True)

    print(f"[quant-compiler] applied qk_diag overlay from {path}: {applied} q/k layers", flush=True)
    return applied


def apply_codebook_overlay_from_env(
    model: torch.nn.Module,
    assignments: dict[str, int],
    device: str = "cuda:0",
    *,
    packed_only: bool = False,
    skip_packed: bool = False,
) -> int:
    """Apply dense or packed precomputed weights to selected 2-bit projections."""
    path = os.environ.get("VLM_CODEBOOK_OVERLAY_PATH", "").strip()
    if not path:
        return 0
    if not os.path.exists(path):
        raise FileNotFoundError(f"VLM_CODEBOOK_OVERLAY_PATH not found: {path}")
    payload = torch.load(path, map_location="cpu")
    weights = payload.get("weights", payload) if isinstance(payload, dict) else {}
    if not isinstance(weights, dict):
        raise ValueError(f"Invalid codebook overlay payload: {path}")

    allowed_suffixes = (
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".o_proj",
        ".gate_proj",
        ".up_proj",
        ".down_proj",
    )
    named_modules = dict(model.named_modules())
    selective_vlm_pattern_raw = os.environ.get(
        "GR00T_CODEBOOK_ACCELERATE_VLM_PATTERN", ""
    ).strip()
    selective_vlm_pattern = (
        re.compile(selective_vlm_pattern_raw) if selective_vlm_pattern_raw else None
    )
    applied = 0
    skipped_non2 = 0
    packed_applied = 0
    selectively_accelerated = 0
    for full, weight in weights.items():
        if not full.endswith(allowed_suffixes):
            continue
        if assignments.get(full) != 2:
            skipped_non2 += 1
            continue
        parent_name, attr = full.rsplit(".", 1)
        parent = named_modules.get(parent_name)
        if parent is None or not hasattr(parent, attr):
            raise KeyError(f"Codebook overlay target not found: {full}")
        old_layer = getattr(parent, attr)
        bias = getattr(old_layer, "bias", None)
        is_packed = isinstance(weight, dict) and weight.get("format") == "packed_vector_codebook_v1"
        if packed_only and not is_packed:
            continue
        if skip_packed and is_packed:
            continue
        if is_packed:
            replacement = PrecomputedPackedVectorCodebookLinear(weight, bias=bias, device=device)
            replacement.runtime_scope = "vlm"
            replacement.runtime_name = full
            if selective_vlm_pattern is not None and selective_vlm_pattern.search(full):
                replacement.runtime_accelerate_vlm = True
                selectively_accelerated += 1
            packed_applied += 1
        else:
            replacement = FixedQKDiagLinear(torch.as_tensor(weight, device=device), bias=bias)
        setattr(parent, attr, replacement)
        applied += 1
        print(f"[quant-compiler] applied codebook overlay to {full}", flush=True)

    expected = payload.get("expected_applied") if isinstance(payload, dict) else None
    expected_packed = (
        payload.get("expected_packed_applied") if isinstance(payload, dict) else None
    )
    if expected_packed is None and expected is not None and all(
        isinstance(weight, dict) and weight.get("format") == "packed_vector_codebook_v1"
        for weight in weights.values()
    ):
        # Backward-compatible strictness for all-packed overlays.  Previously
        # packed_only=True disabled the only expected-count check, so a forced
        # bit spec could silently skip a listed packed layer.
        expected_packed = expected
    if packed_only and expected_packed is not None and applied != int(expected_packed):
        raise RuntimeError(
            f"Packed VLM codebook overlay applied {applied}, expected {expected_packed}: {path}"
        )
    if expected is not None and not (packed_only or skip_packed) and applied != int(expected):
        raise RuntimeError(f"Codebook overlay applied {applied} layers, expected {expected}: {path}")
    print(
        f"[quant-compiler] applied codebook overlay from {path}: "
        f"applied={applied} packed={packed_applied} skipped_non2={skipped_non2} "
        f"selective_vlm_accel={selectively_accelerated}",
        flush=True,
    )
    return applied


def apply_dit_codebook_overlay_from_env(model: torch.nn.Module, device: str = "cuda:0") -> int:
    """Overwrite selected DiT linears with precomputed dense weights.

    Payload: {"weights": {full_name: bf16 weight tensor}, "expected_applied": N}.
    Every listed module must resolve and the final count must match
    expected_applied -- a mismatch is fatal rather than a warning (the adaLN
    found!=replaced incident is why this check is strict)."""
    path = os.environ.get("DIT_CODEBOOK_OVERLAY_PATH", "").strip()
    if not path:
        return 0
    if not os.path.exists(path):
        raise FileNotFoundError(f"DIT_CODEBOOK_OVERLAY_PATH not found: {path}")
    payload = torch.load(path, map_location="cpu")
    weights = payload.get("weights", payload) if isinstance(payload, dict) else {}
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Invalid DiT codebook overlay payload: {path}")
    named_modules = dict(model.named_modules())
    applied = 0
    for full, weight in weights.items():
        if "action_head." not in full:
            raise KeyError(f"DiT overlay entry outside action_head: {full}")
        parent_name, attr = full.rsplit(".", 1)
        parent = named_modules.get(parent_name)
        if parent is None or not hasattr(parent, attr):
            raise KeyError(f"DiT overlay target not found: {full}")
        old_layer = getattr(parent, attr)
        bias = getattr(old_layer, "bias", None)
        if isinstance(weight, dict) and weight.get("format") == "packed_vector_codebook_v1":
            replacement = PrecomputedPackedVectorCodebookLinear(weight, bias=bias, device=device)
            replacement.runtime_scope = "dit"
        else:
            w = torch.as_tensor(weight).to(device=device, dtype=torch.bfloat16)
            replacement = torch.nn.Linear(
                w.shape[1], w.shape[0], bias=bias is not None, device=device, dtype=torch.bfloat16
            )
            replacement.weight.data.copy_(w)
            if bias is not None:
                replacement.bias.data.copy_(bias.detach().to(device=device, dtype=torch.bfloat16))
        setattr(parent, attr, replacement)
        applied += 1
    expected = payload.get("expected_applied") if isinstance(payload, dict) else None
    if expected is not None and applied != int(expected):
        raise RuntimeError(f"DiT overlay applied {applied}, expected {expected}: {path}")
    print(
        f"[quant-compiler] applied DiT codebook overlay from {path}: applied={applied}",
        flush=True,
    )
    return applied


class SingleEmbodimentCategoryLinear(torch.nn.Module):
    """Materialized CategorySpecificLinear for one embodiment.

    GR00T's action head stores several matrices as [num_embodiments, in, out].
    LIBERO inference uses a single embodiment id, so keeping only that slice is
    functionally equivalent for LIBERO and avoids loading the unused categories
    in steady-state memory.
    """

    def __init__(self, source: torch.nn.Module, embodiment_id: int):
        super().__init__()
        self.W = torch.nn.Parameter(source.W.detach()[embodiment_id].clone(), requires_grad=False)
        self.b = torch.nn.Parameter(source.b.detach()[embodiment_id].clone(), requires_grad=False)
        self.num_categories = 1
        self.materialized_embodiment_id = int(embodiment_id)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        del cat_ids
        return torch.matmul(x, self.W) + self.b


def materialize_single_embodiment_action_head(
    model: torch.nn.Module,
    embodiment_id: int,
    device: str = "cuda:0",
) -> int:
    if not hasattr(model, "action_head"):
        print("[quant-compiler] WARNING: model has no action_head; cannot materialize single embodiment", flush=True)
        return 0
    try:
        from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to import CategorySpecificLinear: {exc}", flush=True)
        return 0

    def _replace(module: torch.nn.Module, path: str = "") -> tuple[int, int, int]:
        n_replaced = 0
        bytes_before = 0
        bytes_after = 0
        for name, child in list(module.named_children()):
            full = f"{path}.{name}" if path else name
            if isinstance(child, CategorySpecificLinear):
                if embodiment_id < 0 or embodiment_id >= int(child.W.shape[0]):
                    raise ValueError(
                        f"embodiment_id={embodiment_id} is out of range for {full} "
                        f"with {int(child.W.shape[0])} categories"
                    )
                bytes_before += child.W.numel() * child.W.element_size()
                bytes_before += child.b.numel() * child.b.element_size()
                single = SingleEmbodimentCategoryLinear(child, embodiment_id).to(
                    device=child.W.device,
                    dtype=child.W.dtype,
                )
                bytes_after += single.W.numel() * single.W.element_size()
                bytes_after += single.b.numel() * single.b.element_size()
                setattr(module, name, single)
                n_replaced += 1
                print(
                    f"[quant-compiler] materialized {full} for embodiment_id={embodiment_id} "
                    f"shape {tuple(child.W.shape)} -> {tuple(single.W.shape)}",
                    flush=True,
                )
            else:
                child_n, child_before, child_after = _replace(child, full)
                n_replaced += child_n
                bytes_before += child_before
                bytes_after += child_after
        return n_replaced, bytes_before, bytes_after

    n_replaced, bytes_before, bytes_after = _replace(model.action_head)
    if n_replaced:
        print(
            f"[quant-compiler] single-embodiment action head materialized: "
            f"embodiment_id={embodiment_id}, modules={n_replaced}, "
            f"bytes {bytes_before} -> {bytes_after} "
            f"(saved {(bytes_before - bytes_after) / (1024**3):.4f} GiB)",
            flush=True,
        )
    else:
        print("[quant-compiler] no CategorySpecificLinear modules found to materialize", flush=True)
    return n_replaced


class PackedAffineEmbedding(torch.nn.Module):
    """Compact group-wise affine quantized embedding for inference."""

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
        weight = emb.weight.detach().to(torch.float32)
        vocab, dim = weight.shape
        pad = (group_size - (dim % group_size)) % group_size
        padded = F.pad(weight, (0, pad)) if pad else weight
        groups = padded.shape[1] // group_size
        grouped = padded.reshape(vocab, groups, group_size)
        minv = grouped.amin(dim=2)
        maxv = grouped.amax(dim=2)
        qmax = (1 << nbits) - 1
        scales = ((maxv - minv) / max(qmax, 1)).clamp_min(1e-8)
        zeros = minv
        q = torch.round((grouped - zeros.unsqueeze(-1)) / scales.unsqueeze(-1)).clamp(0, qmax).to(torch.uint8)
        q = q.reshape(vocab, padded.shape[1])
        if pad:
            q = q[:, :dim]
        if nbits == 4:
            if dim % 2 != 0:
                q = F.pad(q, (0, 1))
            q_store = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
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
        num_groups = self.scales.shape[1]
        true_group = (dim + num_groups - 1) // num_groups
        pad = num_groups * true_group - dim
        qf = q.to(torch.float32)
        if pad:
            qf = F.pad(qf, (0, pad))
        qg = qf.reshape(vocab, num_groups, true_group)
        weight = qg * self.scales.to(torch.float32).unsqueeze(-1) + self.zeros.to(torch.float32).unsqueeze(-1)
        weight = weight.reshape(vocab, num_groups * true_group)[:, :dim]
        return weight.to(torch.bfloat16)

    def dequantize_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Dequantize only rows referenced by this embedding lookup.

        A GR00T request uses hundreds of token ids while the vocabulary has
        151,936 rows. Materializing the full FP32 table creates three 1.24 GB
        intermediates. Direct row gathering preserves the packed storage and
        elementwise affine reconstruction while making temporary memory scale
        with the request rather than vocabulary size.
        """
        if self.nbits == 4:
            packed = self.qweight[input_ids]
            q = torch.empty(
                (*packed.shape[:-1], packed.shape[-1] * 2),
                device=packed.device,
                dtype=torch.uint8,
            )
            q[..., 0::2] = packed & 0x0F
            q[..., 1::2] = packed >> 4
            q = q[..., : self.embedding_dim]
        else:
            q = self.qweight[input_ids]

        dim = q.shape[-1]
        num_groups = self.scales.shape[1]
        true_group = (dim + num_groups - 1) // num_groups
        pad = num_groups * true_group - dim
        qf = q.to(torch.float32)
        if pad:
            qf = F.pad(qf, (0, pad))
        grouped = qf.reshape(*input_ids.shape, num_groups, true_group)
        scales = self.scales[input_ids].to(torch.float32).unsqueeze(-1)
        zeros = self.zeros[input_ids].to(torch.float32).unsqueeze(-1)
        rows = grouped * scales + zeros
        rows = rows.reshape(*input_ids.shape, num_groups * true_group)
        return rows[..., :dim].to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.dequantize_rows(input_ids)


def apply_embedding_quant_from_env(model: torch.nn.Module, group_size: int = 64) -> bool:
    bits_raw = os.environ.get("VLM_EMBED_BITS", "").strip()
    if not bits_raw:
        return False
    bits = int(bits_raw)
    if bits >= 16:
        return False
    parent = model.get_submodule("backbone.model.model.language_model")
    emb = parent.embed_tokens
    if not isinstance(emb, torch.nn.Embedding):
        print(f"[quant-compiler] embed_tokens already materialized as {type(emb).__name__}; skipping", flush=True)
    else:
        before = emb.weight.numel() * emb.weight.element_size()
        parent.embed_tokens = PackedAffineEmbedding.from_embedding(emb, nbits=bits, group_size=group_size)
        after = sum(t.numel() * t.element_size() for t in parent.embed_tokens.buffers())
        print(
            f"[quant-compiler] quantized embed_tokens to {bits}-bit: "
            f"bytes {before} -> {after} (saved {(before - after) / (1024**3):.4f} GiB)",
            flush=True,
        )
        del emb
    drop_raw = os.environ.get("VLM_DROP_LM_HEAD", "").strip().lower()
    if drop_raw in ("1", "true", "yes", "y") and hasattr(model.backbone.model, "lm_head"):
        model.backbone.model.lm_head = torch.nn.Identity()
        print("[quant-compiler] dropped unused lm_head via VLM_DROP_LM_HEAD", flush=True)
    return True


def apply_vision_quant_from_env(model: torch.nn.Module, group_size: int = 64, device: str = "cuda:0") -> int:
    bits_raw = os.environ.get("VLM_VISION_BITS", "").strip()
    if not bits_raw:
        return 0
    bits = int(bits_raw)
    if bits >= 16:
        return 0
    try:
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to import HQQ for vision quantization: {exc}", flush=True)
        return 0

    def _replace(root: torch.nn.Module, full_name: str, new_module: torch.nn.Module) -> None:
        parent = root
        parts = full_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("backbone.model.model.visual.") and isinstance(module, torch.nn.Linear)
    ]
    for name, module in candidates:
        cfg = BaseQuantizeConfig(nbits=bits, group_size=group_size, axis=1)
        hqq = HQQLinear(module, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
        _replace(model, name, hqq)
    print(f"[quant-compiler] quantized visual/projector linears to {bits}-bit: count={len(candidates)}", flush=True)
    return len(candidates)


def _normalize_rotation_entry(
    layer_name: str,
    layer_payload,
    device: str,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(layer_payload, dict):
        return None
    input_rotation = layer_payload.get("input_rotation")
    output_rotation = layer_payload.get("output_rotation")
    head_rotation = layer_payload.get("head_rotation")
    if input_rotation is None and output_rotation is None and head_rotation is None:
        return None
    entry: dict[str, torch.Tensor] = {}
    if input_rotation is not None:
        entry["input_rotation"] = torch.as_tensor(
            input_rotation, dtype=torch.float32, device=device
        ).contiguous()
    if output_rotation is not None:
        entry["output_rotation"] = torch.as_tensor(
            output_rotation, dtype=torch.float32, device=device
        ).contiguous()
    if head_rotation is not None:
        entry["head_rotation"] = torch.as_tensor(
            head_rotation, dtype=torch.float32, device=device
        ).contiguous()
    if "head_rotation_transpose" in layer_payload:
        entry["head_rotation_transpose"] = bool(layer_payload["head_rotation_transpose"])
    if "weight_only" in layer_payload:
        entry["weight_only"] = bool(layer_payload["weight_only"])
    return entry


def _load_rotation_payload(path: str, device: str) -> dict[str, dict[str, torch.Tensor]]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[quant-compiler] WARNING: rotation file not found: {path}", flush=True)
        return {}
    try:
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        print(f"[quant-compiler] WARNING: failed to load rotation file {path}: {exc}", flush=True)
        return {}

    if not isinstance(payload, dict):
        print(f"[quant-compiler] WARNING: invalid rotation payload type: {type(payload)}", flush=True)
        return {}

    raw_layers = payload.get("linear_layers", payload)
    if not isinstance(raw_layers, dict):
        print(f"[quant-compiler] WARNING: invalid rotation payload contents in {path}", flush=True)
        return {}

    rotation_map: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, layer_payload in raw_layers.items():
        entry = _normalize_rotation_entry(layer_name, layer_payload, device)
        if entry is not None:
            rotation_map[layer_name] = entry

    print(
        f"[quant-compiler] loaded explicit rotation payload from {path}: {len(rotation_map)} layers",
        flush=True,
    )
    return rotation_map


def quantize_linears_to_bnb_nf4(module: torch.nn.Module, device: str = "cuda:0", path: str = "") -> int:
    import bitsandbytes as bnb
    n_replaced = 0
    for name, child in list(module.named_children()):
        full = f"{path}.{name}" if path else name
        if isinstance(child, torch.nn.Linear):
            if _should_skip(full, child):
                continue
            has_bias = child.bias is not None
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )
            new_linear.weight = bnb.nn.Params4bit(
                child.weight.data.contiguous().to(torch.bfloat16),
                requires_grad=False,
                quant_type="nf4",
            )
            if has_bias:
                new_linear.bias = torch.nn.Parameter(child.bias.data.to(torch.bfloat16))
            new_linear = new_linear.to(device)
            setattr(module, name, new_linear)
            n_replaced += 1
        else:
            n_replaced += quantize_linears_to_bnb_nf4(child, device, path=full)
    return n_replaced


def quantize_linears_with_hqq(
    module: torch.nn.Module,
    nbits: int = 4,
    group_size: int = 64,
    device: str = "cuda:0",
    path: str = "",
) -> int:
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    quant_cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
    n_replaced = 0
    for name, child in list(module.named_children()):
        full = f"{path}.{name}" if path else name
        if isinstance(child, torch.nn.Linear):
            if _should_skip(full, child):
                continue
            hqq_linear = HQQLinear(
                child,
                quant_config=quant_cfg,
                compute_dtype=torch.bfloat16,
                device=device,
                del_orig=True,
            )
            setattr(module, name, hqq_linear)
            n_replaced += 1
        else:
            n_replaced += quantize_linears_with_hqq(child, nbits, group_size, device, path=full)
    return n_replaced


class CSBQuantWrapper(torch.nn.Module):
    """CSB Quantization Wrapper for DiT layers.
    Applies runtime activation scaling x' = x / S, where S is the 
    pre-calculated SmoothQuant-style channel scale factor.
    """
    def __init__(self, hqq_layer: torch.nn.Module, scale: torch.Tensor):
        super().__init__()
        self.hqq_layer = hqq_layer
        self.register_buffer("scale", scale.to(hqq_layer.device, dtype=torch.bfloat16))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale activations per channel: x' = x / S
        # Since scale is a 1D tensor [in_features], it will automatically broadcast on [B, S, in_features] or [B, in_features]
        return self.hqq_layer(x / self.scale)


class HybridOutlierHQQLinear(torch.nn.Module):
    """HQQ Linear Layer with 0.5% high-precision Outlier Safety Net.
    
    Splits the weight matrix into a low-bit quantized weight (zeroed at outlier columns)
    and a dense high-precision (BF16) outlier weight. During forward pass, merges their
    outputs seamlessly without any runtime decoding overhead.
    """
    def __init__(self, child: torch.nn.Linear, nbits: int, group_size: int, device: str, outlier_fraction: float = 0.005):
        super().__init__()
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        
        W = child.weight.data.to(device=device, dtype=torch.float32)
        in_features = child.in_features
        out_features = child.out_features
        num_outliers = int(round(in_features * outlier_fraction))
        
        if num_outliers > 0:
            # Outlier detection proxy: max weight magnitude per column
            scales = W.abs().max(dim=0)[0]
            outlier_indices = torch.topk(scales, num_outliers).indices
            
            # Split weight into outlier and non-outlier
            W_out_dense = W[:, outlier_indices].clone()
            W_non_out = W.clone()
            W_non_out[:, outlier_indices] = 0.0
            
            # Quantize the non-outlier weight using HQQ
            temp_linear = torch.nn.Linear(in_features, out_features, bias=child.bias is not None, device=device, dtype=torch.bfloat16)
            temp_linear.weight.data.copy_(W_non_out.to(torch.bfloat16))
            if child.bias is not None:
                temp_linear.bias.data.copy_(child.bias.data.to(torch.bfloat16))
                
            cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
            self.hqq_linear = HQQLinear(temp_linear, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
            
            self.register_buffer("outlier_indices", outlier_indices)
            self.W_out_dense = torch.nn.Parameter(W_out_dense.to(torch.bfloat16))
            self.use_outliers = True
        else:
            cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
            self.hqq_linear = HQQLinear(child, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
            self.use_outliers = False
            
        self.bias = self.hqq_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hqq_linear(x)
        if self.use_outliers:
            x_out = x[..., self.outlier_indices]
            out += torch.matmul(x_out, self.W_out_dense.t())
        return out


class ActQuantInputWrapper(torch.nn.Module):
    """Fake-quantize the input activation before a (weight-quantized) linear.

    Per-token dynamic symmetric quantization: each token vector gets its own
    scale from its absmax, is rounded to a (2^n - 1)-level signed grid, and is
    dequantized back to the compute dtype -- the standard simulated WxA{n}
    setting. Everything so far in this project was weight-only (activations at
    bf16); published "4-bit action-head floor" claims often include activation
    quantization, so this exists to make that comparison like-for-like.
    Enabled via ACT_QUANT_BITS; off by default so existing runs are unchanged.
    """

    def __init__(self, inner: torch.nn.Module, nbits: int):
        super().__init__()
        self.inner = inner
        self.nbits = int(nbits)
        self.qmax = 2 ** (self.nbits - 1) - 1

    def forward(self, x: torch.Tensor, *args, **kwargs):
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.qmax
        xq = torch.round(x / scale).clamp(-self.qmax, self.qmax) * scale
        return self.inner(xq.to(x.dtype), *args, **kwargs)


class OutputScaleCalibrationWrapper(torch.nn.Module):
    def __init__(self, core_layer: torch.nn.Module, scale: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.core_layer = core_layer
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core_layer(x)
        out = out * self.scale
        if self.bias is not None:
            out = out + self.bias
        return out


class MLPProductScaleCalibrationWrapper(torch.nn.Module):
    def __init__(self, mlp_module: torch.nn.Module, scale: torch.Tensor):
        super().__init__()
        self.gate_proj = mlp_module.gate_proj
        self.up_proj = mlp_module.up_proj
        self.down_proj = mlp_module.down_proj
        self.act_fn = mlp_module.act_fn
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        product = self.act_fn(gate) * up
        product = product * self.scale
        return self.down_proj(product)


class BlockResidualAffineCalibrationWrapper(torch.nn.Module):
    def __init__(
        self,
        block_module: torch.nn.Module,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.block_module = block_module
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, *args, **kwargs):
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("BlockResidualAffineCalibrationWrapper requires hidden_states input")

        out = self.block_module(*args, **kwargs)
        out_tensor = out[0] if isinstance(out, tuple) else out
        delta = out_tensor - hidden_states
        delta = delta * self.scale
        if self.bias is not None:
            delta = delta + self.bias
        calibrated = hidden_states + delta
        if isinstance(out, tuple):
            return (calibrated, *out[1:])
        return calibrated


class SparseGivensPrunedHQQLinear(torch.nn.Module):
    """
    AutoQVLA 0-Bit Pruning + Sparse-Givens Action Anchoring (SG-AA).
    
    Drops P% of input channels completely (0-bit pruning).
    Applies Givens Rotation to top K% outlier channels paired with low-variance channels.
    Quantizes the remaining dense matrix using HQQ.
    """
    def __init__(self, core_layer, keep_indices, givens_pairs, bias):
        super().__init__()
        self.core_layer = core_layer
        self.register_buffer("keep_indices", torch.tensor(keep_indices, dtype=torch.long, device=core_layer.device))
        
        if len(givens_pairs) > 0:
            idx1, idx2 = zip(*givens_pairs)
            self.register_buffer("g_idx1", torch.tensor(idx1, dtype=torch.long, device=core_layer.device))
            self.register_buffer("g_idx2", torch.tensor(idx2, dtype=torch.long, device=core_layer.device))
        else:
            self.g_idx1 = None
            self.g_idx2 = None
            
        self.bias = bias
        
    def forward(self, x):
        # 1. 0-bit Pruning (Drop channels, output is a new contiguous view/copy due to advanced indexing)
        x_pruned = x[..., self.keep_indices]
        
        # 2. Sparse-Givens Rotation (In-place on pruned tensor)
        if self.g_idx1 is not None:
            x1 = x_pruned[..., self.g_idx1].clone()
            x2 = x_pruned[..., self.g_idx2].clone()
            
            x_pruned[..., self.g_idx1] = (x1 - x2) * 0.70710678
            x_pruned[..., self.g_idx2] = (x1 + x2) * 0.70710678
            
        # 3. Dense HQQ Forward
        out = self.core_layer(x_pruned)
        return out


@torch.no_grad()
def create_sg_pruned_hqq_layer(
    linear_module, nbits, group_size, device, act_scales, prune_fraction, givens_fraction
):
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    
    in_features = linear_module.in_features
    out_features = linear_module.out_features
    device_obj = torch.device(device)
    
    W = linear_module.weight.data.to(device=device_obj, dtype=torch.float32)
    
    # 1. Calculate Importance Scores (Proxy: W_col_norm * Act_scale)
    w_scales = W.abs().max(dim=0)[0]
    if act_scales is not None:
        importance = w_scales * act_scales.to(device=device_obj, dtype=torch.float32)
    else:
        importance = w_scales
        
    # 2. Identify channels
    num_prune = int(in_features * prune_fraction)
    num_keep = in_features - num_prune
    
    # Align keep count to group_size to avoid HQQ padding assertion error
    if group_size > 0:
        num_keep = max(group_size, (num_keep // group_size) * group_size)
    num_prune = in_features - num_keep

    num_givens = int(in_features * givens_fraction)
    
    sorted_indices = torch.argsort(importance)
    
    # Keep the upper portion
    keep_indices = sorted_indices[num_prune:].tolist()
    
    # Outliers are the top K
    outlier_indices = sorted_indices[-num_givens:].tolist() if num_givens > 0 else []
    
    # Lowest variance available (that are kept and not outliers)
    low_var_indices = []
    if num_givens > 0:
        available_low_var = [idx for idx in keep_indices if idx not in outlier_indices]
        low_var_indices = available_low_var[:num_givens]
        
    # 3. Build Givens Pairs (relative to keep_indices)
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    
    givens_pairs = []
    for out_idx, low_idx in zip(outlier_indices, low_var_indices):
        givens_pairs.append((old_to_new_idx[out_idx], old_to_new_idx[low_idx]))
        
    # 4. Shrink weight matrix and apply Givens offline
    W_kept = W[:, keep_indices].clone()
    
    if len(givens_pairs) > 0:
        idx1, idx2 = zip(*givens_pairs)
        w1 = W_kept[:, idx1].clone()
        w2 = W_kept[:, idx2].clone()
        W_kept[:, idx1] = (w1 - w2) * 0.70710678
        W_kept[:, idx2] = (w1 + w2) * 0.70710678
        
    # 5. Quantize the shrunk, rotated weight matrix
    in_features_new = len(keep_indices)
    temp_linear = torch.nn.Linear(in_features_new, out_features, bias=linear_module.bias is not None, device=device_obj, dtype=torch.bfloat16)
    temp_linear.weight.data.copy_(W_kept.to(torch.bfloat16))
    if linear_module.bias is not None:
        temp_linear.bias.data.copy_(linear_module.bias.data.to(torch.bfloat16))
        
    eff_group_size = group_size
    if in_features_new % group_size != 0:
        for candidate in [64, 32, 16, 8]:
            if candidate <= group_size and in_features_new % candidate == 0:
                eff_group_size = candidate
                break
    if nbits == 16:
        return SparseGivensPrunedHQQLinear(temp_linear, keep_indices, givens_pairs, linear_module.bias)
        
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=eff_group_size, axis=1)
    hqq_linear = HQQLinear(temp_linear, quant_config=cfg, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
    
    # 6. Return module
    return SparseGivensPrunedHQQLinear(hqq_linear, keep_indices, givens_pairs, linear_module.bias)

class ChannelWiseMixedHQQLinear(torch.nn.Module):
    def __init__(self, hqq_4bit, hqq_3bit, mask_4bit):
        super().__init__()
        self.hqq_4bit = hqq_4bit
        self.hqq_3bit = hqq_3bit
        self.register_buffer("mask_4bit", mask_4bit)
        self.in_features = len(mask_4bit)
        if hqq_4bit is not None:
            self.out_features = hqq_4bit.out_features
        elif hqq_3bit is not None:
            self.out_features = hqq_3bit.out_features
        else:
            self.out_features = 0
        self.bias = None

    def forward(self, x):
        out = 0
        if self.hqq_4bit is not None:
            x_4bit = x[..., self.mask_4bit]
            out += self.hqq_4bit(x_4bit)
        if self.hqq_3bit is not None:
            x_3bit = x[..., ~self.mask_4bit]
            out += self.hqq_3bit(x_3bit)
        return out


class NonUniform2BitLinear(torch.nn.Module):
    """Module-wise 2-bit codebook fake-quantized Linear.

    This is intentionally a proxy/research layer: it stores the dequantized
    bf16 weight so we can test whether non-uniform 2-bit reconstruction helps
    GR00T proxy metrics before implementing packed kernels.
    """

    def __init__(
        self,
        linear_module: torch.nn.Linear,
        group_size: int,
        device: str,
        mode: str,
        act_scales: torch.Tensor | None = None,
        shift_mode: str = "",
        sample_size: int = 1_000_000,
        iters: int = 25,
        vector_dim: int = 4,
        codebook_size: int = 256,
        seed: int = 20260611,
        codebook_transform: tuple[str, float] | None = None,
    ):
        super().__init__()
        device_obj = torch.device(device)
        W = linear_module.weight.data.to(device=device_obj, dtype=torch.float32)
        codebook, W_q, shift = self._quantize_weight(
            W,
            group_size=group_size,
            mode=mode,
            act_scales=act_scales,
            shift_mode=shift_mode,
            sample_size=sample_size,
            iters=iters,
            vector_dim=vector_dim,
            codebook_size=codebook_size,
            seed=seed,
            codebook_transform=codebook_transform,
        )
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.mode = mode
        self.register_buffer("codebook", codebook.to(dtype=torch.float32))
        self.register_buffer("shift", shift.to(dtype=torch.float32))
        self.weight = torch.nn.Parameter(W_q.to(dtype=torch.bfloat16), requires_grad=False)
        if linear_module.bias is not None:
            self.bias = torch.nn.Parameter(
                linear_module.bias.data.to(device=device_obj, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.bias = None

    @staticmethod
    def _fixed_nf2_codebook(device: torch.device) -> torch.Tensor:
        # Equal-probability normal-bin midpoints, normalized to [-1, 1].
        p = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device, dtype=torch.float32)
        vals = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2 * p - 1)
        vals = vals / vals.abs().max().clamp_min(1e-8)
        return vals

    @staticmethod
    def _sample_values(
        values: torch.Tensor,
        cols: torch.Tensor,
        sample_size: int,
        act_scales: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flat = values.reshape(-1)
        flat_cols = cols.reshape(-1)
        n = flat.numel()
        if n > sample_size:
            gen = torch.Generator(device=values.device)
            gen.manual_seed(seed)
            idx = torch.randperm(n, device=values.device, generator=gen)[:sample_size]
            flat = flat[idx]
            flat_cols = flat_cols[idx]
        weights = None
        if act_scales is not None:
            scales = act_scales.to(device=values.device, dtype=torch.float32).flatten()
            if scales.numel() >= int(flat_cols.max().item()) + 1:
                weights = scales[flat_cols].clamp_min(1e-8)
                weights = weights / weights.mean().clamp_min(1e-8)
        return flat, weights

    @staticmethod
    def _weighted_kmeans_1d(
        samples: torch.Tensor,
        k: int,
        iters: int,
        weights: torch.Tensor | None = None,
        symmetric: bool = False,
    ) -> torch.Tensor:
        if symmetric:
            x = samples.abs().clamp(0, 1)
            w = weights
            centers = torch.quantile(x, torch.tensor([0.35, 0.85], device=x.device, dtype=torch.float32))
            centers = centers.clamp(1e-4, 1.0)
            for _ in range(iters):
                dist = (x[:, None] - centers[None, :]).abs()
                assign = dist.argmin(dim=1)
                new_centers = []
                for j in range(2):
                    mask = assign == j
                    if mask.any():
                        if w is None:
                            new_centers.append(x[mask].mean())
                        else:
                            ww = w[mask]
                            new_centers.append((x[mask] * ww).sum() / ww.sum().clamp_min(1e-8))
                    else:
                        new_centers.append(centers[j])
                centers = torch.stack(new_centers).sort().values.clamp(1e-4, 1.0)
            return torch.stack([-centers[1], -centers[0], centers[0], centers[1]])

        x = samples.clamp(-1, 1)
        qs = torch.tensor([0.125, 0.375, 0.625, 0.875], device=x.device, dtype=torch.float32)
        centers = torch.quantile(x, qs)
        # Avoid duplicate initialization on degenerate tensors.
        centers = centers + torch.linspace(-1e-4, 1e-4, k, device=x.device)
        for _ in range(iters):
            dist = (x[:, None] - centers[None, :]).abs()
            assign = dist.argmin(dim=1)
            new_centers = []
            for j in range(k):
                mask = assign == j
                if mask.any():
                    if weights is None:
                        new_centers.append(x[mask].mean())
                    else:
                        ww = weights[mask]
                        new_centers.append((x[mask] * ww).sum() / ww.sum().clamp_min(1e-8))
                else:
                    new_centers.append(centers[j])
            centers = torch.stack(new_centers).sort().values.clamp(-1.0, 1.0)
        return centers

    @staticmethod
    def _sample_vectors(
        vectors: torch.Tensor,
        vector_cols: torch.Tensor,
        sample_size: int,
        act_scales: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = vectors.shape[0]
        if n > sample_size:
            gen = torch.Generator(device=vectors.device)
            gen.manual_seed(seed)
            idx = torch.randperm(n, device=vectors.device, generator=gen)[:sample_size]
            vectors = vectors[idx]
            vector_cols = vector_cols[idx]
        weights = None
        if act_scales is not None:
            scales = act_scales.to(device=vectors.device, dtype=torch.float32).flatten()
            if scales.numel() >= int(vector_cols.max().item()) + 1:
                weights = scales[vector_cols].mean(dim=1).clamp_min(1e-8)
                weights = weights / weights.mean().clamp_min(1e-8)
        return vectors, weights

    @staticmethod
    def _assign_vectors_chunked(samples: torch.Tensor, centers: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
        # Bound the temporary distance matrix for large codebooks (for example,
        # d8/K65536 at a nominal 2 bits per weight).
        max_distance_elements = 64 * 1024 * 1024
        chunk_size = min(chunk_size, max(1, max_distance_elements // centers.shape[0]))
        assignments = []
        for start in range(0, samples.shape[0], chunk_size):
            chunk = samples[start : start + chunk_size]
            dist = torch.cdist(chunk, centers, p=2)
            assignments.append(dist.argmin(dim=1))
        return torch.cat(assignments, dim=0)

    @classmethod
    def _weighted_kmeans_nd(
        cls,
        samples: torch.Tensor,
        k: int,
        iters: int,
        weights: torch.Tensor | None = None,
        seed: int = 20260611,
    ) -> torch.Tensor:
        n = samples.shape[0]
        if n < k:
            repeat = (k + n - 1) // max(n, 1)
            return samples.repeat(repeat, 1)[:k].contiguous()

        gen = torch.Generator(device=samples.device)
        gen.manual_seed(seed)
        init_idx = torch.randperm(n, device=samples.device, generator=gen)[:k]
        centers = samples[init_idx].contiguous()
        for _ in range(iters):
            assign = cls._assign_vectors_chunked(samples, centers)
            new_centers = torch.zeros_like(centers)
            if weights is None:
                counts = torch.bincount(assign, minlength=k).to(dtype=torch.float32, device=samples.device)
                new_centers.index_add_(0, assign, samples)
                nonempty = counts > 0
                new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None].clamp_min(1e-8)
            else:
                ww = weights.to(dtype=torch.float32)
                counts = torch.bincount(assign, weights=ww, minlength=k).to(dtype=torch.float32, device=samples.device)
                new_centers.index_add_(0, assign, samples * ww[:, None])
                nonempty = counts > 0
                new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None].clamp_min(1e-8)
            new_centers[~nonempty] = centers[~nonempty]
            centers = new_centers.clamp(-1.0, 1.0).contiguous()
        return centers

    @classmethod
    def _additive_vq_quantize(
        cls,
        vectors: torch.Tensor,
        samples: torch.Tensor,
        k: int,
        iters: int,
        weights: torch.Tensor | None,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Two residual codebooks: log2(k) + log2(k) bits per vector.
        codebook1 = cls._weighted_kmeans_nd(samples, k, iters, weights=weights, seed=seed)
        sample_idx1 = cls._assign_vectors_chunked(samples, codebook1)
        residual_samples = (samples - codebook1[sample_idx1]).contiguous()
        codebook2 = cls._weighted_kmeans_nd(residual_samples, k, iters, weights=weights, seed=seed + 1009)

        idx1 = cls._assign_vectors_chunked(vectors, codebook1)
        residual = (vectors - codebook1[idx1]).contiguous()
        idx2 = cls._assign_vectors_chunked(residual, codebook2)
        q_vectors = codebook1[idx1] + codebook2[idx2]
        codebooks = torch.stack([codebook1, codebook2], dim=0)
        return codebooks.detach(), q_vectors.detach().clamp(-1.0, 1.0)

    @classmethod
    def _quantize_weight(
        cls,
        W: torch.Tensor,
        group_size: int,
        mode: str,
        act_scales: torch.Tensor | None,
        shift_mode: str,
        sample_size: int,
        iters: int,
        vector_dim: int,
        codebook_size: int,
        seed: int,
        codebook_transform: tuple[str, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_features, in_features = W.shape
        shift_mode = shift_mode.strip().lower()
        if shift_mode in ("module", "scalar", "mean"):
            shift = W.mean()
            W_work = W - shift
        elif shift_mode in ("", "none", "0"):
            shift = torch.zeros((), device=W.device, dtype=torch.float32)
            W_work = W
        else:
            raise ValueError(f"Unknown VLM_2BIT_CODEBOOK_SHIFT={shift_mode}")

        mode = mode.strip().lower()
        vector_modes = (
            "vector_kmeans",
            "vq",
            "vector_lloyd",
            "act_vector_kmeans",
            "act_vq",
            "additive_vq",
            "residual_vq",
            "act_additive_vq",
            "act_residual_vq",
        )

        eff_group_size = group_size
        pad_cols = 0
        if mode in vector_modes and group_size % vector_dim == 0:
            pad_cols = (-in_features) % group_size
        elif in_features % group_size != 0:
            for candidate in [64, 32, 16, 8, 4, 2, 1]:
                if candidate <= group_size and in_features % candidate == 0:
                    eff_group_size = candidate
                    break
        W_work_grouped = F.pad(W_work, (0, pad_cols)) if pad_cols else W_work
        grouped_in_features = W_work_grouped.shape[1]
        num_groups = grouped_in_features // eff_group_size
        Wg = W_work_grouped.reshape(out_features, num_groups, eff_group_size)
        scales = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
        normalized = (Wg / scales).clamp(-1.0, 1.0)
        col_ids = torch.arange(grouped_in_features, device=W.device).clamp_max(in_features - 1)
        col_ids = col_ids.reshape(1, num_groups, eff_group_size)
        col_ids = col_ids.expand(out_features, -1, -1)

        if mode in vector_modes:
            if eff_group_size % vector_dim != 0:
                raise ValueError(
                    f"Vector codebook requires group size divisible by vector_dim: "
                    f"group_size={eff_group_size}, vector_dim={vector_dim}"
                )
            vectors = normalized.reshape(-1, vector_dim).contiguous()
            vector_cols = col_ids.reshape(-1, vector_dim).contiguous()
            if pad_cols:
                valid_cols = torch.arange(grouped_in_features, device=W.device) < in_features
                valid_cols = valid_cols.reshape(1, num_groups, eff_group_size).expand(out_features, -1, -1)
                valid_vectors = valid_cols.reshape(-1, vector_dim).all(dim=1)
                sample_vectors = vectors[valid_vectors]
                sample_vector_cols = vector_cols[valid_vectors]
            else:
                sample_vectors = vectors
                sample_vector_cols = vector_cols
            samples, weights = cls._sample_vectors(
                sample_vectors,
                sample_vector_cols,
                sample_size=sample_size,
                act_scales=act_scales
                if mode in ("act_vector_kmeans", "act_vq", "act_additive_vq", "act_residual_vq")
                else None,
                seed=seed,
            )
            if mode in ("additive_vq", "residual_vq", "act_additive_vq", "act_residual_vq"):
                codebook, q_vectors = cls._additive_vq_quantize(
                    vectors,
                    samples,
                    codebook_size,
                    iters,
                    weights=weights,
                    seed=seed,
                )
                Wq_norm = q_vectors.reshape_as(normalized)
            else:
                codebook = cls._weighted_kmeans_nd(samples, codebook_size, iters, weights=weights, seed=seed)
                indices = cls._assign_vectors_chunked(vectors, codebook)
                if codebook_transform is not None:
                    mode_name, transform_value = codebook_transform
                    if mode_name == "offset":
                        codebook = (codebook + transform_value).clamp(-1.0, 1.0)
                    elif mode_name == "spread":
                        center = codebook.mean(dim=0, keepdim=True)
                        codebook = (center + transform_value * (codebook - center)).clamp(-1.0, 1.0)
                Wq_norm = codebook[indices].reshape_as(normalized)
            Wq = Wq_norm * scales
            Wq = Wq.reshape(out_features, grouped_in_features)
            if pad_cols:
                Wq = Wq[:, :in_features]
            Wq = Wq.reshape_as(W) + shift
            return codebook.detach(), Wq.detach(), shift.detach()

        if mode == "nf2":
            codebook = cls._fixed_nf2_codebook(W.device)
        else:
            samples, weights = cls._sample_values(
                normalized,
                col_ids,
                sample_size=sample_size,
                act_scales=act_scales if mode in ("act_lloyd", "activation_lloyd") else None,
                seed=seed,
            )
            if mode in ("lloyd_sym", "sym"):
                codebook = cls._weighted_kmeans_1d(samples, 4, iters, weights=None, symmetric=True)
            elif mode in ("lloyd_asym", "lloyd", "asym"):
                codebook = cls._weighted_kmeans_1d(samples, 4, iters, weights=None, symmetric=False)
            elif mode in ("act_lloyd", "activation_lloyd"):
                codebook = cls._weighted_kmeans_1d(samples, 4, iters, weights=weights, symmetric=False)
            else:
                raise ValueError(f"Unknown VLM_2BIT_CODEBOOK_MODE={mode}")

        if codebook_transform is not None:
            mode_name, transform_value = codebook_transform
            if mode_name == "offset":
                codebook = (codebook + transform_value).clamp(-1.0, 1.0)
            elif mode_name == "spread":
                center = codebook.mean()
                codebook = (center + transform_value * (codebook - center)).clamp(-1.0, 1.0)

        dist = (normalized[..., None] - codebook.view(1, 1, 1, 4)).abs()
        indices = dist.argmin(dim=-1)
        Wq = codebook[indices] * scales.squeeze(2).unsqueeze(-1)
        Wq = Wq.reshape_as(W) + shift
        return codebook.detach(), Wq.detach(), shift.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class PackedVectorCodebook2BitLinear(torch.nn.Module):
    """Vector-codebook Linear that stores codebook indices instead of W_q.

    This is the first "real memory" version for the d4/k256 codebook path:
    one uint8 index represents a vector of 4 normalized weights, so the index
    stream costs 2 bits per weight before scale/codebook metadata. The forward
    path reconstructs a temporary dense weight and then calls F.linear; that is
    intentionally slower than a fused kernel, but it lets us verify GPU memory
    accounting without changing the learned/proxy quantization math.
    """

    def __init__(
        self,
        linear_module: torch.nn.Linear,
        group_size: int,
        device: str,
        mode: str,
        act_scales: torch.Tensor | None = None,
        shift_mode: str = "",
        sample_size: int = 1_000_000,
        iters: int = 25,
        vector_dim: int = 4,
        codebook_size: int = 256,
        seed: int = 20260611,
        codebook_transform: tuple[str, float] | None = None,
    ):
        super().__init__()
        mode = mode.strip().lower()
        if mode not in ("vector_kmeans", "vq", "vector_lloyd", "act_vector_kmeans", "act_vq"):
            raise ValueError(f"PackedVectorCodebook2BitLinear supports vector kmeans modes only, got {mode}")
        if vector_dim <= 0:
            raise ValueError(f"vector_dim must be positive, got {vector_dim}")
        if codebook_size > 256:
            raise ValueError("packed vector path stores one uint8 index per vector; codebook_size must be <= 256")

        device_obj = torch.device(device)
        W = linear_module.weight.data.to(device=device_obj, dtype=torch.float32)
        (
            codebook,
            indices,
            scales,
            shift,
            out_features,
            in_features,
            eff_group_size,
        ) = self._quantize_weight_to_indices(
            W,
            group_size=group_size,
            mode=mode,
            act_scales=act_scales,
            shift_mode=shift_mode,
            sample_size=sample_size,
            iters=iters,
            vector_dim=vector_dim,
            codebook_size=codebook_size,
            seed=seed,
            codebook_transform=codebook_transform,
        )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(eff_group_size)
        self.vector_dim = int(vector_dim)
        self.mode = mode
        self.register_buffer("codebook", codebook.to(dtype=torch.bfloat16))
        self.register_buffer("indices", indices.to(dtype=torch.uint8))
        self.register_buffer("scales", scales.to(dtype=torch.bfloat16))
        self.register_buffer("shift", shift.to(dtype=torch.bfloat16))
        if linear_module.bias is not None:
            self.bias = torch.nn.Parameter(
                linear_module.bias.data.to(device=device_obj, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.bias = None

    @classmethod
    def _quantize_weight_to_indices(
        cls,
        W: torch.Tensor,
        group_size: int,
        mode: str,
        act_scales: torch.Tensor | None,
        shift_mode: str,
        sample_size: int,
        iters: int,
        vector_dim: int,
        codebook_size: int,
        seed: int,
        codebook_transform: tuple[str, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        out_features, in_features = W.shape
        shift_mode = shift_mode.strip().lower()
        if shift_mode in ("module", "scalar", "mean"):
            shift = W.mean()
            W_work = W - shift
        elif shift_mode in ("", "none", "0"):
            shift = torch.zeros((), device=W.device, dtype=torch.float32)
            W_work = W
        else:
            raise ValueError(f"Unknown VLM_2BIT_CODEBOOK_SHIFT={shift_mode}")

        eff_group_size = group_size
        if in_features % group_size != 0:
            for candidate in [64, 32, 16, 8, 4, 2, 1]:
                if candidate <= group_size and in_features % candidate == 0:
                    eff_group_size = candidate
                    break
        if eff_group_size % vector_dim != 0:
            raise ValueError(
                f"Packed vector codebook requires group size divisible by vector_dim: "
                f"group_size={eff_group_size}, vector_dim={vector_dim}"
            )

        num_groups = in_features // eff_group_size
        Wg = W_work.reshape(out_features, num_groups, eff_group_size)
        scales = Wg.abs().amax(dim=2).clamp_min(1e-8)
        normalized = (Wg / scales.unsqueeze(-1)).clamp(-1.0, 1.0)
        col_ids = torch.arange(in_features, device=W.device).reshape(1, num_groups, eff_group_size)
        col_ids = col_ids.expand(out_features, -1, -1)

        vectors = normalized.reshape(-1, vector_dim).contiguous()
        vector_cols = col_ids.reshape(-1, vector_dim).contiguous()
        samples, weights = NonUniform2BitLinear._sample_vectors(
            vectors,
            vector_cols,
            sample_size=sample_size,
            act_scales=act_scales if mode in ("act_vector_kmeans", "act_vq") else None,
            seed=seed,
        )
        codebook = NonUniform2BitLinear._weighted_kmeans_nd(
            samples,
            codebook_size,
            iters,
            weights=weights,
            seed=seed,
        )
        if codebook_transform is not None:
            mode_name, transform_value = codebook_transform
            if mode_name == "offset":
                codebook = (codebook + transform_value).clamp(-1.0, 1.0)
            elif mode_name == "spread":
                center = codebook.mean(dim=0, keepdim=True)
                codebook = (center + transform_value * (codebook - center)).clamp(-1.0, 1.0)

        indices = NonUniform2BitLinear._assign_vectors_chunked(vectors, codebook)
        if int(indices.max().item()) >= 256:
            raise RuntimeError("packed vector codebook produced an index outside uint8 range")

        num_vectors_per_group = eff_group_size // vector_dim
        indices = indices.reshape(out_features, num_groups, num_vectors_per_group).contiguous()
        return (
            codebook.detach(),
            indices.detach(),
            scales.detach(),
            shift.detach(),
            out_features,
            in_features,
            eff_group_size,
        )

    def apply_weight_scale_(self, scale_multiplier: float) -> None:
        self.scales.mul_(float(scale_multiplier))
        self.shift.mul_(float(scale_multiplier))
        if self.bias is not None:
            # Existing fake-quant scale tuning only scaled the weight. Keep bias
            # unchanged for semantic compatibility with earlier experiments.
            pass

    def dequantize_weight(self) -> torch.Tensor:
        out_features = self.out_features
        num_groups = self.scales.shape[1]
        num_vectors_per_group = self.group_size // self.vector_dim
        q_vectors = self.codebook[self.indices.long()]
        W_norm = q_vectors.reshape(out_features, num_groups, num_vectors_per_group * self.vector_dim)
        W = W_norm * self.scales.unsqueeze(-1)
        return (W.reshape(out_features, self.in_features) + self.shift).to(dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dequantize_weight(), self.bias)


@torch.no_grad()
def optimize_blockwise_reconstruction(
    linear_module,
    nbits,
    group_size,
    device,
    act_scales,
    outlier_fraction,
    apply_wht=True,
    forced_4bit_groups=None,
    rotation_entry=None,
):
    """Grid-searches the optimal AWQ scaling alpha to minimize activation reconstruction MSE."""
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    from gr00t.eval.rotated_linear import MatrixRotatedHQQLinear, RotatedHQQLinear
    from gr00t.eval.matmul_had import has_fast_hadamard, matmul_hadU, matmul_hadUt
    
    in_features = linear_module.in_features
    out_features = linear_module.out_features
    device_obj = torch.device(device)
    use_explicit_rotation = rotation_entry is not None
    input_rotation = None
    output_rotation = None
    head_rotation = None
    head_rotation_transpose = False
    weight_only_rotation = False
    if use_explicit_rotation:
        input_rotation = rotation_entry.get("input_rotation")
        output_rotation = rotation_entry.get("output_rotation")
        head_rotation = rotation_entry.get("head_rotation")
        head_rotation_transpose = bool(rotation_entry.get("head_rotation_transpose", False))
        weight_only_rotation = bool(rotation_entry.get("weight_only", False))
        if input_rotation is not None:
            input_rotation = input_rotation.to(device=device_obj, dtype=torch.float32)
        if output_rotation is not None:
            output_rotation = output_rotation.to(device=device_obj, dtype=torch.float32)
        if head_rotation is not None:
            head_rotation = head_rotation.to(device=device_obj, dtype=torch.float32)
        apply_wht = False

    if apply_wht and not has_fast_hadamard():
        print(
            "[quant-compiler] fast_hadamard_transform unavailable; disabling WHT rotation for this layer",
            flush=True,
        )
        apply_wht = False
    
    # 1. Select Outliers
    W = linear_module.weight.data.to(device=device_obj, dtype=torch.float32)
    outlier_indices = None
    if outlier_fraction > 0:
        num_outliers = int(round(in_features * outlier_fraction))
        if num_outliers > 0:
            scales = act_scales.to(device=device_obj, dtype=torch.float32) if act_scales is not None \
                     else W.abs().max(dim=0)[0]
            outlier_indices = torch.topk(scales, num_outliers).indices.tolist()

    W_out_dense = None
    if outlier_indices is not None and len(outlier_indices) > 0:
        W_out_dense = W[:, outlier_indices].clone()
        W_non_out = W.clone()
        W_non_out[:, outlier_indices] = 0
    else:
        W_non_out = W

    # 2. Walsh-Hadamard Transform applied to weights offline
    Wr = W_non_out.clone()
    use_hadamard = False
    if use_explicit_rotation:
        if input_rotation is not None:
            Wr = torch.matmul(Wr, input_rotation)
        if output_rotation is not None:
            Wr = torch.matmul(output_rotation.t(), Wr)
        if head_rotation is not None:
            had_dim = head_rotation.shape[0]
            if head_rotation_transpose:
                init_shape = Wr.shape
                temp = Wr.reshape(-1, init_shape[-1] // had_dim, had_dim)
                temp = torch.matmul(temp, head_rotation)
                Wr = temp.reshape(init_shape)
            else:
                Wr_t = Wr.t()
                transposed_shape = Wr_t.shape
                temp = Wr_t.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                temp = torch.matmul(temp, head_rotation)
                Wr = temp.reshape(transposed_shape).t()
    elif apply_wht:
        try:
            # Rotate input dimension (left)
            Wr_T_rot1 = matmul_hadUt(Wr.t())
            # Rotate output dimension (right)
            Wr = matmul_hadUt(Wr_T_rot1.t())
            use_hadamard = True
            del Wr_T_rot1
        except Exception as e:
            print(f"[quant-compiler] WHT rotation bypassed due to shape: {e}", flush=True)
            Wr = W_non_out.clone()
            use_hadamard = False

    # 3. AWQ equivalent scaling grid-search
    best_scales_factor = None
    # Skip AWQ scaling if we applied WHT, because WHT already balances the variance
    # and applying unrotated act_scales to rotated weights is mathematically invalid.
    if act_scales is not None and not use_hadamard:
        # B=32 samples
        B = 32
        raw_x = torch.randn(B, in_features, device=device_obj, dtype=torch.float32)
        # Prevent zero-activation channels from creating infinite weight scaling
        scales = act_scales.to(device=device_obj, dtype=torch.float32).clamp(min=1e-5)
        X = raw_x * scales.view(1, -1)

        # Ground truth output
        Y_gt = torch.matmul(X, W.t())

        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_alpha = 0.5
        best_mse = 1e9

        for alpha in alphas:
            # Apply scaling
            weight_scales = Wr.abs().max(dim=0)[0].clamp(min=1e-5)
            scales_factor = (scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
            
            # FIX: Prevent outlier columns from getting massive scale factors
            if outlier_indices is not None and len(outlier_indices) > 0:
                scales_factor[outlier_indices] = 1.0
            
            # Equivalent weight scaling
            Wr_scaled = Wr / scales_factor.view(1, -1)
            
            eff_group_size = group_size
            if in_features % group_size != 0:
                for candidate in [32, 16, 8, 4, 2, 1]:
                    if candidate <= group_size and in_features % candidate == 0:
                        eff_group_size = candidate
                        break

            # Handle Channel-wise Mixed Precision (Float nbits)
            if isinstance(nbits, float):
                num_groups = in_features // eff_group_size
                num_4bit_groups = int(round(num_groups * (nbits - 3.0)))
                num_4bit_groups = max(0, min(num_groups, num_4bit_groups))
                
                group_importance = scales.view(num_groups, eff_group_size).sum(dim=1)
                topk_indices = torch.topk(group_importance, num_4bit_groups).indices.tolist()
                
                mask_4bit = torch.zeros(in_features, dtype=torch.bool, device=device_obj)
                for g_idx in topk_indices:
                    mask_4bit[g_idx * eff_group_size : (g_idx + 1) * eff_group_size] = True
                    
                in_4bit = mask_4bit.sum().item()
                in_3bit = in_features - in_4bit
                
                hqq_4bit, hqq_3bit = None, None
                
                if in_4bit > 0:
                    Wr_4bit = Wr_scaled[:, mask_4bit]
                    temp_4bit = torch.nn.Linear(in_4bit, out_features, bias=False, device=device_obj, dtype=torch.bfloat16)
                    temp_4bit.weight.data.copy_(Wr_4bit.to(torch.bfloat16))
                    cfg_4 = BaseQuantizeConfig(nbits=4, group_size=eff_group_size, axis=1)
                    hqq_4bit = HQQLinear(temp_4bit, quant_config=cfg_4, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
                    
                if in_3bit > 0:
                    Wr_3bit = Wr_scaled[:, ~mask_4bit]
                    temp_3bit = torch.nn.Linear(in_3bit, out_features, bias=False, device=device_obj, dtype=torch.bfloat16)
                    temp_3bit.weight.data.copy_(Wr_3bit.to(torch.bfloat16))
                    cfg_3 = BaseQuantizeConfig(nbits=3, group_size=eff_group_size, axis=1)
                    hqq_3bit = HQQLinear(temp_3bit, quant_config=cfg_3, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
                
                # Reconstruct for MSE evaluation
                Wr_hat = torch.zeros_like(Wr_scaled, dtype=torch.float32)
                if hqq_4bit is not None:
                    Wr_hat[:, mask_4bit] = hqq_4bit.dequantize().to(torch.float32)
                if hqq_3bit is not None:
                    Wr_hat[:, ~mask_4bit] = hqq_3bit.dequantize().to(torch.float32)
            else:
                temp_linear = torch.nn.Linear(in_features, out_features, bias=False, device=device_obj, dtype=torch.bfloat16)
                temp_linear.weight.data.copy_(Wr_scaled.to(torch.bfloat16))
                cfg = BaseQuantizeConfig(nbits=nbits, group_size=eff_group_size, axis=1)
                try:
                    hqq = HQQLinear(temp_linear, quant_config=cfg, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
                    Wr_hat = hqq.dequantize().to(torch.float32)
                except Exception as e:
                    print(f"HQQ failed: {e}")
                    continue

            try:
                # Scale back
                Wr_hat = Wr_hat * scales_factor.view(1, -1)
                
                # Apply inverse WHT to reconstruct W_hat
                if use_hadamard:
                    W_hat_T_rot1 = matmul_hadU(Wr_hat.t())
                    W_hat = matmul_hadU(W_hat_T_rot1.t())
                elif use_explicit_rotation:
                    W_hat = Wr_hat.clone()
                    if head_rotation is not None:
                        had_dim = head_rotation.shape[0]
                        if head_rotation_transpose:
                            init_shape = W_hat.shape
                            temp = W_hat.reshape(-1, init_shape[-1] // had_dim, had_dim)
                            temp = torch.matmul(temp, head_rotation.t())
                            W_hat = temp.reshape(init_shape)
                        else:
                            W_hat_t = W_hat.t()
                            transposed_shape = W_hat_t.shape
                            temp = W_hat_t.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                            temp = torch.matmul(temp, head_rotation.t())
                            W_hat = temp.reshape(transposed_shape).t()
                    if output_rotation is not None:
                        W_hat = torch.matmul(output_rotation, W_hat)
                    if input_rotation is not None:
                        W_hat = torch.matmul(W_hat, input_rotation.t())
                else:
                    W_hat = Wr_hat.clone()
                
                # Apply outlier preservation
                if outlier_indices is not None and W_out_dense is not None:
                    W_hat[:, outlier_indices] = W_out_dense
                
                Y_hat = torch.matmul(X, W_hat.t())
                mse = torch.mean((Y_gt - Y_hat) ** 2).item()
            except Exception:
                mse = 1e9

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
            
            del temp_linear
            torch.cuda.empty_cache()
            
        print(f"  [layer-opt] best AWQ alpha: {best_alpha:.2f} (mse: {best_mse:.6f})", flush=True)
        # Apply final scaling with best alpha
        weight_scales = Wr.abs().max(dim=0)[0].clamp(min=1e-5)
        best_scales_factor = (scales.pow(best_alpha) / weight_scales.pow(1 - best_alpha)).clamp(min=1e-5)
        
        # FIX: Prevent outlier columns from getting massive scale factors
        if outlier_indices is not None and len(outlier_indices) > 0:
            best_scales_factor[outlier_indices] = 1.0
            
        Wr = Wr / best_scales_factor.view(1, -1)

    # 4. Final HQQ layer creation
    eff_group_size = group_size
    if in_features % group_size != 0:
        for candidate in [32, 16, 8, 4, 2, 1]:
            if candidate <= group_size and in_features % candidate == 0:
                eff_group_size = candidate
                break

    use_forced_groups = forced_4bit_groups is not None and len(forced_4bit_groups) > 0
    if isinstance(nbits, float) or use_forced_groups:
        num_groups = in_features // eff_group_size
        if use_forced_groups:
            mask_4bit = torch.zeros(in_features, dtype=torch.bool, device=device_obj)
            for g_idx in forced_4bit_groups:
                if 0 <= g_idx < num_groups:
                    mask_4bit[g_idx * eff_group_size : (g_idx + 1) * eff_group_size] = True
        else:
            num_4bit_groups = int(round(num_groups * (nbits - 3.0)))
            num_4bit_groups = max(0, min(num_groups, num_4bit_groups))
            
            if act_scales is not None:
                group_importance = act_scales.to(device=device_obj, dtype=torch.float32).view(num_groups, eff_group_size).sum(dim=1)
            else:
                group_importance = torch.ones(num_groups, device=device_obj)
                
            topk_indices = torch.topk(group_importance, num_4bit_groups).indices.tolist()
            
            mask_4bit = torch.zeros(in_features, dtype=torch.bool, device=device_obj)
            for g_idx in topk_indices:
                mask_4bit[g_idx * eff_group_size : (g_idx + 1) * eff_group_size] = True
            
        in_4bit = mask_4bit.sum().item()
        in_3bit = in_features - in_4bit
        
        hqq_4bit, hqq_3bit = None, None
        
        if in_4bit > 0:
            Wr_4bit = Wr[:, mask_4bit]
            temp_4bit = torch.nn.Linear(in_4bit, out_features, bias=False, device=device_obj, dtype=torch.bfloat16)
            temp_4bit.weight.data.copy_(Wr_4bit.to(torch.bfloat16))
            cfg_4 = BaseQuantizeConfig(nbits=4, group_size=eff_group_size, axis=1)
            hqq_4bit = HQQLinear(temp_4bit, quant_config=cfg_4, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
            
        if in_3bit > 0:
            Wr_3bit = Wr[:, ~mask_4bit]
            temp_3bit = torch.nn.Linear(in_3bit, out_features, bias=False, device=device_obj, dtype=torch.bfloat16)
            temp_3bit.weight.data.copy_(Wr_3bit.to(torch.bfloat16))
            cfg_3 = BaseQuantizeConfig(nbits=3, group_size=eff_group_size, axis=1)
            hqq_3bit = HQQLinear(temp_3bit, quant_config=cfg_3, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
            
        hqq_layer = ChannelWiseMixedHQQLinear(hqq_4bit, hqq_3bit, mask_4bit)
    else:
        temp_linear = torch.nn.Linear(in_features, out_features, bias=linear_module.bias is not None, device=device_obj, dtype=torch.bfloat16)
        temp_linear.weight.data.copy_(Wr.to(torch.bfloat16))
        if linear_module.bias is not None:
            bias_data = linear_module.bias.data.to(device=device_obj, dtype=torch.float32)
            if use_explicit_rotation:
                if weight_only_rotation and output_rotation is not None:
                    bias_data = torch.matmul(output_rotation.t(), bias_data)
                elif not weight_only_rotation and head_rotation is not None and not head_rotation_transpose:
                    had_dim = head_rotation.shape[0]
                    temp = bias_data.reshape(-1, had_dim)
                    temp = torch.matmul(temp, head_rotation)
                    bias_data = temp.reshape(-1)
            temp_linear.bias.data.copy_(bias_data.to(torch.bfloat16))
            
        cfg = BaseQuantizeConfig(nbits=nbits, group_size=eff_group_size, axis=1)
        hqq_layer = HQQLinear(temp_linear, quant_config=cfg, compute_dtype=torch.bfloat16, device=device_obj, del_orig=True)
    
    # Scale back output if scaling was applied
    if best_scales_factor is not None:
        # Since W_dequant has scales folded, fold back Best Scales Factor in-place
        if isinstance(nbits, float) or use_forced_groups:
            # Complex to fold back dynamically sliced layers, but best_scales_factor is usually None when WHT is used (SpQR/Channel-mixed main path)
            pass
        else:
            orig_dequantize = hqq_layer.dequantize
            hqq_layer.dequantize = lambda orig=orig_dequantize, s=best_scales_factor.to(torch.bfloat16): orig() * s.view(1, -1)
        
    # Apply WHT wrapper if rotated
    if use_hadamard:
        final_layer = RotatedHQQLinear(hqq_layer)
    elif use_explicit_rotation:
        if weight_only_rotation:
            final_layer = hqq_layer
        else:
            final_layer = MatrixRotatedHQQLinear(
                hqq_layer,
                input_rotation=input_rotation,
                output_rotation=output_rotation,
            )
    else:
        final_layer = hqq_layer
        
    # Re-apply Outlier preservation in forward path
    if outlier_indices is not None and W_out_dense is not None:
        # Define hybrid layer forward with outliers
        class RotatedHybridOutlierHQQLinear(torch.nn.Module):
            def __init__(self, core_layer, outlier_indices, W_out_dense, bias):
                super().__init__()
                import os
                self.core_layer = core_layer
                self.register_buffer("outlier_indices", torch.tensor(outlier_indices, device=device_obj))
                
                # Check for 8-bit simulated outlier quantization
                if os.getenv("VLM_OUTLIER_BITS", "16") == "8":
                    scale = W_out_dense.abs().max(dim=0, keepdim=True)[0] / 127.0
                    scale = torch.clamp(scale, min=1e-8)
                    quant = torch.clamp(torch.round(W_out_dense / scale), -128, 127)
                    W_out_dense = (quant * scale).to(torch.bfloat16)
                    
                self.W_out_dense = torch.nn.Parameter(W_out_dense.to(torch.bfloat16))
                self.bias = bias
                
            def forward(self, x):
                out = self.core_layer(x)
                x_out = x[..., self.outlier_indices]
                out += torch.matmul(x_out, self.W_out_dense.t())
                return out
                
        final_layer = RotatedHybridOutlierHQQLinear(final_layer, outlier_indices, W_out_dense, linear_module.bias)
        
    return final_layer


def quantize_gr00t_vlm27_dit4(model: torch.nn.Module, group_size: int = 64, device: str = "cuda:0") -> int:
    """mixed-precision compiler: VLM LLM and DiT quantization with dynamic bits from env vars."""
    import csv
    import os
    
    vlm_bits_env = os.environ.get("VLM_BITS", "2.7")
    dit_bits_env = os.environ.get("DIT_BITS", "4")
    dit_quant_mode_env = os.environ.get("DIT_QUANT_MODE", "uniform")
    post_calib_payload = _load_post_calibration_payload(os.environ.get("VLM_POST_CALIB_PATH", ""), device)
    linear_post_calib_map = post_calib_payload["linear_layers"]
    mlp_product_calib_map = post_calib_payload["mlp_product_layers"]
    block_delta_calib_map = post_calib_payload["block_delta_layers"]
    print(f"[quant-compiler] Dynamic sweep config: VLM_BITS={vlm_bits_env}, DIT_BITS={dit_bits_env}, DIT_QUANT_MODE={dit_quant_mode_env}", flush=True)
    
    # 1. Traverse model to find target VLM LLM layers
    vlm_llm_layers = {}
    def find_llm_layers(module, path=""):
        for name, child in module.named_children():
            full = f"{path}.{name}" if path else name
            if isinstance(child, torch.nn.Linear):
                if _should_skip(full, child):
                    continue
                if "language_model.layers." in full:
                    vlm_llm_layers[full] = child
            else:
                find_llm_layers(child, full)
                
    find_llm_layers(model)
    print(f"[quant-compiler] found {len(vlm_llm_layers)} target VLM LLM layers", flush=True)
    
    # 1.5 Traverse model to find target DiT layers
    dit_layers = {}
    skip_vl_self_attention = os.getenv("DIT_SKIP_VL_SELF_ATTENTION", "0") == "1"
    dit_protect_patterns = tuple(_parse_pattern_env("DIT_PROTECT_PATTERNS"))
    # The `.norm` guard below excludes any Linear whose path contains ".norm".
    # Since plain LayerNorm/RMSNorm are not nn.Linear, the only thing it actually
    # excludes is AdaLayerNorm.linear -- the timestep-conditioning modulation
    # producing (scale, shift). That is 32 x [3072, 1536] = 288 MB left at bf16,
    # i.e. ~21% of the deployed model. Opt in to quantize it too. Default off so
    # existing runs are bit-identical.
    include_adaln = os.getenv("DIT_INCLUDE_ADALN", "0") == "1"

    def _is_protected_dit_layer(full: str) -> bool:
        return any(pattern in full for pattern in dit_protect_patterns)

    def find_dit_layers(module, path=""):
        for name, child in module.named_children():
            full = f"{path}.{name}" if path else name
            if isinstance(child, torch.nn.Linear):
                if _should_skip(full, child):
                    continue
                is_dit_model = "action_head.model" in full
                is_vl_self_attention = "action_head.vl_self_attention" in full
                if skip_vl_self_attention and is_vl_self_attention:
                    continue
                if _is_protected_dit_layer(full):
                    continue
                if (is_dit_model or is_vl_self_attention) and (include_adaln or ".norm" not in full):
                    dit_layers[full] = child
            else:
                find_dit_layers(child, full)
                
    find_dit_layers(model)
    suffix = " (skipping action_head.vl_self_attention)" if skip_vl_self_attention else ""
    if dit_protect_patterns:
        suffix += f" (protecting patterns: {', '.join(dit_protect_patterns)})"
    print(f"[quant-compiler] found {len(dit_layers)} target DiT layers{suffix}", flush=True)
    
    # 2. Read sensitivity CSV
    csv_path = "@WORKSPACE@/libero_pro_eval/quantization_analysis/gr00t_sens/single_layer_sensitivity_gr00t_libero_spatial.csv"
    sens_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer_name = row["name"]
                bits = int(row["bits"])
                err = float(row["act_error"])
                if layer_name not in sens_data:
                    sens_data[layer_name] = {}
                sens_data[layer_name][bits] = err
        print(f"[quant-compiler] loaded sensitivity data for {len(sens_data)} layers from {csv_path}", flush=True)
    else:
        print(f"[quant-compiler] WARNING: sensitivity CSV not found at {csv_path}. Using default sensitivities.", flush=True)
        
    # 3. Load activation scales
    act_scales_path = "@WORKSPACE@/libero_pro_eval/quantization_analysis/act_scales.pt"
    act_scales = {}
    if os.path.exists(act_scales_path):
        try:
            act_scales = torch.load(act_scales_path, map_location=device)
            print(f"[quant-compiler] loaded activation scales from {act_scales_path}", flush=True)
        except Exception as e:
            print(f"[quant-compiler] ERROR loading activation scales: {e}", flush=True)
            
    # 3.5 Load TAC scales for DiT if in tac_awq mode
    tac_scales = {}
    if dit_quant_mode_env in ("tac_awq", "csb"):
        tac_scales_path = "@WORKSPACE@/libero_pro_eval/quantization_analysis/act_scales_tac.pt"
        if os.path.exists(tac_scales_path):
            try:
                tac_scales = torch.load(tac_scales_path, map_location=device)
                print(f"[quant-compiler] loaded TAC scales from {tac_scales_path}", flush=True)
            except Exception as e:
                print(f"[quant-compiler] ERROR loading TAC scales: {e}", flush=True)
            
    # 4. Solve Knapsack/ILP for exactly 2.7-bit average LLM layers
    candidates = []
    total_params = 0
    for name, child in vlm_llm_layers.items():
        params = child.in_features * child.out_features
        total_params += params
        
        # Map PyTorch layer name to CSV layer name
        csv_name = name.replace("backbone.model.language_model.layers", "backbone.eagle_model.language_model.model.layers")
        
        # Get sensitivities
        e2 = sens_data.get(csv_name, {}).get(2, 1.0)
        e3 = sens_data.get(csv_name, {}).get(3, 0.1)
        val = max(0.0, e2 - e3)
        candidates.append({
            "name": name,
            "weight": params,
            "value": val,
            "efficiency": val / max(1e-8, params),
        })
        
    # Sort by efficiency descending
    candidates.sort(key=lambda x: -x["efficiency"])
    
    assignments = {}
    vlm_force_4bit_patterns = tuple(_parse_pattern_env("VLM_FORCE_4BIT_PATTERNS"))
    vlm_force_4bit_group_specs = _parse_group_force_env("VLM_FORCE_4BIT_GROUP_SPECS")
    vlm_force_bits_specs = _parse_force_bits_env("VLM_FORCE_BITS_SPECS")
    dit_force_bits_specs = _parse_force_bits_env("DIT_FORCE_BITS_SPECS")
    for label, specs, expected_env in (
        ("VLM", vlm_force_bits_specs, "GR00T_EXPECTED_VLM_FORCE_SPEC_COUNT"),
        ("DiT", dit_force_bits_specs, "GR00T_EXPECTED_DIT_FORCE_SPEC_COUNT"),
    ):
        expected_raw = os.getenv(expected_env, "").strip()
        if expected_raw:
            expected_count = int(expected_raw)
            if len(specs) != expected_count:
                raise RuntimeError(
                    f"{label} force-spec count mismatch: parsed={len(specs)} "
                    f"expected={expected_count} ({expected_env})"
                )

    def _is_vlm_force_4bit(name: str) -> bool:
        return any(pattern in name for pattern in vlm_force_4bit_patterns)

    def _forced_group_ids_for(name: str) -> list[int]:
        out = []
        for pattern, group_ids in vlm_force_4bit_group_specs.items():
            if pattern in name:
                out.extend(group_ids)
        return sorted(set(out))

    def _forced_bits_for(name: str) -> int | None:
        forced = None
        for pattern, bits in vlm_force_bits_specs.items():
            if pattern in name:
                forced = bits
        return forced

    def _forced_dit_bits_for(name: str) -> int | None:
        forced = None
        for pattern, bits in dit_force_bits_specs.items():
            if pattern in name:
                forced = bits
        return forced
    try:
        v_bits_float = float(vlm_bits_env)
        is_fractional_vlm = not v_bits_float.is_integer()
    except ValueError:
        v_bits_float = 3.0
        is_fractional_vlm = False
        
    if is_fractional_vlm and 2.0 < v_bits_float < 3.0:
        target_ratio_3bit = v_bits_float - 2.0
        capacity = target_ratio_3bit * total_params
        current_weight = 0
        
        for c in candidates:
            if current_weight + c["weight"] <= capacity:
                assignments[c["name"]] = 3
                current_weight += c["weight"]
            else:
                assignments[c["name"]] = 2
                
        # Calculate actual average bit-width
        assigned_sum = sum(assignments.get(name, 2) * (child.in_features * child.out_features) for name, child in vlm_llm_layers.items())
        actual_avg = assigned_sum / total_params
        num_3bit = sum(1 for b in assignments.values() if b == 3)
        num_2bit = len(vlm_llm_layers) - num_3bit
        print(f"[quant-compiler] solved VLM Knapsack: target {v_bits_float}, actual average bit = {actual_avg:.4f}  (3-bit layers: {num_3bit}, 2-bit layers: {num_2bit})", flush=True)
    elif is_fractional_vlm and 3.0 < v_bits_float < 4.0:
        target_ratio_4bit = v_bits_float - 3.0
        capacity = target_ratio_4bit * total_params
        current_weight = 0
        
        # We need sensitivities for 3 vs 4 bit!
        # Re-evaluating candidates for 3->4 bit switch
        candidates_34 = []
        for name, child in vlm_llm_layers.items():
            params = child.in_features * child.out_features
            csv_name = name.replace("backbone.model.language_model.layers", "backbone.eagle_model.language_model.model.layers")
            e3 = sens_data.get(csv_name, {}).get(3, 1.0)
            e4 = sens_data.get(csv_name, {}).get(4, 0.1)
            val = max(0.0, e3 - e4)
            candidates_34.append({
                "name": name,
                "weight": params,
                "value": val,
                "efficiency": val / max(1e-8, params),
            })
        candidates_34.sort(key=lambda x: -x["efficiency"])
        
        for c in candidates_34:
            if current_weight + c["weight"] <= capacity:
                assignments[c["name"]] = 4
                current_weight += c["weight"]
            else:
                assignments[c["name"]] = 3
                
        assigned_sum = sum(assignments.get(name, 3) * (child.in_features * child.out_features) for name, child in vlm_llm_layers.items())
        actual_avg = assigned_sum / total_params
        num_4bit = sum(1 for b in assignments.values() if b == 4)
        num_3bit = len(vlm_llm_layers) - num_4bit
        print(f"[quant-compiler] solved VLM Knapsack: target {v_bits_float}, actual average bit = {actual_avg:.4f}  (4-bit layers: {num_4bit}, 3-bit layers: {num_3bit})", flush=True)
    else:
        v_bits = int(v_bits_float)
        for name in vlm_llm_layers.keys():
            assignments[name] = v_bits
        print(f"[quant-compiler] Uniform VLM LLM at {v_bits}-bit", flush=True)

    if vlm_force_4bit_patterns:
        forced = []
        for name in vlm_llm_layers.keys():
            if _is_vlm_force_4bit(name):
                assignments[name] = 4
                forced.append(name)
        if forced:
            assigned_sum = sum(
                assignments.get(name, 3) * (child.in_features * child.out_features)
                for name, child in vlm_llm_layers.items()
            )
            actual_avg = assigned_sum / total_params
            print(
                f"[quant-compiler] forced {len(forced)} VLM layers to 4-bit via VLM_FORCE_4BIT_PATTERNS; "
                f"actual VLM average bit = {actual_avg:.4f}",
                flush=True,
            )
            for layer_name in forced:
                print(f"[quant-compiler] preserving higher VLM precision for {layer_name}", flush=True)

    if vlm_force_bits_specs:
        forced = []
        for name in vlm_llm_layers.keys():
            forced_bits = _forced_bits_for(name)
            if forced_bits is not None:
                assignments[name] = forced_bits
                forced.append((name, forced_bits))
        if forced:
            assigned_sum = sum(
                assignments.get(name, 3) * (child.in_features * child.out_features)
                for name, child in vlm_llm_layers.items()
            )
            actual_avg = assigned_sum / total_params
            print(
                f"[quant-compiler] forced {len(forced)} VLM layers via VLM_FORCE_BITS_SPECS; "
                f"actual VLM average bit = {actual_avg:.4f}",
                flush=True,
            )
            for layer_name, bits in forced:
                print(f"[quant-compiler] forcing {layer_name} to {bits}-bit", flush=True)
        
    # 4.5 Solve Knapsack/ILP for average DiT layers
    dit_assignments = {}
    try:
        d_bits_float = float(dit_bits_env)
        is_fractional = not d_bits_float.is_integer()
    except ValueError:
        d_bits_float = 4.0
        is_fractional = False

    if is_fractional and 3.0 < d_bits_float < 4.0:
        dit_candidates = []
        dit_total_params = 0
        for name, child in dit_layers.items():
            params = child.in_features * child.out_features
            dit_total_params += params
            
            # Get sensitivities (diff between 3-bit and 4-bit)
            e3 = sens_data.get(name, {}).get(3, 1.0)
            e4 = sens_data.get(name, {}).get(4, 0.1)
            val = max(0.0, e3 - e4)
            dit_candidates.append({
                "name": name,
                "weight": params,
                "value": val,
                "efficiency": val / max(1e-8, params),
            })
            
        dit_candidates.sort(key=lambda x: -x["efficiency"])
        
        # Target capacity for 4-bit: e.g. 3.8 means 80% 4-bit
        target_ratio_4bit = d_bits_float - 3.0
        capacity = target_ratio_4bit * dit_total_params
        current_weight = 0
        
        for c in dit_candidates:
            if current_weight + c["weight"] <= capacity:
                dit_assignments[c["name"]] = 4
                current_weight += c["weight"]
            else:
                dit_assignments[c["name"]] = 3
                
        # Calculate actual average bit-width
        assigned_sum = sum(dit_assignments.get(name, 3) * (child.in_features * child.out_features) for name, child in dit_layers.items())
        actual_avg = assigned_sum / dit_total_params
        num_4bit = sum(1 for b in dit_assignments.values() if b == 4)
        num_3bit = len(dit_layers) - num_4bit
        print(f"[quant-compiler] solved DiT Knapsack: target {d_bits_float}, actual average bit = {actual_avg:.4f}  (4-bit layers: {num_4bit}, 3-bit layers: {num_3bit})", flush=True)
    else:
        d_bits = int(d_bits_float)
        for name in dit_layers.keys():
            dit_assignments[name] = d_bits
        print(f"[quant-compiler] Uniform DiT at {d_bits}-bit", flush=True)

    if dit_force_bits_specs:
        forced = []
        for name in dit_layers.keys():
            forced_bits = _forced_dit_bits_for(name)
            if forced_bits is not None:
                dit_assignments[name] = forced_bits
                forced.append((name, forced_bits))
        if forced:
            dit_total_params_force = sum(
                child.in_features * child.out_features for child in dit_layers.values()
            )
            assigned_sum = sum(
                dit_assignments.get(name, 4) * (child.in_features * child.out_features)
                for name, child in dit_layers.items()
            )
            actual_avg = assigned_sum / dit_total_params_force
            print(
                f"[quant-compiler] forced {len(forced)} DiT layers via DIT_FORCE_BITS_SPECS; "
                f"actual DiT average bit = {actual_avg:.4f}",
                flush=True,
            )
            for layer_name, bits in forced:
                print(f"[quant-compiler] forcing {layer_name} to {bits}-bit", flush=True)
    
    # VLM Quant Mode
    import os
    vlm_quant_mode = os.getenv("VLM_QUANT_MODE", "spqr")
    vlm_rotation_mode = os.getenv("VLM_ROTATION_MODE", "wht").strip().lower()
    vlm_rotation_path = os.getenv("VLM_ROTATION_PATH", "").strip()
    vlm_2bit_codebook_mode = os.getenv("VLM_2BIT_CODEBOOK_MODE", "").strip().lower()
    vlm_2bit_codebook_shift = os.getenv("VLM_2BIT_CODEBOOK_SHIFT", "").strip().lower()
    vlm_2bit_codebook_sample = int(os.getenv("VLM_2BIT_CODEBOOK_SAMPLE", "1000000"))
    vlm_2bit_codebook_iters = int(os.getenv("VLM_2BIT_CODEBOOK_ITERS", "25"))
    vlm_2bit_codebook_vector_dim = int(os.getenv("VLM_2BIT_CODEBOOK_VECTOR_DIM", "4"))
    vlm_2bit_codebook_size = int(os.getenv("VLM_2BIT_CODEBOOK_SIZE", "256"))
    vlm_2bit_codebook_group_size = int(os.getenv("VLM_2BIT_CODEBOOK_GROUP_SIZE", str(group_size)))
    vlm_2bit_codebook_seed = int(os.getenv("VLM_2BIT_CODEBOOK_SEED", "20260611"))
    vlm_2bit_codebook_packed = os.getenv("VLM_2BIT_CODEBOOK_PACKED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    vlm_2bit_codebook_scale_specs = _parse_float_spec_env("VLM_2BIT_CODEBOOK_SCALE_SPECS")
    vlm_2bit_codebook_transform_specs = _parse_codebook_transform_env("VLM_2BIT_CODEBOOK_TRANSFORM_SPECS")
    explicit_rotation_map = (
        _load_rotation_payload(vlm_rotation_path, device)
        if vlm_rotation_mode == "file"
        else {}
    )

    # 5. Recursively apply custom mixed quantization
    def apply_custom_gr00t_quant(module, path=""):
        n_replaced = 0
        for name, child in list(module.named_children()):
            full = f"{path}.{name}" if path else name
            if isinstance(child, torch.nn.Linear):
                if _should_skip(full, child):
                    continue
                if full in assignments:
                    bit = assignments[full]
                    forced_group_ids = _forced_group_ids_for(full)
                    vlm_prune_fraction = os.getenv("VLM_PRUNE_FRACTION")
                    vlm_givens_fraction = os.getenv("VLM_GIVENS_FRACTION")
                    
                    if bit == 16 and vlm_prune_fraction is None and not forced_group_ids:
                        continue
                        
                    scale_key = full
                    if scale_key.startswith("backbone.model."):
                        scale_key = scale_key[len("backbone.model."):]
                    scale_key_candidates = [scale_key]
                    # Qwen3-VL language layers appear as
                    # `backbone.model.model.language_model.layers.*` in GR00T,
                    # while the saved activation-scale table uses
                    # `model.layers.*`.
                    if full.startswith("backbone.model.model.language_model."):
                        scale_key_candidates.append(
                            "model." + full[len("backbone.model.model.language_model.") :]
                        )
                    if scale_key.startswith("model.language_model."):
                        scale_key_candidates.append(
                            "model." + scale_key[len("model.language_model.") :]
                        )

                    layer_scales = None
                    for candidate_scale_key in scale_key_candidates:
                        layer_scales = act_scales.get(candidate_scale_key)
                        if layer_scales is not None:
                            break
                    if layer_scales is not None:
                        layer_scales = layer_scales.to(device)
                        
                    if vlm_quant_mode in ("channel_mixed", "bridge_align"):
                        bit = float(os.getenv("VLM_BITS", "3.2"))
                        outlier_fraction = 0.0
                    else:
                        env_fraction = os.getenv("VLM_OUTLIER_FRACTION")
                        if env_fraction is not None:
                            outlier_fraction = float(env_fraction)
                        else:
                            outlier_fraction = 0.025 if bit >= 3 else 0.015
                        
                    if forced_group_ids:
                        print(
                            f"[quant-compiler] optimizing {full} at base {bit}-bit with forced 4-bit groups {forced_group_ids} "
                            f"(outlier={outlier_fraction}, mode={vlm_quant_mode})",
                            flush=True,
                        )
                    else:
                        print(f"[quant-compiler] optimizing {full} at {bit}-bit (outlier={outlier_fraction}, mode={vlm_quant_mode})", flush=True)
                    
                    if vlm_prune_fraction is not None and vlm_givens_fraction is not None:
                        # Selective Protection Logic
                        is_attention = any(k in full for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
                        
                        if is_attention:
                            layer_prune_fraction = 0.0
                            layer_givens_fraction = 0.0
                            print(f"[quant-compiler] Selective Protection: {full} is ATTENTION layer. Pruning=0.0", flush=True)
                        else:
                            layer_prune_fraction = float(vlm_prune_fraction)
                            layer_givens_fraction = float(vlm_givens_fraction)
                            print(f"[quant-compiler] Selective Pruning: {full} is MLP/Other layer. Pruning={layer_prune_fraction}", flush=True)
                            
                        if layer_prune_fraction == 0.0 and bit == 16:
                            continue
                            
                        optimized_layer = create_sg_pruned_hqq_layer(
                            child, nbits=bit, group_size=group_size, device=device,
                            act_scales=layer_scales, 
                            prune_fraction=layer_prune_fraction, 
                            givens_fraction=layer_givens_fraction
                        )
                        print(f"  -> Applied SparseGivensPrunedHQQLinear (Prune: {layer_prune_fraction}, Givens: {layer_givens_fraction})", flush=True)
                    elif vlm_quant_mode == "uniform":
                        apply_wht = vlm_rotation_mode == "wht"
                        rotation_entry = explicit_rotation_map.get(full) if vlm_rotation_mode == "file" else None
                        if int(bit) == 2 and apply_wht and rotation_entry is None and vlm_2bit_codebook_mode:
                            from gr00t.eval.matmul_had import has_fast_hadamard, matmul_hadUt
                            from gr00t.eval.rotated_linear import RotatedHQQLinear

                            if not has_fast_hadamard():
                                print(
                                    "[quant-compiler] fast_hadamard_transform unavailable; falling back to unrotated NonUniform2BitLinear",
                                    flush=True,
                                )
                                temp_child = child
                                codebook_cls = PackedVectorCodebook2BitLinear if vlm_2bit_codebook_packed else NonUniform2BitLinear
                                optimized_layer = codebook_cls(
                                    temp_child,
                                    group_size=vlm_2bit_codebook_group_size,
                                    device=device,
                                    mode=vlm_2bit_codebook_mode,
                                    act_scales=None,
                                    shift_mode=vlm_2bit_codebook_shift,
                                    sample_size=vlm_2bit_codebook_sample,
                                    iters=vlm_2bit_codebook_iters,
                                    vector_dim=vlm_2bit_codebook_vector_dim,
                                    codebook_size=vlm_2bit_codebook_size,
                                    seed=vlm_2bit_codebook_seed,
                                    codebook_transform=_matched_codebook_transform(
                                        full, vlm_2bit_codebook_transform_specs
                                    ),
                                )
                            else:
                                W = child.weight.data.to(device=device, dtype=torch.float32)
                                Wr_T_rot1 = matmul_hadUt(W.t())
                                Wr = matmul_hadUt(Wr_T_rot1.t())
                                temp_child = torch.nn.Linear(
                                    child.in_features,
                                    child.out_features,
                                    bias=False,
                                    device=device,
                                    dtype=torch.bfloat16,
                                )
                                temp_child.weight.data.copy_(Wr.to(torch.bfloat16))
                                codebook_cls = PackedVectorCodebook2BitLinear if vlm_2bit_codebook_packed else NonUniform2BitLinear
                                core_layer = codebook_cls(
                                    temp_child,
                                    group_size=vlm_2bit_codebook_group_size,
                                    device=device,
                                    mode=vlm_2bit_codebook_mode,
                                    act_scales=None,
                                    shift_mode=vlm_2bit_codebook_shift,
                                    sample_size=vlm_2bit_codebook_sample,
                                    iters=vlm_2bit_codebook_iters,
                                    vector_dim=vlm_2bit_codebook_vector_dim,
                                    codebook_size=vlm_2bit_codebook_size,
                                    seed=vlm_2bit_codebook_seed,
                                    codebook_transform=_matched_codebook_transform(
                                        full, vlm_2bit_codebook_transform_specs
                                    ),
                                )
                                optimized_layer = RotatedHQQLinear(core_layer)
                                if child.bias is not None:
                                    optimized_layer.bias = torch.nn.Parameter(
                                        child.bias.data.to(device=device, dtype=torch.bfloat16),
                                        requires_grad=False,
                                    )
                            codebook_cpu = (
                                optimized_layer.hqq_layer.codebook.detach().cpu()
                                if hasattr(optimized_layer, "hqq_layer")
                                else optimized_layer.codebook.detach().cpu()
                            )
                            codebook_preview = codebook_cpu.reshape(codebook_cpu.shape[0], -1)[:4].tolist()
                            print(
                                f"  -> Applied WHT+{'PackedVectorCodebook2BitLinear' if vlm_2bit_codebook_packed else 'NonUniform2BitLinear'} mode={vlm_2bit_codebook_mode} "
                                f"shift_mode={vlm_2bit_codebook_shift or 'none'} "
                                f"codebook_shape={list(codebook_cpu.shape)} "
                                f"seed={vlm_2bit_codebook_seed} "
                                f"codebook_preview={codebook_preview}",
                                flush=True,
                            )
                        elif int(bit) == 2 and not apply_wht and rotation_entry is None and vlm_2bit_codebook_mode:
                            codebook_cls = PackedVectorCodebook2BitLinear if vlm_2bit_codebook_packed else NonUniform2BitLinear
                            optimized_layer = codebook_cls(
                                child,
                                group_size=vlm_2bit_codebook_group_size,
                                device=device,
                                mode=vlm_2bit_codebook_mode,
                                act_scales=layer_scales,
                                shift_mode=vlm_2bit_codebook_shift,
                                sample_size=vlm_2bit_codebook_sample,
                                iters=vlm_2bit_codebook_iters,
                                vector_dim=vlm_2bit_codebook_vector_dim,
                                codebook_size=vlm_2bit_codebook_size,
                                seed=vlm_2bit_codebook_seed,
                                codebook_transform=_matched_codebook_transform(
                                    full, vlm_2bit_codebook_transform_specs
                                ),
                            )
                            codebook_cpu = optimized_layer.codebook.detach().cpu()
                            codebook_preview = codebook_cpu.reshape(codebook_cpu.shape[0], -1)[:4].tolist()
                            print(
                                f"  -> Applied {'PackedVectorCodebook2BitLinear' if vlm_2bit_codebook_packed else 'NonUniform2BitLinear'} mode={vlm_2bit_codebook_mode} "
                                f"shift_mode={vlm_2bit_codebook_shift or 'none'} "
                                f"shift={float(optimized_layer.shift.detach().cpu()):.8f} "
                                f"codebook_shape={list(codebook_cpu.shape)} "
                                f"seed={vlm_2bit_codebook_seed} "
                                f"codebook_preview={codebook_preview}",
                                flush=True,
                            )
                            codebook_transform = _matched_codebook_transform(
                                full, vlm_2bit_codebook_transform_specs
                            )
                            if codebook_transform is not None:
                                mode_name, transform_value = codebook_transform
                                print(
                                    f"  -> Applied 2bit codebook transform "
                                    f"{mode_name}:{transform_value:.6f} to {full}",
                                    flush=True,
                                )
                            scale_multiplier = _matched_float_spec(
                                full, vlm_2bit_codebook_scale_specs, default=1.0
                            )
                            if scale_multiplier != 1.0:
                                if hasattr(optimized_layer, "apply_weight_scale_"):
                                    optimized_layer.apply_weight_scale_(float(scale_multiplier))
                                else:
                                    optimized_layer.weight.data.mul_(float(scale_multiplier))
                                print(
                                    f"  -> Applied 2bit codebook weight scale multiplier "
                                    f"alpha={scale_multiplier:.6f} to {full}",
                                    flush=True,
                                )
                        elif apply_wht or rotation_entry is not None:
                            optimized_layer = optimize_blockwise_reconstruction(
                                child,
                                nbits=int(bit),
                                group_size=group_size,
                                device=device,
                                act_scales=None,
                                outlier_fraction=0.0,
                                apply_wht=apply_wht,
                                forced_4bit_groups=None,
                                rotation_entry=rotation_entry,
                            )
                        else:
                            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
                            cfg = BaseQuantizeConfig(nbits=int(bit), group_size=group_size, axis=1)
                            hqq_linear = HQQLinear(child, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
                            optimized_layer = hqq_linear
                    else:
                        if bit == 16 and not forced_group_ids:
                            continue
                        apply_wht = vlm_rotation_mode == "wht"
                        rotation_entry = explicit_rotation_map.get(full) if vlm_rotation_mode == "file" else None
                        optimized_layer = optimize_blockwise_reconstruction(
                            child, nbits=bit, group_size=group_size, device=device,
                            act_scales=layer_scales,
                            outlier_fraction=outlier_fraction,
                            apply_wht=apply_wht,
                            forced_4bit_groups=forced_group_ids,
                            rotation_entry=rotation_entry,
                        )
                    post_calib = linear_post_calib_map.get(full)
                    if post_calib is not None:
                        optimized_layer = OutputScaleCalibrationWrapper(
                            optimized_layer,
                            scale=post_calib["scale"],
                            bias=post_calib.get("bias"),
                        )
                        print(f"  -> Applied output calibration to {full}", flush=True)
                    setattr(module, name, optimized_layer)
                    n_replaced += 1
                elif "action_head.model" in full or (
                    "action_head.vl_self_attention" in full and not skip_vl_self_attention
                ):
                    # AdaLN modulation layers ('.norm' in name) are bypassed here to
                    # preserve temporal stability. Note this is the *second* place the
                    # guard appears -- find_dit_layers has its own -- so gating only
                    # that one makes adaLN show up in the "found" count while never
                    # actually being replaced. Both must agree.
                    if ".norm" in full and not include_adaln:
                        continue
                    if _is_protected_dit_layer(full):
                        print(f"[quant-compiler] preserving {full} in bf16 (DIT_PROTECT_PATTERNS)", flush=True)
                        continue
                        
                    if dit_bits_env == "16":
                        continue
                        
                    dit_bit = dit_assignments.get(full, 4)

                    if dit_bit >= 16:
                        # Spec-listed bf16 passthrough: DIT_FORCE_BITS_SPECS
                        # entries at 16 keep the linear intact so a later
                        # DIT_CODEBOOK_OVERLAY_PATH pass can overwrite it.
                        continue

                    if dit_quant_mode_env == "uniform":
                        # DiT layer -> Uniform HQQ
                        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
                        cfg = BaseQuantizeConfig(nbits=dit_bit, group_size=group_size, axis=1)
                        hqq_linear = HQQLinear(child, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
                        setattr(module, name, hqq_linear)
                        n_replaced += 1
                        print(f"[quant-compiler] optimizing {full} at {dit_bit}-bit (mode: uniform)", flush=True)
                    elif dit_quant_mode_env == "hqq_awq":
                        # Proxy AWQ without WHT and without outliers
                        optimized_layer = optimize_blockwise_reconstruction(
                            child, nbits=dit_bit, group_size=group_size, device=device,
                            act_scales=None, outlier_fraction=0.0, apply_wht=False
                        )
                        setattr(module, name, optimized_layer)
                        n_replaced += 1
                        print(f"[quant-compiler] optimizing {full} at {dit_bit}-bit (mode: hqq_awq, proxy scale)", flush=True)
                    elif dit_quant_mode_env == "full_vlm":
                        # Full VLM setup on DiT
                        optimized_layer = optimize_blockwise_reconstruction(
                            child, nbits=dit_bit, group_size=group_size, device=device,
                            act_scales=None, outlier_fraction=0.005, apply_wht=True
                        )
                        setattr(module, name, optimized_layer)
                        n_replaced += 1
                        print(f"[quant-compiler] optimizing {full} at {dit_bit}-bit (mode: full_vlm, proxy scale)", flush=True)
                    elif dit_quant_mode_env == "tac_awq":
                        # Timestep-Aware Calibration (TAC) AWQ
                        layer_tac_scale = tac_scales.get(full)
                        if layer_tac_scale is not None:
                            layer_tac_scale = layer_tac_scale.to(device)
                        else:
                            print(f"[quant-compiler] WARNING: No TAC scale found for {full}. Falling back to proxy scale.", flush=True)
                            
                        dit_outlier_fraction = 0.015 if dit_bit == 3 else 0.005
                        optimized_layer = optimize_blockwise_reconstruction(
                            child, nbits=dit_bit, group_size=group_size, device=device,
                            act_scales=layer_tac_scale, outlier_fraction=dit_outlier_fraction, apply_wht=False
                        )
                        setattr(module, name, optimized_layer)
                        n_replaced += 1
                        print(f"[quant-compiler] optimizing {full} at {dit_bit}-bit (mode: tac_awq)", flush=True)
                    elif dit_quant_mode_env == "csb":
                        # Channel-wise Salience Balancing (CSB) SmoothQuant
                        layer_tac_scale = tac_scales.get(full)
                        if layer_tac_scale is not None:
                            layer_tac_scale = layer_tac_scale.to(device=device, dtype=torch.float32)
                            weight_fp32 = child.weight.data.to(torch.float32)
                            weight_scales = weight_fp32.abs().max(dim=0)[0].clamp_min(1e-6)
                            
                            # SmoothQuant scale vector: S = sqrt(act_scale / weight_scale)
                            alpha = 0.5
                            S = (layer_tac_scale.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(1e-4, 1e4)
                            
                            # Scale weights: W' = W * S (columns multiplied by S)
                            scaled_weight = weight_fp32 * S.unsqueeze(0)
                            child.weight.data.copy_(scaled_weight.to(child.weight.dtype))
                            
                            # Quantize using standard HQQ
                            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
                            cfg = BaseQuantizeConfig(nbits=dit_bit, group_size=group_size, axis=1)
                            hqq_linear = HQQLinear(child, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
                            
                            # Wrap in CSBQuantWrapper
                            csb_wrapped = CSBQuantWrapper(hqq_linear, S)
                            setattr(module, name, csb_wrapped)
                            n_replaced += 1
                            print(f"[quant-compiler] optimizing {full} at {dit_bit}-bit (mode: csb, scale shape={S.shape})", flush=True)
                        else:
                            from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
                            cfg = BaseQuantizeConfig(nbits=dit_bit, group_size=group_size, axis=1)
                            hqq_linear = HQQLinear(child, quant_config=cfg, compute_dtype=torch.bfloat16, device=device, del_orig=True)
                            setattr(module, name, hqq_linear)
                            n_replaced += 1
                            print(f"[quant-compiler] WARNING: No CSB scales found for {full}. Falling back to uniform HQQ.", flush=True)
            else:
                n_replaced += apply_custom_gr00t_quant(child, full)
        return n_replaced

    def apply_mlp_product_calibration(module, path=""):
        n_wrapped = 0
        for name, child in list(module.named_children()):
            full = f"{path}.{name}" if path else name
            calib = mlp_product_calib_map.get(full)
            if calib is not None and all(hasattr(child, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
                wrapped = MLPProductScaleCalibrationWrapper(child, scale=calib["scale"])
                setattr(module, name, wrapped)
                n_wrapped += 1
                print(f"[quant-compiler] applied mlp product calibration to {full}", flush=True)
                continue
            n_wrapped += apply_mlp_product_calibration(child, full)
        return n_wrapped

    def apply_block_delta_calibration(module, path=""):
        n_wrapped = 0
        for name, child in list(module.named_children()):
            full = f"{path}.{name}" if path else name
            calib = block_delta_calib_map.get(full)
            if calib is not None:
                wrapped = BlockResidualAffineCalibrationWrapper(
                    child,
                    scale=calib["scale"],
                    bias=calib.get("bias"),
                )
                setattr(module, name, wrapped)
                n_wrapped += 1
                print(f"[quant-compiler] applied block delta calibration to {full}", flush=True)
                continue
            n_wrapped += apply_block_delta_calibration(child, full)
        return n_wrapped

    # Install packed overlays before traversal so their target linears are not
    # needlessly quantized once and immediately discarded.
    apply_codebook_overlay_from_env(model, assignments, device=device, packed_only=True)
    total_replaced = apply_custom_gr00t_quant(model)
    total_replaced += apply_mlp_product_calibration(model)
    total_replaced += apply_block_delta_calibration(model)
    apply_qkdiag_overlay_from_env(model, assignments, device=device)
    apply_codebook_overlay_from_env(model, assignments, device=device, skip_packed=True)
    apply_dit_codebook_overlay_from_env(model, device=device)

    # Paper-main safety gate: verify the *installed* module layout, including
    # the HQQ bit width.  Merely checking ``isinstance(HQQLinear)`` cannot
    # distinguish an accidental HQQ3/HQQ4 swap.  A 16-bit force entry may
    # remain an nn.Linear during calibration or be replaced by the packed
    # codebook overlay during deployment; both are intentional.
    strict_quant_layout = os.environ.get(
        "GR00T_STRICT_QUANT_LAYOUT", "0"
    ).strip().lower() in (
        "1", "true", "yes", "y",
    )
    if strict_quant_layout:
        named_installed = dict(model.named_modules())

        def _installed_hqq_bits(module):
            cfg = getattr(module, "quant_config", None)
            if not isinstance(cfg, dict):
                return None
            weight_cfg = cfg.get("weight_quant_params")
            if not isinstance(weight_cfg, dict) or "nbits" not in weight_cfg:
                return None
            return int(weight_cfg["nbits"])

        strict_errors = []
        forced_maps = (("VLM", vlm_force_bits_specs), ("DiT", dit_force_bits_specs))
        for scope_name, specs in forced_maps:
            for full, expected_bits in specs.items():
                module = named_installed.get(full)
                if module is None:
                    strict_errors.append(f"{scope_name}:{full}:missing")
                    continue
                if expected_bits in (3, 4):
                    actual_bits = _installed_hqq_bits(module)
                    if actual_bits != expected_bits:
                        strict_errors.append(
                            f"{scope_name}:{full}:expected-HQQ{expected_bits},"
                            f"got-{type(module).__name__}/bits={actual_bits}"
                        )
                elif expected_bits == 16:
                    if not isinstance(
                        module, (torch.nn.Linear, PrecomputedPackedVectorCodebookLinear)
                    ):
                        strict_errors.append(
                            f"{scope_name}:{full}:expected-BF16-or-packed,"
                            f"got-{type(module).__name__}"
                        )
                else:
                    strict_errors.append(
                        f"{scope_name}:{full}:unsupported-strict-bit={expected_bits}"
                    )
        if strict_errors:
            raise RuntimeError(
                "strict quant layout mismatch: " + "; ".join(strict_errors[:12])
            )
        print(
            f"[quant-compiler] strict installed layout PASS: "
            f"VLM specs={len(vlm_force_bits_specs)}, DiT specs={len(dit_force_bits_specs)}",
            flush=True,
        )
    adapter_path = os.environ.get("VLM_TO_DIT_ADAPTER_PATH", "")
    if adapter_path:
        attach_vlm_to_dit_adapter(model, adapter_path, device=device)
    single_embodiment_id = os.environ.get("ACTION_HEAD_SINGLE_EMBODIMENT_ID", "").strip()
    single_embodiment_applied = 0
    if single_embodiment_id:
        single_embodiment_applied = materialize_single_embodiment_action_head(
            model,
            embodiment_id=int(single_embodiment_id),
            device=device,
        )
    embedding_quant_applied = apply_embedding_quant_from_env(
        model, group_size=group_size
    )
    vision_quant_applied = apply_vision_quant_from_env(
        model, group_size=group_size, device=device
    )
    if strict_quant_layout:
        named_fixed = dict(model.named_modules())
        fixed_errors = []
        if single_embodiment_id:
            materialized = [
                (name, module)
                for name, module in named_fixed.items()
                if isinstance(module, SingleEmbodimentCategoryLinear)
            ]
            wrong_ids = [
                name for name, module in materialized
                if int(module.materialized_embodiment_id) != int(single_embodiment_id)
            ]
            if single_embodiment_applied != 7 or len(materialized) != 7 or wrong_ids:
                fixed_errors.append(
                    f"single-embodiment:applied={single_embodiment_applied},"
                    f"installed={len(materialized)},wrong_ids={wrong_ids[:3]}"
                )
        embed_bits_raw = os.environ.get("VLM_EMBED_BITS", "").strip()
        if embed_bits_raw and int(embed_bits_raw) < 16:
            embed = named_fixed.get("backbone.model.model.language_model.embed_tokens")
            if (
                not embedding_quant_applied
                or not isinstance(embed, PackedAffineEmbedding)
                or int(getattr(embed, "nbits", -1)) != int(embed_bits_raw)
            ):
                fixed_errors.append(
                    f"embedding:expected-PackedAffine{embed_bits_raw},"
                    f"got-{type(embed).__name__}/bits={getattr(embed, 'nbits', None)}"
                )
        drop_lm_head = os.environ.get("VLM_DROP_LM_HEAD", "").strip().lower()
        if drop_lm_head in ("1", "true", "yes", "y"):
            lm_head = named_fixed.get("backbone.model.lm_head")
            if not isinstance(lm_head, torch.nn.Identity):
                fixed_errors.append(f"lm_head:not-dropped ({type(lm_head).__name__})")
        vision_bits_raw = os.environ.get("VLM_VISION_BITS", "").strip()
        if vision_bits_raw and int(vision_bits_raw) < 16:
            expected_vision_bits = int(vision_bits_raw)
            remaining = [
                name for name, module in named_fixed.items()
                if name.startswith("backbone.model.model.visual.")
                and isinstance(module, torch.nn.Linear)
            ]
            bad_vision_bits = [
                name for name, module in named_fixed.items()
                if name.startswith("backbone.model.model.visual.")
                and getattr(module, "quant_config", None) is not None
                and _installed_hqq_bits(module) != expected_vision_bits
            ]
            if vision_quant_applied <= 0 or remaining or bad_vision_bits:
                fixed_errors.append(
                    f"vision:applied={vision_quant_applied},remaining={len(remaining)},"
                    f"wrong_bits={len(bad_vision_bits)}"
                )
        vlsa_norm_errors = []
        for block in range(4):
            for norm_name in ("norm1", "norm3"):
                full = (
                    f"action_head.vl_self_attention.transformer_blocks."
                    f"{block}.{norm_name}"
                )
                module = named_fixed.get(full)
                if not isinstance(module, torch.nn.LayerNorm):
                    vlsa_norm_errors.append(f"{full}:{type(module).__name__}")
                    continue
                affine = [value for value in (module.weight, module.bias) if value is not None]
                if any(value.dtype != torch.bfloat16 for value in affine):
                    vlsa_norm_errors.append(
                        f"{full}:{[str(value.dtype) for value in affine]}"
                    )
        if vlsa_norm_errors:
            fixed_errors.append("VLSA norms:" + ",".join(vlsa_norm_errors[:4]))
        if fixed_errors:
            raise RuntimeError("strict fixed-background mismatch: " + "; ".join(fixed_errors))
        print(
            f"[quant-compiler] strict fixed background PASS: "
            f"embed={embed_bits_raw or 'off'} vision={vision_bits_raw or 'off'} "
            f"vision_linears={vision_quant_applied} VLSA_norms=8 "
            f"single_embodiment={single_embodiment_applied}",
            flush=True,
        )
    act_bits_raw = os.environ.get("ACT_QUANT_BITS", "").strip()
    if act_bits_raw:
        # Scope: the same VLM-LLM + DiT linears whose weights this compiler
        # quantizes. Vision/embedding activations stay bf16 -- state this scope
        # explicitly when reporting WxAy numbers.
        act_bits = int(act_bits_raw)
        named_now = dict(model.named_modules())
        n_wrapped = 0
        for full in list(vlm_llm_layers) + list(dit_layers):
            parent_name, _, child_name = full.rpartition(".")
            parent = named_now.get(parent_name)
            if parent is None:
                continue
            child = getattr(parent, child_name, None)
            if child is None or isinstance(child, ActQuantInputWrapper):
                continue
            setattr(parent, child_name, ActQuantInputWrapper(child, act_bits))
            n_wrapped += 1
        print(f"[quant-compiler] wrapped {n_wrapped} linears with A{act_bits} per-token dynamic fake-quant", flush=True)
    print(f"[quant-compiler] completed mixed-precision compile. replaced {total_replaced} total layers (VLM LLM {vlm_bits_env}-bit + DiT {dit_bits_env}-bit)", flush=True)
    return total_replaced
