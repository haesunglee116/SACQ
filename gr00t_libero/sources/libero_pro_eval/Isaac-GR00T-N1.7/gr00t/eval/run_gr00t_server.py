# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    # Smoothness study: post-load quantization knobs
    quant_backbone_4bit: bool = False
    """If True, quantize model.backbone Linear layers to bnb NF4 (4-bit)."""

    quant_action_head_bits: int = 0
    """If 1/2/3/4: HQQ-quantize model.action_head Linear layers to that bit-width."""

    # ILP-style smoothness study: HQQ-based phased quantization (no bitsandbytes dep)
    quant_phase: str = "none"
    """Phase preset: 'none' | 'llm' | 'backbone' | 'backbone_dit'.
    llm: only backbone.model.model.language_model.
    backbone: entire backbone (visual + LLM, lm_head auto-skipped).
    backbone_dit: backbone + action_head.model (DiT).
    """

    quant_bits: int = 4
    """Bit-width for the HQQ phase preset (when quant_phase != none)."""

    quant_group_size: int = 64
    """HQQ group size."""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        
        # Inject TAC Calibrator if environment variable is set
        if os.environ.get("TAC_CALIBRATION_MODE", "0") == "1":
            from gr00t.eval.tac_calibrator import attach_tac_calibrator, setup_auto_export
            attach_tac_calibrator(policy.model)
            out_path = "@WORKSPACE@/libero_pro_eval/quantization_analysis/act_scales_tac.pt"
            setup_auto_export(out_path)
            print(f"[server] TAC Calibration Mode Enabled. Scales will be saved to {out_path} on exit.", flush=True)

        # Smoothness study: post-load quantization
        if config.quant_backbone_4bit or config.quant_action_head_bits > 0:
            import torch
            model = policy.model
            if config.quant_backbone_4bit and hasattr(model, "backbone"):
                from gr00t.eval.cosmos_quant_shim import quantize_linears_to_bnb_nf4  # local helper
                n = quantize_linears_to_bnb_nf4(model.backbone, device="cuda:0")
                print(f"[quant] model.backbone: bnb NF4 on {n} Linear layers", flush=True)
            head_bits = int(config.quant_action_head_bits)
            if head_bits in (1, 2, 3, 4) and hasattr(model, "action_head"):
                if head_bits == 4:
                    from gr00t.eval.cosmos_quant_shim import quantize_linears_to_bnb_nf4
                    n = quantize_linears_to_bnb_nf4(model.action_head, device="cuda:0")
                    print(f"[quant] model.action_head: bnb NF4 on {n} Linear layers", flush=True)
                else:
                    from gr00t.eval.cosmos_quant_shim import quantize_linears_with_hqq
                    n = quantize_linears_with_hqq(model.action_head, nbits=head_bits, device="cuda:0")
                    print(f"[quant] model.action_head: HQQ {head_bits}-bit on {n} Linear layers", flush=True)
        # HQQ-based phase presets (no bitsandbytes dependency)
        if config.quant_phase != "none":
            model = policy.model
            gs = int(config.quant_group_size)
            if config.quant_phase == "gr00t_vlm27_dit4":
                from gr00t.eval.cosmos_quant_shim import quantize_gr00t_vlm27_dit4
                print(f"[quant] phase=gr00t_vlm27_dit4 HQQ mixed-precision, group_size={gs}", flush=True)
                n = quantize_gr00t_vlm27_dit4(model, group_size=gs, device="cuda:0")
                print(f"[quant] Completed gr00t_vlm27_dit4 HQQ on {n} layers", flush=True)
            else:
                from gr00t.eval.cosmos_quant_shim import quantize_linears_with_hqq
                bits = int(config.quant_bits)
                print(f"[quant] phase={config.quant_phase}  HQQ bits={bits} group_size={gs}", flush=True)
                if config.quant_phase == "llm":
                    target = model.backbone.model.model.language_model
                    n = quantize_linears_with_hqq(target, nbits=bits, group_size=gs, device="cuda:0")
                    print(f"[quant] phase=llm  HQQ-W{bits} on {n} LLM Linear layers", flush=True)
                elif config.quant_phase == "backbone":
                    n = quantize_linears_with_hqq(model.backbone, nbits=bits, group_size=gs, device="cuda:0")
                    print(f"[quant] phase=backbone  HQQ-W{bits} on {n} backbone Linear layers", flush=True)
                elif config.quant_phase == "backbone_dit":
                    n_b = quantize_linears_with_hqq(model.backbone, nbits=bits, group_size=gs, device="cuda:0")
                    n_d = quantize_linears_with_hqq(model.action_head.model, nbits=bits, group_size=gs, device="cuda:0")
                    print(f"[quant] phase=backbone_dit  HQQ-W{bits} on {n_b} backbone + {n_d} DiT Linear layers", flush=True)
                else:
                    print(f"[quant] WARN: unknown quant_phase '{config.quant_phase}', skipping", flush=True)

        analysis_dir = os.environ.get("GR00T_ANALYSIS_DIR", "")
        if analysis_dir:
            from gr00t.eval.analysis_probe import attach_analysis_probe

            analysis_tag = os.environ.get("GR00T_ANALYSIS_TAG", config.quant_phase)
            analysis_max_queries = int(os.environ.get("GR00T_ANALYSIS_MAX_QUERIES", "40"))
            attach_analysis_probe(
                policy,
                out_dir=analysis_dir,
                tag=analysis_tag,
                max_queries=analysis_max_queries,
            )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                with open(config.modality_config_path, "r") as f:
                    raw = json.load(f)
                # ReplayPolicy expects ModalityConfig instances, not raw dicts.
                modality_configs = {k: ModalityConfig(**v) for k, v in raw.items()}
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    capture_dir = os.environ.get("GR00T_CAPTURE_REQUESTS_DIR", "")
    if capture_dir:
        from gr00t.eval.request_capture import attach_request_capture

        capture_tag = os.environ.get("GR00T_CAPTURE_TAG", config.quant_phase or "capture")
        capture_max_requests = int(os.environ.get("GR00T_CAPTURE_MAX_REQUESTS", "64"))
        attach_request_capture(
            policy,
            out_dir=capture_dir,
            tag=capture_tag,
            max_requests=capture_max_requests,
        )

    # Seed *after* model loading and post-load quantization. Those setup paths
    # can consume a bit-dependent number of random values; seeding only at
    # process startup would therefore give each quantization point a different
    # diffusion-noise stream. This opt-in keeps ordinary users unchanged.
    inference_seed_raw = os.environ.get("GR00T_INFERENCE_SEED", "").strip()
    if inference_seed_raw:
        import random
        import numpy as np
        import torch

        inference_seed = int(inference_seed_raw)
        def _seed_inference_rng() -> None:
            random.seed(inference_seed)
            np.random.seed(inference_seed % (2**32))
            torch.manual_seed(inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(inference_seed)

        _seed_inference_rng()
        print(
            f"[server] fixed post-compile inference RNG seed={inference_seed}",
            flush=True,
        )
        reset_seed_raw = os.environ.get(
            "GR00T_RESET_INFERENCE_SEED", "0"
        ).strip().lower()
        if reset_seed_raw in ("1", "true", "yes", "y"):
            original_reset = policy.reset

            def _reset_with_fixed_rng(options=None):
                # rollout_policy calls the remote reset endpoint once at the
                # beginning of every task.  Re-seeding here prevents task B's
                # diffusion-noise stream from depending on how early task A's
                # environments succeeded and reset.
                _seed_inference_rng()
                print(
                    f"[server] task-reset inference RNG seed={inference_seed}",
                    flush=True,
                )
                return original_reset(options)

            policy.reset = _reset_with_fixed_rng

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    print(f"\n✓ Server ready — listening on {config.host}:{config.port}\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
