#!/usr/bin/env python3
"""Final, suite-correct mixed-bit allocator for GR00T N1.7.

This is deliberately separate from the 2026-08-23 ablation scripts.  It fixes
four correctness problems that must not be inherited by the paper allocation:

* every candidate delta is averaged over the four matching LIBERO checkpoints;
* the 24 VLSA linears are kept as suite-specific HQQ4 background modules;
* vector-codebook weights are reconstructed from packed state on CUDA in BF16;
* the stopping constraint uses explicit serialized bytes (including VC shift).

Eight workers mean four suites x two mirrored candidate shards.  A candidate is
accepted only after exactly one measurement from every suite has arrived.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import tree

PROJ = Path("@WORKSPACE@/libero_pro_eval")
QA = PROJ / "quantization_analysis"
ALLOC = QA / "alloc_floor_2026-08-06"
REQUEST_DIR = PROJ / "request_replay_runs/calib_hard_2026-08-04/val/requests"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SUITE_ORDER = {suite: index for index, suite in enumerate(SUITES)}
GROUP_SIZE = 64
VECTOR_DIM = 4
HQQ_SENTINEL_K = {90000: 4, 80000: 3}

sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "Isaac-GR00T-N1.7"))
import coord_descent_alloc_setting1_improved_2026_08_23 as infra  # noqa: E402
import coord_descent_floor_scope_persistent_2026_08_23 as legacy  # noqa: E402
from artifact_registry import KB, find_packed  # noqa: E402
from gr00t.policy.server_client import MsgSerializer  # noqa: E402
from measure_hotspot_protection_proxy import collate_request  # noqa: E402

BIT_OF_K = {90000: 4.0, 80000: 3.0, **infra.BIT_OF_K, 8: 0.75}
LADDER = sorted(BIT_OF_K, reverse=True)
DEFAULT_TARGET_PAYLOAD_BYTES = 432_236_942
HOT_SPEC_FILE = QA / "vlm_hotspot11_new_2026_08_18_specs.txt"
UNIFIED_PARAMS_CSV = QA / "coord_descent_setting3_2026_08_22_ckpt1.50.csv"

# Variables read by the quantization compiler or by optional deployment paths.
# They are removed first so a submitted shell cannot silently change the model.
QUANT_ENV_KEYS = {
    "ACTION_HEAD_SINGLE_EMBODIMENT_ID", "ACT_QUANT_BITS",
    "DIT_BITS", "DIT_CODEBOOK_OVERLAY_PATH", "DIT_FORCE_BITS_SPECS",
    "DIT_FORCE_BITS_SPECS_FILE", "DIT_INCLUDE_ADALN", "DIT_PROTECT_PATTERNS",
    "DIT_QUANT_MODE", "DIT_SKIP_VL_SELF_ATTENTION",
    "GR00T_CODEBOOK_ACCELERATE_VLM", "GR00T_CODEBOOK_ACCELERATE_VLM_PATTERN",
    "GR00T_CODEBOOK_LINEAR_BACKEND", "VLM_BITS", "VLM_CODEBOOK_OVERLAY_PATH",
    "VLM_DROP_LM_HEAD", "VLM_EMBED_BITS", "VLM_FORCE_4BIT_GROUP_SPECS",
    "VLM_FORCE_4BIT_PATTERNS", "VLM_FORCE_BITS_SPECS",
    "VLM_FORCE_BITS_SPECS_FILE", "VLM_GIVENS_FRACTION", "VLM_OUTLIER_BITS",
    "VLM_OUTLIER_FRACTION", "VLM_POST_CALIB_PATH", "VLM_PRUNE_FRACTION",
    "VLM_QKDIAG_OVERLAY_PATH", "VLM_QUANT_MODE", "VLM_ROTATION_MODE",
    "VLM_ROTATION_PATH", "VLM_TO_DIT_ADAPTER_PATH", "VLM_VISION_BITS",
    "VLM_2BIT_CODEBOOK_GROUP_SIZE", "VLM_2BIT_CODEBOOK_ITERS",
    "VLM_2BIT_CODEBOOK_MODE", "VLM_2BIT_CODEBOOK_PACKED",
    "VLM_2BIT_CODEBOOK_SAMPLE", "VLM_2BIT_CODEBOOK_SEED",
    "VLM_2BIT_CODEBOOK_SHIFT", "VLM_2BIT_CODEBOOK_SIZE",
    "VLM_2BIT_CODEBOOK_VECTOR_DIM", "VLM_2BIT_CODEBOOK_SCALE_SPECS",
    "VLM_2BIT_CODEBOOK_TRANSFORM_SPECS",
}
FINAL_ENV = {
    "ACTION_HEAD_SINGLE_EMBODIMENT_ID": "2",
    "VLM_BITS": "16",
    "VLM_QUANT_MODE": "uniform",
    "VLM_ROTATION_MODE": "none",
    "DIT_BITS": "4",
    "DIT_QUANT_MODE": "uniform",
    "DIT_INCLUDE_ADALN": "0",
    "DIT_SKIP_VL_SELF_ATTENTION": "0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def hermetic_final_env():
    previous = {key: os.environ.get(key) for key in QUANT_ENV_KEYS}
    for key in QUANT_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(FINAL_ENV)
    try:
        yield
    finally:
        for key in QUANT_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in previous.items():
            if value is not None:
                os.environ[key] = value


def read_hot() -> dict[str, int]:
    output = {}
    for item in HOT_SPEC_FILE.read_text().strip().split(";"):
        name, bits = item.split("=", 1)
        output[name.strip()] = int(bits)
    assert len(output) == 11 and set(output.values()) <= {3, 4}
    return output


def load_candidates(subset_size: int = 0):
    vlm_rows = list(csv.DictReader((ALLOC / "vlm_new11_cg150.csv").open()))
    dit_rows = list(csv.DictReader((ALLOC / "dit_cg150.csv").open()))
    params = {row["layer"]: int(row["params"]) for row in vlm_rows + dit_rows}
    param_rows = {
        row["layer"]: int(row["params"])
        for row in csv.DictReader(UNIFIED_PARAMS_CSV.open())
    }
    hot = read_hot()
    for name in hot:
        params[name] = param_rows[name]
    vlm = [row["layer"] for row in vlm_rows] + list(hot)
    dit = [row["layer"] for row in dit_rows]
    assert len(vlm) == 112 and len(dit) == 226
    assert len(set(vlm + dit)) == 338
    if subset_size:
        assert 2 <= subset_size <= 338
        rng = random.Random(20260824)
        n_vlm = max(1, round(subset_size * len(vlm) / 338))
        n_vlm = min(n_vlm, len(vlm), subset_size - 1)
        vlm = rng.sample(vlm, n_vlm)
        dit = rng.sample(dit, subset_size - n_vlm)
    return vlm, dit, params, hot


def kind_of(name: str) -> str:
    return "dit" if name.startswith("action_head") else "vlm"


def load_suite_batches(policy, suite: str, limit: int):
    """Prepare only this suite's requests while preserving global seed indices."""
    paths = sorted(REQUEST_DIR.glob("*.msgpack"))
    assert len(paths) == 40
    selected = [(index, path) for index, path in enumerate(paths)
                if path.name.startswith(suite + "_")][:limit]
    assert len(selected) == limit
    batches = [None] * len(paths)
    for index, path in selected:
        payload = MsgSerializer.from_bytes(path.read_bytes())
        collated = collate_request(policy, payload["observation"])
        backbone, action = policy.model.prepare_input(collated)
        to_cpu = lambda value: tree.map_structure(
            lambda item: item.cpu() if isinstance(item, torch.Tensor) else item,
            value,
        )
        batches[index] = (payload["request_id"], to_cpu(backbone), to_cpu(action))
    return batches, [index for index, _ in selected]


def payload_bytes(params: int, k: int) -> int:
    """Exact serialized layer payload; candidate bias is constant and omitted."""
    assert params % GROUP_SIZE == 0
    if k in HQQ_SENTINEL_K:
        groups = params // GROUP_SIZE
        metadata = groups * 4  # BF16 scale + BF16 zero per group.
        if HQQ_SENTINEL_K[k] == 4:
            packed = params // 2
        else:
            packed = math.ceil(groups / 10) * GROUP_SIZE * 4
        return packed + metadata
    assert params % VECTOR_DIM == 0
    bits = math.ceil(math.log2(k))
    indices = ((params // VECTOR_DIM) * bits + 7) // 8
    codebook = k * VECTOR_DIM * 2
    scales = (params // GROUP_SIZE) * 2
    shift = 2  # scalar BF16 shift is part of every packed VC payload.
    return indices + codebook + scales + shift


def exact_packed_path(kind: str, suite: str, name: str, k: int) -> Path:
    """Legacy exact-directory resolver; public manifests bind this explicitly."""
    path = find_packed(kind, name, k, suite=suite, init="aware", verify=False)
    expected_parent = (
        f"vlm_packed_u{KB[k]}" if kind == "vlm"
        else f"fits_packed_u{KB[k]}_{suite}"
    )
    assert path.parent.name == expected_parent, (
        f"final allocation forbids fallback artifacts: {path} "
        f"(expected parent {expected_parent})"
    )
    return path


def packed_weight_gpu(kind: str, suite: str, name: str, k: int,
                      device: torch.device) -> torch.Tensor:
    """Deployment-equivalent BF16 gather/mul/add; never reads artifact['dense']."""
    path = exact_packed_path(kind, suite, name, k)
    obj = torch.load(path, map_location="cpu", mmap=True)
    packed = obj["packed"]
    assert int(packed["codebook"].shape[0]) == k, (path, packed["codebook"].shape, k)
    codebook = packed["codebook"].to(device=device, dtype=torch.bfloat16)
    indices = packed["indices"].to(device=device, dtype=torch.long)
    scales = packed["scales"].to(device=device, dtype=torch.bfloat16)
    shift = torch.as_tensor(packed.get("shift", 0.0), device=device,
                            dtype=torch.bfloat16)
    out_features = int(packed["out_features"])
    in_features = int(packed["in_features"])
    group_size = int(packed["group_size"])
    vectors = codebook[indices]
    weight = vectors.reshape(out_features, scales.shape[1], group_size)
    weight.mul_(scales.unsqueeze(-1))
    weight = weight.reshape(out_features, in_features)
    weight.add_(shift)
    return weight


def preflight_exact_artifacts(vlm: list[str], dit: list[str], min_bit: float) -> int:
    """Require every VC state from the canonical per-bit directories."""
    checked = 0
    vc_ks = [k for k in LADDER
             if k not in HQQ_SENTINEL_K and BIT_OF_K[k] >= min_bit - 1e-12]
    for name in vlm:
        for k in vc_ks:
            path = find_packed("vlm", name, k, suite=SUITES[0],
                               init="aware", verify=False)
            assert path.parent.name == f"vlm_packed_u{KB[k]}", path
            checked += 1
    for suite in SUITES:
        for name in dit:
            for k in vc_ks:
                path = find_packed("dit", name, k, suite=suite,
                                   init="aware", verify=False)
                assert path.parent.name == f"fits_packed_u{KB[k]}_{suite}", path
                checked += 1
    return checked


class SuiteReplica:
    def __init__(self, suite: str, device_id: int, requests_per_suite: int,
                 candidates: list[str]):
        assert suite in SUITES
        self.suite = suite
        self.device = torch.device(f"cuda:{device_id}")
        self.cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self.policy = infra.Gr00tPolicy(
            infra.EmbodimentTag.resolve("LIBERO_PANDA"), infra.local_ckpt(suite),
            device=str(self.device), strict=True,
        )
        self.policy.model.eval()
        for parameter in self.policy.model.parameters():
            parameter.requires_grad_(False)

        self.batches, self.subset = load_suite_batches(
            self.policy, suite, requests_per_suite
        )

        # References must come from the corresponding suite checkpoint.  The
        # old refs_val_hard_limit40 cache came entirely from spatial and is not
        # legal here.
        self.refs = {}
        with torch.no_grad():
            for index in self.subset:
                request_id, batch, action_input = self.batches[index]
                action = legacy.decoded_action_only(
                    self.policy,
                    infra.move_to_device(batch, self.device, torch.bfloat16),
                    infra.move_to_device(action_input, self.device, torch.bfloat16),
                    seed=7000 + index,
                ).to(torch.float64)
                self.refs[request_id] = action

        named = dict(self.policy.model.named_modules())
        missing = [name for name in candidates if name not in named]
        assert not missing, f"{suite}: missing candidate modules: {missing[:3]}"
        self.orig_weights = {
            name: named[name].weight.detach().to(self.device, torch.float32).clone()
            for name in candidates
        }
        self.orig_biases = {
            name: (None if getattr(named[name], "bias", None) is None else
                   named[name].bias.detach().to(self.device, torch.bfloat16).clone())
            for name in candidates
        }
        vlsa_targets = [
            name for name, module in named.items()
            if "action_head.vl_self_attention" in name
            and isinstance(module, nn.Linear) and ".norm" not in name
        ]
        assert len(vlsa_targets) == 24, (
            f"{suite}: expected 24 non-norm VLSA linears, found {len(vlsa_targets)}"
        )

        # Hermetic baseline: language candidates stay BF16; DiT model and the
        # 24 VLSA linears become suite-native HQQ4.  Candidate DiT layers are
        # overwritten below, while VLSA remains fixed HQQ4.
        with hermetic_final_env():
            infra.quantize_gr00t_vlm27_dit4(
                self.policy.model, group_size=GROUP_SIZE, device=str(self.device)
            )
        infra.replace_embedding(self.policy.model, 4, GROUP_SIZE, True)
        infra.quantize_visual_linears(
            self.policy.model, nbits=4, group_size=GROUP_SIZE,
            device=str(self.device),
        )
        named_after = dict(self.policy.model.named_modules())
        unreplaced_vlsa = [name for name in vlsa_targets
                           if isinstance(named_after[name], nn.Linear)]
        assert not unreplaced_vlsa, (
            f"{suite}: VLSA HQQ4 replacement missing: {unreplaced_vlsa[:3]}"
        )
        self.parents = {
            name: (self.policy.model.get_submodule(name.rsplit(".", 1)[0]),
                   name.rsplit(".", 1)[1])
            for name in candidates
        }
        gc.collect()
        torch.cuda.empty_cache()

    def put(self, name: str, weight: torch.Tensor) -> None:
        bias = self.orig_biases[name]
        linear = nn.Linear(
            weight.shape[1], weight.shape[0], bias=bias is not None,
            device=self.device, dtype=torch.bfloat16,
        )
        linear.weight.data.copy_(weight.to(self.device, torch.bfloat16))
        if bias is not None:
            linear.bias.data.copy_(bias)
        parent, attr = self.parents[name]
        setattr(parent, attr, linear)

    def weight(self, name: str, k: int) -> torch.Tensor:
        key = (kind_of(name), name, k)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if k in HQQ_SENTINEL_K:
            weight = infra.quantize_weight_hqq(
                self.orig_weights[name], nbits=HQQ_SENTINEL_K[k],
                group_size=GROUP_SIZE, device=str(self.device),
            ).to(torch.bfloat16)
        else:
            weight = packed_weight_gpu(
                kind_of(name), self.suite, name, k, self.device
            ).to(torch.bfloat16)
        # CPU BF16 cache bounds device memory and preserves exact BF16 values.
        cached = weight.detach().cpu().contiguous()
        self.cache[key] = cached
        return cached

    def install(self, state: dict[str, int]) -> None:
        for name, k in state.items():
            self.put(name, self.weight(name, k))

    def measure(self, prepared=None, cached_vlm: bool = False) -> float:
        if prepared is None:
            prepared = legacy.prepare_fast_round(self, self.subset)
        return legacy.measure_fast_exact(
            self, self.subset, prepared, use_cached_vlm=cached_vlm
        )

    def measure_deltas(self, shard: list[str], state: dict[str, int],
                       min_bit: float):
        prepared = legacy.prepare_fast_round(self, self.subset)
        base = self.measure(prepared, False)
        deltas = {}
        for name in shard:
            index = LADDER.index(state[name])
            if index + 1 >= len(LADDER):
                continue
            next_k = LADDER[index + 1]
            if BIT_OF_K[next_k] < min_bit - 1e-12:
                continue
            current = self.weight(name, state[name])
            self.put(name, self.weight(name, next_k))
            error = self.measure(prepared, cached_vlm=kind_of(name) == "dit")
            deltas[name] = error - base
            self.put(name, current)
        return base, deltas


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def worker_main(args) -> None:
    shard = [line for line in Path(args.shard_file).read_text().splitlines() if line]
    all_candidates = [line for line in Path(args.candidate_file).read_text().splitlines() if line]
    work_dir = Path(args.work_dir)
    replica = SuiteReplica(args.suite, args.device, args.requests_per_suite,
                           all_candidates)
    round_index = 0
    while not (work_dir / "stop.flag").exists():
        state_file = work_dir / f"round_{round_index}_state.csv"
        if not state_file.exists():
            time.sleep(0.25)
            continue
        state = {row["layer"]: int(row["k"])
                 for row in csv.DictReader(state_file.open())}
        floor = float(
            (work_dir / f"round_{round_index}_floor.txt").read_text()
        )
        replica.cache.clear()
        gc.collect()
        replica.install(state)
        base, deltas = replica.measure_deltas(shard, state, floor)
        output = work_dir / (
            f"round_{round_index}_result_{args.suite}_shard{args.shard_group}.json"
        )
        atomic_json(output, {
            "suite": args.suite,
            "shard_group": args.shard_group,
            "base_err": base,
            "deltas": deltas,
        })
        round_index += 1


class MultiSuitePool:
    def __init__(self, args, candidates: list[str], work_dir: Path):
        self.args = args
        self.work_dir = work_dir
        self.round = 0
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        candidate_file = work_dir / "candidates.txt"
        candidate_file.write_text("\n".join(candidates) + "\n")
        shards = [candidates[index::args.workers_per_suite]
                  for index in range(args.workers_per_suite)]
        self.procs = []
        device = 0
        for suite in SUITES:
            for shard_group, shard in enumerate(shards):
                shard_file = work_dir / f"shard_{shard_group}.txt"
                if not shard_file.exists():
                    shard_file.write_text("\n".join(shard) + "\n")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker-mode",
                    "--suite", suite, "--device", str(device),
                    "--shard-group", str(shard_group),
                    "--shard-file", str(shard_file),
                    "--candidate-file", str(candidate_file),
                    "--work-dir", str(work_dir),
                    "--requests-per-suite", str(args.requests_per_suite),
                    "--active-min-bit", str(args.min_bit),
                ]
                log_path = work_dir / f"worker_{suite}_shard{shard_group}.log"
                log = log_path.open("w")
                process = subprocess.Popen(command, stdout=log,
                                           stderr=subprocess.STDOUT)
                self.procs.append((suite, shard_group, process, log, log_path))
                device += 1
        expected = len(SUITES) * args.workers_per_suite
        assert device == expected == args.n_gpus

    def measure(self, state: dict[str, int], active_floor: float):
        # Workers read the floor from a small round sidecar, allowing moving
        # floor without restarting processes.
        floor_file = self.work_dir / f"round_{self.round}_floor.txt"
        floor_file.write_text(f"{active_floor:.12g}\n")
        state_file = self.work_dir / f"round_{self.round}_state.csv"
        temporary = state_file.with_suffix(".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "k"])
            for name in sorted(state):
                writer.writerow([name, state[name]])
        temporary.replace(state_file)

        results = []
        for suite, shard_group, process, _, log_path in self.procs:
            path = self.work_dir / (
                f"round_{self.round}_result_{suite}_shard{shard_group}.json"
            )
            while not path.exists():
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"worker {suite}/{shard_group} died rc={return_code}; {log_path}"
                    )
                time.sleep(0.5)
            results.append(json.loads(path.read_text()))

        expected_names = {
            name for name, k in state.items()
            if LADDER.index(k) + 1 < len(LADDER)
            and BIT_OF_K[LADDER[LADDER.index(k) + 1]] >= active_floor - 1e-12
        }
        suite_bases = {}
        suite_deltas = {suite: {} for suite in SUITES}
        seen_workers = set()
        for result in results:
            suite = result["suite"]
            shard_group = int(result["shard_group"])
            key = (suite, shard_group)
            assert suite in SUITES and key not in seen_workers
            seen_workers.add(key)
            base = float(result["base_err"])
            assert math.isfinite(base)
            suite_bases.setdefault(suite, []).append(base)
            for name, delta in result["deltas"].items():
                assert name in expected_names and name not in suite_deltas[suite]
                delta = float(delta)
                assert math.isfinite(delta)
                suite_deltas[suite][name] = delta
        assert len(seen_workers) == len(SUITES) * self.args.workers_per_suite
        for suite in SUITES:
            values = suite_bases[suite]
            assert len(values) == self.args.workers_per_suite
            assert max(values) - min(values) <= 1e-10, (suite, values)
            missing = expected_names.difference(suite_deltas[suite])
            extra = set(suite_deltas[suite]).difference(expected_names)
            assert not missing and not extra, (suite, len(missing), len(extra))
        aggregate = {
            name: sum(suite_deltas[suite][name] for suite in SUITES) / len(SUITES)
            for name in expected_names
        }
        base_by_suite = {suite: sum(values) / len(values)
                         for suite, values in suite_bases.items()}
        base = sum(base_by_suite.values()) / len(SUITES)
        self.round += 1
        return base, aggregate, base_by_suite

    def shutdown(self):
        (self.work_dir / "stop.flag").touch()
        for _, _, process, log, _ in self.procs:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
            log.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--target-payload-bytes", type=int,
                        default=DEFAULT_TARGET_PAYLOAD_BYTES)
    parser.add_argument("--target-nominal-bits", type=float, default=1.50)
    parser.add_argument("--start-k", type=int, default=90000, choices=LADDER,
                        help="Final default is HQQ4; lower values are for smoke tests.")
    parser.add_argument("--min-bit", type=float, default=1.25,
                        choices=sorted(set(BIT_OF_K.values())))
    parser.add_argument("--refresh-every", type=int, default=100)
    parser.add_argument("--checkpoint-step", type=float, default=0.10)
    parser.add_argument(
        "--scope-policy",
        choices=("hard", "atomic-tie-global"),
        default="hard",
        help=(
            "hard reproduces the original higher-actual-bpw scope rule; "
            "atomic-tie-global searches both scopes when no legal move in "
            "the higher-bpw scope can reduce the absolute scope gap."
        ),
    )
    parser.add_argument("--workers-per-suite", type=int, default=2)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--requests-per-suite", type=int, default=10)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Smoke-test stop; zero means run to byte budget.")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--suite", choices=SUITES, default=SUITES[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-group", type=int, default=0)
    parser.add_argument("--shard-file", default="")
    parser.add_argument("--candidate-file", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--active-min-bit", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        worker_main(args)
        return

    assert args.out, "--out is required"
    assert args.n_gpus == len(SUITES) * args.workers_per_suite
    assert 1 <= args.requests_per_suite <= 10
    vlm, dit, params, hot = load_candidates(args.subset_size)
    candidates = vlm + dit
    scopes = {"vlm": vlm, "dit": dit}
    total_params = sum(params[name] for name in candidates)
    artifact_count = preflight_exact_artifacts(vlm, dit, args.min_bit)
    scope_params = {scope: sum(params[name] for name in names)
                    for scope, names in scopes.items()}
    start_k = args.start_k
    if args.resume_from:
        loaded = {row["layer"]: int(row["k"])
                  for row in csv.DictReader(open(args.resume_from))}
        missing = set(candidates).difference(loaded)
        assert not missing, f"resume missing {len(missing)} candidates"
        state = {name: loaded[name] for name in candidates}
    else:
        state = {name: start_k for name in candidates}

    def nominal(names=candidates):
        denominator = sum(params[name] for name in names)
        return sum(params[name] * BIT_OF_K[state[name]] for name in names) / denominator

    def actual(names=candidates):
        denominator = sum(params[name] for name in names)
        return sum(payload_bytes(params[name], state[name]) * 8 for name in names) / denominator

    def total_bytes():
        return sum(payload_bytes(params[name], state[name]) for name in candidates)

    def save(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "params", "bit", "k", "scope",
                             "payload_bytes"])
            for name in sorted(candidates):
                writer.writerow([name, params[name], f"{BIT_OF_K[state[name]]:.2f}",
                                 state[name], kind_of(name),
                                 payload_bytes(params[name], state[name])])

    checkpoints = []
    point = (math.ceil((nominal() - 1e-9) / args.checkpoint_step) - 1) * args.checkpoint_step
    while point > args.target_nominal_bits + 1e-9:
        checkpoints.append(round(point, 2))
        point -= args.checkpoint_step

    def floor_for(target: float):
        legal = [bit for bit in set(BIT_OF_K.values())
                 if args.min_bit - 1e-9 <= bit < target - 1e-9]
        return max(legal) if legal else args.min_bit

    def next_target():
        return checkpoints[0] if checkpoints else args.target_nominal_bits

    floor = floor_for(next_target())
    print(
        f"[final-cd] suites={','.join(SUITES)} candidates={len(candidates)} "
        f"params={total_params} target_bytes={args.target_payload_bytes} "
        f"start_bytes={total_bytes()} floor={floor:.2f} "
        f"VLSA=HQQ4x24/suite vision=HQQ4 embedding=HQQ4 "
        f"packed_decoder=GPU_BF16 shift_accounted=yes "
        f"exact_artifacts={artifact_count}",
        flush=True,
    )
    if args.dry_run:
        print(f"[final-cd] DRY nominal={nominal():.6f} actual={actual():.6f} "
              f"scope=VLM:{actual(vlm):.6f}/DiT:{actual(dit):.6f}")
        return

    work_dir = QA / f"final_multisuite_work_{Path(args.out).stem}"
    pool = MultiSuitePool(args, candidates, work_dir)
    started = time.time()
    history = []
    try:
        base, deltas, bases = pool.measure(state, floor)
        history.append({"step": 0, "proxy": base, "base_by_suite": bases,
                        "bytes": total_bytes(), "nominal": nominal(),
                        "actual": actual(), "floor": floor})
        print(f"[final-cd] round0 proxy={base:.8f} by_suite={bases} "
              f"deltas={len(deltas)} elapsed={time.time()-started:.0f}s", flush=True)
        step = 0
        since_refresh = 0
        global_tie_steps = 0
        def continue_descent():
            if args.max_steps:
                return step < args.max_steps
            return total_bytes() > args.target_payload_bytes

        while continue_descent():
            scope_now = {scope: actual(names) for scope, names in scopes.items()}
            available = {}
            for scope, names in scopes.items():
                rows = []
                for name in names:
                    if name not in deltas:
                        continue
                    index = LADDER.index(state[name])
                    if index + 1 >= len(LADDER):
                        continue
                    next_k = LADDER[index + 1]
                    if BIT_OF_K[next_k] < floor - 1e-12:
                        continue
                    saved = (payload_bytes(params[name], state[name]) -
                             payload_bytes(params[name], next_k))
                    assert saved > 0
                    rows.append((deltas[name] / saved, name, next_k))
                if rows:
                    available[scope] = rows
            if not available:
                raise RuntimeError(
                    f"no legal move at bytes={total_bytes()} target={args.target_payload_bytes}"
                )
            selected_scope = max(available, key=lambda scope: (scope_now[scope], scope))
            selection_rows = available[selected_scope]
            selection_reason = f"higher-{selected_scope}"
            if args.scope_policy == "atomic-tie-global" and len(available) == 2:
                other_scope = next(scope for scope in available if scope != selected_scope)
                gap_before = abs(scope_now[selected_scope] - scope_now[other_scope])
                balancing_rows = []
                for row in selection_rows:
                    _, candidate_name, candidate_k = row
                    saved = (
                        payload_bytes(params[candidate_name], state[candidate_name])
                        - payload_bytes(params[candidate_name], candidate_k)
                    )
                    projected_selected = (
                        scope_now[selected_scope]
                        - saved * 8 / scope_params[selected_scope]
                    )
                    gap_after = abs(projected_selected - scope_now[other_scope])
                    if gap_after < gap_before - 1e-15:
                        balancing_rows.append(row)
                if balancing_rows:
                    selection_rows = balancing_rows
                else:
                    selection_rows = [row for rows in available.values() for row in rows]
                    selection_reason = "atomic-tie-global"
                    global_tie_steps += 1
            cost, name, next_k = min(selection_rows,
                                     key=lambda row: (row[0], row[1]))
            state[name] = next_k
            del deltas[name]
            step += 1
            since_refresh += 1

            while checkpoints and nominal() <= checkpoints[0] + 1e-12:
                reached = checkpoints.pop(0)
                checkpoint_path = Path(args.out).with_suffix("")
                checkpoint_path = Path(str(checkpoint_path) + f"_ckpt{reached:.2f}.csv")
                save(checkpoint_path)
                old_floor = floor
                floor = floor_for(next_target())
                print(f"[final-cd] checkpoint={reached:.2f} step={step} "
                      f"nominal={nominal():.5f} actual={actual():.5f} "
                      f"bytes={total_bytes()} floor={old_floor:.2f}->{floor:.2f}",
                      flush=True)
                if floor != old_floor:
                    since_refresh = args.refresh_every

            if step % 50 == 0:
                print(f"[final-cd] step={step} nominal={nominal():.5f} "
                      f"actual={actual():.5f} bytes={total_bytes()} "
                      f"scope=VLM:{actual(vlm):.5f}/DiT:{actual(dit):.5f} "
                      f"last={name}->{BIT_OF_K[next_k]:.2f} cost={cost:.4e} "
                      f"selection={selection_reason} "
                      f"elapsed={time.time()-started:.0f}s", flush=True)
            if since_refresh >= args.refresh_every or not deltas:
                base, deltas, bases = pool.measure(state, floor)
                since_refresh = 0
                history.append({"step": step, "proxy": base,
                                "base_by_suite": bases, "bytes": total_bytes(),
                                "nominal": nominal(), "actual": actual(),
                                "floor": floor})
                print(f"[final-cd] refresh step={step} proxy={base:.8f} "
                      f"by_suite={bases} deltas={len(deltas)} "
                      f"elapsed={time.time()-started:.0f}s", flush=True)

        final_proxy, _, final_bases = pool.measure(state, floor)
        if not args.max_steps:
            assert total_bytes() <= args.target_payload_bytes
        output = Path(args.out)
        save(output)
        vc_path = output.with_suffix(output.suffix + ".vc_only.csv")
        hqq_path = output.with_suffix(output.suffix + ".hqq_specs.txt")
        with vc_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "params", "bit", "k", "scope",
                             "payload_bytes"])
            for name in sorted(candidates):
                if state[name] in HQQ_SENTINEL_K:
                    continue
                writer.writerow([name, params[name], f"{BIT_OF_K[state[name]]:.2f}",
                                 state[name], kind_of(name),
                                 payload_bytes(params[name], state[name])])
        hqq_path.write_text(";".join(
            f"{name}={HQQ_SENTINEL_K[state[name]]}"
            for name in sorted(candidates) if state[name] in HQQ_SENTINEL_K
        ) + "\n")
        counts = {}
        for k in state.values():
            counts[str(k)] = counts.get(str(k), 0) + 1
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "allocator": str(Path(__file__).resolve()),
            "allocator_sha256": sha256(Path(__file__).resolve()),
            "output_csv": str(output.resolve()),
            "output_csv_sha256": sha256(output),
            "vc_only_csv": str(vc_path.resolve()),
            "vc_only_csv_sha256": sha256(vc_path),
            "hqq_specs": str(hqq_path.resolve()),
            "hqq_specs_sha256": sha256(hqq_path),
            "suites": list(SUITES),
            "workers_per_suite": args.workers_per_suite,
            "requests_per_suite": args.requests_per_suite,
            "candidate_counts": {"vlm": len(vlm), "dit": len(dit),
                                 "total": len(candidates)},
            "candidate_params": total_params,
            "exact_artifacts_preflighted": artifact_count,
            "target_payload_bytes": args.target_payload_bytes,
            "start_k": args.start_k,
            "final_payload_bytes": total_bytes(),
            "final_nominal_bpw": nominal(),
            "final_actual_bpw": actual(),
            "scope_actual_bpw": {"vlm": actual(vlm), "dit": actual(dit)},
            "final_proxy": final_proxy,
            "final_proxy_by_suite": final_bases,
            "state_counts_by_k": counts,
            "min_bit": args.min_bit,
            "moving_floor_checkpoint_step": args.checkpoint_step,
            "refresh_every": args.refresh_every,
            "scope_balanced": True,
            "scope_policy": args.scope_policy,
            "atomic_tie_global_steps": global_tie_steps,
            "packed_reconstruction": "artifact packed -> CUDA BF16 gather/mul/add",
            "vc_shift_bytes_per_layer": 2,
            "fixed_background": {
                "vlsa_24_per_suite": "HQQ4 group64",
                "vision": "HQQ4 group64",
                "embedding": "HQQ4 group64",
                "adapter": "off",
                "adaln": "allocation candidate (32)",
            },
            "hot_spec_sha256": sha256(HOT_SPEC_FILE),
            "history": history,
            "steps": step,
            "elapsed_seconds": time.time() - started,
            "smoke_max_steps": args.max_steps,
        }
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        atomic_json(manifest_path, manifest)
        print(f"[final-cd] FINAL proxy={final_proxy:.8f} by_suite={final_bases} "
              f"nominal={nominal():.6f} actual={actual():.6f} "
              f"bytes={total_bytes()} steps={step}", flush=True)
        print(f"[final-cd] saved {output} and {manifest_path}", flush=True)
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
