# Run the final W1.5A8 model

This entrypoint restores the recorded **LIBERO-10 fixed-input diagnostic** and
runs the complete prepared model: Backbone plus four Action denoising steps.
It retains the original weight/activation/action/RNG/dispatch gates. It is a
latency reproduction command, not a LIBERO simulator or an arbitrary-input
serving API. The kernel configuration targets RTX 4090 (SM89).

The bundle contains the calibrated W1.5 overlays, frozen background tensors,
selection tables, one captured observation, the original policy noise, and the
original INT8 reference outputs. No fitting or rollout is performed. Original
GR00T/Cosmos weights are fetched from pinned NVIDIA repositories using your own
Hugging Face access; model files retain the included NVIDIA Open Model License.

## Environment

Use a separate Linux x86_64 Python 3.10 environment for A8. The A16 example uses
a different PyTorch environment. The recorded A8 versions are PyTorch
`2.11.0+cu129`, vLLM `0.24.0+cu129`, vLLM-Omni `0.24.1`, Transformers `5.19.0`
and Triton `3.6.0`. [external_runtime.json](presets/external_runtime.json) records
the exact vLLM wheel URL, Omni commit and GR00T-port source hashes.
[requirements.txt](requirements.txt) records the relevant observed dependencies;
it is not a fresh-host/container validation claim.

Install the pinned runtime in a new environment, using the CUDA 12.9 PyTorch
wheel index for the `+cu129` packages. Install vLLM from the recorded wheel URL,
and vLLM-Omni from commit `b9e9d236c3f78afd405119a5b686ebebeeb53984` of
`https://github.com/vllm-project/vllm-omni`. Keep the recorded versions when
resolving dependencies. From the SACQ repository root, with that environment
activated, check:

```bash
python -m gr00t_libero.acceleration.runtime_check
```

Prepare the public GR00T code as described in [the A16 guide](../INFERENCE.md#1-install-the-policy-runtime-and-prepare-sources):

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git /absolute/Isaac-GR00T
python gr00t_libero/pipeline.py prepare \
  --workspace /absolute/gr00t-workspace \
  --gr00t-checkout /absolute/Isaac-GR00T
```

A C++ compiler, Ninja and CUDA development toolkits are required. The recorded
SACQ kernel compiler is CUDA 12.4; the Sage build requires CUDA 12.8 for its FP8
instructions. These compiler toolkits are distinct from the CUDA runtime bundled
with PyTorch. First-time compilation can take several minutes. Keep all model
files, environments and build caches on local storage with sufficient free space.

## Build the attention runtime

The original raw-binding oracle and patched package are both required by the
numerical gates. The builders copy sources into new directories and verify the
pinned source/patch hashes. They do not modify the upstream checkout.

```bash
git clone https://github.com/thu-ml/SageAttention.git /absolute/SageAttention
git -C /absolute/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
python -m gr00t_libero.acceleration.build_sage_base \
  --source /absolute/SageAttention --output /absolute/sage-runtime \
  --cuda-home /absolute/cuda-12.8
python -m gr00t_libero.acceleration.build_sage \
  --source /absolute/sage-runtime/current_stream/source \
  --output /absolute/sage-fp32 --cuda-home /absolute/cuda-12.8
```

The `build_sage_base` command writes separate `original/BUILD_RECEIPT.json` and
`current_stream/BUILD_RECEIPT.json` files. The FP32 builder writes its own
`BUILD_RECEIPT.json`. Build receipts alone do not establish numerical agreement;
the model command below runs the original validation gates before timing.

## Download and run

Obtain Cosmos access through its official Hugging Face page and authenticate
with your own account as explained in the A16 guide. Then:

```bash
python -m gr00t_libero.acceleration.download --output /absolute/a8-inputs
```

The downloader checks the archive and every input hash. Existing exact upstream
files can be reused with `--local-model /absolute/libero_10` and
`--local-processor /absolute/Cosmos-snapshot`. It can resume its own interrupted
download, but rejects a directory belonging to a different manifest.

Create `runtime.json` with your local paths:

```json
{
  "python": "/absolute/a8-env/bin/python",
  "local_runtime": "/absolute/a8-build-cache",
  "gr00t": "/absolute/gr00t-workspace/libero_pro_eval/Isaac-GR00T-N1.7",
  "cuda_home": "/absolute/cuda-12.4",
  "sage_current_receipt": "/absolute/sage-runtime/current_stream/BUILD_RECEIPT.json",
  "sage_original_receipt": "/absolute/sage-runtime/original/BUILD_RECEIPT.json",
  "sage_fp32_receipt": "/absolute/sage-fp32/BUILD_RECEIPT.json"
}
```

Run from the SACQ repository root:

```bash
python -m gr00t_libero.acceleration.run \
  --bundle /absolute/a8-inputs \
  --workspace /absolute/a8-kernel-workspace \
  --runtime runtime.json --output /absolute/new-a8-result
```

The kernel workspace is created automatically and can subsequently be reused
if its complete source receipt still verifies. Each output directory must be
new. `--prepare-only` verifies inputs and renders contracts without launching
the model. Do not use Python `-O`, which disables the original assertions.

The result is `orders/forward/int8_fused.json`. It contains synchronized CPU wall
samples/median, CUDA event samples, clock samples, allocated/reserved memory and
the original exactness/dispatch checks. Reference/action arrays and component
validation receipts are under `bootstrap/`. These are raw model action tensors,
not simulator-ready postprocessed action dictionaries.

The recorded procedure uses batch 1, seed 42, five warmup and twenty measured
queries, four denoising steps and the same explicit noise from policy seed
20260824. The observation is LIBERO-10 task 0, environment seed 17, episode 0.
The 50 timed action checks are repetitions of this fixed input/noise, not 50
different observations. CPU wall timing includes synchronization and model
input copies/metadata checks; it excludes preprocessing, output validation,
profiling and simulator work. Clocks are recorded, not locked. Compare paired
runs before attributing small latency differences to code changes.

This does not establish a new success rate, held-out generalization, arbitrary
shape support or a hardware-independent speedup. Keep the historical latency
and earlier rollout results separate in reporting.
