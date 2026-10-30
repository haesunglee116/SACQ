# Generate actions from a saved observation

This is the simulator-free **BF16 / SACQ A16** entrypoint. It loads the original
suite policy, optionally installs the calibrated SACQ overlays and frozen
background, and calls `Gr00tSimPolicyWrapper.get_action` on one captured request.
It preserves the entire request batch and options. There is no task rollout,
action truncation, fitting, timing loop or A8 acceleration in this entrypoint.

**Validated example:** BF16 and representative Spatial W1.5A16 both generated
finite actions on an RTX 4090 and matched the original inference path byte for
byte (all seven action arrays, maximum absolute difference 0). This covers one
published request with five observations and a 16-step action chunk, seed 7000,
and four denoising steps. The existing CUDA host and policy environment were
reused. See [the validation record](../experiments/gr00t_libero/inference_validation.json).
It is not a new success-rate result or a validation of A8 acceleration.

## Ready-to-use reference inputs

After installing the policy environment and preparing sources in step 1 below:

1. Obtain access to [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
   through its official Hugging Face page and accept its terms using your own
   account. Authenticate in the policy environment with
   `python -c "from huggingface_hub import login; login()"`.
2. Download the exact inputs and run the example:

```bash
/absolute/policy-env/bin/python gr00t_libero/download_inference.py \
  --output /absolute/spatial-sacq-example

/absolute/policy-env/bin/python gr00t_libero/inference.py \
  --bundle /absolute/spatial-sacq-example --workspace /absolute/prepared-workspace \
  --output /absolute/new-action-output
```

No export JSON needs to be written for this reference example. The committed
[preset](presets/inference_spatial_w1p5a16.json) pins upstream revisions, file
hashes, the published artifact archive and all example settings. The archive is
available in [the inference release](https://github.com/haesunglee116/SACQ/releases/tag/inference-spatial-v1).
Original model weights come from NVIDIA's repositories, rather than being
relicensed or bundled in Git. Allow roughly 30 GB for inputs, download cache and
intermediates, in addition to the policy environment.

For BF16, give the downloader `--condition bf16` and a different output directory.
This example still downloads the same SACQ/example-observation archive, but the
BF16 runner does not install its quantized weights. The condition is recorded in
`inference.json`; do not edit a SACQ configuration to relabel its results.

The downloader can resume its own interrupted directory. It checks every file
and rejects a conflicting manifest. `--local-model /exact/checkpoint` and
`--local-processor /exact/Cosmos-snapshot` reuse existing files only when their
hashes match. A different checkpoint is not substituted silently. Run the
inference command only after `BUNDLE_READY` is printed.

Model artifacts retain the NVIDIA Open Model License included in the archive;
SACQ's source Apache-2.0 license does not replace those terms. Metadata containing
internal paths was removed from published tensors; all retained tensor values,
dtypes, shapes and installation fields were checked unchanged.

## 1. Install the policy runtime and prepare sources

Use a Linux CUDA host and the [policy environment installation recipe](environments/POLICY_INSTALL.md).
That recipe includes an installer script and pinned dependencies. The numerical
core's `pip install -e .` alone is insufficient. No simulator installation is
needed for this command.

Obtain the public GR00T checkout and prepare it at its final destination:

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git /absolute/Isaac-GR00T
python gr00t_libero/pipeline.py prepare \
  --workspace /absolute/prepared-workspace \
  --gr00t-checkout /absolute/Isaac-GR00T
```

The checkout must contain pinned revision
`4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e`. The preparer uses that revision and
bundled overlays, not the checkout's current HEAD. Use a full clone.

## 2. Obtain or export an inference bundle

A bundle contains local `model/` and `processor/` directories, a captured
`request.msgpack`, `inference.json`, and (for SACQ) the calibrated overlays,
installation specifications and frozen background tensors. Every file is hashed.
File symlinks in Hugging Face snapshots are materialized. Directory symlinks
are rejected. Export requires enough disk space for another copy of all inputs.

The reference download above is ready to use. To export a different model after
completing the pipeline, supply this explicit JSON:

```json
{
  "schema": "sacq_inference_export_v1",
  "condition": "sacq",
  "suite": "SUITE_MATCHING_YOUR_CHECKPOINT",
  "model": "/absolute/suite-checkpoint",
  "processor": "/absolute/Cosmos-Reason2-2B-snapshot",
  "seed": "REPLACE_WITH_EXPLICIT_INTEGER",
  "denoising_steps": "REPLACE_WITH_CHECKPOINT_INTEGER",
  "request": {"path": "/absolute/request.msgpack", "sha256": "EXACT_SHA256"},
  "workshop_model": "COPY_THE_COMPLETE_OBJECT_FROM_YOUR_CALIBRATED_EVALUATION_PLAN"
}
```

The strings marking integers and `workshop_model` are documentation placeholders,
not executable defaults. Use the exact `workshop_model` from the plan generated
by `jobs.py evaluation`: it contains `vlm`, `dit`, `vlm_spec`, `dit_spec`, `gptq`,
`expected_sacq_count`, and `expected_gptq_count`. The export preserves those
installation choices and verifies the existing tensor hashes. It does not
choose a bit rate or replace a missing model with another checkpoint.
For BF16, set `condition` to `bf16` and omit `workshop_model`.

```bash
python gr00t_libero/inference_bundle.py \
  --spec /absolute/export-inputs.json --output /absolute/new-inference-bundle
```

The original files are untouched. A failed export retains `EXPORT_FAILED.json`
and cannot be reused as a completed bundle. Use a new output directory for retry.
The completed bundle can be moved without rewriting its internal paths.
A hash inventory checks file identity, not authorship: obtain bundles from a
trusted provider, since the original quantized tensor loader uses PyTorch pickle.

The seed controls this single-request example's Python/NumPy/PyTorch random
state immediately before the call. It is **not** the archived per-episode noise
stream used for reported success rates. Denoising steps are checked against the
loaded model; the runner does not silently change them. A demo bundle does not
constitute approval to reuse its observations for held-out evaluation.

## 3. Check inputs, then generate actions

From the source release root:

```bash
python gr00t_libero/inference.py \
  --bundle /absolute/inference-bundle --workspace /absolute/prepared-workspace --check

/absolute/policy-env/bin/python gr00t_libero/inference.py \
  --bundle /absolute/inference-bundle --workspace /absolute/prepared-workspace \
  --output /absolute/new-inference-output
```

The first command uses only the standard library. `INPUTS_VERIFIED` means file
and configuration checks passed; it is not a GPU validation result. The second
requires a CUDA GPU visible as device 0 (select the physical GPU using your
scheduler or `CUDA_VISIBLE_DEVICES`). Run it in a fresh process without inherited
quantization/capture environment overrides and without `python -O`.
All model loading is offline; a missing dependency must be supplied explicitly.

Outputs:

- `actions.npz`: named action arrays with the original batch and action horizon.
- `result.json`: completion status, shapes, input/source identities, model
  installation audit, seed, denoising steps, GPU and torch version.
- `failure.json`: failure details if execution fails after creating its output.

Existing output directories are never overwritten. A failure or a partial output
is not success. Only `result.json` with `COMPLETE` records a returned finite
nonempty action output. It does not establish equivalence to the historical
harness, success rate, throughput or the final A8 speedup.

## Supported scope

The published example completes the standalone BF16/SACQ A16 path for one
representative Spatial model. Additional suites and bit settings need their own
explicit artifacts and validation. A8 bootstrap/runtime integration and latency
measurement remain separate work. The earlier success-rate and speed tables are
not updated by this software validation.
