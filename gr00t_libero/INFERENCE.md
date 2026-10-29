# Generate actions from a saved observation

This is the simulator-free **BF16 / SACQ A16** entrypoint. It loads the original
suite policy, optionally installs the calibrated SACQ overlays and frozen
background, and calls `Gr00tSimPolicyWrapper.get_action` on one captured request.
It preserves the entire request batch and options. There is no task rollout,
action truncation, fitting, timing loop or A8 acceleration in this entrypoint.

**Current validation:** bundle relocation, integrity checks and failure handling
have CPU tests. The newly added entrypoint has not yet passed a real GPU forward.
Earlier GPU checks used a different harness. A reference inference bundle has not
yet been published; the calibration release contains observations, not weights.

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

Until a downloadable reference bundle is published, the model owner or a user
who completed the pipeline must export one. Supply this explicit JSON:

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

## Scope of the remaining work

Publish representative inference bundles or equivalent immutable download
instructions; run the new entrypoint on GPU and compare with the original policy
path for the same input and RNG state. A8 bootstrap/runtime integration and
latency measurement are separate from this A16 loader. No new benchmark result
is claimed by adding these commands.
