# GR00T–LIBERO execution runbook

Run these commands from the extracted source distribution, not from the installed
numerical wheel alone. This is the execution order and artifact contract; it is
not a bundled model or dataset. Model weights, captures with episode provenance,
fitted candidates and frozen background payloads are not included. See
[validation scope](../VALIDATION.md) for the checks actually completed.

## Recorded SACQ2 reference preset

`presets/gr00t_libero_sacq2_seed17.json` records an actual archived SACQ2
configuration selected by the author as the paper reference preset.
It is distinct from the validated 338-codebook software fixture.
The bundled allocation CSV has 338 candidates: 331 codebooks and seven scalar
slots. Downstream jobs use 111 frozen GPTQ payloads. Nominal BPW is
1.9996885382059801 over the candidate scope.

The preset supplies index settings, calibration settings (640 ordered requests,
576 training / 64 validation, 1,152 updates, codebook LR 1e-4, scale LR 1e-5),
and the pinned seed17 rollout settings. It preserves fitting/search provenance
as metadata; the executable reference route **reuses the exact allocation**.
Fresh fitting/allocation commands can also be prepared as described below.
Neither route downloads inputs or launches experiments automatically.

Prepare the external input bindings in the existing builder format. Omit the
preset-owned `allocation`, `job.index_settings`, `job.training` and
`plan.rollout` fields; the helper fills those fields and rejects conflicting
values. Model, processor, bank/background records, ordered observations,
workspace, output paths and genuine reset/noise/provenance inputs remain your
explicit bindings. It does not turn unknown episode provenance into a PASS.

```bash
python gr00t_libero/preset.py index --bindings index-inputs.json --output index-spec.json
python gr00t_libero/jobs.py index --spec index-spec.json --output-dir /absolute/new/index-job
# After actual index completion, bind those shard jobs and receipts:
python gr00t_libero/preset.py calibration --bindings calibration-inputs.json --output calibration-spec.json
python gr00t_libero/jobs.py calibration --spec calibration-spec.json --output-dir /absolute/new/calibration-job
# After actual calibration completion, bind its job and completion:
python gr00t_libero/preset.py evaluation --bindings evaluation-inputs.json --output evaluation-spec.json
python gr00t_libero/jobs.py evaluation --spec evaluation-spec.json --output-dir /absolute/new/evaluation-job
```

These commands prepare job specifications; execute them through the pipeline
commands below. The reference helper does not launch GPUs or choose a different
model, task subset, request ordering or scientific protocol.

## Preparing fitting and allocation from the reference settings

Use `preset_stages.py` to generate commands with the recorded scientific arguments.
Bindings contain **paths only**, not replacement hyperparameters. Example fitting bindings:

```json
{
  "workspace": "/absolute/prepared-workspace",
  "processor": "/absolute/processor",
  "models": {
    "libero_10": "/absolute/models/libero_10",
    "libero_goal": "/absolute/models/libero_goal",
    "libero_object": "/absolute/models/libero_object",
    "libero_spatial": "/absolute/models/libero_spatial"
  },
  "output_root": "/absolute/new/fitting-results",
  "vlm_requests": "/absolute/original-64-vlm-requests",
  "dit_requests": "/absolute/original-64-dit-requests"
}
```

```bash
python gr00t_libero/preset_stages.py fitting --bindings fitting-inputs.json --output-dir /absolute/new/fitting-plan
```

The plan contains 44 ordered commands: four metric collections, eight shared
VLM fits, and 32 suite-specific DiT fits. Its 8,128 candidate payloads cover all
eight K values. Use `stage-row.json` as the `fitting` stage in your execution
configuration (see pipeline commands below); its entrypoints refer to the
prepared workspace. After fitting finishes, build the bank with `bank.py
--build-spec /absolute/new/fitting-plan/inventory-spec.json`, as documented below.

For allocation bindings, retain `workspace`, `processor`, and all four `models`;
choose a new `output_root`, and replace `vlm_requests`/`dit_requests` with
`bank_manifest` (the completed bank inventory JSON) and `allocation_requests`
(the original 40 captures, ten filenames prefixed with each suite name).

```bash
python gr00t_libero/preset_stages.py allocation --bindings allocation-inputs.json --output-dir /absolute/new/allocation-plan
```

Use that `stage-row.json` for the allocation stage. The original search goes to
**1.25 nominal bits**; SACQ2 uses its intermediate **`allocation_ckpt2.00.csv`**,
not the final `allocation.csv`. This preserves the recorded search trajectory.
The downstream `preset.py` helper deliberately reuses the bundled archival CSV;
a newly computed allocation is a separate artifact and must be explicitly bound
through the ordinary job builders if it differs. Fresh commands do not guarantee
identical allocations with different input artifacts. Original captures, models,
frozen background payloads and scientific provenance are still external inputs.

## Before starting

Prepare the policy and simulator environments using [ENVIRONMENT.md](ENVIRONMENT.md),
then materialize the pinned public source trees using the commands in
[README.md](README.md#prepare-sources). Keep the source directory separate from
the new prepared workspace and run output directories. Prepared workspaces
contain absolute import paths: run `prepare` at the final destination rather
than moving or copying a prepared tree. Bind the evaluation `sim_source` to
that workspace's `gr00t_final_seed17_20260922/simulator.py`; do not reuse a
simulator template from another prepared directory.

Supply these inputs before launching model work:

| Input | Source and responsibility |
|---|---|
| Four suite checkpoints and processor | Obtain the exact checkpoints selected by your protocol. Record their immutable identities; another GR00T checkpoint is not an interchangeable substitute. |
| Development captures | Capture from the selected development episodes. Use [CAPTURE_INPUTS.md](CAPTURE_INPUTS.md) to preserve request order, original noise positions and episode provenance. |
| Development/evaluation episode lists | Supply separate lists for fitting, allocation, calibration train, calibration validation and rollout. Run the public sample audit against your explicit task registry. |
| Scientific protocol | Record actual settings and the genuine sample-audit receipt. The driver does not choose settings or grant scientific approval. |
| Fitting configurations | Explicit path bindings and original fitter arguments, including metric, layers, K values, statistics and random settings. CLI defaults are not a declaration of your protocol. |
| Frozen background payloads | Supply the selected payloads and bit precision used outside the SACQ codebooks. The index builder does not generate them. |
| Rollout inputs | Explicit tasks, official reset journal, saved noise table, simulator paths and the accepted worker profile. The current pinned evaluator supports the documented seed17 contract. |

Do not use a directory name as evidence of checkpoint, seed, layer scope or
sample identity. Historical files with unknown provenance can support an
explicitly approved diagnostic, but cannot establish held-out reproduction.

## Create the execution configuration

Create `execution.json` with all five stage keys, as shown in
[Explicit command configuration](README.md#explicit-command-configuration).
Use absolute paths throughout. Initially, downstream rows can be empty while
their inputs do not yet exist; only execute a stage after filling its complete
`commands`, `inputs` and `outputs` lists. Each command is an argv list, not a shell
command string. Shell variables and wildcards inside JSON are not expanded.

The `python` field selects the policy interpreter. The evaluation plan separately
selects the simulator interpreter. Activate the policy environment for commands
that deserialize tensor payloads. The orchestration and job-building tools
themselves use the Python standard library.

| Stage | Command entrypoint in the prepared workspace | Command argv |
|---|---|---|
| fitting | `release_stage.py` | `--stage-config`, absolute binding JSON, then explicit original fitter arguments |
| allocation | `release_stage.py` | `--stage-config`, absolute binding JSON, then explicit original allocator arguments |
| index | `gr00t_final_seed17_20260922/index_worker.py` | Absolute generated `job.json` |
| calibration | `gr00t_final_seed17_20260922/calibration_worker.py` | Absolute generated `job.json`; optional explicit torchrun launcher |
| evaluation | `gr00t_final_seed17_20260922/rollout_worker.py` | Absolute generated plan, explicit task-index string |

For every row, register its job/binding JSON as a hashed input. A builder's
`inputs.json` supplies its discovered file dependencies. Include the exact
expected completion files and payloads in `outputs`, not a directory or glob.
Downstream `from_stage` references work only for files listed in the upstream
driver completion receipt.

The command pattern for each ready stage is:

```bash
python gr00t_libero/pipeline.py check --config execution.json --stage fitting
python gr00t_libero/pipeline.py run --config execution.json --stage fitting
```

Replace `fitting` with the next stage name only after preparing the handoff below.
`check` is input preflight, not model execution. The driver runs commands
sequentially and is not a GPU scheduler; obtain the required resources through
your execution environment before running it. It does not continue monitoring
after its process exits.

## 1. Fitting to allocation

The fitting row may contain multiple explicit metric and VLM/DiT fitting
commands. Define the full intended candidate inventory first. A selected-K bank
is insufficient when allocation requests additional K values. Run the fitting
row and retain its output receipts.

Declare completed fit groups with schema `sacq_fitted_inventory_build_v1`:
each group supplies scope, suite, initialization, K, output directory and exact
layer names. See [Build allocation inventory](README.md#build-allocation-inventory-from-completed-fits).

```bash
python gr00t_libero/bank.py --build-spec /absolute/fitted-groups.json \
  --output-dir /absolute/new/bank-inventory --check-payloads
```

Bind the generated `bank.json` reference to `bank_manifest` in the allocation
binding JSON. Include that JSON and the inventory's input references in the
allocation execution row. Supply all four suite checkpoint paths and the exact
allocation request directory. The original allocator expects four-suite workers;
the release adapter does not make it a one-suite allocator.

A `--dry-run` validates inputs and reports initialization only. Do not register
that result as a completed allocation or feed it downstream as a target-bit
solution. For the actual stage, declare the allocation CSV and the required
companion outputs, then run `check` and `run` for `allocation`.

## 2. Allocation to index correction

Create the explicit index builder spec described in
[Build index and calibration jobs](README.md#build-index-and-calibration-jobs).
Reference the actual allocation CSV and selected candidate bank. Supply original
request positions, explicit index settings, target layers and frozen background
records. For layer shards, every shard retains the approved observation list.

```bash
python gr00t_libero/jobs.py index --spec /absolute/index-spec.json \
  --output-dir /absolute/new/index-job
```

Add each generated job command and its inputs to the index execution row. Keep
worker output directories distinct from job-specification directories. Run the
index stage. Its actual `COMPLETE.json` files and corrected payload hashes form
the calibration handoff; do not manufacture receipts for old loose payloads.

## 3. Index correction to calibration

Create the calibration spec with the same allocation/model bindings and actual
index shard job/completion references. Supply the full ordered `all_records`,
the original trainer/cache launcher references, and all explicit training fields.
Calibration's positional split must agree with the declared sample audit.

```bash
python gr00t_libero/jobs.py calibration --spec /absolute/calibration-spec.json \
  --output-dir /absolute/new/calibration-job
```

Register the generated job and dependencies, then run the calibration stage.
Retain `TRAINING_COMPLETE.json` and its hashed VLM, DiT and best-state outputs.
The selected best state may be step zero if optimization did not improve the
selection metric; successful process completion does not mean quality improved.

An identical job may resume its owned output directories. A changed job requires
new directories; do not transplant `refs.pt` or optimizer state between jobs.
Teacher references are inputs to calibration, not quantized checkpoints.

## 4. Calibration to evaluation

Prepare explicit tasks and genuine reset/noise inputs first. Build an evaluation
spec referencing the actual calibration job and completion, plus the matching
VLM/DiT specifications. Do not supply the derived `workshop_model` field.

Generate those specifications with `installation_specs.py` using the actual
calibrated allocation and explicit fixed-background templates, as described in
[Prepare an evaluation plan](README.md#prepare-an-evaluation-plan). A stale scalar
entry can conflict with a newly selected codebook. VLM and DiT use different
compiler selectors; do not fill both files with the same nominal bit value.

```bash
python gr00t_libero/jobs.py evaluation --spec /absolute/evaluation-spec.json \
  --output-dir /absolute/new/evaluation-job
```

After the plan exists, append its reference to an input-only stage protocol
extension. Preserve all base scientific settings, approval and sample-audit
fields, and the existing input prefix. Register that extension in the evaluation
row and place the exact same bytes at the rollout root's `protocol.json`.
See [Explicit command configuration](README.md#explicit-command-configuration)
for the extension rules and `manage_evaluation_receipts: true` option.

Register the plan, task file and dependencies. Use explicit task commands and
expected output files for the approved sample set. Run the evaluation stage;
let the driver manage its own execution interval. Do not handwrite a RUNNING
receipt to bypass a worker check. Execution FINISHED means commands and declared
outputs passed, not that benchmark coverage or accuracy was independently proven.

## 5. Aggregate and retain provenance

Build an expected-episode manifest from the actual selected tasks, plans and
episode identities. Select the intended `episode_micro` or `task_macro` rule
explicitly. Do not glob every result under an output directory into a final table.

```bash
python gr00t_libero/results.py --manifest /absolute/expected-episodes.json \
  --output /absolute/new/final-summary.json
```

Use `--allow-partial` only for a progress report. Preserve the protocol, source
receipt, job inputs, driver receipts, worker completions and expected-episode
manifest with the summary. Missing episodes cannot be silently counted as
failures or excluded from the final denominator.

## If execution stops

Inspect the affected stage's `pipeline_receipts/<stage>.json` and numbered logs.
A nonzero command, missing declared output or input drift blocks downstream
execution. Establish process termination before resolving stale RUNNING records;
a failed status query is not evidence that a worker stopped. Do not rewrite
completed-stage inputs to force continuation. Worker checkpoint resumption and
driver completion bookkeeping are separate mechanisms.

A changed evaluation plan produces a different protocol identity. Even after a
failed launch, use a fresh rollout root and rebuild its task/plan/protocol
references; do not overwrite or delete the previous execution receipt. The four
completed upstream stages can remain unchanged when their inputs and outputs
retain the same identities. A terminal failed receipt is not permission to
replace it with a different protocol.

## Optional final A8 kernel source preparation

For source inspection and isolated staging, follow the
[acceleration integration guide](acceleration/README.md). This step does not
change the A16 pipeline above or authorize a new experiment. The A8 hook requires
the separately supplied compatible INT8 bootstrap and its model/input contracts.
Do not enable inference-only kernels in fitting or calibration.
