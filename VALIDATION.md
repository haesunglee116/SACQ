# Validation scope

## A8 runtime preparation — 2026-10-30

- Full CPU suite: **214 tests passed**. Bootstrap staging/hash/path guards and unchanged kernel installation-order tests pass.
- Source distribution built and unpacked successfully; release audit and pipeline source staging pass.
- The portable Sage builder reproduced the original patched CUDA and binding hashes and compiled successfully with the recorded PyTorch/toolkit.
- All 40 recorded attention input files (16 Action self-attention, 24 Vision attention) gave byte-exact output against the existing FP32 current-stream extension on RTX 4090.
- No new inputs, fitting, rollout or latency measurement. Existing runtime reused; fresh-host installation and the full relocated model are not validated.
- Machine-readable evidence: [a8_runtime_validation.json](experiments/gr00t_libero/a8_runtime_validation.json).

The older entries below describe validation at their recorded dates.

## Final kernel source integration — 2026-10-29

- Full CPU suite: **204 tests passed**, including six acceleration integration tests.
- All 96 exported source files match hashes listed in the final experiment contract before path-only transformation. Eight frozen selection files retain their choices.
- Installer statement order matches the original selected worker's AST.
- Isolated source staging and refusal of source/selection tampering tested, including a forged receipt.
- Latency denominators and success-count aggregation checked. A16 uses its separate paired BF16 control.
- Source release audit and local documentation-link checks passed.
- Source archive and wheel built; source archive unpacking, hash verification and existing pipeline source staging passed.
- **No GPU rebuild, numerical rerun or success-rate rerun of the exported overlay.** The compatible full INT8 bootstrap remains external.
- Top-level owner license remains undecided; recommendation in LICENSE_REVIEW.md.


## Repository/data update — 2026-10-29

- Full CPU test discovery: **198 tests passed**, including three new dataset-verifier tests.
- Source release check: **175 files checked, no errors**.
- Calibration manifest: all 2,560 request positions, split counts, unique paths and sizes checked.
- Four observation archives: all payloads reread after compression and matched against the published SHA-256 file manifest.
- Local documentation links and Git whitespace checks passed.
- No scientific experiment was rerun and no inference/training implementation was changed.
- Dataset assets are uploaded to calibration-v1; all four GitHub asset sizes and SHA-256 digests match the verified local archives.
- Package builds were not rerun: the default Python lacks the build frontend.


## Public five-stage driver completion (2026-10-07)

The public `pipeline.run` entrypoint executed **all five stages**, each with
its own actual COMPLETE receipt: one VLM layer fitted at K=512; one packed-bank
allocation move over four suites; 338-layer index correction in four shards;
one calibration optimizer update; and the original full rollout worker over
50 official episodes of one Spatial task. The public results builder expanded
the expected set from the actual plan, and the aggregator found 50/50 records
with no missing episodes. Jobs 280419, 280433 and 280437 contain this chain.

The rollout's fixed-weight cache and CUDA Graph passed three full-action checks
and exact capture with unchanged RNG. This is a **software diagnostic**, not
held-out benchmark evidence or a quantization-quality result. An existing multi-K
bank supplied candidates other than the newly fitted layer. Allocation's HQQ
background and the downstream frozen GPTQ background are explicitly different
fixture inputs; full-bank training and identical-background paper reproduction
were not tested. Completion also does not assert bitwise repeatability across
GPU placement or equivalence to the preceding diagnostic's success count.

Two launch mistakes were caught and preserved: the plan referenced another
prepared workspace's simulator template; then a changed plan attempted to reuse
the old execution root. The corrected run bound the current template and used a
new rollout root, retaining the four completed upstream stages. No failing
receipt was relabeled and no scientific sample identity was fabricated.
Prepared workspaces contain absolute paths and must be prepared at their final
location. Exact private logs, driver receipts and aggregate are retained in
`dist/public_driver_chain_20261007/` (excluded from distribution).

## Connected packed-bank stages and installation selectors (2026-10-07)

Job **280326** ran one real allocation move from K=512 across all four suites.
The candidate manifest included one layer produced by the public fitting worker;
other candidates came from the user-approved existing multi-K bank. All 338
layers were codebook-backed. One action layer moved to K=256, yielding nominal
BPW 2.249689; this is not a completed target-1.5-bit allocation.

Job **280327** processed all 338 selected layers in four index shards in 78 s.
The calibration builder joined their completion receipts and the real trainer
performed one optimizer update. Validation error increased from 0.142075 to
0.143093, so the initial best checkpoint was correctly exported.

The full rollout worker exposed stale installation specs in the diagnostic
fixture. VLM codebooks must be omitted from the forced-bit specification (the
global overlay selector is 2); DiT codebooks require selector 16. These are
compiler selectors, not nominal-bit declarations. The public
`gr00t_libero.installation_specs` command derives both files from the allocation,
calibration job and explicit background templates. The evaluation builder rejects
conflicting codebook entries. Failed attempts are preserved; a fresh rollout
using generated specs completed installation and all **50 diagnostic episodes**
(job **280333**, 476 seconds, exit 0). The original worker wrote
`COMPLETE_DIAGNOSTIC`; the execution receipt is `FINISHED`. Fixed-weight caching
and CUDA Graph were enabled after three full-action checks and an exact capture
check with unchanged RNG. This is software validation, not benchmark evidence.

This diagnostic reuses archived requests with unknown episode provenance and
uses the existing frozen GPTQ background for index/calibration, while allocation
uses its original fixed HQQ background. It checks software handoffs, not an
identical-background scientific experiment or full-bank fitting from scratch.
Private receipts are under `dist/allocation_codebook_r2_20261007/`.

## Packed-bank runtime path correction (2026-10-07)

A subsequent K=512 codebook diagnostic (job 280325) exposed a second historical
directory assertion in the runtime decoder, beyond the already corrected
preflight. The public adapter now binds both checks to exact manifest identity
and checksum lookup; the legacy directory resolver retains its original checks.
The BF16 gather/multiply/add reconstruction is unchanged. Two regression tests
compare reconstructed values exactly and retain mutation/fallback rejection.
GPU retry **280326 passed**: all four suites evaluated packed K=512 candidates,
and one action layer moved to K=256. The reconstruction-path regression suite
passed; subsequent installation-spec tests bring the CPU suite to **186 tests**.

## Actual four-suite allocation step (2026-10-07)

User-approved disposable diagnostic job **280324** completed successfully on four
RTX 4090 GPUs in **520 seconds**. The public stage adapter invoked the original
allocator with four suite checkpoints, ten existing requests per suite and one
allocation move. All 338 candidate deltas were computed; final proxy measurement,
CSV and companion manifests were written and their hashes checked.

Mean proxy moved from 0.11277832 to 0.10968223. One language `v_proj` layer
changed from HQQ4 to HQQ3, leaving nominal BPW 3.998893. This is a software
execution check, not a target-1.5-bit result or evidence of quantization quality.
The alternative bank was preflighted, but the selected move did not numerically
use its codebooks. Packed-bank selection was subsequently verified in job 280326 (below). The
complete five-stage public-driver orchestration remains unverified. Private receipts are in ignored
`dist/allocation_one_step_20261007/`.

## Consistent index shards at calibration handoff (2026-10-07)

The calibration builder now rejects completed index shards with different
explicit correction settings (including damping, block columns, noise seed and
TF32 flags). Previously it compared requests and bank/model identities but did
not compare this settings object. Three synthetic regression tests cover a
consistent disjoint join, mismatched settings and missing settings. They do not
rerun index correction or calibration, and no numerical worker was modified.

## Actual multi-K bank allocation preflight (2026-10-07)

The prepared public stage adapter and original allocator completed a dry-run
with exit code zero. Exact manifest lookup checked **7,112 candidate artifacts**
(file SHA-256 and codebook size) for the original 338-layer layout and four
LIBERO suites. Separately, all 13,520 entries in the alternative bank had their
packed headers inspected. Only the required 7,112 entries received the allocator
preflight hash checks; this does not claim all-bank byte verification.

The alternative bank comes from a later seed6 metric-v2 fitting run. It is not
established as the submitted paper bank. The dry-run performed no model forward,
allocation move or GPU job, and was invoked through the stage adapter rather
than the five-stage driver. Its starting nominal 4.0 BPW is an initialization
report, **not** a completed 1.5-bit allocation. Evidence is retained in ignored
`dist/allocation_bank_discovery_20261007/`.

## Actual calibration-to-evaluation connection (2026-10-07)

The public evaluation builder consumed job 280111's actual training completion
and exported overlays. Job **280112** loaded those newly generated VLM/DiT
artifacts, produced finite policy actions, and executed **eight simulator steps**
for the archived Spatial task-1 / episode-0 fixture. Initial observation and
noise-manifest hashes matched the earlier fixture. This connects the actual
337-layer index output to calibration, evaluation-plan construction, model
installation and policy/simulator RPC.

The RPC harness still bypasses the full rollout driver and runs one action chunk;
this is not a completed episode, success-rate result, acceleration gate or
fitting/allocation-to-rollout reproduction. The exported checkpoint was the
initial best state selected by the one-update calibration fixture.


## Actual index-to-calibration connection (2026-10-07)

Job **280106** ran the index worker over all **337 codebook layers** in the
existing Spatial fixture, using one archived request, and finished in **220 s**.
Every output reported its index-only invariant check. The public calibration
builder consumed that actual completion receipt and all corrected artifacts.
Job **280111** then ran the real calibration worker with two archived requests,
one validation request and one training request, one optimizer update, codebook
LR `1e-4`, scale LR `1e-5`, and the existing frozen background. It finished in
**32 s** and emitted its completion receipt plus VLM/DiT overlays and best state.

The initial validation error was 0.282903 and the endpoint 0.283674, so export
correctly retained the initial best state. This tests the connected workers and
export behavior, not calibration quality or success-rate preservation. The
input bank had already been corrected in an earlier experiment; these disposable
outputs are not new paper weights. Full fitting/allocation-to-rollout validation
and execution through the public evaluation driver remain incomplete.


## Evaluation execution-receipt lifecycle (2026-10-07)

The full CPU suite has **174 passing tests**. The public driver can now manage
the execution receipt required by the pinned rollout worker, under an explicit
configuration option and an exact checked-protocol binding. Four new tests cover
RUNNING during invocation followed by FINISHED, failure cleanup, concurrent and
unresolved RUNNING rejection, and protocol identity drift. Invocations in these
tests are mocked. Real rollout execution through this newly connected driver
remains to be tested; the earlier policy/simulator RPC smoke does not prove it.


## Late-bound downstream protocol inputs (2026-10-07)

At this stage, the complete CPU suite had **170 passing tests**. Three additional driver tests
cover late input registration, preservation of earlier stage fingerprints,
rejection of scientific-setting/input-removal changes, and all five stages with
a generated evaluation plan. Subprocesses are mocked in this test: it proves
orchestration bookkeeping, not real-model connected execution. The pinned
rollout worker still requires the actual launch execution-receipt lifecycle to
be integrated with the public driver.


## Portable bank-to-allocation handoff (2026-10-07)

At this stage, the full CPU suite had **167 passing tests**. Inspection found that the
legacy allocator still asserted historical parent directory names even when
using the public exact bank manifest. The stage adapter now checks the same
candidate/rate/suite coverage through manifest identities and codebook headers.
Three tests exercise renamed paths, missing suites/rates, modified bytes and
incorrect codebook sizes. The allocation objective and update loop are unchanged;
these handoff tests do not claim a real-model allocation run.


## Explicit noise-input preparation (2026-10-07)

At this stage, the full CPU suite had **164 passing tests**. The public noise CLI calls
the unchanged bundled generator. Synthetic CPU tests compare every generated
tensor with the original episode/call draw order, check unchanged global RNG,
and reject existing output directories, missing settings and Torch version
mismatches. This does not establish CUDA or cross-runtime regeneration equality.
No scientific noise table or rollout was regenerated for this check.


## Public reset-input generation (2026-10-07)

At this stage, the complete CPU suite had **161 passing tests**, including four reset
contract-boundary tests.

Job **280074** completed the public reset-only helper in the fresh simulator
environment and public source assembly. All **50 Spatial task-1 initial
observations matched** the archived fixture, and regenerated RNG journal SHA-256
was identical: `6bdbd70ee94f79b22edf4e4255b3543eae136dc78d5921529cd3895e9e49838c`.
The helper reported 68.79 seconds, zero policy calls and zero environment steps.
Its reset-loop AST was compared with the original reset audit after the original
seed17/env1 wrapper transformations. Environment, Python and NumPy seeds are all
17; the old wrapper string substitution affected both Python and NumPy calls.
This proves reset preparation for that fixture, not all 40 tasks, a complete
rollout, success rates or the connected quantization pipeline.


## Public sample-audit preparation (2026-10-07)

At this stage, the complete CPU suite had **157 passing tests**. Nine new auditor tests
cover episode overlap across policy calls, validation-only leakage, task coverage,
duplicate rollout identities, missing provenance, changed manifest hashes, edited
receipts and failed CLI output. A pipeline integration test confirms that a
manifest changed after receipt creation blocks execution. These use synthetic
identities, not newly audited scientific datasets. The public auditor checks the
supplied registry and declared sample identities; it does not prove their origin
or independently certify canonical benchmark completeness.


## Actual fitting adapter (2026-10-07)

At this fitting check, the complete CPU suite had **147 passing tests**, including the five
ordered capture-input preparation tests.

Job **280070** completed in 23 seconds using the fresh policy environment and
public upstream source assembly. The actual `release_stage.py` VLM fitter
consumed one archived request and fitted language layer 0's down projection
with its explicit K=512 candidate, group64/vector4 and metric-v2 configuration.
The generated packed artifact then passed `bank.py` inventory construction and
the codebook-size payload check. Numerical fitting parameters were explicit in
the disposable fixture command; the resulting artifact is not a paper model.
This verifies a real fitting-to-bank handoff, not all candidate fits, DiT metric
collection, multi-suite allocation or complete pipeline reproduction.

## Public upstream source assembly and index execution (2026-10-06)

The current CPU suite has **142 passing tests**. Source preparation now exports
public GR00T commit `4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e` and LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`, then applies the bundled source
overlays. It no longer requires the private research-fork commit. The resulting
receipt covers **1,481 files**. Missing revisions fail without leaving a partial
workspace; mutable checkout files are not used in place of the pinned revisions.

With both fresh dependency environments, job **280055** passed the same
Spatial task-1 / episode-0 policy-to-simulator RPC fixture using this assembled
tree. Initial observation SHA-256 remained
`0fc1d5fa9b7f4abe9c2ac2cc0c0e135f420363cc9a79c0de1ff0717e3b633b61`.
The earlier assembly attempt exposed an omitted locally added LIBERO object
module; it is now included with provenance. No untracked simulator assets or
extra task files were copied. This is still one eight-step action chunk with
external archived checkpoints/reset/noise inputs, not a full episode or benchmark.

Job **280056** exercised the real index job builder and worker with one archived
request and two selected layers (language down projection and DiT feed-forward
output). Both corrected artifacts completed with index-only invariants checked.
The input bank was the already-corrected bank used by the earlier backward
fixture. Outputs are disposable software-test artifacts, not new paper weights,
an algorithm comparison or evidence of full-layer index/calibration completion.

## Fresh installation continuation (2026-10-06)

The current CPU suite has **139 passing tests**, including six installer and
simulator-configuration failure-boundary tests added after the 133-test run.
The policy installation steps now have an executable, dry-run-capable script.
Those constituent commands were tested previously; this is not a claim that
the entire new installer was rerun to create another policy environment.

A new Python 3.10.19 simulator venv contains **79 compatible distributions**.
The initial RPC attempt (job 280044) failed before simulation because robosuite
needed an undeclared `termcolor` import. Follow-up import checks also found the
shared `/tmp/robosuite.log` permission conflict and BDDL's `future` import.
The final dependency list and configuration helper address those failures.

Job **280047** then passed with both fresh policy and simulator venvs: the
archived Spatial task-1 / episode-0 reset observation matched exactly, SACQ
produced finite actions, and RPC executed eight environment steps and closed
normally. The original saved policy noise and reset journal were reused.
This remains a one-chunk software check, not a completed benchmark episode,
accuracy result, five-stage reproduction or acceleration validation. Existing
host drivers, checkpoints, prepared GR00T and locally modified LIBERO/assets
were reused; the LIBERO source limitation is documented in the environment guide.

## Working-tree cleanup (2026-10-06)

The initial cleanup driver suite had 18 passing synthetic tests, including required job
fields, duplicate stage-config rejection, workspace consistency, calibration
counts, authorized evaluation-plan registration, matching task indices, and
failure receipts for spawn errors, nonzero exits and missing outputs.
These tests use mocked subprocesses and do not launch scientific experiments.
Numerical implementation files and archived worker sources are unchanged.
Source preparation into a fresh temporary workspace also passed, including the
exported-source hash checks. An additional full-suite run in the shared Python
3.10.19 / PyTorch 2.7.1+cu128 / NumPy 1.26.4 environment was stopped during
`test_calibration_gradients_and_decode` after prolonged filesystem I/O waits;
only the preceding bit-packing test had completed. This run is not counted as
a full numerical-core validation or a numerical failure.

## Job-builder validation (2026-10-06)

The index/calibration configuration builder adds 16 synthetic tests. Together
with the 18 driver tests, all **34 tests passed** using system Python 3.12 and
only the standard library:

```bash
python3 -m unittest discover -s tests -p 'test_[jp]*.py' -v
```

The builder tests exercise exact allocation-to-bank matching, original request
positions, corrected-bank linkage, derived layer counts, changed file rejection,
missing/overlapping shards, mismatched source jobs, index-only completion evidence,
and actual CLI output generation with refusal to overwrite an existing directory.
Tiny text files stand in for weights; these tests deliberately do not deserialize
checkpoints, execute numerical calibration, or establish scientific validity.
No archived worker or numerical implementation was changed in this step.

## Portable adapter and packaging preparation (2026-10-06)

The current complete CPU suite has **133 passing tests** in Python 3.11.14,
PyTorch 2.14.0+cpu and NumPy 2.4.6. This includes the 13 numerical-core tests
that were previously interrupted in the shared GPU environment, plus driver,
job linkage, explicit training configuration, evaluation-plan preparation,
static runtime/fitter equivalence, checkpoint copy tests, and explicit episode
aggregation tests. Additional checks cover manifest-driven installation counts
and per-layer frozen precision, while retaining the original installation math.

The active calibration worker uses an explicit trainer callback and output
path instead of source-string insertion. Removing only those integration
extensions produces the same normalized trainer AST as the submitted snapshot.
The materialized rollout and simulator functions similarly match the original
composed function ASTs after removing the explicit operational parameters.
These comparisons establish source-operation preservation, not GPU numerical
or renderer equivalence; a real-model check remains separate.

Source and wheel builds succeeded. The wheel was installed into an isolated
target directory, imported outside the checkout, and passed all 13 core tests.
An extracted source archive passed the 85-test suite before the explicit bank
inventory was added. Packaging
includes the integration data manifests and static runtimes but excludes model
weights and replay captures. GitHub CI is configured but has not been run remotely.

The CPU example also ran through fitting, index correction and eight calibration
updates. Its random-tensor losses are demonstration outputs, not paper results.
The source scanner checks Python syntax, exported-source hashes and known local
artifact/credential patterns. It is not a proof of licensing or complete secret
absence. The repository and owner-code license remain deferred by the owner;
no top-level license, remote, or publication approval has been invented.

## Historical submission validation

The following records describe the submitted snapshot, not a new end-to-end
validation of this working tree.


This export was tested with Python 3.11.14, PyTorch 2.14.0+cpu and NumPy 2.4.6
in a separate CPU environment. This is the **export-test environment**, not
the environment claimed for the original GPU experiments.

Thirteen synthetic unit tests passed:

1. Euclidean, diagonal and full-block fitting paths.
2. Index-only reassignment and preservation of fixed payload tensors.
3. Full second-moment accumulation across all input rows.
4. Damped inverse-factor construction and invalid-input rejection.
5. SwiGLU downstream diagonal weighting.
6. GELU derivative against autograd (exact and tanh variants).
7. CPU index-packing round trips across byte boundaries.
8. Packed reconstruction equality and two differentiable calibration updates.
9. Inverse-factor construction for correlated inputs.
10. Interrupted index correction, exact resume and provenance-mismatch rejection.
11. Separate Q/K/V codebook preservation in fused payloads.
12. Deterministic CPU full-block fitting on identical inputs.
13. Optional FP32 gather forward equality and a simple gradient check.

Seven additional synthetic driver tests check missing job rejection, operation
mismatch, changed or unhashed stage configurations, prior configuration/output
drift, and a mocked two-stage run using an upstream output reference. These tests
do not launch a model and their synthetic audit declarations are test fixtures.
The total is 20 tests.

All tests are run after extracting the ZIP into a fresh temporary directory.
The thirteen numerical core tests were also run outside the source tree against
a non-editable installed package; the driver tests require the supplementary tree.
Python source syntax and ZIP integrity are checked. Bytecode, Git metadata,
weights, replay data and internal export provenance paths are excluded.

These tests do not establish end-to-end action equality, simulator success
rates, CUDA equivalence, peak/resident GPU memory, or benchmark reproducibility.
The standalone tests are not a substitute for those validations.

Only the stated CPU environment has been validated. The package's permissive
dependency bounds are not evidence that every permitted version works. For the
tested environment install PyTorch 2.14.0+cpu from its CPU distribution index and
NumPy 2.4.6. These versions are not asserted to be the original GPU runtime pins.

The tested numerical code is extracted without changing function bodies.
Reference snapshots have identifying workspace prefixes replaced by a neutral
placeholder. Automatic pattern scans cover known identifiers, private paths,
email addresses, IPv4 addresses and common credential formats. Such scans are
not a proof of complete anonymity; final human review remains appropriate.

## Source archive completeness

A packaging check found that the first source archive omitted a pinned `.txt`
layer-spec file even though the working-tree checks passed. The source manifest
rules now include these files. `tools/check_distribution.py` unpacks a trusted
local source archive, runs its own source/hash checks, and prepares a fresh source
workspace from that archive. CI runs this check after building distributions.
This covers data-file omissions that numerical unit tests alone cannot detect.


## Fitted-bank portability (2026-10-06)

Seven additional tests cover explicit shared/suite-specific candidates, missing
candidate rejection, duplicate/ambiguous identities, checksum caching and file
changes, manifest identity, codebook-size headers, and registration as a hashed
allocation-stage input. All 92 CPU tests pass. These tests use synthetic files;
no fitting, allocation, calibration or rollout experiment was launched.


## Dependency inventory consistency (2026-10-06)

Removed an outdated OFT source entry that was absent from the GR00T-focused
release. The release check now requires complete, unique reference-source
coverage, existing declared bundled reference modules, and existing in-bundle
paths in the unresolved-import inventory. Four tests cover missing, added,
duplicate and escaping paths. These checks validate inventory consistency;
they do not establish that external model or simulator dependencies import.


## Reconstruction cache lifetime (2026-10-06)

Added an explicit context manager around the unchanged source `QuantLinear`.
It clears cached reconstructed weights on entry and exit so independent requests
and optimizer updates cannot accidentally reuse the prior request's cache.
Parameters and gradient mode must remain unchanged within the scope. Three CPU
tests cover reuse within one request, fresh reconstruction after an optimizer
step, inference followed by training, backward after context exit, and cleanup
on exceptions in nested modules. The synthetic example uses this public API.
No historical trainer or quantization arithmetic was modified.


## Pinned runtime source preparation (2026-10-06)

The optional `prepare --gr00t-checkout` path exports the declared upstream
commit from an existing local Git object database, then applies public overlays.
Three synthetic Git-repository tests verify that uncommitted/untracked files
are ignored, overlays win without duplicate receipt entries, missing commits
leave no partial workspace, and existing destinations remain untouched. Every
resulting receipt digest is checked. This tests source assembly, not installation
or model execution of the actual GR00T runtime.


## Actual GR00T source assembly (2026-10-06)

The first check using the actual pinned local GR00T checkout failed because its
`AGENTS.md` is a symbolic link to `CLAUDE.md`. The exporter now materializes
internal links to regular archived files; external, missing, cyclic and directory
targets fail. Two synthetic tests cover internal materialization and rejection
of an escaping link. Repeating preparation against the actual reference commit
successfully produced 1,075 source files with zero receipt hash mismatches and
the policy module present. No Git metadata was copied. The temporary workspace
was removed afterward. Dependencies, submodules, models and simulator execution
were not tested or installed by this source-assembly check.


## Calibration cache identity (2026-10-06)

The original trainer's shared cache key includes request path, size and timestamp
but does not fully bind the model/training job. The exported worker now separates
cache roots by its exact job digest and binds fresh local/durable output directories
to that identity. Four tests exercise same-job continuation, rejection of changed
jobs or unowned legacy outputs, isolated cache roots and coincident output paths.
This does not change the trainer's reference computation, RNG, loss or optimizer.
The integration worker and its exported hash changed; archived sources did not.


## Initial SimplER integration (2026-10-06)

The separate-QKV runtime was copied verbatim into the numerical package. Four
CPU tests verify its source digest, equality to three independent ordinary
reconstructions, frozen indices and exact BF16 reconstruction after synthetic
optimizer updates/export/reload, and ordinary-format dispatch. Shared cache
lifetime management now recognizes this layer. These are format and arithmetic
checks, not validation of a SimplER model, selected protocol or success rate.


## First-release scope decision (2026-10-06)

The owner selected GR00T-LIBERO for the first release. The initial SimplER QKV
module, tests and interface notes were preserved in a separate local draft and
removed from this distribution. Its four tests remain historical draft evidence;
the current LIBERO/core distribution has 108 tests. The shared cache helper no
longer imports the deferred QKV module. This scope change does not alter existing
LIBERO algorithms, data or experiment outputs.


## DiT checkpoint binding correction (2026-10-06)

An execution-path review found that assigning `farm.CKPT` in the outer adapter
was ineffective: fitter `main` overwrote it with the historical suite default.
The public adapter now passes a keyword-only `checkpoint` parameter, applied
inside `main` after argument parsing. A regression test executes the actual
argument-parser/checkpoint-binding prefix with model I/O excluded. It verifies
both the explicit override and the unchanged direct-call historical fallback,
and checks the adapter forwards the supplied model path. The source AST
comparison restores only the explicit checkpoint expression and strict-metric
hook and still matches the original fitter's arithmetic. No historical fitting
job was run or modified by this fix.


## Fitting-to-allocation inventory builder (2026-10-06)

Five tests exercise explicit candidate expansion without glob-selected scope,
missing and duplicate candidate rejection, path-name validation, DiT filenames
and optional codebook header checks, and CLI output consumed by `PackedBank`.
The CLI test checks every emitted file digest and verifies that rebuilding into
an existing output directory does not overwrite it. Inputs are synthetic;
this does not establish completeness of any real experimental candidate bank.


## Allocation-bank to worker-job connection (2026-10-06)

Four tests verify that direct manifest selection produces the same index job
as the explicit selected-artifact input, that completed-index calibration linkage
also produces the same job, that ambiguous/implicit selection is rejected, and
that suite mismatches cannot silently choose another suite's DiT candidate.
Unused candidates are not opened. This validates file selection and linkage,
not fitting accuracy or experimental sample coverage.


## Rollout-plan to aggregation linkage (2026-10-06)

Four tests cover expansion of a synthetic pinned-profile plan into 50 exact
expected episodes, consumption by the existing final aggregator, changed-plan
and duplicate-task rejection, incompatible condition rejection and unsupported
protocol rejection. The synthetic completed records establish schema linkage,
not actual simulator execution or success rates. The builder does not infer
benchmark-wide task coverage from the supplied plan subset.


## Explicit model-worker index settings (2026-10-06)

The public index worker now reads damping, block size, noise seed base, moment
chunk size, thread count and TF32 flags from a required `index_settings` object.
A reference profile transcribes the previous constants. Three tests cover
reference/overridden values, invalid or missing settings and worker call-site
bindings. Existing job-builder and pipeline tests also cover the required object.
The compensation solver and accumulation implementation are unchanged; no real
index correction or experiment was executed by this interface change.


## Model-runtime import attempt (2026-10-06)

A temporary source workspace was assembled from the actual pinned GR00T checkout.
An offline subprocess using the existing shared model Python environment attempted
to import torch, transformers, the GR00T policy and exported index/calibration
modules with GPU visibility disabled. It reached the 50-second timeout without
stdout/stderr. The cause was not established; model-runtime imports remain
unverified. No model was constructed and no experiment was launched. The process
terminated and the temporary workspace was removed.

The public `runtime_check.py` provides this import-only probe for a prepared
runtime with an explicit timeout. Three mocked tests cover subprocess options,
import failure reporting and timeout reporting. They do not count as successful
actual model-runtime imports or a fresh-environment installation test.


A subsequent bounded probe started Python 3.10.19 successfully, while importing
PyTorch alone exceeded 25 seconds. A separate 20-second probe captured a Python
stack after 10 seconds at `torch.__init__` loading the native `torch._C` module.
This narrows the observed delay to PyTorch native module loading, before the
GR00T adapters are imported; it does not identify the underlying cause or prove
that waiting longer would fail. Neither probe requested model execution.

The import probe now emits `python_started`, `import_started`, and
`import_completed` JSON events, flushing each event before moving on. One real
stdlib subprocess test confirms that an unavailable module is reported after the
last successful import; the three existing wrapper tests remain mocked.

Calibration job preparation now rejects non-`.msgpack` request paths, matching
the original loader glob. A regression test covers rejection before model
execution. This checks file selection compatibility, not payload validity.


## Extended dependency probe (2026-10-06)

A 150-second offline probe completed `import torch` (2.7.1+cu128), then timed
out while importing Transformers. Repeated tracebacks moved through Python
module loading; live process observations included `cl_sync_io_wait`. Thus the
initial torch import was slow, not a confirmed permanently stuck import. The
observed shared-filesystem wait contributes to the delay. Transformers, GR00T
policy/worker imports and model execution are still unverified. No experiment
or environment mutation was performed.

The runtime environment guide records source-declared GR00T requirements and
observed local HQQ/safetensors/dm-tree versions separately. It is not a newly
validated dependency lock or simulator installation recipe.


## Prepared-source import probe (2026-10-06)

After exporting the pinned GR00T revision and applying the release overlays into
a fresh temporary workspace, a 300-second offline import probe completed torch
in 65.55 seconds and Transformers in 230.01 seconds. It then began importing
`gr00t.policy.gr00t_policy` before the overall timeout. No stderr was emitted.
The two dependency imports passed; the policy and worker imports remain
unverified because nearly all of the total timeout was spent on dependencies.
No model was constructed and no benchmark was launched. A live process was
observed waiting in `cl_sync_io_wait`; this is operational I/O evidence, not
proof of numerical correctness or model/runtime compatibility.


The runtime probe's optional live JSONL progress file is covered by real stdlib
subprocess tests for successful imports, timeout with a started-but-unfinished
module, and preservation of an existing log. These tests do not import a VLA
model or establish model-runtime readiness.


## Completed prepared-source import check (2026-10-06)

The extended 900-second probe **passed** all requested imports: `torch`,
`transformers`, `gr00t.policy.gr00t_policy`, `index_worker`,
`calibration_worker`, and `fixed_model`. Total subprocess time was 864.93 seconds;
reported module import times were 98.99, 232.41 and 373.99 seconds for torch,
Transformers and GR00T policy respectively. Worker module imports were each
under 0.003 seconds. Stderr was empty and the child exited with code 0.

This supersedes the earlier timeout-only outcome for this selected module list.
The worker `main` functions were not called, so their function-local imports,
checkpoint loading, forward/backward computation and simulator execution remain
unverified. Importing the policy also started a PyTorch Inductor compile-worker
subprocess; no compilation or model computation was explicitly requested by the
probe. GPU visibility and Hugging Face offline flags remained set throughout.
The CPU suite and this import check are distinct evidence.


After observing an import-created compile worker, timeout cleanup was updated
to terminate the probe process group on POSIX. A Linux regression test starts a
sleeping child, triggers the parent timeout, and confirms the child is no longer
running. Existing import outcomes and progress-file tests also pass. The previous
six-module real-environment import pass predates this process-management change;
no extra model or scientific validation is inferred from this regression test.


## Real-model GPU forward smoke (2026-10-06)

Slurm job 279640 completed successfully on one RTX 4090 using a freshly
prepared public-source workspace and the pinned GR00T revision. The two
checkpoint shard hashes and copied quantization artifact hashes matched the
archived input identities. One existing LIBERO Spatial capture was used,
containing a batch of five observations; its original position-zero noise
seed was 7000. This was a software test, not a new benchmark or held-out result.

- Strict BF16 checkpoint loading completed.
- BF16 decoded-action forward returned finite output of shape `[5, 16, 7]`.
- The public installer loaded calibrated SACQ overlays with the declared
  337 SACQ layers and 105 frozen GPTQ background layers.
- SACQ decoded-action forward returned finite output of the same shape.
- Each condition was run twice with the same input and noise. Maximum absolute
  repeat difference was exactly zero for both conditions.

The private reproducibility receipt, exact fixture, harness, and live log are
retained under ignored `dist/gpu_smoke_20261006/`; no private model or capture
is included in the distribution. This checks prepared source, model loading,
installation and action calculation. It does not establish equality with an
archived action output, calibration gradient/update correctness, index-worker
execution, fitting/allocation execution, simulator readiness or rollout quality.
Existing built archives predate this documentation update.


## Real-model calibration backward smoke (2026-10-06)

Slurm job 279753 completed successfully on one RTX 4090. This disposable
software test used the same prepared public-source runtime, the original
trainer's `QuantLinear` and decoded-action forward, all 337 corrected SACQ
bank layers, and the frozen background from the previous installation smoke.
It used the first archived training capture (original request position 1,
noise seed 7001), containing five observations. Artifact hashes matched the
archived manifest. The BF16 teacher was computed before quantization.

The archived optimizer configuration was retained: AdamW, codebook LR 1e-4,
scale LR 1e-5, weight decay 0, and FP32 codebook gradients. One disposable
optimizer update was performed on that one captured batch. This deliberately
does not reproduce the archived six-request global-batch training schedule.

- All 674 codebook/scale parameter tensors received finite, nonzero gradients.
- All 674 parameter tensors changed after the optimizer update and stayed finite.
- All assignment indices were unchanged.
- Frozen model parameters had identical content hashes before and after the update.
- Post-update action output remained finite with shape `[5, 16, 7]`.
- Test-input MSE changed from 0.0138150351 to 0.0136141842. This is a plumbing
  check, not validation-set improvement or a success-rate result.

Private fixture, harness, receipt and log are retained under ignored
`dist/gpu_backward_20261006/`. This does not test the complete calibration
worker/controller, distributed gradient accumulation, checkpoint resume,
checkpoint selection, full fitting/allocation/index execution or simulator
rollouts. Existing distribution archives predate these validation notes.

## Simulator reset and rendering smoke (2026-10-06)

Slurm job 279864 passed using the separate existing simulator interpreter and
node-local LIBERO source/assets. The prepared public GR00T environment class
and official-state adapter loaded LIBERO Spatial task 1, official episode 0,
with reset seed 17 from the archived reset journal. The simulator state matched
the official initial state exactly. Both camera observations were finite
256x256 RGB arrays; the complete observation hash matched the archived record.

This checks construction, official-state reset and rendering for one software
fixture. No policy RPC, action step, full episode or success-rate evaluation
was performed. It is not a new benchmark result or proof of all-task coverage.
Private evidence is retained in ignored `dist/simulator_smoke_20261006/`.

## Checkpoint continuation and backward determinism (2026-10-06)

The default-runtime fresh-process continuation check did **not** pass bitwise
equality. It used the original trainer's atomic save helper and codebook/scale
plus AdamW state format. The input tensors, BF16 teacher, frozen parameter
hashes and pre-update loss matched between processes. Nevertheless, following
backward/update states and actions differed. A diagnostic replay observed a
maximum action difference of 0.02734375; this is a numerical action difference,
not a success-rate difference.

The discrepancy also occurs without saving or restarting: two backward passes
at the same fixed state within one process produced identical forward outputs
but different gradients in all 674 trainable tensors. The largest absolute
gradient difference in that diagnostic was 0.000732421875. Consequently this
failure cannot be attributed to checkpoint serialization alone.

A diagnostic-only configuration enabled all three controls together:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python;
- `FLASH_ATTENTION_DETERMINISTIC=1` before model execution;
- `torch.use_deterministic_algorithms(True)`.

With these controls, all 674 repeated gradient tensors were bitwise identical
(job 279913). This combined test does not isolate which individual kernel or
control accounts for each difference. The release's numerical defaults and
archived experiments were not changed. The diagnostic configuration is not
presented as equivalent in training speed or scientific outcomes to the
historical configuration. Private diagnostic evidence is retained under
ignored `dist/gpu_resume_20261006/`.

The subsequent fresh-process test with the same three deterministic controls
**passed** (job 279915): two uninterrupted updates and a save after update one
followed by a fresh-process update two produced exactly equal codebook/scale
parameters, complete AdamW state and decoded actions. This is a two-update
software fixture, not a full training schedule, distributed resume test, or
validation of the complete worker/controller checkpoint orchestration.


## Real policy-to-simulator RPC and action execution (2026-10-06)

Slurm job 279917 passed on one RTX 4090. A SACQ policy in the policy interpreter
communicated over a Unix socket with the public `simulator_profile.py` /
`simulator_runtime.py` subprocess in the separate simulator interpreter.
The original message serialization helpers, GR00T simulation policy wrapper,
official reset adapter and multi-step environment wrapper were used.

The fixture retained LIBERO Spatial task 1, official episode 0, environment
seed 17 and the existing saved policy-noise table (seed 20260824). The simulator
validated its 50-state reset journal and built renderer references; all 50
reference observation hashes matched the archive. Only episode 0 was passed
to the policy and stepped. The initial observation hash was
`0fc1d5fa9b7f4abe9c2ac2cc0c0e135f420363cc9a79c0de1ff0717e3b633b61`.

The policy produced finite action arrays for all seven action fields. The
first eight actions were transmitted through the existing RPC format and
executed by the simulator. The reported step count was exactly eight, and
the returned numerical observation arrays were finite. The close command
terminated the simulator process normally.

This is a one-call connection fixture, not a full rollout, success-rate result
or full-worker scheduling test. No fixed-weight caching/CUDA Graph acceleration
was enabled or validated in this software smoke. Private exact inputs, harness,
logs and receipt remain in ignored `dist/rpc_smoke_20261006/`. Public numerical
and evaluation source defaults were not changed.


## Clean-venv distribution validation (2026-10-06)

The current source was built into a new wheel and source archive under
`dist/clean_install_20261006/`. A new Python 3.11.14 virtual environment with
`include-system-site-packages = false` installed torch 2.14.0+cpu, NumPy 2.4.6
and the built wheel from cached distributions. The existing development venv
was not copied and no editable SACQ installation was used. Dependency checking
reported all 12 installed packages compatible.

- Outside both source trees, isolated Python imported `sacq` from the new
  environment's `site-packages`, and passed 13 core plus 3 cache tests.
- The freshly extracted source archive passed all 133 CPU tests in that environment.
- Its documented `python -m examples.cpu_pipeline` example completed successfully.
- Archive integrity/source checks passed for 128 files, and preparation staged
  66 adapter/helper sources without a supplied GR00T checkout.

The exact tested archive hashes, logs and receipt are retained in the ignored
validation directory. This verifies clean installation of the numerical package
and CPU/source tooling. It does not establish a fresh GR00T/CUDA/flash-attention
or simulator dependency installation, and does not run the full scientific
pipeline. The tested archives predate this added validation note.


## Clean policy-runtime venv and real GPU forward (2026-10-06)

A new Python 3.10.19 environment was installed from distributions and official
source, without copying the existing policy venv or enabling system site
packages. All 140 original distribution versions matched after installation.
The missing DeepSpeed dependency `nvidia-ml-py==13.615.71` and the SACQ wheel
were added; dependency checking passed for all 142 installed packages.

Slurm job 279954 completed in 19 seconds on an RTX 4090. The fresh environment
loaded the prepared public sources and existing checkpoint, installed the
calibrated SACQ weights, and passed both BF16 and SACQ action forward checks.
Both produced finite `[5,16,7]` outputs and zero maximum repeat difference for
the same captured input and noise. Torch, Transformers, FlashAttention and
SACQ imports were asserted to originate in the new venv.

Installation issues were exposed and resolved: the remote FlashAttention
wheel fetch timed out (downloaded locally before installation); the
fast-hadamard-transform isolated build lacked torch; its PyPI source archive
also omitted `csrc/fast_hadamard_transform.cpp`. Building official v1.1.0 source
at commit `1cc807efbd6cc001df359822d60bf6052dd66859` after installing torch
succeeded. A CUDA-index torchcodec local-version variant was explicitly
replaced with PyPI `0.4.0` to match the observed environment.

This is a fresh **policy venv** test on the existing host. The host driver,
CUDA 12.4 build toolkit, Python interpreter, pinned GR00T sources, checkpoints
and captures were reused. It is not a blank-host/container install, fresh
simulator install or a full pipeline test. Backward/resume/RPC were not rerun
in this new venv. See `gr00t_libero/environments/POLICY_INSTALL.md` for the
tested dependencies and required installation order. Private logs and exact
receipt are under ignored `dist/clean_policy_20261006/`.

## Single-request inference packaging (2026-10-29)

Added a simulator-free BF16/SACQ A16 entrypoint and relocatable model-input
bundle exporter. Four CPU tests cover relocation, refusal to overwrite, file
tampering, unlisted model files, quantized-artifact bindings, path escape, missing
settings and nested export rejection. These are packaging tests, not GPU forward
or numerical-equivalence evidence. The new entrypoint has not been GPU-validated.
The old smoke fixture references a node-local `/tmp` checkpoint unavailable on
the current host; no substitute checkpoint or benchmark protocol was selected.
No model bundle has been published by this change.

The complete CPU suite passed **208 tests**. The source distribution built
successfully and includes both entrypoints, the inference guide and new tests.
The publication source/license checker reported no errors.

## Standalone inference GPU validation (2026-10-30)

The new standalone entrypoint now passed BF16 and representative Spatial
W1.5A16 GPU inference on RTX 4090. A fresh prepared public source tree and
metadata-cleaned artifacts were compared against the original research policy
wrapper and installer using the previously published request `sp1_req0015`,
seed 7000, four denoising steps, and the exact existing checkpoint shard hashes.
All seven `[5,16,1]` action arrays matched byte for byte, maximum difference 0,
for both conditions. The existing policy runtime and CUDA host were reused.
No success rollout, timing benchmark, new fitting, A8 test or all-suite test was
performed. This supersedes the prior unverified status for this single example.

The published tensor export removes unused metadata only and checks equality
of every retained tensor, dtype, shape and installation field. The ready-to-use
downloader pins NVIDIA model revisions and all input hashes; three new CPU
tests exercise resume identity, unsafe archive rejection, wrong model rejection
and refusal to overwrite unrelated directories.

All **211 CPU tests** passed from the built source distribution in the node-local
policy runtime. Source-distribution content checks passed. The initial shared
runtime test process was stopped during filesystem I/O wait and was not counted
as a completed test run.

The published artifact archive was then downloaded anonymously through the
public downloader into a new directory, checked against the pinned SHA-256,
combined with exact-hash local copies of upstream models, and executed again
on GPU. All returned action arrays remained byte-identical. This exercises
public artifact download/restore; it is not a fresh download of gated Cosmos
weights or a new-host installation.
