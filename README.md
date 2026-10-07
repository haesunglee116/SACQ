# SACQ — GR00T–LIBERO

GR00T–LIBERO implementation of SACQ. This repository contains the numerical
core, model integration, execution tools and validation documentation.
The code has been uploaded to the project repository; public release and the
author-code license remain pending the authors' decision.

This supplement exposes the numerical implementation and selected experiment
source snapshots. It is **not yet an end-to-end reproduction release**. No model
weights, private replay captures, simulator assets, credentials, machine launchers,
Git history, or author contact information are included.

The first public release targets **GR00T–LIBERO only**. See
`gr00t_libero/README.md` for the expanded local dependency bundle, path-explicit
fitting/allocation adapter, five-stage driver, index/calibration/evaluation job builders, explicit result aggregation,
and remaining integration limits.
SimplER and OpenVLA-OFT integration are deferred to later releases.
Calibration data is currently external, not an accidentally omitted artifact.

A recorded SACQ2 reference preset and spec-preparation command are described in
[the runbook](gr00t_libero/RUNBOOK.md#recorded-sacq2-reference-preset). The author has selected the submitted SACQ2 configuration as the reference;
external inputs remain
required.

## Pipeline

```mermaid
flowchart LR
    A[External captures and statistics] --> B[Codebook fitting]
    B --> C[Bit allocation]
    C --> D[Index correction]
    D --> E[Codebook and scale calibration]
    E --> F[Policy rollout]
    F --> G[Explicit episode aggregation]
```

The CPU numerical core is tested independently. Model-level execution additionally
requires externally supplied models, data, runtime dependencies and explicit
per-experiment settings; the diagram does not imply turnkey reproduction.

The public five-stage driver completed a real-model diagnostic connecting one fitting artifact, a packed-bank
allocation step, 338-layer index correction, one calibration update and a full
50-episode rollout. This reuses an existing candidate bank and external inputs;
it does not establish full-bank fitting or paper-result reproduction. See
`VALIDATION.md` for the exact limits and resolved installation issues.

## What runs independently

`sacq/` contains source-extracted implementations of:

- normalized vector-codebook fitting with Euclidean, diagonal and full-block metrics;
- downstream SwiGLU diagonal weighting and GELU derivatives;
- full, uncentered input second-moment accumulation;
- fixed-codebook/fixed-scale index reassignment with second-order error compensation;
- differentiable codebook/scale reconstruction and integer index bit packing.

The numerical function bodies are extracted from the research sources. Packaging
does not introduce a replacement quantization algorithm. Synthetic tests are
software checks only; they are not new benchmark measurements.

## Installation and tests

Use Python 3.10 or later and install PyTorch appropriate for your platform, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m examples.cpu_pipeline
```

The test suite and tiny example use CPU tensors and do not load a VLA model or
contact a service. The example shows fitting, index correction, and eight
codebook/scale updates on random tensors; it is not a benchmark or paper preset.
Index correction is not guaranteed to improve every toy reconstruction objective.
See `VALIDATION.md` for the environment actually tested for this export.

## Source snapshots, not turnkey launchers

`reference_sources/` provides the metric collectors/fitters, actual multi-suite
allocator, calibration trainer, and a GR00T index adapter for inspection.
These snapshots still require project-specific helper modules, model runtimes,
replay manifests and artifacts not included here. Their anonymized
`/data/workspace` paths are **placeholders**, not working download locations.
Do not execute their historical CLI defaults as a final experiment protocol.
Older introductory comments/defaults may describe an earlier launch; final runs
used explicit overrides. This directory is not installed by the Python package.

`REFERENCE_DEPENDENCIES.json` lists statically observed imports, including missing
project helpers. This is an inventory, not a complete dependency lock; dynamic
file loaders and simulator assets also need resolution.

Only load trusted checkpoints. Research snapshot and resume loaders use
`torch.load(..., weights_only=False)` and may execute Python pickle payloads.
No untrusted downloaded checkpoint is needed for the included synthetic tests.

The standalone `QuantLinear` preserves the source's within-request weight cache.
Use `sacq.cache.reconstruction_cache_scope(model)` around each independent
request, with optimizer steps outside the scope. This clears reconstructed
weights on entry and exit, including exceptions, without changing parameters
or detaching the returned loss. For manual lifetime management use
`clear_reconstruction_cache(model)`. Never reuse a
stale reconstructed weight after changing codebooks or scales. The optional
FP32 gather-gradient path changes floating-point reduction order; it is not
claimed to be bitwise-equivalent training.

## Scientific boundaries

- Normalized fitting and reconstructed-weight error are different objectives.
  `scale_squared_weighting=False` retains the normalized-coordinate fitting
  convention. Do not silently enable it while reproducing structure-aware fitting.
- SwiGLU downstream weighting is diagonal; it does not retain correlations
  between coordinates within a codeword. It is not a full downstream Hessian.
- Index correction uses a full input second moment and updates indices only.
  Model-level fixed-quantized-input collection is in the adapter snapshot;
  the standalone solver does not select a model or calibration dataset.
- The code supports configurations beyond any one reported experiment.
  Seeds, background quantization, request counts and optimization settings must
  be taken from a verified per-experiment contract, not inferred from defaults.
- A replay request may contain multiple observations. Requests are not necessarily
  episodes or temporal sequences. No independence assumption is implied.
- Legacy development episode provenance is incomplete for some experiments.
  Diagnostic-only results must not be relabeled as held-out evidence.
- Packed tensor storage, model resident GPU allocation and inference peak GPU
  memory are distinct quantities. Bit packing alone does not establish a
  runnable low-memory inference implementation.

## Real-runtime validation status

Fresh policy and simulator environments passed real-model checks. The public
five-stage driver then completed a diagnostic using one newly fitted candidate
and an existing bank, a four-suite allocation step, 338-layer index correction,
one calibration update and 50 rollout episodes. Explicit aggregation found no
missing episodes. The source archive passed 186 tests at that release; current
preset additions bring the local suite to 195 passing tests.

This validates software integration within the tested scope, not full-bank
fitting, fresh-host container installation or paper success-rate reproduction.
See [validation evidence and limits](VALIDATION.md). The archived SACQ2 reference
preset has a different 331-codebook selection and has not been rerun by the new
preset command.

## Reproduction inputs and supported scope

The five-stage driver and explicit aggregation have completed a real-model
software diagnostic. Installation guides, input-preparation helpers and source/
wheel packages are available. The diagnostic reused an existing multi-K bank,
fitted one new candidate, ran one allocation move and one calibration update,
and completed 50 evaluation episodes. It is not full paper-result reproduction.

Scientific reproduction still requires the selected checkpoint, captures,
background artifacts and their episode provenance. These are external inputs,
not included in the repository. See [the runbook](gr00t_libero/RUNBOOK.md) and
[validation record](VALIDATION.md) for the exact supported path and limits.

The first release supports the pinned GR00T–LIBERO model and evaluation layout.
Other model layouts and seeds are not promised by the existing adapters.

Third-party materials retain their applicable license terms; see
`THIRD_PARTY_NOTICES.md`. Redistribution licensing for the original research
code remains to be confirmed by its rights holders. This draft does not invent
an open-source license or claim that all external artifacts may be redistributed.

## Distribution and automated checks

The installable distribution is named `sacq-research`; its Python import remains
`sacq`. The wheel contains the numerical package and notices. Use the repository
or source archive for model adapters, examples and tests.

```bash
python tools/release_check.py
python -m pip install build
python -m build
```

GitHub Actions is configured for CPU tests, the synthetic example, source checks,
package builds and an installed-wheel check outside the checkout. The workflow is triggered by pushes and pull requests. Its GitHub run status
must be checked in the Actions tab; local test results do not establish CI success.
Repository preparation status is recorded in `RELEASE_READINESS.json`.
See [draft release notes](RELEASE_NOTES.md) for the initial release contents and limits.
