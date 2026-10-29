# Release notes

## Owner-approved license — 2026-10-29

- Applied Apache-2.0 to original SACQ code and documentation.
- Included the top-level license in source and wheel distributions and declared the SPDX identifier in package metadata.
- Retained third-party terms; dataset/model licensing and repository visibility are unchanged.


## Final inference source integration — 2026-10-29

- Added a configuration/source/evidence map for final W1.5A8 latency and the earlier W1.5A8 success evaluation.
- Added 96 original-contract-pinned kernel/helper/fixture files and eight frozen selection tables.
- Added isolated source staging and an opt-in installer preserving the selected installation order.
- Retained output/RNG assertions, source guards, inference-only restrictions and original hash provenance.
- Recorded the external bootstrap requirements and lack of GPU revalidation after packaging.
- Added an owner-facing Apache-2.0 recommendation; license and repository visibility remain unchanged.


## Repository update — 2026-10-29

- Reorganized the entry README around installation, data and the execution runbook.
- Added exact manifests for 2,560 calibration requests across four LIBERO suites.
- Added a standalone dataset checksum verifier.
- Observation archives are available in the calibration-v1 GitHub Release; uploaded sizes and SHA-256 digests match the verified originals.
- Model kernels and scientific experiment settings are unchanged.

## 0.1.0 — research source release

The first release packages the SACQ numerical implementation and the selected
GR00T–LIBERO research pipeline. SimplER and OpenVLA-OFT integration are deferred.
The submitted supplement and historical experiment directories are preserved
outside this working tree.

### Included

- CPU-usable vector-codebook fitting, structure-aware metric helpers, input
  second-moment statistics, index correction and differentiable codebook/scale
  reconstruction.
- A synthetic example and CPU tests that require no model weights or simulator.
- GR00T–LIBERO adapters connecting explicit fitting/allocation inputs, corrected
  index shards, calibration outputs and evaluation plans.
- Builders for fitted-bank inventories, stage jobs and expected result manifests.
- Explicit index/calibration settings and source/artifact receipts.
- Public upstream GR00T/LIBERO source assembly with versioned research overlays,
  separate tested dependency lists, policy installer and simulator configuration helper.
- Ordered capture-input preparation that retains explicit per-observation
  episode provenance without choosing samples or manufacturing approval.
- Reconstruction-cache lifetime helpers, job-specific calibration caches, and
  an import probe with optional live progress logging.
- A source archive, an installable numerical-core wheel, and a GitHub Actions
  workflow for software tests and package checks.

### What a user still supplies

The selected calibration set is documented in [the dataset guide](data/libero_calibration/README.md).
Observation archives are distributed separately from the source checkout.

Model checkpoints and processors, calibration captures and their provenance,
fitted or frozen background artifacts as applicable, simulator assets, and
explicit experiment/episode manifests are external inputs. See
[the input guide](gr00t_libero/CAPTURE_INPUTS.md) and
[the environment guide](gr00t_libero/ENVIRONMENT.md).

The model adapters retain the selected research layout and pinned evaluation
contract. This release is not a generic adapter for arbitrary model layouts or
benchmarks. It is not yet a turnkey fresh-environment reproduction package.
Software tests and successful imports do not establish benchmark accuracy,
held-out evaluation status, GPU memory savings or full model compatibility.
See [validation evidence](VALIDATION.md) for what was actually checked.

### Distribution contents

The wheel installs `sacq` and its notices. The repository/source archive also
contains examples, tests, model adapters, reference snapshots and documentation.
Use the source archive or repository for the GR00T–LIBERO pipeline.

No model weights, calibration captures, private credentials or simulator assets
are included. Third-party terms remain separate; see
[third-party notices](THIRD_PARTY_NOTICES.md).

### Publication status

Source code is available in the SACQ GitHub repository. Calibration dataset assets are available in the calibration-v1 release.
No package-index release is claimed by this update. Original SACQ code and
documentation use Apache-2.0; third-party terms are retained.
