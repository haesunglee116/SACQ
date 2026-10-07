# Release notes

## 0.1.0 — draft, not published

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

The owner has deferred the GitHub destination and the license for original
research code. No release tag, remote repository or published package is claimed.
These notes are a draft to accompany the first approved publication, not a
statement that redistribution rights or full reproduction have been finalized.
