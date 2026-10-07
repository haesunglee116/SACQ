# Third-party materials

The numerical vector-sampling and Euclidean k-means helpers were extracted from
the research checkout's modified `gr00t/eval/cosmos_quant_shim.py`. The full
GR00T repository and its weights are not redistributed. The checkout's Apache
2.0 license text is retained in `third_party/Apache-2.0.txt`; extraction and
repackaging are modifications for this supplement. Provenance hashes are in
`SOURCE_HASHES.json`. These helpers should not be represented as a separately
verified, unmodified upstream implementation.

PyTorch is an external dependency and is not bundled. Model-level snapshots
reference GR00T, HQQ and other packages that must be obtained separately under
their own licenses. Import statements do not grant redistribution rights to
their models, datasets or code. No third-party author or copyright notice has
been intentionally removed for anonymization.

This is a publication preparation tree; the owner has deferred the license decision. Complete the dependency/license inventory before a
general-purpose public release; do not assume one license covers all snapshots.

The GR00T–LIBERO integration also contains modified checkout overlays for
`cosmos_quant_shim.py`, `rollout_policy.py` and `run_gr00t_server.py`, together with
the checkout dependency manifest and license. Additional runtime overlays retain
the research observation/reset adapter, environment mapping, model hooks and
request-capture helper. They are distributed as source
overlays, not the complete model repository. Private workspace prefixes have
been replaced for relocation; see the integration source manifest for hashes.

## Distribution boundaries

| Material | Included | License evidence / pending work |
|---|---|---|
| Extracted numerical helpers and modified GR00T overlays | Source | Bundled Apache-2.0 text and original checkout license are retained; source provenance is recorded |
| Original research algorithms, adapters, tests and documentation | Source | Owner-selected top-level license is pending |
| Full GR00T checkout and model weights | No | Obtain separately under their upstream terms |
| PyTorch, NumPy, Transformers, HQQ, dm-tree, safetensors | No | External runtime dependencies; this bundle does not relicense them |
| Modified LIBERO Python source overlays | 13 modified files and one required added module | Upstream MIT notice retained in `third_party/LIBERO-MIT.txt`, per-file provenance in `gr00t_libero/LIBERO_OVERLAY.json`; owner-code license decision also applies to original additions |
| Full LIBERO simulator/assets, tasks and calibration captures | No | Obtain/prepare separately; no data redistribution permission is asserted here |

The wheel installs only the `sacq` numerical package plus notices. The source
archive also contains the integration adapters, reference snapshots, tests and
documentation. A license notice in the wheel does not license unrelated external
models or private calibration data.

The LIBERO overlays preserve tracked Python changes from the research simulator,
including definitions for additional task variants. Their presence does not
extend the supported release benchmark beyond the four original LIBERO suites.
Additional untracked assets and task files are not included. Source preparation
exports the pinned upstream commit and applies these overlays explicitly; the
result must not be described as an unmodified upstream LIBERO checkout.
