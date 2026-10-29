# Final W1.5A8 inference kernel overlay

This opt-in integration exports the kernel implementation used by the recorded
**14.066 ms / 1.513x** fixed-input experiment. It includes 96 pinned source files,
eight frozen launch-selection tables, an installer preserving the selected
installation order, and a source-preparation tool. It is separate from the
existing A16 training/evaluation pipeline and does not change its defaults.

**Integration level:** source packaging and CPU staging/guard tests. The
relocated package has not been compiled or revalidated on a GPU. It requires a
compatible existing INT8 bootstrap; this is not a standalone model loader or a
one-command reproduction of the recorded timing. See [recorded experiments](../../experiments/gr00t_libero/README.md).

## What is included

| Component | Selected source |
|---|---|
| Native-order INT8 epilogue, stage-6 GEMM | `int8_ordered_epilogue_20261028_r1.py`, `codex_gemm_stage6_20261029_r1.py` |
| Frozen small/large GEMMs, split-K | `int8_small_m_frozen_20261028_r2.py`, `int8_large_m_frozen_20261028_r1.py`, `int8_splitk_frozen_20261028_r1.py` |
| Direct packed weight decode, vectorized stores | `codex_codebook_vec4rcpftz_*`, `codex_affine_vec8rcpftz_*`, `codex_qkv_vec4rcpftz_*` |
| Grouped FF decode with 128-thread blocks | `codex_ffdecode_vec4cta128_*` |
| Norm3 register packing, AdaLN producer fusion | `codex_norm3_regpack_*`, `codex_adaln_sharedmax_*` |
| Activation reuse, GELU/SwiGLU, FF-down residual | `int8_shared_padded_activation_*`, `codex_gelu_lookup_w32_*`, `int8_swiglu_lookup_fusion_*`, `int8_ffdown_residual_epilogue_*` |
| RMS, Q/K normalization and RoPE | `codex_language_rms_*`, `codex_qknorm_rope_*` |
| Attention preparation, scale/copy removal | `claude_int8_attention_path_*` (`c1=True`, `c2=False`, `c3=True`) |
| Vision normalization and direct segment quantization | `int8_vision_norm_producer_*`, `codex_vision_segment_producer_*` |

Exact filenames and original/exported hashes are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
The source tree preserves its relative research layout because several modules
load neighboring CUDA files or source-identity fixtures. Path placeholders are
the only changes to exported kernel files. The additional installer and staging
utility are new integration code. Embedded CUDA and PTX arithmetic is unchanged.

## Inspect and stage without a GPU

Run from the repository root:

```bash
python -m gr00t_libero.acceleration.prepare --check
python -m gr00t_libero.acceleration.prepare \
  --workspace /absolute/new-kernel-workspace \
  --local-runtime /absolute/local-runtime-root
```

Preparation verifies hashes, writes a new isolated workspace, copies the frozen
choices, and records rendered source hashes. It refuses to overwrite an existing
workspace. It does not launch Slurm, load a model, build CUDA, or alter scientific
settings. The local-runtime root replaces the original machine-local prefix for
build directories and pinned runtime paths; it must contain the corresponding
runtime tree when used on a GPU.

## Integrate into the compatible inference bootstrap

The explicit hook is:

```python
from gr00t_libero.acceleration.install import configure

with torch.inference_mode():
    counters = configure(
        workspace=prepared_workspace,
        bootstrap_contract=existing_approved_bootstrap_contract,
        trace_directory=trace_directory,
        bridge=loaded_INT8_r5_bridge,
    )
    # Continue the existing bridge.build(...) and its validation/capture gates.
```

The bridge must be loaded under the module name `rollout_bridge_r1`, as in the
[original worker](reference/final_worker.py). Its source is included for identity
checking in [reference/rollout_bridge_perrow_int8_r5.py](reference/rollout_bridge_perrow_int8_r5.py).
Its `build()` requires the external `bootstrap_policy_perrow_int8_r5.py` and its
approved model/input contracts. Those historical bootstrap and full-model graph
construction dependencies are **not yet packaged by this overlay**. Use an
existing verified runtime; do not substitute the A16 pipeline's model loader.

The installer checks the prepared source hashes, frozen selections, compatible
bridge, projection recipe, attention recipe, inference mode and SM89. It rejects
conflicting imported modules and repeated installation. Use one fresh process
per configuration. Never use Python `-O`, which disables the original assertions.
The returned counters are validation state, **not a PASS receipt**.

Before timing or rollout, retain the original worker's gates: decoded weight
bytes, actual activation bytes/scales, intermediate native outputs, whole-model
action bytes against the original INT8 reference, RNG state, dispatch coverage,
and allocated/reserved memory. Changing hardware, library builds, model shapes,
checkpoint or inputs requires corresponding validation. Source staging alone
never authorizes execution or certifies equivalence.

## External requirements and boundaries

- NVIDIA SM89 / RTX 4090 is the recorded target. These choices are not generic
  autotuning recommendations for another GPU or model.
- Matching PyTorch/CUDA, Triton, Transformers, diffusers, GR00T, vLLM and
  vLLM-Omni runtime; source guards deliberately reject incompatible implementations.
- The existing modified SageAttention build with FP32+FP32 accumulation and
  current-stream handling. Installing an arbitrary stock version is insufficient.
- CUTLASS headers, model checkpoint, compressed SACQ/background payloads, original
  INT8 reference actions, observation and explicit policy-noise fixtures.
- Existing backbone/whole-model graph and canonical-noise bootstrap.

[DEPENDENCIES.json](DEPENDENCIES.json) records static imports. Some inherited
helpers contain standalone probe entry points; those probes are not supported
public launchers. No model weights, compiled libraries, source trees of external
packages or full experimental artifact directories are bundled.

This is inference-only. Do not enable frozen-weight reuse, forward monkeypatches
or inference CUDA Graphs in codebook fitting or calibration training. The final
kernel combination was not rerun for success rate; see the separate latency and
success records rather than attributing a new rollout result to this export.
