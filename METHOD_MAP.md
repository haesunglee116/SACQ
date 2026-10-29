# Reading the implementation

| Stage | Standalone code | Model-level reference |
|---|---|---|
| Structure-aware fitting | `sacq/vector_quantization.py`, `sacq/metrics.py` | `reference_sources/metric_v2/` |
| Mixed-bit search | Source inspection only in this draft | `reference_sources/allocation/coord_descent_final_multisuite_2026_08_24.py` |
| Full input statistics | `sacq/statistics.py` | `reference_sources/adapters/` |
| Index correction | `sacq/index_correction.py` | `reference_sources/adapters/` |
| Action calibration | `sacq/calibration_layers.py` (layer component only) | `reference_sources/calibration/` |

`FullMoment.total / FullMoment.rows` gives the uncentered second moment.
The accumulator itself stores the sum, so callers must normalize explicitly.

`compensate_packed` accepts an explicit original weight matrix, input second
moment and fixed packed codebook payload. It returns a modified payload and
diagnostic report. Do not interpret the layer-output diagnostic as the
action-level allocation/calibration objective.

The allocator's reference file is the suite-correct multi-suite implementation,
not an independently rewritten greedy allocator. Its selection, cost and
refresh rules should be inspected there. Its runtime helper dependencies are
not all bundled; it is deliberately not advertised as a standalone command.

`train_centroid_ft_2026_08_05.py` contains more optional modes than the final
paper conditions. An option's presence in this source is not evidence that a
reported experiment used it. The action decoder is included as a companion
reference in `train_spinquant_r1r2_vla_objective.py`.

## Public integration entrypoints

| Purpose | Entry point |
|---|---|
| Prepare a fresh source workspace | `gr00t_libero/pipeline.py prepare` |
| Bind fitting/allocation paths | `gr00t_libero/stage.py` (materialized as `release_stage.py`) |
| Build index/calibration/evaluation jobs | `gr00t_libero/jobs.py` |
| Validate explicit calibration fields | `gr00t_libero/configuration.py` |
| Run one dependency-checked stage | `gr00t_libero/pipeline.py run` |
| Aggregate an explicit episode set | `gr00t_libero/results.py` |
| Exercise the numerical core without a model | `python -m examples.cpu_pipeline` |
| Check source packaging and known private-artifact patterns | `tools/release_check.py` |

The active model adapters now use explicit calibration callbacks, a strict-metric
fitter argument and static rollout/simulator functions. Original reference
snapshots remain available for comparison; they are not alternative supported
launchers. See the integration README for supported legacy layout boundaries.

## Final inference implementation

The [experiment map](experiments/gr00t_libero/README.md) links recorded latency
and success results to immutable settings and source identities.
The [opt-in A8 overlay](gr00t_libero/acceleration/README.md) contains the selected
INT8/CUDA/Triton implementation and frozen launch choices. Its source staging and
installer are CPU-tested; the relocated runtime has not been GPU-revalidated.
