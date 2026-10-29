# GR00T–LIBERO recorded experiments

This page links recorded outcomes to the configuration and implementation that
produced them. These are archived research measurements, not a rerun of the
newly packaged code. The source identifiers and original SHA-256 values in the
JSON records refer to immutable research artifacts; they are not downloadable
model files or runnable absolute paths.

## Final inference comparison

| Configuration | Wall median (ms) | Speedup vs historical original BF16 | Speedup vs matched optimized BF16 |
|---|---:|---:|---:|
| Original BF16, historical measurement | 56.035 | 1.000x | — |
| Optimized BF16, final A8 control | 21.284 | 2.633x | 1.000x |
| SACQ W1.5A16, common kernel port | 20.203 | 2.774x | 1.053x |
| SACQ W1.5A8, final kernel combination | 14.066 | 3.984x | 1.513x |

Source: [latency_final.json](latency_final.json). Ratios use unrounded values.
A16 used its own paired BF16 control, **21.271 ms**; the displayed 21.284 ms is
paired with A8. The original 56.035 ms measurement is a historical cross-run
comparison. GPU clocks were recorded but not matched; **1.513x is the observed
ratio, not a clock-normalized causal estimate**. Reverse-order A8 was 14.119 ms
and 1.506x; forward order is the designated reporting order.

The metric is synchronized whole-model CPU wall time on an RTX 4090, batch 1,
four denoising steps, five warmup iterations and twenty measured iterations.
Vision, language and action computation are included. Preprocessing, simulator
steps and network/RPC time are excluded. The observation and policy noise are
fixed; timing seed is 42. This uses GR00T input preparation but is not a claim to
reproduce NVIDIA's published throughput metric.

## Previously evaluated W1.5A8 implementation

| Suite | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Spatial | 469 | 500 | 93.8% |
| Object | 481 | 500 | 96.2% |
| Goal | 477 | 500 | 95.4% |
| Long / LIBERO-10 | 444 | 500 | 88.8% |
| **All four** | **1,871** | **2,000** | **93.55%** |

Source: [success_w1p5a8.json](success_w1p5a8.json). This record retains its
**diagnostic_only** status. Environment seed 17, policy seed 20260824,
episodes 0–49 per task, one environment, action chunk 8, limit 720 steps and
four denoising steps are recorded in its original contract.

The success evaluation predates the final 1.513x kernel combination. The final
latency worker matched the original INT8 reference action bytes and all four RNG
states on the timing fixture, in both orders. Its fifty checks repeat that
fixture across wall/GPU timing calls; they are not fifty distinct observations.
There was **no new 2,000-episode rollout of the final combination**. Keep these
two experiment IDs distinct when presenting accuracy and speed.

## Settings → implementation → evidence

| Record | Configuration | Implementation / evidence |
|---|---|---|
| Final W1.5A8 latency | [Frozen recipe](../../gr00t_libero/acceleration/presets/final_w1p5a8.json) | [Kernel integration guide](../../gr00t_libero/acceleration/README.md), [original worker export](../../gr00t_libero/acceleration/reference/final_worker.py), [latency record](latency_final.json) |
| Earlier W1.5A8 success | [Recorded protocol and results](success_w1p5a8.json) | Original worker and suite contract SHA-256 identities in the record |
| Checkpoint / fixture identity | [Model input inventory](MODEL_INPUTS.json) | External weights and payloads required; not bundled here |
| Existing SACQ2 pipeline preset | [SACQ2 preset](../../gr00t_libero/presets/gr00t_libero_sacq2_seed17.json) | Existing fitting-to-rollout pipeline; **not** the W1.5A8 checkpoint |

W1.5 describes the selected compressed SACQ weight representation; A8 does not
mean every tensor is INT8. There are 441 INT8 projections (dynamic per-token
activations and per-output-row weights), and 40 low-precision attention modules:
16 action self-attention and 24 vision attention. Attention uses INT8 Q/K and
FP8 P/V, with the recorded FP32+FP32 accumulation. Language attention and action
cross-attention are not part of that 40-module attention expansion.

The A8 row-scale metadata remains part of the runtime's storage accounting.
“No additional memory” refers to the final kernel improvements relative to the
existing A8 implementation, not to A8 having no overhead relative to A16.
