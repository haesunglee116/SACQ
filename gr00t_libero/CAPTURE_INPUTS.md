# Inputs for the pinned GR00T–LIBERO pipeline

This document describes the exported consumers, not a newly chosen experimental
protocol. Captures, models, teacher tensors and simulator assets are not shipped.
A fresh reproduction still needs an explicitly selected dataset and its episode
provenance. A capture hook and ordered-manifest preparation CLI are supplied;
they do not choose or launch development episodes for the user.

## Capture and prepare explicit inputs

The prepared GR00T server includes `gr00t.eval.request_capture`. To capture
requests during an explicitly configured development run, set all three
environment variables for that server:

- `GR00T_CAPTURE_REQUESTS_DIR`: a new, empty destination dedicated to one process;
- `GR00T_CAPTURE_TAG`: a unique tag for this process/run;
- `GR00T_CAPTURE_MAX_REQUESTS`: your explicit approved request limit.

The server must otherwise use the selected original model, processor, policy
configuration and approved development rollout inputs. Capturing requests does
not change the observations passed to `get_action`, and no episode is selected
by the hook. The legacy hook restarts its counter at process startup, so never
reuse the same destination/tag when restarting; it is not a resumable collector.

The hook produces `.msgpack` requests and a JSONL index. Its index **does not
contain complete episode provenance**. Record that separately at the simulator
caller, for every observation in each request batch. A five-observation request
needs five identities, even if their task strings are the same. Do not infer
episode IDs or seeds from file names, request counters or language descriptions.

Create a JSON spec with `schema: "sacq_capture_inputs_v1"` and an ordered
`requests` list. Every request declares `path` (absolute `.msgpack` path),
`request_id` (the payload's actual ID), `position` (its original index noise
position), and `observations`. Every observation supplies `benchmark_family`,
`task`, `base_seed`, `episode_id`, and `policy_call`. Seed/episode/call/position
values must be explicit nonnegative integers. An optional request `sha256`
is checked against the file if supplied.

```bash
python gr00t_libero/captures.py --spec /absolute/capture-provenance.json \
  --output /absolute/new-prepared-inputs.json
```

The output's `records` preserves the supplied order and original positions and
adds file hashes. It rejects duplicate IDs/positions/paths and missing episode
identities. It does not decode the model payload or certify the caller's
provenance declarations. Keep the output as the input-preparation record; it
is **not** a sample-disjointness PASS or an execution authorization.

Use the approved list as the index builder's `records`. Calibration separately
needs its complete, approved `all_records` list in the intended order. Its
noise/split positions are list offsets, not arbitrary `position` metadata;
the preparation tool deliberately does not silently reconcile different lists.
Keep fitting/allocation/calibration/evaluation episode manifests separate and
run the experiment's episode-disjointness audit before claiming held-out results.

## Audit the declared episode lists

Use `python gr00t_libero/sample_audit.py --spec /absolute/audit-spec.json
--output /absolute/new-audit.json` (one shell command). The spec uses schema
`sacq_sample_audit_spec_v1`, a `task_registry` file reference, and `stages`
containing exactly `fitting`, `allocation`, `calibration_train`,
`calibration_validation`, and `rollout`. Each reference is an object with
absolute `path` and `sha256`.

The registry uses schema `sacq_task_registry_v1` and a `tasks` list of
`{"benchmark_family": "...", "task": "..."}` objects. Obtain these identities
from the canonical task registry for the explicitly selected protocol; declaring
a subset here does not prove that the full benchmark is covered or authorize
omissions. Each stage manifest is either a prepared capture manifest described
above or schema `sacq_episode_identities_v1` with a `samples` list containing
explicit `benchmark_family`, `task`, `base_seed`, and `episode_id` values.
Split manifests must describe the actual consumer split, not simply repeat the
unsplit capture list under different labels.

The auditor compares rollout episodes with every development stage, ignoring
policy-call differences. It checks every stage against the supplied registry,
rejects duplicate rollout episodes, and records calibration train/validation
episode overlap separately. That overlap is not a claim of independent validation.
Development/rollout overlap or missing/unexpected tasks yields `FAIL` and a
nonzero exit code. It never generates approval for overlapping diagnostic data.

The resulting `sacq_sample_audit_v1` receipt binds all manifest hashes. The
pipeline recomputes it before execution, rejecting changed inputs and edited
results. This verifies declared identities, not their truth or linkage to model
payloads; preserve original capture provenance and protocol approval alongside it.

## What to supply

| Input | Consumer | Meaning |
|---|---|---|
| Original suite checkpoint and matching processor | Fitting, allocation, index and calibration | External model directories; bind immutable model revisions in the experiment manifest |
| Serialized observation requests | Proxy/statistics collectors, index, calibration | Observations accepted by that model's processor, not ground-truth action labels |
| Fitted candidate bank | Allocation and index | Codebooks, scales and indices for the explicitly selected layer/candidate pairs |
| Frozen background artifacts | Index, calibration and rollout | Explicit layer, precision and reconstructed weight payloads used outside the SACQ bank |
| Index completion receipts | Calibration job builder | Corrected artifacts linked to the exact input job; these do not contain calibrated codebooks |
| Calibration completion receipt | Evaluation job builder | VLM/DiT overlays and selected codebook state |
| Episode plans, noise and reset evidence | Rollout | Separate evaluation inputs; calibration requests do not define a rollout episode set |

See [the integration guide](README.md) for the builders and file-reference schemas.
A path plus SHA-256 identifies bytes. It does not establish that those bytes came
from an approved checkpoint, task, episode or split.

## Prepare pinned rollout reset journals

Run `gr00t_libero/reset_inputs.py` with the separate simulator Python, using
`--contract /absolute/reset-spec.json --task-index N`. It consumes schema
`sacq_reset_inputs_v1` with absolute `repo` (prepared GR00T directory),
`adapter_directory` (prepared `gr00t_final_seed17_20260922`), and `output`.
Each task supplies its exact `suite`, `task`, registry `task_id`, `language`,
and `initial_file` path/SHA-256 reference. Bind the initial-state and BDDL files
in `inputs`, and the prepared `seed17_adapter.py`, `libero_env.py` and
`multistep_wrapper.py` in `sources`, using the same path/SHA-256 reference format.

All `settings` are explicit and must match the supported historical path:

```json
{
  "environment_seed": 17, "python_seed": 17, "numpy_seed": 17,
  "n_envs": 1, "episode_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
  "n_action_steps": 8, "max_episode_steps": 720,
  "terminate_on_success": false, "policy_forward_calls": 0,
  "environment_steps": 0
}
```

The helper creates `tasks/SUITE/TASK/reference.pkl` and `result.json`. It refuses
an existing task output directory. It calls no policy and forbids environment
steps, comparing official `reset → set_init_state` observations against the
adapter and wrapper for every episode. `PASS` certifies that reset comparison
only; it does not authorize a rollout, prove development/evaluation separation,
or produce success-rate evidence. A failure preserves `FAIL` rather than a
usable reset receipt. Use a fresh output when retrying; never relabel a failed
or incomplete journal as complete. The rollout plan's `reset_audit_root` and
per-task contract hashes must point to these actual completed outputs.

This is a static port of the historical seed17 reset wrapper, including its
NumPy seed transformation. It does not expose arbitrary seed or parallelism
changes as supported reproduction settings. Journals contain pickle data;
consume only trusted, hash-bound files. Cross-renderer observation handling
remains the rollout runtime's existing same-process reference procedure.

## Prepare fixed episode noise

Before policy rollout, fixed episode noise can be prepared with
`python gr00t_libero/noise_inputs.py --spec /absolute/noise-spec.json` in the
policy environment. The JSON schema is `sacq_noise_inputs_v1`; all fields are
required: absolute new `output`, integer `seed`, positive `episodes` and `calls`,
positive integer `shape` list, `dtype` (`float32`, `float16` or `bfloat16`),
`device` (`cpu` or an explicit CUDA device), and exact `torch_version` including
any build suffix. Use values from the selected rollout/model contract; the tool
has no scientific defaults and does not infer action shape or episode count.

It calls the original bundled generator, drawing in episode-then-policy-call
order with a separate Torch generator. The output contains per-episode tensors
and a final hash manifest. Existing destinations are rejected; a partial
directory without the final manifest is not a usable completed table. A runtime
version mismatch fails before generating files. Device, dtype and runtime can
change generated tensors: for exact replay, retain the original table and its
hashes instead of assuming that matching seeds guarantee equivalent regeneration.
Generation is input preparation, not execution authorization. The reset journal,
noise table, scientific sample manifest and rollout plan must refer to the same
explicit protocol before running the worker.

## Serialized request contract

Calibration reads files ending in **`.msgpack`**. Each is decoded by the pinned
GR00T `gr00t.policy.server_client.MsgSerializer.from_bytes`, rather than a plain
JSON decoder. The payload must contain:

- `request_id`: a unique, hashable identifier within the calibration input list;
  a stable string is suitable. The teacher cache is keyed by this payload value,
  not the filename.
- `observation`: the observation mapping expected by the chosen policy.

Preserve the original arrays, image/state fields, prompt, shapes and dtypes.
`normalize_flat_observation`, the selected processor's `process_observation`,
and the model's `prepare_input` determine the actual accepted structure. There
is no model-independent RGB/state schema supplied by this release. Renaming an
arbitrary file to `.msgpack` does not convert or validate its contents.

The standard-library job builder hashes files but does **not** deserialize them
or prove model compatibility. Its extension check prevents files being silently
omitted by the calibration loader. Duplicate payload request IDs are rejected by
the trainer when it checks the reference/request mapping.

## Order is part of the input

Index jobs use `records`, each with `path`, `sha256` and explicit `position`.
The index worker sets request noise from `index_settings.noise_seed_base +
position`. Layer shards must use the same approved records and positions.

Calibration jobs use `all_records` in the approved order. The worker supplies
that list to the original trainer. The trainer stages numerically prefixed
symlinks and the loader reads them in sorted order. The pinned implementation
then assigns validation by `i % val_every == 0` and training to the other
positions. Teacher and student action noise use `7000 + i`. These are facts about
the pinned trainer, not suggested new seed or split values.

Consequently, reordering requests can change both the train/validation split and
the action noise. Do not sort, shuffle, deduplicate, oversample or replace the
list merely to make a launch succeed. Request-level separation also does not
establish episode-level separation: keep task, episode, base seed and policy-call
provenance in the external dataset manifest.

## Teacher references and prepared batches

Before installing quantized layers, calibration runs the original teacher and
stores detached float32 action tensors in `refs.pt`, keyed by `request_id`.
The teacher computation itself uses the pinned model/runtime precision; storing
float32 references does not make this a float32 model evaluation.

`refs.pt` is a derived teacher cache. It is neither a dataset of successful
rollouts nor a calibrated checkpoint. `codebook_state.pt` and the exported
VLM/DiT overlays are calibration outputs instead.

The prepared-batch cache stores processor/model inputs derived from observations.
It is distinct from `refs.pt`. Both depend on the selected model, preprocessing
and request identities. The public worker isolates their cache paths per job and
requires matching ownership for resumed output directories. Use fresh output
locations for changed jobs. Do not copy an old `refs.pt` into a new job directory
to bypass this ownership boundary.

## Source locations

The relevant consumers are shipped under `sources/`:

- `gr00t_final_seed17_20260922/index_worker.py`: request decoding and noise positions.
- `gr00t_final_seed17_20260922/calibration_worker.py`: ordered request injection and cache bindings.
- `libero_pro_eval/measure_dit_scalar_vs_vector_2026_07_17.py`: calibration batch loader.
- `libero_pro_eval/measure_hotspot_protection_proxy.py`: observation collation.
- `libero_pro_eval/quantization_analysis/workshop_metricv2_calibration_20260919/source/train_centroid_ft_2026_08_05.py`: reference generation, split and loss.

Fitting and allocation also have stage-specific statistics/candidate inputs;
this document does not replace their explicit configuration or claim that a
calibration capture list alone is enough to run the entire pipeline.
