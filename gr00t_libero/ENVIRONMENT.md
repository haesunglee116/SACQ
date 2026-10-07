# GR00T–LIBERO runtime environments

The CPU package environment and the model runtime are separate. Installing
`sacq-research` installs the numerical core; it does not install GR00T, LIBERO,
renderer assets, checkpoints, or observation captures.

## Policy and calibration runtime

A fresh Linux x86_64 / Python 3.10 policy venv now has a successful real-model
BF16/SACQ forward smoke. See the [tested dependency set and installation
order](environments/POLICY_INSTALL.md). This required staged installation and
an official-source build for fast-hadamard-transform; directly installing the
original freeze did not succeed. The simulator environment remains separate.

Prepare the pinned GR00T sources as described in [README.md](README.md), then use
the staged checkout's dependency manifest:

`libero_pro_eval/Isaac-GR00T-N1.7/pyproject.toml`

The bundled manifest declares Python **3.10**, torch **2.7.1**, torchvision
**0.22.1**, transformers **4.57.3**, NumPy **1.26.4**, and flash-attn
**2.7.4.post1**. It also declares the remaining GR00T dependencies and platform
conditions. These are source requirements, not evidence that a fresh installation
has passed this release's tests.

Read its `[tool.uv.sources]` and index sections when installing: these specify
CUDA-wheel sources and a platform-specific flash-attention wheel. An installer
that ignores those sections does not reproduce the same package resolution.
The manifest contains unpinned ranges too, so it is not a complete environment
lock. Do not replace it with the small core's broad `torch>=2.2` requirement.

Exported helpers additionally import HQQ. The inspected local model environment
has **hqq 0.2.8.post1**, **safetensors 0.7.0**, and **dm-tree 0.1.9**, as reported
by installed distribution metadata. `tree` is the import name for `dm-tree`.
These observations help reconstruct the research environment; they are not a
newly tested compatibility matrix or a portable lockfile. HQQ is not listed in
the bundled upstream GR00T dependency list and must be supplied separately.

The static import inventory is [UNRESOLVED_IMPORTS.json](UNRESOLVED_IMPORTS.json).
It does not cover every dynamic import or native shared-library dependency.

## Simulator runtime

The evaluation plan has an explicit `sim_python` separate from the policy
interpreter. The simulator needs its own compatible LIBERO/robosuite/MuJoCo and
rendering installation, assets and approved initial-state/noise/reset inputs.
A successful policy import is not a simulator installation check. A separate
fresh simulator venv passed a real SACQ policy/RPC/eight-action-chunk
check together with the fresh policy venv, followed by a complete 50-episode
software diagnostic on one Spatial task. Its 79-distribution dependency set is
[simulator-linux-py310-20261006.txt](environments/simulator-linux-py310-20261006.txt).
This is an observed Linux x86_64 / Python 3.10.19 environment, not a container
or proof that every benchmark episode is reproducible.

```bash
uv venv /absolute/new-simulator-env --python 3.10.19
uv pip install --python /absolute/new-simulator-env/bin/python --link-mode=copy \
  -r gr00t_libero/environments/simulator-linux-py310-20261006.txt
uv pip check --python /absolute/new-simulator-env/bin/python
/absolute/new-simulator-env/bin/python gr00t_libero/environments/configure_simulator.py \
  --libero /absolute/LIBERO --config-dir /absolute/new-libero-config
```

Set `LIBERO_CONFIG_PATH=/absolute/new-libero-config` for the simulator process;
put the explicit LIBERO source root on its `PYTHONPATH`. With both checkout
arguments, `pipeline.py prepare` materializes the pinned LIBERO source and
overlays at `libero_pro_eval/Isaac-GR00T-N1.7/external_dependencies/LIBERO`
inside the prepared workspace. Use that assembled directory for `--libero`
above and the simulator's source path. Preparation without `--libero-checkout`
does not provide a complete simulator source/assets tree.
Do not install the policy's full GR00T dependency set into this simulator venv.
The environment separation is intentional: simulator torch is 2.5.1 and
gymnasium is 0.29.1, whereas the policy has different requirements.

The configuration helper changes only robosuite 1.4.0's logger filename to a
per-user path and records the source hashes. Its default `/tmp/robosuite.log`
can belong to another user and prevent imports. Physics, rendering and random
settings are unchanged. `termcolor` and `future` are included explicitly because
fresh runtime imports needed them despite the package resolver reporting no
missing dependencies.

**Source identity:** the research LIBERO tree descends from commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` with local Python changes. The release
now exports that pinned upstream commit and applies 13 modified Python files
plus the required object-definition module, all recorded in `LIBERO_OVERLAY.json`.
This assembled source/assets tree passed the same one-chunk RPC fixture with
the exact archived initial observation. Extra untracked tasks/assets from the
research checkout were not copied. Use `pipeline.py prepare --libero-checkout`
as documented in the integration guide; a plain unmodified upstream checkout
is not the tested identity. The model checkpoints and original reset/noise
journals are still external inputs. The subsequent full-worker diagnostic
covered all 50 official initial states of that same single task, not all suites.

## Check an existing environment

```bash
python gr00t_libero/runtime_check.py \
  --workspace /absolute/prepared/workspace \
  --python /absolute/model-runtime/bin/python \
  --timeout-seconds 150
```

This checks selected imports with offline flags and GPU visibility disabled.
It does not install packages, construct a model, load weights or run evaluation.
A pass establishes import availability only. The timeout is an operational
choice, not a scientific setting or recommended model execution time limit.

On the inspected shared filesystem, Python startup completed but a live PyTorch
native import was observed in `cl_sync_io_wait`. Short import timeouts therefore
must not be treated as proof of a missing/broken dependency. Inspect the last
progress event and process state; do not start duplicate probes while the
original is still running. No claim is made that the I/O wait explains every
runtime delay or that the environment is fully validated.

Use [VALIDATION.md](../VALIDATION.md) for the checks actually completed. A future
installation recipe or container must be tested before it is described as a
reproducible model environment.
