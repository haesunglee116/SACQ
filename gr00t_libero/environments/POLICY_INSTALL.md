# Tested policy environment

The adjacent `policy-linux-py310-20261006.txt` records the tested dependency
set for Linux x86_64, Python 3.10.19 and torch 2.7.1+cu128. It is a full observed
research environment, not a minimal inference dependency list or a universal
platform lock. It excludes the old editable GR00T installation. Use the pinned
prepared GR00T source described in the parent README.

A newly installed venv passed BF16 and SACQ GPU forward checks. The host already
had an NVIDIA driver and CUDA 12.4 development toolkit. A separate fresh
simulator venv subsequently passed a policy/RPC/action-chunk check with this
policy venv. The host and source/assets were reused; no fresh-host container
test is claimed. See ../../VALIDATION.md and ../ENVIRONMENT.md.

## Installation order

The installation steps below are also available as a script. Build the SACQ
wheel first (`python -m build --wheel` at the release root), then inspect:

```bash
python gr00t_libero/environments/install_policy.py \
  --env /absolute/new-policy-env --artifacts /absolute/new-build-artifacts \
  --sacq-wheel /absolute/sacq_research-0.1.0-py3-none-any.whl --dry-run
```

Remove `--dry-run` to install. Both destination paths must be new and separate.
The script verifies the FlashAttention checksum, enforces Python 3.10.19,
uses the pinned source build, and records completed commands and failures in
`install_receipt.json`. It needs `uv`, `curl`, `git`, a CUDA development toolkit,
a C++ toolchain and substantial download/disk space. It does not install models
or configure an experiment. A dependency receipt is not a model validation pass.
The constituent commands have been exercised in the fresh environment below;
the script has separate failure-boundary tests.

Use a new venv with system site packages disabled. Commands below use `uv`,
which must already be installed. Choose your own writable paths:

```bash
runtime_env=/absolute/path/to/new-policy-env
artifact_dir=/absolute/path/to/build-artifacts
uv venv "$runtime_env" --python 3.10
mkdir -p "$artifact_dir"
```

1. Download the exact FlashAttention wheel URL in the requirements file before
   installation. Direct remote-wheel reads timed out in uv's 30-second window.
   Downloading with curl and installing the local file succeeded. The tested
   wheel's SHA-256 is
   `f09928f2c7e4dfbe6efca5a997862898495263c87afe662c18fb1936eb192eab`.
   Preserve the full wheel filename, including its Python/platform suffix.
2. Create a working copy of the requirements without `deepspeed` and
   `fast-hadamard-transform`. Replace the FlashAttention URL with that verified
   local wheel's absolute `file://` URI. Install this foundation first:

   ```bash
   UV_HTTP_TIMEOUT=120 uv pip install --python "$runtime_env/bin/python" \
     --link-mode=copy --extra-index-url https://download.pytorch.org/whl/cu128 \
     --index-strategy unsafe-best-match -r /absolute/path/to/foundation.txt
   ```

3. The CUDA index can select `torchcodec==0.4.0+cu128` for a `==0.4.0`
   requirement. Restore the tested PyPI version explicitly:

   ```bash
   uv pip install --python "$runtime_env/bin/python" --no-deps \
     --reinstall-package torchcodec --index-url https://pypi.org/simple \
     'torchcodec===0.4.0'
   uv pip install --python "$runtime_env/bin/python" --no-build-isolation \
     'deepspeed==0.17.6'
   ```

4. Build fast-hadamard-transform from the pinned official source. The PyPI
   1.1.0 source archive lacked a required C++ file; its isolated build also
   lacked torch. Do not silently substitute a newer version.

   ```bash
   git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
     "$artifact_dir/fast-hadamard-transform"
   git -C "$artifact_dir/fast-hadamard-transform" checkout --detach \
     1cc807efbd6cc001df359822d60bf6052dd66859
   MAX_JOBS=4 FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
     uv pip install --python "$runtime_env/bin/python" \
       --no-deps --no-build-isolation --link-mode=copy \
       "$artifact_dir/fast-hadamard-transform"
   ```

5. Install the SACQ wheel and run `uv pip check --python
   "$runtime_env/bin/python"`. The tested environment included
   `nvidia-ml-py==13.615.71`, required by DeepSpeed but absent from the original
   freeze. This addition is already in the adjacent requirements file.

The tested result contained 142 distributions including SACQ. Compare installed
versions with the requirement list, including local version suffixes. A package
check alone does not establish native CUDA compatibility; follow it with the
prepared-source import check and an explicitly configured model smoke test.
Existing checkpoints, replay inputs and simulator assets are not supplied here.
