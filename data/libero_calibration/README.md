# LIBERO calibration set

Original observations were collected periodically during rollouts with a BF16
policy using environment seed **1234**. Within each suite, the collected requests
were shuffled with selection seed **0**, and **640 requests** were selected.
Every tenth request in the selected order is used for validation: **576 training
requests and 64 validation requests per suite**.

Each request contains five observations. Counts are matched across suites, not
across individual tasks. Preserve the published request order when using the set.

| Suite | Training requests | Validation requests | Observations | Uncompressed size |
|---|---:|---:|---:|---:|
| Spatial | 576 | 64 | 3,200 | 1.260 GB |
| Object | 576 | 64 | 3,200 | 1.260 GB |
| Goal | 576 | 64 | 3,200 | 1.260 GB |
| Long (LIBERO-10) | 576 | 64 | 3,200 | 1.260 GB |
| **Total** | **2,304** | **256** | **12,800** | **5.038 GB** |

## Files and downloads

[manifest.json](manifest.json) lists the exact files, SHA-256 hashes, sizes,
request IDs, ordering, train/validation split and task labels.

The dataset is available as one archive per suite in the
[calibration-v1 release](https://github.com/haesunglee116/SACQ/releases/tag/calibration-v1),
separately from the Git source tree. Model checkpoints, teacher caches and intermediate training outputs
are not included in these observation archives.

[archives.json](archives.json) records archive filenames, compressed sizes and
SHA-256 hashes. The four archives total **2.263 GB**. Every archived request was
checked against the file manifest after compression.


| Suite | Download | Compressed size |
|---|---|---:|
| libero_10 | [sacq_calibration_libero_10_20261029.tar.gz](https://github.com/haesunglee116/SACQ/releases/download/calibration-v1/sacq_calibration_libero_10_20261029.tar.gz) | 518.4 MB |
| libero_goal | [sacq_calibration_libero_goal_20261029.tar.gz](https://github.com/haesunglee116/SACQ/releases/download/calibration-v1/sacq_calibration_libero_goal_20261029.tar.gz) | 512.1 MB |
| libero_object | [sacq_calibration_libero_object_20261029.tar.gz](https://github.com/haesunglee116/SACQ/releases/download/calibration-v1/sacq_calibration_libero_object_20261029.tar.gz) | 714.7 MB |
| libero_spatial | [sacq_calibration_libero_spatial_20261029.tar.gz](https://github.com/haesunglee116/SACQ/releases/download/calibration-v1/sacq_calibration_libero_spatial_20261029.tar.gz) | 517.2 MB |

After obtaining and extracting the archives into a directory, verify the files:

```bash
python tools/verify_calibration_data.py --data-root /path/to/extracted/data
```

Expected layout:

```text
data/
  libero_spatial/requests/0000__....msgpack
  libero_object/requests/0000__....msgpack
  libero_goal/requests/0000__....msgpack
  libero_10/requests/0000__....msgpack
```

See [the input format guide](../../gr00t_libero/CAPTURE_INPUTS.md) for serialized
observations and teacher/prepared-input caches. Dataset file verification checks
bytes and layout; it does not execute fitting, calibration or evaluation.
