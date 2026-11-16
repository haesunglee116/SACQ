# GPTQ scalar candidates during allocation

New searches evaluate GPTQ candidates at both 3 and 4 bits. The selected scalar
weights are reused byte-for-byte in index correction and action calibration.
Missing candidates are errors: the allocator never falls back to HQQ.

This changes the search method. Existing published allocations and success rates
were obtained with the historical HQQ-candidate search followed by same-bit GPTQ
replacement. They are not results of this new GPTQ-candidate search. Historical
sources in `reference_sources/` and existing preset CSVs remain archival.

## Candidate preparation

Use the same scalar GPTQ recipe as the existing downstream preparation:
untouched suite-specific BF16 model, original 576 ordered training requests
(positions 0–639 excluding every tenth position), no validation consumption,
noise seed `7000 + original position`, group size 64, block size 128,
damping 0.01, asymmetric quantization, no act-order, TF32 enabled.
The original policy's denoising steps and request contents are preserved.

Run from the repository with the GR00T runtime installed:

```sh
python -m gr00t_libero.prepare_scalar_bank run /absolute/path/job.json
python -m gr00t_libero.prepare_scalar_bank merge \
  --output /absolute/path/scalar_bank.json /absolute/path/shard/COMPLETE.json
```

The job has explicit fields: `workspace` (staged runtime workspace), `suite`,
`out`, `model`, `processor`, `trainer` (path/SHA256 reference), `model_files`
and `processor_files` (path/SHA256 references), `records` (576 ordered
path/SHA256/position entries), and `targets` (layer, shape, bits, scope).
Each shard contains unique layer names at one chosen precision per layer.
Run separate shards for 3-bit and 4-bit candidates. All four suites require
both precisions for every allocation candidate layer (112 VLM + 226 DiT).
Identical layer names across suites do not imply identical GPTQ statistics.
Layer sharding bounds Hessian memory; it preserves each layer's complete
request stream. Merge all shard receipts into one bank manifest.

The outputs store dequantized BF16 weights, as the existing preparation did.
Their on-disk sizes are **not** packed GPTQ deployment sizes. Search bit accounting
retains the historical packing convention and target budget; this change does
not claim a new physical packing format. Candidate preparation time and memory
must be included when reporting preparation costs for this revised pipeline.

## Allocation and downstream binding

Add `scalar_bank_manifest` to allocation preset path bindings. The generated
stage config records its path and SHA256. Every intermediate and final CSV
contains `scalar_method=GPTQ` and has a `.scalar_bank.json` sidecar binding the
CSV hash to its candidate bank. Keep that sidecar with the CSV.

For downstream job specifications, use the selected suite/layer/bit entries
from this bank in the `gptq` list, alongside the existing fixed Vision background.
The job builder checks exact artifact hashes. Regenerating selected scalar
weights after allocation is unnecessary and a different artifact is rejected.
Historical CSVs without the GPTQ marker retain their legacy reproduction path.

Fixed VLSA, Vision and embedding treatment during search is unchanged. This
revision concerns the scalar **allocation candidates**, not every HQQ operation
in the research code. No new rollout accuracy or full-GPU timing is claimed by
this code change; those require fresh runs.
