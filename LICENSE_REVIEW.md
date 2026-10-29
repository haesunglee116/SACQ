# License decision and scope

**Owner approved Apache-2.0 for original SACQ code and documentation.**
The top-level [LICENSE](LICENSE) and package metadata implement this decision.
Repository visibility remains unchanged. Existing authorship and third-party
notices are preserved.

Apache-2.0 permits commercial use and redistribution, contains an express patent
license from contributors, and requires preservation of applicable notices and
identification of modified files. The existing GR00T-derived materials already
retain Apache-2.0 terms. See the [official license, sections 2–4](https://www.apache.org/licenses/LICENSE-2.0).

MIT is a shorter permissive alternative that also allows commercial use while
requiring preservation of its copyright and license notice. It does not include
the same express patent provisions. See the [MIT text](https://choosealicense.com/licenses/mit/).
Neither choice means commercial use is prohibited.

## Scope of the decision

The owner-selected license applies to original SACQ code and documentation.
It does not replace third-party licenses or grant rights to external model
weights, simulator assets or the separately distributed calibration observations.

## Included and external materials

| Material | Current handling |
|---|---|
| Original SACQ algorithms, integration, tests, docs | Apache-2.0 |
| GR00T-derived code/overlays | Existing Apache-2.0 text and provenance retained |
| LIBERO overlays | MIT notice retained |
| Two CUTLASS-derived MMA headers in acceleration sources | NVIDIA BSD notice retained at the top of each header; modified header is identified in the source manifest |
| Transformers/diffusers method fixtures and adapted attention code | Source-identity fixtures / adaptations; Apache-2.0 attribution below and existing license text retained |
| SageAttention-derived preparation arithmetic | Identified in source docstrings and third-party notices; Apache-2.0 text retained |
| vLLM, vLLM-Omni, PyTorch, CUDA, Triton, HQQ, external model weights | Not relicensed or bundled as complete packages; obtain under upstream terms |
| Calibration observations in GitHub Release assets | Distributed separately; dataset licensing must be specified separately from the code license, accounting for source simulator/assets |

The kernel source-manifest hashes establish provenance and modifications; they
do not alone establish ownership. The recorded performance does not imply a
new license grant for external dependencies or model assets.
