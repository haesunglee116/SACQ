"""Install the final SACQ1.5 payload, never the historical n140 benchmark model.

The caller supplies an existing rollout plan and node-local model/codec assets.
No checkpoint, bit allocation, group size or teacher is chosen here.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "gr00t_final_seed17_20260922"
sys.path.insert(0, str(FINAL))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_plan(path, suite):
    plan = json.loads(Path(path).read_text())
    if plan["condition"] != "sacq1.5" or plan["selected_suites"] != [suite]:
        raise ValueError("Expected final SACQ1.5 plan for the requested suite")
    model = plan["workshop_model"]
    for ref in [model[k] for k in ("vlm", "dit", "vlm_spec", "dit_spec")] + model["gptq"]:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError(f"Final artifact digest mismatch: {ref['path']}")
    return plan


def build_policy(plan, suite, device):
    from transformers import AutoProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration
    sys.path.insert(0, plan["repo"])
    # Same offline Cosmos resolution as the final resident/rollout workers.
    for cls in (AutoProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration):
        original = cls.from_pretrained
        def local(cls, name, *args, _original=original, **kwargs):
            if str(name) == "nvidia/Cosmos-Reason2-2B":
                name = plan["cosmos"]
            return _original(name, *args, **kwargs)
        cls.from_pretrained = classmethod(local)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag
    checkpoint = Path(plan["models"][suite]).resolve()
    if not str(checkpoint).startswith("/tmp/") or not checkpoint.is_dir():
        raise ValueError("Checkpoint must already be staged on node-local /tmp")
    policy = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, str(checkpoint),
                         device=str(device), strict=True)
    policy.model.eval().requires_grad_(False)
    return policy


def install(policy, plan, suite, backend, fused_dictionary=False):
    from install_packed_memory_reference import install as pack
    from gr00t.eval.cosmos_quant_shim import PrecomputedPackedVectorCodebookLinear
    report = pack(policy.model, plan, suite)
    names = [name for name, layer in policy.model.named_modules()
             if type(layer) is PrecomputedPackedVectorCodebookLinear]
    # The receipt lists only newly packed layers. The original installation
    # may already contain packed payloads, which pack_installed_sacq skips.
    converted = {item['layer'] for item in report['sacq']['layers']}
    if not converted.issubset(names):
        raise ValueError("A newly packed layer is absent from the installed model")
    report['installed_codebook_layers'] = len(names)
    report['already_packed_layers'] = sorted(set(names) - converted)
    for name in names:
        layer = policy.model.get_submodule(name)
        if layer.packed_indices is None:
            raise ValueError(f"Unpacked SACQ indices remain: {name}")
        layer.runtime_backend = backend
        layer.runtime_accelerate_vlm = False
    if fused_dictionary:
        from exact_dictionary_decode import install as fuse
        report["fused_dictionary_layers"] = fuse(policy.model)
    report["backend"] = backend
    report["full_policy_output_verified"] = False
    return report
