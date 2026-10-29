"""Single captured-request inference; no simulator, rollout or timing claim.

Run in a fresh policy-runtime process. Only load trusted model artifacts.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Bundle path escapes root: ' + name)
    return path


def preflight(bundle, workspace):
    bundle, workspace = Path(bundle).resolve(), Path(workspace).resolve()
    cfg = json.loads((bundle / 'inference.json').read_text())
    if cfg.get('schema') != 'sacq_single_request_v1':
        raise ValueError('Unsupported inference bundle')
    if cfg.get('condition') not in ('bf16', 'sacq', 'sacq1.5'):
        raise ValueError('This entrypoint supports BF16 and SACQ A16 only')
    for key in ('seed', 'denoising_steps'):
        if type(cfg.get(key)) is not int or cfg[key] < (1 if key == 'denoising_steps' else 0) or (key == 'seed' and cfg[key] >= 2**32):
            raise ValueError('Explicit valid ' + key + ' required')
    files = cfg['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Nonempty file inventory required')
    for name, digest in files.items():
        if sha(resolve(bundle, name)) != digest:
            raise ValueError('Input hash mismatch: ' + name)
    for key in ('model', 'processor'):
        directory = resolve(bundle, cfg[key])
        actual = {str(p.relative_to(bundle)) for p in directory.rglob('*') if p.is_file()}
        listed = {name for name in files if resolve(bundle, name).is_relative_to(directory)}
        if not actual or actual != listed:
            raise ValueError('Incomplete ' + key + ' inventory')
    if cfg['request'] not in files:
        raise ValueError('Request missing from inventory')
    plan = dict(condition=cfg['condition'], selected_suites=[cfg['suite']])
    if cfg['condition'] != 'bf16':
        model = copy.deepcopy(cfg['workshop_model'])
        refs = [model[k] for k in ('vlm', 'dit', 'vlm_spec', 'dit_spec')] + model['gptq']
        for ref in refs:
            if files.get(ref['path']) != ref['sha256']:
                raise ValueError('Unbound quantized artifact: ' + ref['path'])
            ref['path'] = str(resolve(bundle, ref['path']))
        plan['workshop_model'] = model
    elif 'workshop_model' in cfg:
        raise ValueError('BF16 bundle must not contain a quantized model')
    receipt = json.loads((workspace / 'SOURCE_RECEIPT.json').read_text())
    if not receipt.get('sources'):
        raise ValueError('Empty source receipt')
    for row in receipt['sources']:
        if sha(resolve(workspace, row['file'])) != row['sha256']:
            raise ValueError('Staged source drift: ' + row['file'])
    if not (workspace / 'libero_pro_eval/Isaac-GR00T-N1.7/gr00t/policy/gr00t_policy.py').is_file():
        raise ValueError('Prepare a complete pinned GR00T checkout first')
    return cfg, plan


def run(bundle, workspace, output):
    if sys.flags.optimize:
        raise ValueError('Python -O is unsupported: original installers use assertions')
    cfg, plan = preflight(bundle, workspace)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        forbidden = ('VLM_', 'DIT_', 'ACT_QUANT', 'ACTION_HEAD_', 'TAC_', 'GR00T_CODEBOOK_', 'GR00T_CAPTURE_')
        if any(k.startswith(forbidden) for k in os.environ):
            raise ValueError('Remove inherited quantization/capture environment controls')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        workspace, bundle = Path(workspace).resolve(), Path(bundle).resolve()
        sys.path[:0] = [str(workspace / p) for p in (
            'libero_pro_eval/Isaac-GR00T-N1.7', 'libero_pro_eval',
            'gr00t_final_seed17_20260922', 'libero_transfer_quantstats_20260922',
            'simplerenv_n17_feasibility_2026_08_27')]
        import random
        import numpy as np
        import torch
        import transformers
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required by the pinned model installer')
        for cls in (transformers.AutoProcessor, transformers.Qwen3VLProcessor,
                    transformers.Qwen3VLForConditionalGeneration):
            original = cls.from_pretrained
            def local(cls, name, *args, _original=original, **kwargs):
                if str(name) == 'nvidia/Cosmos-Reason2-2B':
                    name = str(resolve(bundle, cfg['processor']))
                return _original(name, *args, **kwargs)
            cls.from_pretrained = classmethod(local)
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.server_client import MsgSerializer
        from model_install import install
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        print('Loading model', flush=True)
        core = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA,
                           str(resolve(bundle, cfg['model'])), device='cuda:0', strict=True)
        core.model.eval().requires_grad_(False)
        if {p.dtype for p in core.model.parameters()} != {torch.bfloat16}:
            raise ValueError('Expected the original BF16 checkpoint before installation')
        if core.model.action_head.num_inference_timesteps != cfg['denoising_steps']:
            raise ValueError('Checkpoint denoising setting differs from explicit bundle setting')
        with torch.inference_mode():
            audit = install(core.model, plan, cfg['suite'])
            policy = Gr00tSimPolicyWrapper(core)
            request = MsgSerializer.from_bytes(resolve(bundle, cfg['request']).read_bytes())
            random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
            torch.cuda.manual_seed_all(cfg['seed'])
            print('Generating actions', flush=True)
            actions, _ = policy.get_action(request['observation'], request['options'])
        arrays = {k: np.asarray(v) for k, v in actions.items()}
        if not arrays or any(v.size == 0 or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Empty or nonfinite actions')
        np.savez(output / 'actions.npz', **arrays)
        result = dict(status='COMPLETE', scope='single_request_inference_not_benchmark',
                      condition=cfg['condition'], request_id=request['request_id'],
                      config_sha256=sha(bundle / 'inference.json'), entrypoint_sha256=sha(__file__),
                      source_receipt_sha256=sha(workspace / 'SOURCE_RECEIPT.json'),
                      seed=cfg['seed'], denoising_steps=cfg['denoising_steps'],
                      actions={k: list(v.shape) for k, v in arrays.items()},
                      actions_sha256=sha(output / 'actions.npz'), install=audit,
                      gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('status', 'condition', 'actions')}))
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps(dict(status='FAILED', error=str(error))))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true', help='Hash/input checks only; no GPU imports')
    a = p.parse_args()
    if a.check:
        cfg, _ = preflight(a.bundle, a.workspace)
        print(json.dumps(dict(status='INPUTS_VERIFIED', condition=cfg['condition'], gpu_tested=False)))
    else:
        if a.output is None:
            p.error('--output is required for inference')
        run(a.bundle, a.workspace, a.output)


if __name__ == '__main__':
    main()
