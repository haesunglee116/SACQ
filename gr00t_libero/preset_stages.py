"""Prepare original SACQ2 fitting/search commands from path-only bindings.

No model is loaded and no command is executed. The archival search runs to 1.25;
its checkpoint2.00 is the SACQ2 selection, not its final allocation.csv.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
try:
    from .preset import PRESET
except ImportError:
    from preset import PRESET


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def path(value):
    p = Path(value)
    if not p.is_absolute():
        raise ValueError('Bindings require absolute paths')
    return p


def prepare(stage, bindings, output_dir):
    if stage not in ('fitting', 'allocation'):
        raise ValueError('Expected fitting or allocation')
    preset = json.loads(PRESET.read_text())
    allocation = PRESET.parent/preset['allocation']['file']
    if ref(allocation)['sha256'] != preset['allocation']['sha256']:
        raise ValueError('Preset allocation checksum mismatch')
    common = {'workspace', 'processor', 'models', 'output_root'}
    extra = {'vlm_requests', 'dit_requests'} if stage == 'fitting' else {'bank_manifest', 'allocation_requests'}
    if set(bindings) != common | extra:
        raise ValueError('Expected path-only bindings: '+', '.join(sorted(common | extra)))
    workspace = path(bindings['workspace']); processor = path(bindings['processor'])
    root = path(bindings['output_root'])
    models = {s: str(path(v)) for s,v in bindings['models'].items()}
    if set(models) != set(preset['suites']):
        raise ValueError('All four suite model paths required')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    with (PRESET.parent/preset['allocation']['file']).open() as f:
        rows = list(csv.DictReader(f))
    commands = []; outputs = []; groups = []; inputs = []
    configs = {}
    def command(name, config, argv):
        cfg = output/(name+'.json'); configs[cfg] = config
        commands.append(dict(entrypoint='release_stage.py', argv=['--stage-config',str(cfg),*map(str,argv)]))
    base = dict(workspace=str(workspace),processor=str(processor))
    if stage == 'fitting':
        vlm_requests = path(bindings['vlm_requests']); dit_requests = path(bindings['dit_requests'])
        names = {s: [r['layer'] for r in rows if r['scope']==s] for s in ('vlm','dit')}
        for suite in preset['suites']:
            metrics = root/'metrics'/suite
            command('metrics_'+suite, dict(base,operation='metrics',checkpoint=models[suite]),
                    ['--out-dir',metrics,'--ckpt',models[suite],'--request-dir',dit_requests,
                     '--limit',64,'--calib-calls',48,'--device','cuda:0','--ff-up-activation-curvature','diag_mean'])
            outputs.extend(map(str,[metrics/'metrics.pt',metrics/'consumption.json']))
        for k in preset['candidate_k']:
            dest = root/'bank'/f'vlm_k{k}'
            command(f'vlm_k{k}',dict(base,operation='vlm',checkpoint=models['libero_spatial']),
                    ['--out-dir',dest,'--request-dir',vlm_requests,'--limit',64,'--max-tokens',preset['fitting']['vlm_max_tokens'],
                     '--seed',20260713,'--sample-size',50000,'--iters',8,'--codebook-size',k,'--device','cuda:0',
                     '--only-layers',output/'vlm.txt','--swiglu-down-curvature','diag','--packed','--packed-only','--skip-existing-packed'])
            groups.append(dict(scope='vlm',suite='shared',init='aware',k=k,layers=names['vlm'],directory=str(dest)))
            outputs.extend(str(dest/(n+'.pt')) for n in names['vlm'])
            for suite in preset['suites']:
                dest = root/'bank'/f'dit_{suite}_k{k}'
                command(f'dit_{suite}_k{k}',dict(base,operation='dit',checkpoint=models[suite]),
                        ['--suite',suite,'--k',k,'--metrics-dir',root/'metrics'/suite,'--fits-dir',dest,
                         '--modules-file',output/'dit.txt','--packed','--packed-only','--device','cuda:0',
                         '--ff-up-activation-curvature','diag_mean','--proj-out-structure-aware','none'])
                groups.append(dict(scope='dit',suite=suite,init='aware',k=k,layers=names['dit'],directory=str(dest)))
                outputs.extend(str(dest/(n+'__ck.pt')) for n in names['dit'])
        # Match original sorted directory consumption; require exactly the recorded 64.
        for folder in (vlm_requests, dit_requests):
            files=sorted(folder.glob('*.msgpack'))
            if len(files)!=64:
                raise ValueError('Fitting input directory must contain exactly the recorded 64 requests')
            inputs.extend(ref(p) for p in files)
    elif stage == 'allocation':
        bank = path(bindings['bank_manifest']); requests = path(bindings['allocation_requests'])
        files = sorted(requests.glob('*.msgpack'))
        if len(files) != 40 or any(sum(p.name.startswith(s + '_') for p in files) != 10 for s in preset['suites']):
            raise ValueError('Allocation requires 40 requests, ten named for each suite')
        inputs.extend(ref(p) for p in files)
        settings = preset['allocation']['origin_search']
        command('allocation',dict(base,operation='allocation',models=models,bank_manifest=ref(bank),
                                  allocation_requests=str(requests),allocation_work=str(root/'work')),
                ['--out',root/'allocation.csv','--target-payload-bytes',settings['target_payload_bytes'],
                 '--target-nominal-bits',settings['target_nominal_bits'],'--start-k',settings['start_k'],
                 '--min-bit',settings['min_bit'],'--refresh-every',settings['refresh_every'],
                 '--checkpoint-step',settings['checkpoint_step'],'--scope-policy',settings['scope_policy'],
                 '--workers-per-suite',2,'--n-gpus',8,'--requests-per-suite',10])
        inputs.append(ref(bank));outputs=[str(root/'allocation.csv'),str(root/'allocation.csv.manifest.json'),str(root/'allocation_ckpt2.00.csv')]
    else:
        raise ValueError('Expected fitting or allocation')
    output.mkdir(parents=True)
    for p,config in configs.items():
        p.write_text(json.dumps(config,indent=2)+'\n');inputs.append(ref(p))
    if stage=='fitting':
        for scope in ('vlm','dit'):
            p=output/(scope+'.txt');p.write_text('\n'.join(names[scope])+'\n');inputs.append(ref(p))
        (output/'inventory-spec.json').write_text(json.dumps(dict(schema='sacq_fitted_inventory_build_v1',groups=groups),indent=2)+'\n')
    row=dict(commands=commands,inputs=inputs,outputs=outputs)
    (output/'stage-row.json').write_text(json.dumps(row,indent=2)+'\n')
    (output/'preset-receipt.json').write_text(json.dumps(dict(preset=ref(PRESET),stage=stage,executed=False),indent=2)+'\n')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fitting','allocation'))
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    row=prepare(args.stage,json.loads(args.bindings.read_text()),args.output_dir)
    print(f'Prepared {len(row["commands"])} commands; no experiment launched')


if __name__=='__main__':main()
