"""Path-explicit fitting/allocation adapter around the original algorithms."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value

def manifest_preflight(allocator, bank, vlm, dit, min_bit):
    """Preserve candidate coverage without historical directory-name heuristics."""
    ks=[k for k in allocator.LADDER
        if k not in allocator.HQQ_SENTINEL_K and allocator.BIT_OF_K[k]>=min_bit-1e-12]
    checked=0
    for scope,names,suites in (('vlm',vlm,allocator.SUITES[:1]),('dit',dit,allocator.SUITES)):
        for suite in suites:
            for name in names:
                for k in ks:
                    bank.find_packed(scope,name,k,suite=suite,init='aware',verify=True,exact=True)
                    checked+=1
    return checked

def manifest_packed_path(bank, kind, suite, name, k):
    """Bind runtime reconstruction to the same exact inventory as preflight."""
    return bank.find_packed(kind,name,k,suite=suite,init='aware',verify=True,exact=True)

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--stage-config',default=os.environ.get('SACQ_RELEASE_STAGE_CONFIG'))
    args,rest=p.parse_known_args()
    if not args.stage_config:p.error('--stage-config is required')
    cfgpath=Path(args.stage_config).resolve();cfg=json.loads(cfgpath.read_text())
    workspace=Path(cfg['workspace']).resolve();b=workspace/'libero_pro_eval'
    sys.path[:0]=[str(b),str(b/'Isaac-GR00T-N1.7')]
    processor=cfg['processor']
    if not Path(processor).is_dir():raise ValueError('Explicit local processor checkpoint missing')
    import transformers
    for cls in (transformers.AutoProcessor,transformers.Qwen3VLProcessor,transformers.Qwen3VLForConditionalGeneration):
        old=cls.from_pretrained
        def local(cls,name,*a,_old=old,**kw):
            return _old(processor if str(name)=='nvidia/Cosmos-Reason2-2B' else name,*a,**kw)
        cls.from_pretrained=classmethod(local)
    source=b/'quantization_analysis/workshop_fixedalloc_refit_20260919/source'
    stage=cfg['operation']
    if stage in ('vlm','dit','metrics'):
        model=Path(cfg['checkpoint']).resolve()
        if not model.is_dir():raise ValueError('Explicit suite checkpoint missing')
        name={'vlm':'vlm_fitter.py','dit':'dit_fitter.py','metrics':'dit_metrics.py'}[stage]
        m=module(source/name,'release_'+stage)
        if stage=='vlm':
            if '--swiglu-down-curvature' not in rest or rest[rest.index('--swiglu-down-curvature')+1]!='diag':
                raise ValueError('Final metric-v2 path requires explicit downstream diagonal weighting')
            if '--only-layers' not in rest:raise ValueError('Explicit full candidate layer list required')
            m.local_ckpt=lambda:str(model)
        elif stage=='dit':
            if '--ff-up-activation-curvature' not in rest or rest[rest.index('--ff-up-activation-curvature')+1]!='diag_mean':
                raise ValueError('Explicit metric-v2 activation curvature required')
            # The fitter receives this binding after parsing its suite arguments.
        else:
            if '--ckpt' not in rest or Path(rest[rest.index('--ckpt')+1]).resolve()!=model:
                raise ValueError('Metric checkpoint must match the explicit stage configuration')
            if '--ff-up-activation-curvature' not in rest or rest[rest.index('--ff-up-activation-curvature')+1]!='diag_mean':
                raise ValueError('Explicit metric-v2 activation curvature required')
        sys.argv=[str(source/name),*rest]
        if stage=='dit':
            m.main(strict_metrics=True, checkpoint=model)
        else:
            m.main()
    elif stage=='allocation':
        import artifact_registry as registry
        if 'bank_manifest' in cfg:
            from release_bank import PackedBank
            bank=PackedBank.from_ref(cfg['bank_manifest'])
            registry.find_packed=bank.find_packed
        else:
            registry.ROT=Path(cfg['bank']);registry.FS2=Path(cfg['bank'])
        import coord_descent_final_multisuite_2026_08_24 as a
        if 'bank_manifest' in cfg:
            a.preflight_exact_artifacts=lambda vlm,dit,min_bit:manifest_preflight(a,bank,vlm,dit,min_bit)
            a.exact_packed_path=lambda kind,suite,name,k:manifest_packed_path(bank,kind,suite,name,k)
        models=cfg['models']
        if set(models)!=set(a.SUITES):raise ValueError('All four suite checkpoints required')
        if not all(Path(v).is_dir() for v in models.values()):raise ValueError('Missing suite checkpoint')
        a.find_packed=registry.find_packed
        a.infra.find_packed=registry.find_packed
        a.infra.local_ckpt=lambda suite:str(models[suite])
        a.REQUEST_DIR=Path(cfg['allocation_requests']);a.QA=Path(cfg['allocation_work'])
        # Spawned allocation workers re-enter this same adapter and path binding.
        os.environ['SACQ_RELEASE_STAGE_CONFIG']=str(cfgpath)
        a.__file__=__file__
        sys.argv=[__file__,*rest];a.main()
    else:raise ValueError('Supported operations: metrics, vlm, dit, allocation')

if __name__=='__main__':main()
