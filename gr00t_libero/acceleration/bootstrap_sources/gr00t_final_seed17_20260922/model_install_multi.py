"""Same calibrated overlay installer, allocation-derived scalar bits and counts."""
import importlib.util,inspect,sys
from pathlib import Path
def install(model,plan,suite):
    assert plan['condition'] in ['sacq3','sacq2']
    old=Path(__file__).resolve().parent.parent/'libero_transfer_quantstats_20260922'
    sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('calibrated_multi_install',old/'gr00t_rollout_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=inspect.getsource(module.install)
    replacements={
        "assert len(m['gptq'])==105":"assert len(m['gptq'])==plan['gptq_count']",
        "p['bits']==(3 if name=='action_head.model.proj_out_2' else 4)":"p['bits']==r['bits']",
        "count=337,gptq_frozen=105":"count=plan['sacq_count'],gptq_frozen=plan['gptq_count']"
    }
    for old_text,new_text in replacements.items():
        assert source.count(old_text)==1,old_text
        source=source.replace(old_text,new_text)
    exec(compile(source,__file__,'exec'),module.__dict__)
    return module.install(model,plan,suite)
