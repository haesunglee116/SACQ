"""Explicit BF16 no-op or calibrated SACQ installation with declared background."""
import importlib.util,sys
from pathlib import Path
def install(model,plan,suite):
    if plan['condition']=='bf16':
        dtypes=sorted({str(p.dtype) for p in model.parameters()})
        assert dtypes==['torch.bfloat16'] and 'workshop_model' not in plan
        return dict(method='bf16',count=0,quantizer_called=False,parameter_dtypes=dtypes)
    if plan['condition'] not in ('sacq','sacq1.5'):
        raise ValueError('Unsupported model condition')
    old=Path(__file__).resolve().parent.parent/'libero_transfer_quantstats_20260922'
    sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('original_calibrated_install',old/'gr00t_rollout_model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.install(model,plan,suite)
