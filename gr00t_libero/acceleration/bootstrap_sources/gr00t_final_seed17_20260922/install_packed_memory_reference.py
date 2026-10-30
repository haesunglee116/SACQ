"""Full-policy installation adapter for the approved packed-memory mode.

This does not install rollout acceleration. Fresh process and whole-policy output
validation are the caller's responsibility; no measured-memory value is inferred.
"""
from pack_gptq_reference import pack_installed_gptq
from pack_sacq_reference import pack_installed_sacq

def install(model,plan,suite):
    condition=plan['condition']
    assert condition in ['sacq1.5','sacq2','sacq3','gptq2_baseline','gptq3_baseline','hqq2_baseline','hqq3_baseline']
    if condition=='sacq1.5':
        from model_install import install as original
    elif condition in ['sacq2','sacq3']:
        from model_install_multi import install as original
    elif condition=='gptq2_baseline':
        from model_install_gptq_baseline_r2 import install as original
    elif condition=='gptq3_baseline':
        from model_install_gptq3_baseline import install as original
    else:
        from model_install_hqq_baseline import install as original
    original_report=original(model,plan,suite)
    if condition.startswith('sacq'):
        scalar=plan['workshop_model']['gptq']
    elif condition.startswith('gptq'):
        scalar=plan['baseline_model']['weights']
    else:
        scalar=plan['baseline_model']['vision']
    scalar_report=pack_installed_gptq(model,scalar)
    sacq_report=pack_installed_sacq(model)
    model.eval().requires_grad_(False)
    return dict(condition=condition,original_install=original_report,
        scalar=scalar_report,sacq=sacq_report,full_policy_output_verified=False,
        resident_memory_measured=False,scalar_representation='custom_exact_level_dictionary',
        sacq_representation='native_packed_index_reference',dense_cache=False,cuda_graph=False)
