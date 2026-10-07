"""Exact existing official7 env1 worker, approved model install + guarded acceleration."""
import atexit,os,sys
from pathlib import Path
from io_utils import load,save,sha
R=Path(__file__).resolve().parent;W=R.parent;B=W/'libero_pro_eval'
def main(path,task_index):
    plan=load(path);root=Path(plan['root']);assert plan['diagnostic_only'] and not plan['held_out']
    assert plan['rollout']==dict(environment_seed=17,policy_seed=20260824,episodes_per_task=50,n_envs=1,action_steps=8,max_episode_steps=720,termination='official_done_or_720',success='success_any',settling_steps=0)
    assert plan['worker_task_index']==task_index
    for ref in plan['sources']:assert sha(ref['path'])==ref['sha256'],ref['path']
    receipt=load(root/'EXECUTION_RECEIPT.json')
    assert receipt['execution_status']=='RUNNING' and receipt['protocol_sha256']==sha(root/'protocol.json')
    protocol=load(root/'protocol.json');assert dict(path=str(path),sha256=sha(path)) in protocol['inputs']
    sys.path[:0]=[str(R),str(B),plan['repo'],str(W/'simplerenv_n17_feasibility_2026_08_27'),str(Path(plan['worker_source']).parent)]
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    for key in list(os.environ):
        if key.startswith(('VLM_','DIT_','SACQ_','ACTION_HEAD_','ACT_QUANT','TAC_')):os.environ.pop(key)
    from rollout_acceleration_default_20260920 import AsyncProfile,DefaultAcceleration
    profile=AsyncProfile(root/'performance',f'task{task_index}');accel=DefaultAcceleration(profile)
    atexit.register(profile.save,True)
    canonical=B/'quantization_analysis/workshop_suitealloc_long500_s7_env1_20260919/source/worker.py'
    if sha(plan['worker_source'])!=sha(canonical):
        raise ValueError('Materialized runtime requires the pinned worker template')
    import rollout_runtime as worker
    Path(plan['socket_root']).mkdir(parents=True,exist_ok=True)
    sys.argv=[str(canonical),'--plan',str(path),'--gpu',str(task_index)]
    worker.main(PROFILE=profile,ACCEL=accel,PROFILE_WRAPPER=str(R/'simulator_profile.py'))
    task=load(root/'tasks.json')[task_index];rows=[load(p) for p in (root/'episodes/final_seed17'/task['suite']/task['task']).glob('*.json')]
    assert len(rows)==50 and {r['episode_id'] for r in rows}==set(range(50))
    assert all(r['base_seed']==17 and r['policy_seed']==20260824 and r['n_envs']==1 for r in rows)
    save(root/'completed'/f'{task_index}.json',dict(status='COMPLETE_DIAGNOSTIC',episodes=50,successes=sum(r['success'] for r in rows),task=task['task'],suite=task['suite'],acceleration=accel.audit()))
if __name__=='__main__':main(Path(sys.argv[1]),int(sys.argv[2]))

