"""BF16-only worker; preserved canonical episode/call noise and true batch inference."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args(); plan = json.loads(Path(a.plan).read_text()); root = Path(plan['root'])
    if plan.get('rollout_authorized') is not True:
        raise RuntimeError('No rollout authorization; preparation only')
    if plan.get('method') != 'suitealloc_workshopcal' or plan.get('reset_identity_status') != 'PASS':
        raise ValueError('BF16 execution contract and reset identity PASS required')
    if any(k.startswith(('VLM_', 'DIT_', 'ACT_QUANT', 'ACTION_HEAD_', 'TAC_')) for k in os.environ):
        raise ValueError('Inherited quantization controls are forbidden')
    import torch
    import numpy as np
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.data.embodiment_tags import EmbodimentTag
    from transformers import AutoProcessor, Qwen3VLProcessor, Qwen3VLForConditionalGeneration
    from libero_episode_noise_injection_2026_09_09 import NoiseInjection
    from fixed_episode_noise_2026_09_09 import read_episode
    from vast12_hqq_sim_2026_09_09 import send, receive
    # Resolve the already hash-verified dependency locally; no weight/config changes.
    for cls in (Qwen3VLForConditionalGeneration, Qwen3VLProcessor, AutoProcessor):
        original = cls.from_pretrained
        def local(cls, name, *args, _original=original, **kwargs):
            if str(name) == 'nvidia/Cosmos-Reason2-2B': name = plan['cosmos']
            return _original(name, *args, **kwargs)
        cls.from_pretrained = classmethod(local)
    torch.set_num_threads(2)
    tasks = json.loads((root/'tasks.json').read_text())
    jobs = [(1.5, plan['worker_task_index'], 0)]
    active_key = None; core = None; selected_task = None; injector = None; noises = None
    sim_socket = str(Path(plan['socket_root'])/f'sim_{a.gpu}_{os.getpid()}.sock')
    log = (root/f'sim_{a.gpu}.log').open('a')
    sim = subprocess.Popen([plan['sim_python'], plan['sim_source'], '--plan', a.plan, '--socket', sim_socket],
        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONPATH':plan['repo']+':'+plan['repo']+'/external_dependencies/LIBERO'})
    sock = socket.socket(socket.AF_UNIX)
    deadline = time.monotonic()+600
    while not Path(sim_socket).exists():
        if sim.poll() is not None: raise RuntimeError('Simulator failed; see '+str(log.name))
        if time.monotonic()>deadline: raise TimeoutError('Simulator startup')
        time.sleep(.2)
    sock.connect(sim_socket); sock.settimeout(600)
    def rpc(v): send(sock, v); return receive(sock)
    def state(**v): atomic(root/f'workers/{a.gpu}.json', dict(time=time.time(), pid=os.getpid(), gpu=a.gpu, **v))
    try:
        while True:
            available = sorted(jobs, key=lambda j: (active_key != (j[0], tasks[j[1]]['suite']), j))
            chosen = None; lock = None
            for bits, task_index, first in available:
                task = tasks[task_index]; key = f'suitealloc_workshopcal/{task["suite"]}/{task["task"]}/{first:03d}'
                if (root/'chunks'/f'{key}.json').exists(): continue
                path = root/'claims'/f'{key}.lock'; path.parent.mkdir(parents=True, exist_ok=True)
                candidate = path.open('a')
                try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: candidate.close(); continue
                if (root/'chunks'/f'{key}.json').exists(): candidate.close(); continue
                chosen=(bits, task_index, first, key); lock=candidate; break
            if chosen is None:
                state(status='QUEUE_PASS_FINISHED'); return
            bits, task_index, first, key = chosen; task = tasks[task_index]
            state(status='MODEL_LOADING' if active_key != (bits, task['suite']) else 'PREPARING_EPISODES', key=key)
            if active_key != (bits, task['suite']):
                if injector: injector.close()
                injector=None; core=None; policy=None; gc.collect(); torch.cuda.empty_cache()
                core = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, plan['models'][task['suite']], device='cuda:0', strict=True)
                core.model.eval()
                for par in core.model.parameters(): par.requires_grad_(False)
                audit = {'method': 'bf16', 'quantizer_called': False, 'parameter_dtypes': sorted({str(p.dtype) for p in core.model.parameters()})}
                if audit['parameter_dtypes'] != ['torch.bfloat16']:
                    raise ValueError('Original BF16 model dtype mismatch: '+str(audit))
                from workshop_suitealloc_bowl_20260919 import install
                state(status='CALIBRATED_INSTALLING', key=key)
                audit=install(core.model,plan,task['suite'])
                atomic(root/f'install/{a.gpu}_suitealloc_workshopcal_{task["suite"]}.json',audit)
                install_sha=hashlib.sha256((root/f'install/{a.gpu}_suitealloc_workshopcal_{task["suite"]}.json').read_bytes()).hexdigest()
                print('CALIBRATED_INSTALL_PASS',task['suite'],audit['count'],flush=True)
                policy = Gr00tSimPolicyWrapper(core); injector = NoiseInjection(core.model.action_head)
                shape = [1, core.model.action_head.config.action_horizon, core.model.action_head.action_dim]
                table = Path(plan['noise_root'])/('x'.join(map(str,shape)))
                (root/'noise').mkdir(exist_ok=True)
                with (root/'noise/table.lock').open('a') as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    if not (table/'manifest.json').exists():
                        raise FileNotFoundError('Canonical saved noise required; regeneration forbidden')
                    manifest=json.loads((table/'manifest.json').read_text())
                    assert manifest['seed']==20260824 and manifest['episodes']==50 and manifest['calls']==90
                noises = [read_episode(table, e) for e in range(50)]
                active_key = (bits, task['suite'])
            if selected_task != task_index:
                if selected_task is not None:
                    # Discard process-global simulator/renderer state from the prior task.
                    send(sock, {'op':'close'}); sock.close(); sim.wait(timeout=120)
                    Path(sim_socket).unlink(missing_ok=True)
                    sim = subprocess.Popen([plan['sim_python'], plan['sim_source'], '--plan', a.plan, '--socket', sim_socket],
                        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONPATH':plan['repo']+':'+plan['repo']+'/external_dependencies/LIBERO'})
                    sock = socket.socket(socket.AF_UNIX); deadline=time.monotonic()+600
                    while not Path(sim_socket).exists():
                        if sim.poll() is not None:raise RuntimeError('Fresh task simulator failed')
                        if time.monotonic()>deadline:raise TimeoutError('Fresh task simulator startup')
                        time.sleep(.2)
                    sock.connect(sim_socket);sock.settimeout(600)
                initial = root/'official_reset_locks'/task['suite']/(task['task']+'.lock'); initial.parent.mkdir(parents=True, exist_ok=True)
                with initial.open('a') as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    manifest_path = Path(plan['reset_audit_root'])/'tasks'/task['suite']/task['task']/'reference.pkl'
                    if not manifest_path.is_file():
                        raise FileNotFoundError('Canonical reset records required')
                    ready=rpc({'op':'task','task_index':task_index})
                selected_task=task_index
            result_dir = root/'episodes'/'suitealloc_workshopcal'/task['suite']/task['task']
            pending=[e for e in range(first,first+50) if not (result_dir/f'{e:03d}.json').exists()]
            active={}; obs={}; calls={}; successes={}; initial_hash={}
            def reset_free():
                pairs=[]
                for slot in range(1):
                    if slot not in active and pending:
                        e=pending.pop(0); active[slot]=e; calls[slot]=0; successes[slot]=False; pairs.append((slot,e))
                if pairs:
                    for slot, observation, h in rpc({'op':'reset','items':pairs}):
                        expected=plan['paired_initial_observation_sha256'].get(task['task']+'/'+str(active[slot]))
                        if expected is not None and h != expected:
                            raise ValueError(('Single-env changed paired initial observation',active[slot],h,expected))
                        obs[slot]=observation; initial_hash[slot]=h
            reset_free()
            while active:
                slots=sorted(active)
                batch={k: tuple(obs[s][k] for s in slots) if isinstance(obs[slots[0]][k],str)
                    else np.stack([obs[s][k] for s in slots]) for k in obs[slots[0]]}
                noise=torch.cat([noises[active[s]][calls[s]] for s in slots], dim=0)
                with torch.inference_mode(), injector.selected(noise): actions,_=policy.get_action(batch)
                updates=rpc({'op':'step','items':[(s,{k:v[i,:8] for k,v in actions.items()}) for i,s in enumerate(slots)]})
                for s, observation, done, success, steps in updates:
                    obs[s]=observation; calls[s]+=1; successes[s] |= success
                    if done:
                        e=active[s]
                        atomic(result_dir/f'{e:03d}.json', dict(status='COMPLETE', method='suitealloc_workshopcal', suite=task['suite'],task=task['task'],
                            weight_bits=plan['quantization'][task['suite']]['full_scope_nominal_bpw'], language_action_nominal_bits=plan['quantization'][task['suite']]['retained_language_action_nominal_bpw'], group_size=64, quantization_install_sha256=install_sha, base_seed=7, initial_state_source='LIBERO_official_pruned_init', policy_seed=20260824, episode_id=e, success=successes[s], steps=steps, calls=calls[s],
                            initial_observation_sha256=initial_hash[s], noise_manifest_sha256=hashlib.sha256((table/'manifest.json').read_bytes()).hexdigest(),
                            protocol_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(), n_envs=1,true_batch=True,time=time.time()))
                        print('EPISODE_COMPLETE',key,e,successes[s],flush=True); del active[s]
                reset_free(); state(status='ROLLOUT_RUNNING', key=key, episodes_remaining=len(active)+len(pending), calls=calls)
            atomic(root/'chunks'/f'{key}.json', {'status':'COMPLETE','episodes':list(range(first,first+50)),'time':time.time()})
            lock.close()
    finally:
        try: send(sock, {'op':'close'})
        except Exception: pass
        sock.close()
        try: sim.wait(timeout=120)
        except subprocess.TimeoutExpired:
            sim.terminate()
            try: sim.wait(timeout=10)
            except subprocess.TimeoutExpired: sim.kill(); sim.wait()
            print('SIMULATOR_SHUTDOWN_FORCED_AFTER_WORKER_EXIT',flush=True)
        log.close()


if __name__ == '__main__': main()
