"""Official50 simulator with exact per-slot reset-history reconstruction.

This is an operational recovery of r2.  Scientific states, RNG journals,
observations, action stepping, and success criteria are unchanged.  When an
environment that has executed actions no longer reproduces the reset-only
reference, rebuild a five-environment group, retain the same canonical slot,
and replay the immutable reset prefix before accepting the observation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import socket
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--socket', required=True)
    a = p.parse_args(); plan = json.loads(Path(a.plan).read_text()); root = Path(plan['root'])
    sys.path[:0] = [plan['repo'], str(Path(plan['repo'])/'external_dependencies/LIBERO'), str(Path(__file__).parent)]
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
    sys.path.insert(0, '@WORKSPACE@/libero_pro_eval')
    import numpy as np
    from seed17_adapter import official_env_class, state_sha, observation_sha
    from vast12_hqq_sim_2026_09_09 import send, receive
    repo = Path(plan['repo']); lib = module('official_libero_r3', repo/'gr00t/eval/sim/LIBERO/libero_env.py')
    wrap = module('official_multistep_r3', repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py')
    tasks = json.loads((root/'tasks.json').read_text())
    suites = {name:lib.benchmark.get_benchmark_dict()[name]() for name in plan['suites']}
    adapter = official_env_class(lib.LiberoEnv)
    task = records = states = None
    envs = []

    def make(current):
        return wrap.MultiStepWrapper(adapter(current['bddl'], current['language']), np.array([0]), np.array([0]),
            n_action_steps=8, max_episode_steps=720, terminate_on_success=False)

    def reset_one(env, ep):
        saved = records[ep]
        random.setstate(saved['rng'][0]); np.random.set_state(saved['rng'][1])
        obs, info = env.reset(seed=saved['reset_seed'], options=dict(episode_id=ep,
            initial_state=np.asarray(states[ep]), initial_state_sha256=saved['state_sha256']))
        base_obs = {k:(v if isinstance(v,str) else v[0]) for k,v in obs.items()}
        return obs, observation_sha(base_obs)

    def reconstruct_slot(slot, ep):
        # The canonical audit created five environments together.  Preserve that
        # creation order and keep the matching slot; a single isolated make()
        # is not equivalent for the affected drawer fixture.
        group = [make(task) for _ in range(plan['rollout']['n_envs'])]
        chosen = group[slot]
        try:
            obs = actual = None
            for prior in range(ep + 1):
                obs, actual = reset_one(chosen, prior)
                if actual != records[prior]['observation_sha256']:
                    raise ValueError(('Official reset history mismatch after five-env reconstruction',
                        task['task'], ep, prior, slot))
            envs[slot].close(); envs[slot] = chosen; group[slot] = None
            return obs, actual
        finally:
            for env in group:
                if env is not None: env.close()

    sock_path = Path(a.socket); sock_path.parent.mkdir(parents=True, exist_ok=True); sock_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX); server.bind(str(sock_path)); server.listen(1); conn, _ = server.accept()
    try:
        while True:
            request = receive(conn)
            if request['op'] == 'close': break
            if request['op'] == 'task':
                for env in envs: env.close()
                task = tasks[request['task_index']]
                audit_dir = Path(plan['reset_audit_root'])/'tasks'/task['suite']/task['task']
                result = json.loads((audit_dir/'result.json').read_text())
                if result['status'] != 'PASS' or result['contract_sha256'] != plan['reset_task_contract_sha256'][task['suite']+'/'+task['task']]:
                    raise ValueError('Official state audit missing or stale')
                path = audit_dir/'reference.pkl'
                if hashlib.sha256(path.read_bytes()).hexdigest() != result['reference_rng_sha256']:
                    raise ValueError('Reset RNG journal changed')
                records = pickle.loads(path.read_bytes()); states = suites[task['suite']].get_task_init_states(task['task_id'])
                if [r['episode_id'] for r in records] != list(range(50)):
                    raise ValueError('Official episode coverage mismatch')
                if any(state_sha(np.asarray(states[e])) != records[e]['state_sha256'] for e in range(50)):
                    raise ValueError('Official states changed since audit')
                if plan.get('render_reference_mode') != 'same_process_saved_rng_exact':
                    raise ValueError('Explicit renderer reference contract required')
                reference = lib.LiberoEnv(task['bddl'], task['language'])
                original_hashes = [r['observation_sha256'] for r in records]
                try:
                    for e, saved in enumerate(records):
                        random.setstate(saved['rng'][0]); np.random.set_state(saved['rng'][1])
                        if saved['reset_seed'] is not None: reference._env.seed(saved['reset_seed'])
                        reference._env.reset(); raw = reference._env.set_init_state(np.asarray(states[e]))
                        if not np.array_equal(reference._env.get_sim_state(), np.asarray(states[e])):
                            raise ValueError('Renderer reference changed official state')
                        saved['observation_sha256'] = observation_sha(reference._process_observation(raw))
                finally: reference.close()
                renderer_record = dict(host=os.uname().nodename, pid=os.getpid(),
                    original_journal_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),
                    episodes=[dict(episode_id=e, state_sha256=r['state_sha256'],
                        original_observation_sha256=original_hashes[e], renderer_observation_sha256=r['observation_sha256'])
                        for e,r in enumerate(records)])
                renderer_path=root/'renderer_references'/str(os.getpid())/task['suite']/(task['task']+'.json')
                renderer_path.parent.mkdir(parents=True,exist_ok=True)
                temp=renderer_path.with_suffix('.tmp'); temp.write_text(json.dumps(renderer_record,indent=2)); temp.replace(renderer_path)
                print('RENDERER_REFERENCE_READY', task['task'], sum(original_hashes[e]!=r['observation_sha256'] for e,r in enumerate(records)), flush=True)
                envs = [make(task) for _ in range(plan['rollout']['n_envs'])]
                send(conn, {'status':'READY', 'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            elif request['op'] == 'reset':
                output = []
                for slot, ep in request['items']:
                    obs, actual = reset_one(envs[slot], ep)
                    if actual != records[ep]['observation_sha256']:
                        obs, actual = reconstruct_slot(slot, ep)
                    output.append((slot, obs, actual))
                send(conn, output)
            elif request['op'] == 'step':
                output = []
                for slot, action in request['items']:
                    obs, reward, done, truncated, info = envs[slot].step(action)
                    output.append((slot, obs, bool(done or truncated), bool(np.any(info['success'])), len(envs[slot].reward)))
                send(conn, output)
            else: raise ValueError('Unknown RPC')
    finally:
        for env in envs: env.close()
        conn.close(); server.close(); sock_path.unlink(missing_ok=True)


if __name__ == '__main__': main()

