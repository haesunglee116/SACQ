"""Isolated LIBERO simulator, exact reset RNG journal and batched local RPC."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import socket
import struct
import sys

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_obs(obs):
    h = hashlib.sha256()
    for k, v in sorted(obs.items()):
        h.update(k.encode())
        if isinstance(v, str):
            h.update(v.encode())
        else:
            a = np.asarray(v)
            h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('wb') as f:
        pickle.dump(value, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        chunk = s.recv(n-len(b))
        if not chunk: raise EOFError('RPC disconnected')
        b.extend(chunk)
    return bytes(b)


def receive(s):
    return pickle.loads(recv_exact(s, struct.unpack('!Q', recv_exact(s, 8))[0]))


def send(s, data):
    data = pickle.dumps(data, protocol=5)
    s.sendall(struct.pack('!Q', len(data)) + data)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    p.add_argument('--socket'); p.add_argument('--registry-only', action='store_true')
    a = p.parse_args(); plan = json.loads(Path(a.plan).read_text())
    repo = Path(plan['repo'])
    config_dir = Path(plan['root'])/'libero_config'
    config_dir.mkdir(parents=True, exist_ok=True)
    libroot = repo/'external_dependencies/LIBERO/libero/libero'
    paths = dict(benchmark_root=str(libroot), bddl_files=str(libroot/'bddl_files'),
        init_states=str(libroot/'init_files'), datasets=str(libroot.parent/'datasets'), assets=str(libroot/'assets'))
    for key in ('benchmark_root','bddl_files','init_states','assets'):
        assert Path(paths[key]).is_dir(), (key, paths[key])
    target = config_dir/'config.yaml'
    content = json.dumps(paths, indent=2)+'\n'
    if target.exists():
        assert json.loads(target.read_text()) == paths
    else:
        temp = target.with_name('config.tmp.'+str(os.getpid()))
        temp.write_text(content); temp.replace(target)
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    lib = load_module('libero_explicit_env', repo/'gr00t/eval/sim/LIBERO/libero_env.py')
    wrap = load_module('libero_explicit_multistep', repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py')
    suites = {s: lib.benchmark.get_benchmark_dict()[s]() for s in plan['suites']}
    tasks = []
    for suite, b in suites.items():
        assert b.get_num_tasks() == 10
        for i in range(10):
            t = b.get_task(i)
            tasks.append(dict(suite=suite, task=t.name, task_id=i, language=t.language,
                bddl=os.path.join(lib.get_libero_path('bddl_files'), t.problem_folder, t.bddl_file)))
    if a.registry_only:
        dest = Path(plan['root'])/'tasks.json'
        assert not dest.exists() or json.loads(dest.read_text()) == tasks
        dest.write_text(json.dumps(tasks, indent=2)); return
    assert tasks == json.loads((Path(plan['root'])/'tasks.json').read_text())
    def make(task):
        env = lib.LiberoEnv(task['bddl'], task['language'])
        return wrap.MultiStepWrapper(env, np.array([0]), np.array([0]),
            n_action_steps=plan['rollout']['action_steps'],
            max_episode_steps=plan['rollout']['max_episode_steps'],
            terminate_on_success=False)
    sock = socket.socket(socket.AF_UNIX); sock.bind(a.socket); sock.listen(1)
    conn, _ = sock.accept(); envs = []; task = None; records = None
    try:
        while True:
            request = receive(conn)
            if request['op'] == 'close': break
            if request['op'] == 'task':
                for env in envs: env.close()
                task = tasks[request['task_index']]
                path = Path(plan['root'])/'initial_states'/task['suite']/(task['task']+'.pkl')
                if not path.exists():
                    reference = make(task); records = []
                    # Explicit seed7 sequential reset stream, no per-task offset.
                    random.seed(plan['rollout']['environment_seed'])
                    np.random.seed(plan['rollout']['environment_seed'])
                    for e in range(plan['rollout']['episodes_per_task']):
                        before = (random.getstate(), np.random.get_state())
                        obs, info = reference.reset(seed=plan['rollout']['environment_seed'] if e == 0 else None)
                        records.append(dict(episode_id=e, rng=before,
                            reset_seed=plan['rollout']['environment_seed'] if e == 0 else None,
                            initial_observation_sha256=digest_obs(obs)))
                    reference.close()
                    save(path, records)
                with path.open('rb') as f: records = pickle.load(f)
                assert [r['episode_id'] for r in records] == list(range(50))
                envs = [make(task) for _ in range(plan['rollout']['n_envs'])]
                send(conn, {'status':'READY', 'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            elif request['op'] == 'reset':
                output = []
                for slot, ep in request['items']:
                    r = records[ep]; random.setstate(r['rng'][0]); np.random.set_state(r['rng'][1])
                    obs, info = envs[slot].reset(seed=r['reset_seed'])
                    actual = digest_obs(obs)
                    if actual != r['initial_observation_sha256']:
                        raise ValueError(('Initial observation restore mismatch', task['task'], ep, actual, r['initial_observation_sha256']))
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
        conn.close(); sock.close()


if __name__ == '__main__': main()
