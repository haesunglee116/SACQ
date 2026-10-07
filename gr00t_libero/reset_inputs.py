"""Prepare reset-only journals for the pinned seed17/env1 rollout.

Static port of the historical reset audit with its seed17 adapter substitutions.
No policy forward or environment steps are allowed. A PASS is reset equivalence,
not scientific sample approval. Environment, Python and NumPy seeds are explicitly 17.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class gate:
    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def atomic(path, value):
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n')
        temp.replace(path)


def validate(c, task_index):
    gate.require(c.get('schema') == 'sacq_reset_inputs_v1', 'Explicit reset input schema required')
    expected = dict(environment_seed=17, python_seed=17, numpy_seed=17, n_envs=1,
                    episode_ids=list(range(50)), n_action_steps=8, max_episode_steps=720,
                    terminate_on_success=False, policy_forward_calls=0, environment_steps=0)
    gate.require(c.get('settings') == expected, 'Only the pinned historical reset settings are supported')
    gate.require(isinstance(c.get('tasks'), list) and 0 <= task_index < len(c['tasks']), 'Explicit task index required')
    for key in ('repo', 'adapter_directory', 'output'):
        gate.require(Path(c[key]).is_absolute(), 'Absolute paths required')
    refs = c.get('sources', []) + c.get('inputs', [])
    gate.require(bool(c.get('sources')) and bool(c.get('inputs')), 'Hashed sources and inputs required')
    bound = {}
    for ref in refs:
        path = Path(ref['path'])
        gate.require(path.is_absolute() and gate.sha(path) == ref['sha256'], 'Changed source/input')
        bound[str(path.resolve())] = ref['sha256']
    repo = Path(c['repo']); adapter = Path(c['adapter_directory'])
    for path in (adapter/'seed17_adapter.py', repo/'gr00t/eval/sim/LIBERO/libero_env.py',
                 repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py'):
        gate.require(str(path.resolve()) in bound, 'Reset implementation must be a hashed source')
    task = c['tasks'][task_index]
    for key in ('suite', 'task'):
        gate.require(isinstance(task.get(key), str) and task[key] not in ('', '.', '..')
                     and '/' not in task[key] and '\\' not in task[key], 'Invalid task path component')
    gate.require(type(task.get('task_id')) is int and task['task_id'] >= 0, 'Explicit task ID required')
    gate.require(isinstance(task.get('language'), str) and bool(task['language']), 'Explicit task language required')
    initial = task['initial_file']
    gate.require(bound.get(str(Path(initial['path']).resolve())) == initial['sha256'], 'Initial states must be bound inputs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--task-index', type=int, required=True)
    args = p.parse_args()
    c = json.loads(Path(args.contract).read_text())
    validate(c, args.task_index)
    task = c['tasks'][args.task_index]
    out = Path(c['output']) / 'tasks' / task['suite'] / task['task']
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(c['repo'])
    assets = repo/'external_dependencies/LIBERO/libero/libero'
    sys.path[:0] = [str(repo), str(assets.parents[1]), str(Path(c['adapter_directory']))]
    cfg = out/'libero_config'
    cfg.mkdir(exist_ok=True)
    (cfg/'config.yaml').write_text(json.dumps({k: str(assets/v) for k,v in {
        'benchmark_root': '.', 'bddl_files':'bddl_files', 'init_states':'init_files',
        'datasets':'../datasets', 'assets':'assets'}.items()}))
    os.environ.update(LIBERO_CONFIG_PATH=str(cfg), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      MUJOCO_EGL_DEVICE_ID='0')
    # Match the earlier audited simulator renderer wrapper; cgroups retain GPU isolation.
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    import numpy as np
    from seed17_adapter import official_env_class, state_sha, observation_sha, episode_batches
    lib = module('official_libero', repo/'gr00t/eval/sim/LIBERO/libero_env.py')
    wrap = module('official_multistep', repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py')
    bench = lib.benchmark.get_benchmark_dict()[task['suite']]()
    t = bench.get_task(task['task_id'])
    gate.require(t.name == task['task'] and t.language == task['language'], 'Task registry mismatch')
    gate.require(gate.sha(assets/'init_files'/t.problem_folder/t.init_states_file) == task['initial_file']['sha256'], 'State file changed')
    bddl_path = assets/'bddl_files'/t.problem_folder/t.bddl_file
    gate.require(any(Path(r['path']).resolve() == bddl_path.resolve() and gate.sha(bddl_path) == r['sha256']
                     for r in c['inputs']), 'BDDL must be a bound input')
    states = bench.get_task_init_states(task['task_id'])
    gate.require(len(states) >= 50, 'Official states missing')
    states = [np.asarray(s).copy() for s in states[:50]]
    bddl = str(assets/'bddl_files'/t.problem_folder/t.bddl_file)
    adapter = official_env_class(lib.LiberoEnv)
    def make():
        return wrap.MultiStepWrapper(adapter(bddl, t.language), np.array([0]), np.array([0]),
            n_action_steps=8, max_episode_steps=720, terminate_on_success=False)
    def forbid_step(*a, **kw):
        raise RuntimeError('Reset audit must never call environment.step')
    records = []
    envs = []
    started = time.monotonic()
    try:
        reference = lib.LiberoEnv(bddl, t.language)
        try:
            reference._env.step = forbid_step
            random.seed(17)
            np.random.seed(17)
            for e, state in enumerate(states):
                before = (random.getstate(), np.random.get_state())
                if e == 0:
                    reference._env.seed(17)
                reference._env.reset()
                # Independent public LIBERO reference: reset -> set_init_state.
                raw = reference._env.set_init_state(state)
                gate.require(np.array_equal(reference._env.get_sim_state(), state), 'Reference state mismatch')
                obs = reference._process_observation(raw)
                records.append(dict(episode_id=e, rng=before, reset_seed=17 if e == 0 else None,
                    state_sha256=state_sha(state), observation_sha256=observation_sha(obs)))
        finally:
            reference.close()
        # Persist the canonical env0 reset RNG context; no success outcomes saved.
        temp = out/'reference.pkl.tmp'
        temp.write_bytes(pickle.dumps(records, protocol=5))
        temp.replace(out/'reference.pkl')
        envs = [make() for _ in range(1)]
        for env in envs:
            env.unwrapped._env.step = forbid_step
        checks = []
        last_reset = [-1]*1
        prefix_checks = 0
        for batch in [[(0, e)] for e in range(50)]:
            for slot,e in batch:
                r = records[e]
                # Reset-only audit: each slot advances through the same canonical
                # reset prefix, verifying every intervening observation. No actions
                # occur, so replayed prefixes can be retained instead of rebuilt.
                for prior in range(last_reset[slot]+1,e+1):
                    saved=records[prior]
                    random.setstate(saved['rng'][0]); np.random.set_state(saved['rng'][1])
                    obs,_ = envs[slot].reset(seed=saved['reset_seed'], options=dict(episode_id=prior,
                        initial_state=states[prior], initial_state_sha256=saved['state_sha256']))
                    check_obs={k:(v if isinstance(v,str) else v[0]) for k,v in obs.items()}
                    gate.require(observation_sha(check_obs)==saved['observation_sha256'],
                        f'Canonical prefix mismatch episode{e} prior{prior} slot{slot}')
                    prefix_checks += 1
                last_reset[slot]=e
                # MultiStepWrapper adds one observation-history dimension.
                base_obs = {k:(v if isinstance(v,str) else v[0]) for k,v in obs.items()}
                actual = observation_sha(base_obs)
                gate.require(actual == r['observation_sha256'], f'Observation mismatch episode{e} slot{slot}')
                gate.require(len(envs[slot].reward) == len(envs[slot].done) == 0, 'Nonzero action counters')
                checks.append(dict(episode_id=e, slot=slot, official_state_sha256=r['state_sha256'],
                    reference_observation_sha256=r['observation_sha256'], observed_sha256=actual))
            gate.atomic(out/'progress.json',dict(checked=len(checks), total=50, elapsed=time.monotonic()-started))
        gate.atomic(out/'result.json',dict(status='PASS', task=task['task'], suite=task['suite'],
            episodes=50, checks=checks, canonical_prefix_observations_verified=prefix_checks, environment_steps=0, policy_forward_calls=0,
            n_envs=1, contract_sha256=gate.sha(args.contract), elapsed_seconds=time.monotonic()-started,
            reference_rng_sha256=gate.sha(out/'reference.pkl')))
    except Exception as e:
        gate.atomic(out/'result.json',dict(status='FAIL',error=repr(e),task=task['task'],
            contract_sha256=gate.sha(args.contract)))
        raise
    finally:
        for env in envs: env.close()


if __name__ == "__main__":
    main()
