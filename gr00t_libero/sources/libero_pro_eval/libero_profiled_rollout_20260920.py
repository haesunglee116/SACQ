import atexit, inspect, sys
from pathlib import Path
from rollout_exact_acceleration_20260920 import Profile, PolicyAcceleration
SELF=Path(__file__).resolve()

def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Instrumentation anchor drift: {old!r}')
    return source.replace(old, new)


def transform_worker(source):
    source = once(source,
        "    def state(**v): atomic(root/f'workers/{a.gpu}.json', dict(time=time.time(), pid=os.getpid(), gpu=a.gpu, **v))",
        "    heartbeat = [0., None]\n    def state(**v):\n        now = time.monotonic()\n        status = v.get('status')\n        if status == 'ROLLOUT_RUNNING' and status == heartbeat[1] and now-heartbeat[0] < 10:\n            return\n        with PROFILE.phase('heartbeat_write'):\n            atomic(root/f'workers/{a.gpu}.json', dict(time=time.time(), pid=os.getpid(), gpu=a.gpu, **v))\n        heartbeat[:] = [now, status]")
    source = once(source, "    def rpc(v): send(sock, v); return receive(sock)",
        "    def rpc(v):\n        with PROFILE.phase('rpc_' + v['op']):\n            PROFILE.timed('rpc_send', send, sock, v)\n            return PROFILE.timed('rpc_receive_wait', receive, sock)")
    source = once(source,
        '                with torch.inference_mode(), injector.selected(noise): actions,_=policy.get_action(batch)',
        '                def inference():\n                    with torch.inference_mode(), injector.selected(noise):\n                        return policy.get_action(batch)\n                actions,_=ACCEL(core.model, inference)')
    # Run the same simulator main through a timing-only wrapper. The immutable
    # plan, simulator source SHA, reset history and socket protocol are unchanged.
    source = source.replace("plan['sim_python'], plan['sim_source'], '--plan'",
        "plan['sim_python'], PROFILE_WRAPPER, 'sim', '--source', plan['sim_source'], '--plan'")
    return source


def simulator(source_path, args):
    import importlib.util
    import json
    plan_path = Path(args[args.index('--plan')+1])
    plan = json.loads(plan_path.read_text())
    profile = Profile(Path(plan['root'])/'performance_v1', 'simulator')
    atexit.register(profile.save, True)
    spec = importlib.util.spec_from_file_location('profiled_original_simulator', source_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    source = inspect.getsource(m.main)
    source = once(source, "            request = receive(conn)",
        "            PROFILE.save()\n            request = PROFILE.timed('waiting_for_policy', receive, conn)")
    source = once(source, '                    obs, reward, done, truncated, info = envs[slot].step(action)',
        "                    obs, reward, done, truncated, info = PROFILE.timed('environment_step', envs[slot].step, action)")
    source = once(source, '                    obs, actual = reset_one(envs[slot], ep)',
        "                    obs, actual = PROFILE.timed('environment_reset', reset_one, envs[slot], ep)")
    source = once(source, '                        obs, actual = reconstruct_slot(slot, ep)',
        "                        obs, actual = PROFILE.timed('reset_reconstruction', reconstruct_slot, slot, ep)")
    source = source.replace('send(conn, ', "PROFILE.timed('response_send', send, conn, ")
    m.__dict__['PROFILE'] = profile
    exec(compile(source, str(SELF), 'exec'), m.__dict__)
    sys.argv = [str(source_path), *args]
    m.main()

