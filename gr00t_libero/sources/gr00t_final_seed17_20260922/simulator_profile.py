"""Static profiled simulator adapter; physical EGL device is operational only."""
import argparse
import atexit
import json
import os
from pathlib import Path
import sys
from io_utils import sha

B=Path(__file__).resolve().parent.parent/'libero_pro_eval'
sys.path.insert(0,str(B))


def main():
    if len(sys.argv)<2 or sys.argv[1]!='sim':
        raise ValueError('Expected sim --source ... --plan ... --socket ...')
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    args,rest=parser.parse_known_args(sys.argv[2:])
    canonical=Path(__file__).resolve().parent/'simulator.py'
    if sha(args.source)!=sha(canonical):
        raise ValueError('Materialized simulator requires the pinned template')
    gpu=os.environ.get('SIM_GPU_PHYSICAL','0')
    if gpu not in [str(i) for i in range(8)]:
        raise ValueError('Expected explicit visible physical GPU index 0..7')
    from rollout_exact_acceleration_20260920 import Profile
    import simulator_runtime
    plan_path=Path(rest[rest.index('--plan')+1])
    plan=json.loads(plan_path.read_text())
    profile=Profile(Path(plan['root'])/'performance_v1','simulator')
    atexit.register(profile.save,True)
    sys.argv=[str(canonical),*rest]
    simulator_runtime.main(PROFILE=profile,egl_device=gpu)


if __name__=='__main__':
    main()
