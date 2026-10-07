"""Default operational wrapper for the existing Long task adapter.

Pin this file plus its imported acceleration sources in the launch receipt.
The underlying approved task plan/checkpoint/sample gates run unchanged.
"""
import atexit
from pathlib import Path
import sys
import time

import libero_profiled_rollout_20260920 as adapter
from rollout_acceleration_default_20260920 import AsyncProfile,DefaultAcceleration
from rollout_display_heartbeat_20260920 import DisplayHeartbeat

adapter.Profile=AsyncProfile
adapter.PolicyAcceleration=DefaultAcceleration
old_transform=adapter.transform_worker


def transform(source):
    source=old_transform(source)
    old="            atomic(root/f'workers/{a.gpu}.json', dict(time=time.time(), pid=os.getpid(), gpu=a.gpu, **v))"
    new="            publish_display(root/f'workers/{a.gpu}.json', dict(time=time.time(), pid=os.getpid(), gpu=a.gpu, **v))"
    assert source.count(old)==1
    source=source.replace(old,new)
    source=source.replace('\n','\n    from libero_default_rollout_20260920 import publish_display\n',1)
    source=adapter.once(source,"                audit=install(core.model,plan,task['suite'])",
        "                audit=install(core.model,plan,task['suite'])\n                core.model.eval(); core.model.requires_grad_(False)")
    return source


_writers={}
def publish_display(path,value):
    key=str(path)
    if key not in _writers:
        _writers[key]=DisplayHeartbeat(path)
        atexit.register(_writers[key].close)
    _writers[key].publish(value)


adapter.transform_worker=transform
if __name__=='__main__':
    if sys.argv[1]!='worker':raise ValueError('worker TASK')
    adapter.worker(int(sys.argv[2]))
