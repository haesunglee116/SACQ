"""Asynchronous *display only* status. Never use for episode/checkpoint receipts.

No torch/numpy/random imports and no scientific computation. A single pending
snapshot is coalesced; errors are explicit in `.error`. Completion and restart
journals must continue using their original synchronous durable writers.
"""
import copy
import json
import os
from pathlib import Path
import threading


class DisplayHeartbeat:
    def __init__(self, path):
        self.path=Path(path)
        self.condition=threading.Condition()
        self.pending=None
        self.closing=False
        self.error=None
        self.writes=0
        self.thread=threading.Thread(target=self._run,daemon=True,name='display-heartbeat')
        self.thread.start()

    def publish(self, value):
        with self.condition:
            if self.closing:raise RuntimeError('Heartbeat already closed')
            if self.error:raise RuntimeError('Heartbeat writer failed: '+self.error)
            self.pending=copy.deepcopy(value)
            self.condition.notify()

    def write(self, value):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_name(self.path.name+'.display_tmp.'+str(os.getpid()))
        tmp.write_text(json.dumps(value))
        tmp.replace(self.path)

    def _run(self):
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.closing:self.condition.wait()
                    if self.pending is None and self.closing:return
                    value,self.pending=self.pending,None
                self.write(value)
                self.writes+=1
        except Exception as exc:self.error=repr(exc)

    def close(self, timeout=10):
        with self.condition:
            self.closing=True;self.condition.notify()
        self.thread.join(timeout)
        return not self.thread.is_alive() and self.error is None
