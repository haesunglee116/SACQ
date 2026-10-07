"""Capture policy requests so they can be replayed offline against multiple models."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

from gr00t.policy.server_client import MsgSerializer


def _language_preview(observation: dict[str, Any]) -> str | None:
    for key in ("task", "annotation.human.coarse_action"):
        value = observation.get(key)
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
    return None


class RequestCapture:
    def __init__(self, out_dir: str, tag: str, max_requests: int = 64):
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.max_requests = max_requests
        self.count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.out_dir / f"requests_{tag}_index.jsonl"

    def enabled(self) -> bool:
        return self.count < self.max_requests

    def save(self, observation: dict[str, Any], options: dict[str, Any] | None) -> None:
        if not self.enabled():
            return
        self.count += 1
        rec_id = f"{self.tag}_req{self.count:04d}"
        payload = {
            "request_id": rec_id,
            "observation": observation,
            "options": options,
        }
        request_path = self.out_dir / f"{rec_id}.msgpack"
        request_path.write_bytes(MsgSerializer.to_bytes(payload))

        meta = {
            "request_id": rec_id,
            "path": str(request_path),
            "language": _language_preview(observation),
            "keys": sorted(observation.keys()),
        }
        with self.index_path.open("a") as f:
            f.write(json.dumps(meta, sort_keys=True) + "\n")


def attach_request_capture(policy: Any, out_dir: str, tag: str = "capture", max_requests: int = 64) -> RequestCapture:
    capture = RequestCapture(out_dir=out_dir, tag=tag, max_requests=max_requests)
    original_get_action = policy.get_action

    @functools.wraps(original_get_action)
    def traced_get_action(observation: dict[str, Any], options: dict[str, Any] | None = None):
        if capture.enabled():
            capture.save(observation, options)
        return original_get_action(observation, options)

    policy.get_action = traced_get_action
    print(
        f"[capture] attached request capture tag={tag} max_requests={max_requests} dir={capture.out_dir}",
        flush=True,
    )
    return capture
