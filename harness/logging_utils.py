"""Structured JSON-lines logging."""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, is_dataclass


class JsonlLogger:
    def __init__(self, path: str | None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            open(path, "w").close()

    def log(self, event: str, **fields) -> None:
        rec = {"ts": time.time(), "event": event}
        for k, v in fields.items():
            rec[k] = asdict(v) if is_dataclass(v) else v
        line = json.dumps(rec, default=str)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


NULL_LOGGER = JsonlLogger(None)
