"""Shared construction context handed to plugin factories."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from harness.types import RunConfig
from harness.logging_utils import JsonlLogger, NULL_LOGGER


@dataclass
class Context:
    cfg: RunConfig
    llm: Any
    embedder: Any
    logger: JsonlLogger = NULL_LOGGER
    benchmark: Any = None
    representation: Any = None
    agent: Any = None
    task_lookup: dict = field(default_factory=dict)  # task_id -> task text
    learner: Any = None   # set by the runner before the gate is built (governance check 1 reads
                          # learner.external_provenance); not part of any plugin interface
