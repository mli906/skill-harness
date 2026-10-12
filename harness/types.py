"""Canonical data types (verbatim from spec)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Step:
    observation: str
    action: str
    tool_result: str | None
    meta: dict          # tokens, latency, etc.


@dataclass
class Trace:
    task_id: str
    steps: list[Step]
    outcome: bool       # externally verified success/failure
    failure_mode: str | None
    final_output: str
    tokens: int


@dataclass
class Skill:
    id: str
    content: str        # SKILL.md / NL / code — representation-defined
    representation: str  # "skill_md" | "nl_rule" | "code" | ...
    provenance: list[str]  # source task_ids — required for auditability
    version: int


@dataclass
class SkillLibrary:
    skills: list[Skill]
    # retrieve() and inject() live on Representation, not here


@dataclass
class RunConfig:
    learner: str
    gate: str
    representation: str
    benchmark: str
    base_model: str
    seed: int
    retrieval_k: int


@dataclass
class Scorecard:
    success_rate: float
    avg_steps: float
    tokens_eval: int
    tokens_build: int        # cost to produce the library
    flips_w2c: int           # improvement: wrong→correct
    flips_c2w: int           # regression: correct→wrong
    library_size: int
    auditability: float      # lineage completeness
    unauthorized_actions: int  # finance safety metric
    promotion_rejections: int
