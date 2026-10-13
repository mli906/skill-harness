"""RawTraceRAGBaseline (`learner: raw_trace_rag`): the no-distillation control.
Stores each raw training trace as one library item (provenance = its task_id),
no LLM call (last_build_tokens = 0). Pair with `representation: raw_trace`."""
from __future__ import annotations
from harness.interfaces.learner import Learner
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary

MAX_STEPS_SHOWN = 30


def render_trace(task: str, t: Trace) -> str:
    lines = [f"Task: {task.splitlines()[-1] if task else t.task_id}",
             f"Outcome: {'SUCCESS' if t.outcome else 'FAILURE'} in {len(t.steps)} steps"]
    for s in t.steps[:MAX_STEPS_SHOWN]:
        res = (s.tool_result or "").replace("\n", " ")[:160]
        lines.append(f"> {s.action}\n  {res}")
    return "\n".join(lines)


@register("learner", "raw_trace_rag")
class RawTraceRAGBaseline(Learner):
    def __init__(self, task_lookup: dict):
        self.task_lookup = task_lookup
        self.last_build_tokens = 0

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.task_lookup)

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        v = 1 + max((s.version for s in current.skills), default=0)
        return [Skill(id=f"trace_{t.task_id}", content=render_trace(self.task_lookup.get(t.task_id, ""), t),
                      representation="raw_trace", provenance=[t.task_id], version=v) for t in traces]
