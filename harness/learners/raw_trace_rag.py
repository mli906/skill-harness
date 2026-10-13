"""RawTraceRAGBaseline (`learner: raw_trace_rag`): the no-distillation control.
Stores each raw training trace as one library item (provenance = its task_id),
no LLM call (last_build_tokens = 0). Pair with `representation: raw_trace`."""
from __future__ import annotations
from harness.interfaces.learner import Learner, C1_FIXED_TRACES
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
    contract = C1_FIXED_TRACES
    def __init__(self, task_lookup: dict):
        self.task_lookup = task_lookup
        self.last_build_tokens = 0

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.task_lookup)

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        v = 1 + max((s.version for s in current.skills), default=0)
        seen: dict[str, int] = {}
        out = []
        for t in traces:   # several rollouts of one task (frozen corpus, k>1) get unique ids
            n = seen.get(t.task_id, 0); seen[t.task_id] = n + 1
            sid = f"trace_{t.task_id}" + (f"_r{n}" if n else "")
            out.append(Skill(id=sid, content=render_trace(self.task_lookup.get(t.task_id, ""), t),
                             representation="raw_trace", provenance=[t.task_id], version=v))
        return out
