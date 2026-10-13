"""RawTraceRepresentation (`representation: raw_trace`): top-k embedding
retrieval over raw traces (matching on the task goal line), injected as
labelled example trajectories ahead of the prompt."""
from __future__ import annotations
from harness.registry import register
from harness.representations.skill_md import SkillMdRepresentation
from harness.types import Skill, SkillLibrary


def _goal(text: str) -> str:
    for ln in text.splitlines():
        if ln.startswith("Task:") or "Your task is to" in ln:
            return ln
    return text[:300]


@register("representation", "raw_trace")
class RawTraceRepresentation(SkillMdRepresentation):
    def retrieve(self, task: str, lib: SkillLibrary, k: int) -> list[Skill]:
        if k <= 0 or not lib.skills:
            return []
        q = self._emb(_goal(task))
        return sorted(lib.skills, key=lambda s: (-self.embedder.cosine(q, self._emb(_goal(s.content))), s.id))[:k]

    def inject(self, skills: list[Skill], prompt: str) -> str:
        if not skills:
            return prompt
        ex = "\n\n".join(f"<example_trajectory id=\"{s.id}\">\n{s.content}\n</example_trajectory>" for s in skills)
        return f"# Past trajectories on similar tasks (raw, may include failures)\n{ex}\n\n{prompt}"
