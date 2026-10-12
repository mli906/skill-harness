from __future__ import annotations
from harness.interfaces.gate import Gate
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary


@register("gate", "noop")
class NoOpGate(Gate):
    """Admit everything."""
    def admit(self, candidates: list[Skill], current: SkillLibrary,
              probe: list[Trace]) -> SkillLibrary:
        self.last_rejections, self.last_tokens = 0, 0
        return SkillLibrary(skills=list(current.skills) + list(candidates))
