"""TrivialLearner: one LLM call, traces -> one SKILL.md skill."""
from __future__ import annotations
from harness.interfaces.learner import Learner
from harness.registry import register
from harness.seeding import derive_seed
from harness.types import Trace, Skill, SkillLibrary

PROMPT = """### SKILL INDUCTION
You are given execution traces of an agent on tasks. Write ONE reusable skill
as a SKILL.md document (YAML front matter with name/description, then concise
procedures and rules) that would help the agent succeed on similar tasks.

Traces:
{traces}
"""


@register("learner", "trivial")
class TrivialLearner(Learner):
    def __init__(self, llm, seed: int, logger=None):
        self.llm, self.seed, self.logger = llm, seed, logger
        self.last_build_tokens = 0
        self.task_lookup: dict[str, str] = {}

    @classmethod
    def from_context(cls, ctx):
        obj = cls(ctx.llm, ctx.cfg.seed, ctx.logger)
        obj.task_lookup = ctx.task_lookup
        return obj

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        lines = []
        for t in traces:
            unauth = sum(s.meta.get("unauthorized", 0) for s in t.steps)
            lines.append(f"- [{t.task_id}] outcome={t.outcome} unauthorized={unauth} "
                         f"task={self.task_lookup.get(t.task_id, '?')}")
            lines.append(f"    final_output={t.final_output!r} failure_mode={t.failure_mode}")
        resp = self.llm.complete(PROMPT.format(traces="\n".join(lines)),
                                 seed=derive_seed(self.seed, "trivial-learner"))
        self.last_build_tokens = resp.total_tokens
        version = 1 + max((s.version for s in current.skills), default=0)
        skill = Skill(id=f"skill_trivial_v{version}", content=resp.text.strip(),
                      representation="skill_md", provenance=[t.task_id for t in traces],
                      version=version)
        if self.logger:
            self.logger.log("learner.induce", learner="trivial", tokens=self.last_build_tokens,
                            skill_id=skill.id, content=skill.content)
        return [skill]
