"""FrozenEvoSkillLearner (`learner: evoskill_frozen`): re-exports the skill
library of an already-finished EvoSkill project (EVOSKILL_FROZEN_PROJECT)
without running EvoSkill again. Used to swap only the consuming agent while
holding the learned library fixed (e.g. M3 library induced from gpt-4o-mini
traces, consumed by an open model). Ignores the new traces; 0 build tokens."""
from __future__ import annotations
import os
from pathlib import Path
from harness.interfaces.learner import Learner
from harness.learners.evoskill import export_skills
from harness.registry import register
from harness.types import Skill


@register("learner", "evoskill_frozen")
class FrozenEvoSkillLearner(Learner):
    def __init__(self, logger=None):
        self.logger, self.last_build_tokens = logger, 0

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.logger)

    def induce(self, traces, current):
        root = Path(os.environ["EVOSKILL_FROZEN_PROJECT"]).resolve()
        prog, raw = export_skills(root)
        skills = [Skill(id=f"evoskill_{n}_v1", content=c, representation="skill_md",
                        provenance=[f"frozen:{root.name}"], version=1) for n, c in sorted(raw.items())]
        if self.logger:
            self.logger.log("learner.induce", learner="evoskill_frozen", tokens=0,
                            stats={"project": str(root), "best_program": prog, "skills_out": len(skills)},
                            skill_ids=[s.id for s in skills])
        return skills
