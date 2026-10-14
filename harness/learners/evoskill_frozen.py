"""FrozenEvoSkillLearner (`learner: evoskill_frozen`): re-exports the skill
library of an already-finished EvoSkill project (EVOSKILL_FROZEN_PROJECT)
without running EvoSkill again. Used to swap only the consuming agent while
holding the learned library fixed (e.g. M3 library induced from gpt-4o-mini
traces, consumed by an open model). Ignores the new traces; 0 build tokens.

Provenance carries forward the skill's REAL origin lineage, recovered from the
saved project (same rule as the EvoSkill learner: provenance = task_ids of the
traces it was built from):
  1. ``.evoskill/origin_provenance.json`` (explicit list of origin task_ids), else
  2. the ``uid`` column of ``data/dataset.csv`` (one row per input trace; uid ==
     Trace.task_id by construction in evoskill.write_project), else
  3. ``["origin_unavailable"]`` — an explicit, never-auditable label.
When the recovered ids are disjoint from this run's input traces, the skill is
declared external via ``self.external_provenance[skill_id]`` (read by the runner's
three-state ``audit`` event) and ``external_provenance: true`` in the logged stats.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from harness.interfaces.learner import Learner
from harness.learners.evoskill import export_skills
from harness.provenance import ORIGIN_UNAVAILABLE
from harness.registry import register
from harness.types import Skill


def recover_origin_ids(root: Path) -> tuple[list[str], str]:
    """Return (origin task_ids in first-seen order, source label)."""
    exp = root / ".evoskill" / "origin_provenance.json"
    if exp.is_file():
        ids = json.loads(exp.read_text())
        if isinstance(ids, dict):
            ids = ids.get("task_ids", [])
        ids = list(dict.fromkeys(str(i) for i in ids if i))
        if ids:
            return ids, str(exp.relative_to(root))
    ds = root / "data" / "dataset.csv"
    if ds.is_file():
        with open(ds, newline="") as f:
            ids = list(dict.fromkeys(r["uid"] for r in csv.DictReader(f) if r.get("uid")))
        if ids:
            return ids, "data/dataset.csv:uid"
    return [ORIGIN_UNAVAILABLE], "unavailable"


@register("learner", "evoskill_frozen")
class FrozenEvoSkillLearner(Learner):
    def __init__(self, logger=None):
        self.logger, self.last_build_tokens = logger, 0
        self.external_provenance: dict[str, dict] = {}

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.logger)

    def origin_task_ids(self) -> list[str]:
        """Origin lineage, resolvable before the run (checked by splits.check_frozen_origin)."""
        return recover_origin_ids(Path(os.environ["EVOSKILL_FROZEN_PROJECT"]).resolve())[0]

    def induce(self, traces, current):
        root = Path(os.environ["EVOSKILL_FROZEN_PROJECT"]).resolve()
        prog, raw = export_skills(root)
        origin, src = recover_origin_ids(root)
        skills = [Skill(id=f"evoskill_{n}_v1", content=c, representation="skill_md",
                        provenance=list(origin), version=1) for n, c in sorted(raw.items())]
        run_ids = {t.task_id for t in traces}
        external = src != "unavailable" and not set(origin) & run_ids
        self.external_provenance = ({s.id: {"origin_project": str(root), "source": src, "n_origin_ids": len(origin)}
                                     for s in skills} if external else {})
        if self.logger:
            self.logger.log("learner.induce", learner="evoskill_frozen", tokens=0,
                            stats={"project": str(root), "best_program": prog, "skills_out": len(skills),
                                   "origin_provenance_source": src, "origin_task_ids": origin,
                                   "external_provenance": external},
                            skill_ids=[s.id for s in skills])
        return skills
