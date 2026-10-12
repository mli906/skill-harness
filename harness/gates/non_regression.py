"""NonRegressionGate: admit a candidate only if replaying the held-out probe
set with it does not lower overall pass rate nor (if the benchmark exposes
categories) any per-category pass rate, relative to the current library."""
from __future__ import annotations
from collections import defaultdict
from harness.interfaces.gate import Gate
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary


@register("gate", "non_regression")
class NonRegressionGate(Gate):
    def __init__(self, agent, representation, benchmark, k: int, task_lookup: dict, logger=None):
        self.agent, self.rep, self.bench, self.k = agent, representation, benchmark, k
        self.task_lookup, self.logger = task_lookup, logger

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.agent, ctx.representation, ctx.benchmark, ctx.cfg.retrieval_k,
                   ctx.task_lookup, ctx.logger)

    def _cat(self, task: str) -> str:
        f = getattr(self.bench, "category", None)
        return f(task) if callable(f) else "_all"

    def _replay(self, tasks: list[str], lib: SkillLibrary) -> dict[str, bool]:
        res = {}
        for t in tasks:
            tr = self.agent.run_task(t, lib, self.rep, self.k)
            self.last_tokens += tr.tokens
            res[t] = tr.outcome
        return res

    def _rates(self, res: dict[str, bool]) -> tuple[float, dict[str, float]]:
        by = defaultdict(list)
        for t, ok in res.items():
            by[self._cat(t)].append(ok)
        overall = sum(res.values()) / max(1, len(res))
        return overall, {c: sum(v) / len(v) for c, v in by.items()}

    def admit(self, candidates: list[Skill], current: SkillLibrary,
              probe: list[Trace]) -> SkillLibrary:
        self.last_rejections, self.last_tokens = 0, 0
        tasks = [self.task_lookup[p.task_id] for p in probe]
        lib = SkillLibrary(skills=list(current.skills))
        base = self._replay(tasks, lib)
        b_all, b_cat = self._rates(base)
        for cand in candidates:
            trial = SkillLibrary(skills=lib.skills + [cand])
            c_all, c_cat = self._rates(self._replay(tasks, trial))
            drops = {c: (b_cat[c], c_cat[c]) for c in b_cat if c_cat[c] < b_cat[c]}
            ok = c_all >= b_all and not drops
            if self.logger:
                self.logger.log("gate.decision", gate="non_regression", skill_id=cand.id,
                                admitted=ok, base_rate=b_all, cand_rate=c_all,
                                base_by_cat=b_cat, cand_by_cat=c_cat, category_drops=drops)
            if ok:
                lib, b_all, b_cat = trial, c_all, c_cat
            else:
                self.last_rejections += 1
        return lib
