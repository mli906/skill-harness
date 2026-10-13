"""BayesianGate (`gate: bayesian`): admit a candidate only if a lower credible
bound on its probe success probability beats the current library's probe rate.

Beta-Bernoulli posterior ported verbatim from DataArcTech/Bayesian-Agent @ 4b69b4e (core/algorithms/beta_bernoulli.py). One-shot credible-bound admission adapted to the harness Gate contract; the repo's native policy is online/incremental, not one-shot.

Ported (bayesian_agent/core/algorithms/beta_bernoulli.py @ 4b69b4ed02d166c8d1673ea67e2ac836ac377896, MIT):
  L11-21  BetaBernoulliState(alpha=1.0, beta=1.0) + success_probability = alpha/(alpha+beta)
  L23-29  update(event): "success" -> alpha += 1; "failure"/"failed"/"error" -> beta += 1
  L31-40  to_dict / from_dict
The only change is L8's import of bayesian_agent.core.evidence.TrajectoryEvidence: update()
only reads `event.outcome`, so a minimal TrajectoryEvidence stand-in with that one field is
defined here instead of pulling in the repo's evidence module.
Not ported: the online policy (explore/retire/patch/split/compress), categorical backend,
harness and adapters.

Adapted (one-shot admission, Gate contract):
  1. p0 = probe pass rate with the CURRENT library, replayed exactly like NonRegressionGate
     (same probe tasks, same agent/seed/representation/k); its tokens count into last_tokens.
  2. per candidate: replay probe with current + candidate -> a successes, b failures;
     posterior = BetaBernoulliState() updated once per outcome = Beta(1+a, 1+b).
  3. admit iff scipy.stats.beta.ppf(ALPHA, 1+a, 1+b) > p0 (ALPHA default 0.05;
     env BAYESIAN_GATE_ALPHA or ctor arg). Candidates are tested sequentially like
     NonRegressionGate: an admitted candidate joins the library and p0 becomes its rate.
  4. last_rejections = number rejected; last_tokens = all probe-replay tokens.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from scipy.stats import beta as beta_dist
from harness.interfaces.gate import Gate
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary

DEFAULT_ALPHA = 0.05


@dataclass
class TrajectoryEvidence:
    """Stand-in for bayesian_agent.core.evidence.TrajectoryEvidence (only `.outcome` is read)."""
    outcome: str


# ---- verbatim port: beta_bernoulli.py L11-40 -------------------------------------------------
@dataclass
class BetaBernoulliState:
    """Conjugate posterior for Bernoulli success/failure evidence."""

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def success_probability(self) -> float:
        denom = self.alpha + self.beta
        return self.alpha / denom if denom else 0.0

    def update(self, event: TrajectoryEvidence) -> "BetaBernoulliState":
        outcome = event.outcome.strip().lower()
        if outcome == "success":
            self.alpha += 1.0
        elif outcome in {"failure", "failed", "error"}:
            self.beta += 1.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "posterior_success": self.success_probability,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BetaBernoulliState":
        return cls(alpha=float(raw.get("alpha", 1.0)), beta=float(raw.get("beta", 1.0)))
# ---- end verbatim port ------------------------------------------------------------------------


@register("gate", "bayesian")
class BayesianGate(Gate):
    def __init__(self, agent, representation, benchmark, k: int, task_lookup: dict, logger=None,
                 alpha: float | None = None):
        self.agent, self.rep, self.bench, self.k = agent, representation, benchmark, k
        self.task_lookup, self.logger = task_lookup, logger
        self.alpha = float(alpha if alpha is not None else os.environ.get("BAYESIAN_GATE_ALPHA", DEFAULT_ALPHA))
        self.last_rejections, self.last_tokens = 0, 0

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.agent, ctx.representation, ctx.benchmark, ctx.cfg.retrieval_k,
                   ctx.task_lookup, ctx.logger)

    def _replay(self, tasks: list[str], lib: SkillLibrary) -> dict[str, bool]:
        res = {}
        for t in tasks:
            tr = self.agent.run_task(t, lib, self.rep, self.k)
            self.last_tokens += tr.tokens
            res[t] = tr.outcome
        return res

    def admit(self, candidates: list[Skill], current: SkillLibrary,
              probe: list[Trace]) -> SkillLibrary:
        self.last_rejections, self.last_tokens = 0, 0
        tasks = [self.task_lookup[p.task_id] for p in probe]
        lib = SkillLibrary(skills=list(current.skills))
        base = self._replay(tasks, lib)
        p0 = sum(base.values()) / max(1, len(base))
        for cand in candidates:
            trial = SkillLibrary(skills=lib.skills + [cand])
            res = self._replay(tasks, trial)
            post = BetaBernoulliState()
            for ok in res.values():
                post.update(TrajectoryEvidence("success" if ok else "failure"))
            lower = float(beta_dist.ppf(self.alpha, post.alpha, post.beta))
            ok = lower > p0
            if self.logger:
                self.logger.log("gate.decision", gate="bayesian", skill_id=cand.id, admitted=ok,
                                base_rate=p0, cand_successes=int(post.alpha - 1), cand_failures=int(post.beta - 1),
                                posterior=post.to_dict(), credible_alpha=self.alpha, lower_bound=lower)
            if ok:
                lib = trial
                p0 = sum(res.values()) / max(1, len(res))
            else:
                self.last_rejections += 1
        return lib
