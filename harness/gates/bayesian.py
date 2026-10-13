"""BayesianGate (`gate: bayesian`): admit a candidate only if the posterior
probability that it beats the current library on the probe is high enough.

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

Adapted (one-shot admission, Gate contract) -- two-posterior comparison:
  1. Baseline: replay the probe with the CURRENT library exactly like NonRegressionGate
     (same probe tasks, agent/seed/representation/k) -> a0 successes, b0 failures;
     theta_base ~ Beta(1+a0, 1+b0) (BetaBernoulliState updated once per outcome).
  2. Candidate: replay with current + candidate -> a, b; theta_cand ~ Beta(1+a, 1+b).
  3. Admit iff P(theta_cand > theta_base) >= 1 - ALPHA (ALPHA default 0.05; ctor arg or
     env BAYESIAN_GATE_ALPHA). P is estimated by Monte Carlo with N=20000 draws from each
     posterior, numpy Generator seeded with derive_seed(run seed, "bayesian-gate", index, id).
  4. Sequential over candidates: an admitted skill joins the library and its probe
     counts become the new baseline. last_rejections = number rejected; last_tokens =
     all probe-replay tokens (baseline + candidates). Each decision is logged with both
     posteriors' alpha/beta and P(better).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping
import numpy as np
from harness.interfaces.gate import Gate
from harness.registry import register
from harness.seeding import derive_seed
from harness.types import Trace, Skill, SkillLibrary

DEFAULT_ALPHA = 0.05
MC_DRAWS = 20000


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


def posterior(outcomes) -> BetaBernoulliState:
    post = BetaBernoulliState()
    for ok in outcomes:
        post.update(TrajectoryEvidence("success" if ok else "failure"))
    return post


def p_better(cand: BetaBernoulliState, base: BetaBernoulliState, rng: np.random.Generator,
             n: int = MC_DRAWS) -> float:
    """Monte Carlo estimate of P(theta_cand > theta_base)."""
    return float(np.mean(rng.beta(cand.alpha, cand.beta, n) > rng.beta(base.alpha, base.beta, n)))


@register("gate", "bayesian")
class BayesianGate(Gate):
    def __init__(self, agent, representation, benchmark, k: int, task_lookup: dict, logger=None,
                 alpha: float | None = None, seed: int = 0):
        self.agent, self.rep, self.bench, self.k = agent, representation, benchmark, k
        self.task_lookup, self.logger, self.seed = task_lookup, logger, seed
        self.alpha = float(alpha if alpha is not None else os.environ.get("BAYESIAN_GATE_ALPHA", DEFAULT_ALPHA))
        self.last_rejections, self.last_tokens = 0, 0

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.agent, ctx.representation, ctx.benchmark, ctx.cfg.retrieval_k,
                   ctx.task_lookup, ctx.logger, seed=ctx.cfg.seed)

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
        base = posterior(self._replay(tasks, lib).values())
        for i, cand in enumerate(candidates):
            trial = SkillLibrary(skills=lib.skills + [cand])
            post = posterior(self._replay(tasks, trial).values())
            rng = np.random.default_rng(derive_seed(self.seed, "bayesian-gate", i, cand.id))
            pb = p_better(post, base, rng)
            ok = pb >= 1 - self.alpha
            if self.logger:
                self.logger.log("gate.decision", gate="bayesian", skill_id=cand.id, admitted=ok,
                                base_posterior=base.to_dict(), cand_posterior=post.to_dict(),
                                p_better=pb, credible_alpha=self.alpha, mc_draws=MC_DRAWS)
            if ok:
                lib, base = trial, post
            else:
                self.last_rejections += 1
        return lib
