"""BayesianGate with a fake agent (no API spend).

Fake agent: a probe task passes iff some skill in the library lists that task in its
content (comma-separated), or the task is in the agent's `base_pass` set. Each run costs 7 tokens."""
import pytest
from scipy.stats import beta as B
from harness.gates.bayesian import BayesianGate, BetaBernoulliState, TrajectoryEvidence
from harness.types import Skill, SkillLibrary, Trace


class FakeAgent:
    def __init__(self, base_pass=()): self.base_pass = set(base_pass)
    def run_task(self, task, lib, rep, k):
        ok = task in self.base_pass or any(task in s.content.split(",") for s in lib.skills)
        return Trace(task_id="id_" + task, steps=[], outcome=ok, failure_mode=None, final_output="", tokens=7)


def _gate(n, base_pass=()):
    tasks = [f"p{i}" for i in range(n)]
    probe = [Trace("id_" + t, [], False, None, "", 0) for t in tasks]
    g = BayesianGate(FakeAgent(base_pass), None, None, 3, {"id_" + t: t for t in tasks})
    return g, tasks, probe


def _skill(sid, passes):
    return Skill(id=sid, content=",".join(passes), representation="skill_md", provenance=["x"], version=1)


def test_beta_bernoulli_port():
    s = BetaBernoulliState()
    assert s.success_probability == 0.5
    s.update(TrajectoryEvidence("success")).update(TrajectoryEvidence("Failure ")).update(TrajectoryEvidence("success"))
    assert (s.alpha, s.beta) == (3.0, 2.0) and s.success_probability == 0.6
    assert BetaBernoulliState.from_dict(s.to_dict()) == s


def test_all_pass_candidate_admitted():
    # probe n=4, baseline p0=0.25 (1/4); candidate 4/4 -> ppf(0.05,5,1)=0.549 > 0.25
    g, tasks, probe = _gate(4, base_pass={"p0"})
    lib = g.admit([_skill("good", tasks)], SkillLibrary(skills=[]), probe)
    assert [s.id for s in lib.skills] == ["good"] and g.last_rejections == 0


def test_mostly_failing_candidate_rejected_and_counted():
    # n=4, p0=0.25; candidate 1/4 -> ppf(0.05,2,4)=0.076 < 0.25
    g, tasks, probe = _gate(4, base_pass={"p0"})
    lib = g.admit([_skill("bad", ["p0"]), _skill("bad2", [])], SkillLibrary(skills=[]), probe)
    assert lib.skills == [] and g.last_rejections == 2


def test_last_tokens_counts_baseline_and_candidate_replays():
    g, tasks, probe = _gate(4)
    g.admit([_skill("a", tasks), _skill("b", [])], SkillLibrary(skills=[]), probe)
    assert g.last_tokens == 7 * 4 * 3 > 0     # baseline + 2 candidates, 4 tasks each


def test_tiny_probe_single_pass_is_conservative():
    # n=1, baseline passes (p0=1.0): even a passing candidate (ppf(0.05,2,1)=0.224) is rejected
    g, tasks, probe = _gate(1, base_pass={"p0"})
    assert g.admit([_skill("c", tasks)], SkillLibrary(skills=[]), probe).skills == []
    # n=2, p0=0.5: a perfect 2/2 still only gives ppf(0.05,3,1)=0.368 < 0.5 -> rejected
    g, tasks, probe = _gate(2, base_pass={"p0"})
    assert g.admit([_skill("c", tasks)], SkillLibrary(skills=[]), probe).skills == [] and g.last_rejections == 1
    assert B.ppf(0.05, 2, 1) == pytest.approx(0.05 ** 0.5)


def test_alpha_configurable(monkeypatch):
    monkeypatch.setenv("BAYESIAN_GATE_ALPHA", "0.4")
    g, tasks, probe = _gate(2, base_pass={"p0"})     # ppf(0.4,3,1)=0.737 > 0.5 -> admitted
    assert g.alpha == 0.4 and len(g.admit([_skill("c", tasks)], SkillLibrary(skills=[]), probe).skills) == 1
