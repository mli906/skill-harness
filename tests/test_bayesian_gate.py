"""BayesianGate (two-posterior rule) with a fake agent (no API spend), fixed MC seed.

Fake agent: a probe task passes iff it is in the agent's `base_pass` set or some skill in
the library lists it in its content (comma-separated). Each run costs 7 tokens.
Exact P(theta_cand > theta_base) (numerical integration) noted per case; ALPHA=0.05 => need >= 0.95."""
import pytest
from harness.gates.bayesian import BayesianGate, BetaBernoulliState, TrajectoryEvidence
from harness.types import Skill, SkillLibrary, Trace


class FakeAgent:
    def __init__(self, base_pass=()): self.base_pass = set(base_pass)
    def run_task(self, task, lib, rep, k):
        ok = task in self.base_pass or any(task in s.content.split(",") for s in lib.skills)
        return Trace(task_id="id_" + task, steps=[], outcome=ok, failure_mode=None, final_output="", tokens=7)


class Log:
    def __init__(self): self.ev = []
    def log(self, event, **kw): self.ev.append((event, kw))


def _gate(n, base_pass=(), seed=0):
    tasks = [f"p{i}" for i in range(n)]
    probe = [Trace("id_" + t, [], False, None, "", 0) for t in tasks]
    g = BayesianGate(FakeAgent(base_pass), None, None, 3, {"id_" + t: t for t in tasks}, logger=Log(), seed=seed)
    return g, tasks, probe


def _skill(sid, passes):
    return Skill(id=sid, content=",".join(passes), representation="skill_md", provenance=["x"], version=1)


def _decide(n, base_k, cand_k):
    g, tasks, probe = _gate(n, base_pass=set(f"p{i}" for i in range(base_k)))
    lib = g.admit([_skill("c", tasks[:cand_k])], SkillLibrary(skills=[]), probe)
    return bool(lib.skills), g


def test_beta_bernoulli_port():
    s = BetaBernoulliState()
    assert s.success_probability == 0.5
    s.update(TrajectoryEvidence("success")).update(TrajectoryEvidence("Failure ")).update(TrajectoryEvidence("success"))
    assert (s.alpha, s.beta) == (3.0, 2.0) and s.success_probability == 0.6
    assert BetaBernoulliState.from_dict(s.to_dict()) == s


def test_4of4_vs_1of4_admitted():               # exact P = 0.97619
    ok, g = _decide(4, 1, 4)
    assert ok and g.last_rejections == 0
    d = g.logger.ev[-1][1]
    assert d["p_better"] == pytest.approx(0.97619, abs=0.01)
    assert d["base_posterior"]["alpha"] == 2 and d["cand_posterior"]["alpha"] == 5


def test_1of4_and_0of4_vs_1of4_rejected():      # exact P = 0.5 and 0.22222
    g, tasks, probe = _gate(4, base_pass={"p0"})
    lib = g.admit([_skill("same", []), _skill("worse", [])], SkillLibrary(skills=[]), probe)
    assert lib.skills == [] and g.last_rejections == 2
    g2, tasks, probe = _gate(4)                 # 0/4 vs 1/4 needs a probe where cand < base
    class Worse(FakeAgent):
        def run_task(self, task, lib, rep, k):
            ok = (task == "p0") and not lib.skills
            return Trace("id_" + task, [], ok, None, "", 7)
    g2.agent = Worse()
    assert g2.admit([_skill("w", [])], SkillLibrary(skills=[]), probe).skills == [] and g2.last_rejections == 1
    assert g2.logger.ev[-1][1]["p_better"] == pytest.approx(0.22222, abs=0.01)


def test_small_probe_rejected():
    assert not _decide(1, 1, 1)[0]              # n=1: 1/1 vs 1/1, exact P = 0.5
    assert not _decide(1, 0, 1)[0]              # n=1: 1/1 vs 0/1, exact P = 0.83333
    assert not _decide(2, 1, 2)[0]              # n=2: 2/2 vs 1/2, exact P = 0.8


def test_regression_zero_baseline():
    assert not _decide(4, 0, 0)[0]              # 0/4 vs 0/4, exact P = 0.5 -> REJECTED
    assert _decide(4, 0, 4)[0]                  # 4/4 vs 0/4, exact P = 0.99603 -> ADMITTED


def test_sequential_baseline_update_and_tokens():
    g, tasks, probe = _gate(4, base_pass={"p0"})
    lib = g.admit([_skill("a", tasks), _skill("b", tasks)], SkillLibrary(skills=[]), probe)
    assert [s.id for s in lib.skills] == ["a"] and g.last_rejections == 1   # b: 4/4 vs new base 4/4
    assert g.logger.ev[-1][1]["base_posterior"]["alpha"] == 5
    assert g.last_tokens == 7 * 4 * 3 > 0


def test_deterministic_and_alpha_configurable(monkeypatch):
    p = [_decide(4, 2, 4)[1].logger.ev[-1][1]["p_better"] for _ in range(2)]
    assert p[0] == p[1] and p[0] == pytest.approx(0.91667, abs=0.01)        # M4 Trace2Skill case
    monkeypatch.setenv("BAYESIAN_GATE_ALPHA", "0.1")
    assert _decide(4, 2, 4)[0]                  # 0.917 >= 0.90
