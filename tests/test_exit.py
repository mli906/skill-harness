"""Milestone M0 exit test."""
from dataclasses import fields
import pytest
from harness import registry
from harness.runner import run
from harness.types import RunConfig, Scorecard

SEEDS = range(10)


def cfg(gate, seed):
    return RunConfig(learner="trivial", gate=gate, representation="skill_md",
                     benchmark="synthetic", base_model="stub", seed=seed, retrieval_k=3)


def _check(sc: Scorecard):
    for f in fields(Scorecard):
        v = getattr(sc, f.name)
        assert v is not None, f.name
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f.name
    assert 0 <= sc.success_rate <= 1 and 0 <= sc.auditability <= 1
    assert sc.tokens_eval > 0 and sc.tokens_build > 0 and sc.avg_steps >= 1


@pytest.mark.parametrize("gate", ["noop", "non_regression"])
def test_end_to_end(gate, tmp_path):
    _check(run(cfg(gate, 0), log_path=str(tmp_path / "log.jsonl")))
    assert (tmp_path / "log.jsonl").read_text().count("\n") > 5


def test_deterministic(tmp_path):
    a = run(cfg("non_regression", 1), log_path=str(tmp_path / "a.jsonl"))
    b = run(cfg("non_regression", 1), log_path=str(tmp_path / "b.jsonl"))
    assert a == b


def test_gate_difference_over_seed_sweep(tmp_path):
    found = []
    for s in SEEDS:
        sc_n = run(cfg("noop", s), log_path=str(tmp_path / f"n{s}.jsonl")); lib_n = run.last_library
        sc_r = run(cfg("non_regression", s), log_path=str(tmp_path / f"r{s}.jsonl")); lib_r = run.last_library
        _check(sc_n); _check(sc_r)
        assert sc_r.library_size <= sc_n.library_size
        if [x.id for x in lib_r.skills] != [x.id for x in lib_n.skills]:
            assert sc_r.promotion_rejections >= 1
            found.append(s)
    print("seeds where NonRegressionGate library != NoOpGate library:", found)
    assert found


def test_registry_only_milestone_plugins():
    assert registry.available("learner") == ["trivial"]
    assert registry.available("gate") == ["non_regression", "noop"]
    assert registry.available("benchmark") == ["synthetic"]
    assert registry.available("representation") == ["skill_md"]
    assert registry.available("agent") == ["react"]
