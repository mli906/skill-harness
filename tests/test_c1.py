"""C1 frozen-corpus track: contracts, env guard, corpus validation, Trace2Skill mapping (offline)."""
import json
import os
from pathlib import Path
import pytest
from harness import registry, corpus
from harness.env_guard import forbid_env, EnvAccessError, check_env_allowed
from harness.interfaces.learner import Learner, C1_FIXED_TRACES, C2_SELF_COLLECT, CONTRACTS
from harness.seeding import task_id
from harness.types import Step, Trace, SkillLibrary


def test_contract_attribute_values():
    registry._load_all()
    L = registry._REG["learner"]
    assert Learner.contract == C2_SELF_COLLECT and set(CONTRACTS) == {C1_FIXED_TRACES, C2_SELF_COLLECT}
    for n in ("trace2skill", "raw_trace_rag", "noskill"):
        assert L[n].contract == C1_FIXED_TRACES, n
    assert L["evoskill"].contract == C2_SELF_COLLECT
    assert all(c.contract in CONTRACTS for c in L.values())


class FakeBench:
    def __init__(self): self.calls = 0
    def env_reset(self, task): self.calls += 1; return "obs", ["look"]
    def env_step(self, task, a): self.calls += 1; return "obs", True, True, []


class EnvTouchingC1(Learner):
    contract = C1_FIXED_TRACES
    def __init__(self, bench): self.bench = bench
    def induce(self, traces, current):
        self.bench.env_reset("x"); return []


def test_c1_learner_cannot_touch_env_and_guard_restores():
    b = FakeBench()
    with forbid_env(b):
        with pytest.raises(EnvAccessError):
            EnvTouchingC1(b).induce([], SkillLibrary(skills=[]))
        with pytest.raises(EnvAccessError):
            b.env_step("x", "look")
        with pytest.raises(EnvAccessError):
            check_env_allowed()            # what alfworld.make_env calls (covers subprocesses)
    assert b.env_reset("x") == ("obs", ["look"]) and b.calls == 1
    check_env_allowed()


def test_alfworld_make_env_refuses_under_guard(monkeypatch):
    from harness.benchmarks.alfworld import make_env
    monkeypatch.setenv("HARNESS_ENV_FORBIDDEN", "1")
    with pytest.raises(EnvAccessError):
        make_env("/nonexistent/game.tw-pddl", 5)


def _write_corpus(d: Path, tasks, k=2, leak=None):
    recs = []
    for i, t in enumerate(tasks):
        for r in range(k):
            recs.append({"task_id": task_id(t), "rollout": r, "task": t, "category": "c", "outcome": (i + r) % 2 == 0,
                         "failure_mode": None, "final_output": "WON", "tokens": 5,
                         "steps": [{"observation": "o", "action": "look", "tool_result": "r", "meta": {}}],
                         "artifacts": {}, "gold": {"goal": "g", "expert_walkthrough": ["look"]}})
    body = "".join(json.dumps(x, sort_keys=True) + "\n" for x in recs)
    (d / "corpus.jsonl").write_text(body)
    import hashlib
    n_s = sum(x["outcome"] for x in recs)
    (d / "manifest.json").write_text(json.dumps({"n_traces": len(recs), "n_tasks": len(tasks), "k": k,
        "success_failure_ratio": f"{n_s}:{len(recs)-n_s}", "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "excluded_probe_ids": [], "excluded_test_ids": []}))


def test_corpus_validation_ok_and_detects_leakage(tmp_path):
    train, probe, test = ["a", "b", "p"], ["p"], ["z"]
    _write_corpus(tmp_path, ["a", "b"])
    v = corpus.validate(str(tmp_path), train, probe, test)
    assert v["all_ok"] and v["has_success_and_failure"]
    assert len(corpus.to_traces(corpus.load(str(tmp_path))[0])) == 4
    _write_corpus(tmp_path, ["a", "p"])
    v = corpus.validate(str(tmp_path), train, probe, test)
    assert not v["no_probe_leak"] and not v["all_ok"]
    _write_corpus(tmp_path, ["a", "z"])
    v = corpus.validate(str(tmp_path), train, probe, test)
    assert not v["no_test_leak"] and not v["train_only_ok"]


def test_trace2skill_mapping_offline(tmp_path):
    from harness.learners.trace2skill import Trace2SkillLearner, read_skill_dir, instance_name
    t1 = Trace("t_aaa", [Step("o1", "go to desk 1", "You arrive", {"thought": "find it"})], True, None, "WON", 3)
    t2 = Trace("t_aaa", [Step("o1", "look", "nothing", {})], False, "max_steps", "NOT_WON", 3)
    recs = [{"task_id": "t_aaa", "rollout": r, "gold": {"goal": "g", "task_type": "x", "expert_walkthrough": ["go to desk 1"]}} for r in (0, 1)]
    L = Trace2SkillLearner(0, {"t_aaa": "ALFWorld game x\nYour task is to: g"}, corpus_records=recs)
    names = L.write_logs([t1, t2], tmp_path)
    assert names == ["alfworld_agent_t-aaa-r0_SUCCEED.md", "alfworld_agent_t-aaa-r1_FAILED.md"]
    # Trace2Skill's own filename parser takes the id after the last '_'
    assert all(n[:-3].rsplit("_", 1)[0].split("_")[-1] == instance_name("t_aaa", i) for i, n in enumerate(names))
    body = (tmp_path / names[0]).read_text()
    assert "Expert walkthrough: go to desk 1" in body and "Tool call (env command): go to desk 1" in body
    d = tmp_path / "skill"; d.mkdir(); (d / "SKILL.md").write_text("---\nname: s\n---\nX"); (d / "ref.md").write_text("R")
    assert read_skill_dir(d).startswith("---\nname: s") and "## file: ref.md" in read_skill_dir(d)


def test_noskill_learner_empty_library():
    registry._load_all()
    L = registry._REG["learner"]["noskill"]()
    assert L.induce([], SkillLibrary(skills=[])) == [] and L.last_build_tokens == 0
