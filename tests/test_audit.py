"""Regression tests for flip accounting, probe disjointness, provenance."""
import json
import pytest
from harness.llm.stub import StubClient
from harness.runner import run
from harness.seeding import task_id
from harness.types import RunConfig

SEEDS = range(10)


def _run(gate, seed, tmp_path):
    p = tmp_path / f"{gate}_{seed}.jsonl"
    sc = run(RunConfig("trivial", gate, "skill_md", "synthetic", "stub", seed, 3), log_path=str(p))
    ev = [json.loads(l) for l in p.read_text().splitlines()]
    return sc, run.last_library, ev


def _one(ev, name):
    return next(e for e in ev if e["event"] == name)


@pytest.mark.parametrize("gate", ["noop", "non_regression"])
@pytest.mark.parametrize("seed", SEEDS)
def test_paired_flips_and_disjointness_and_provenance(gate, seed, tmp_path):
    sc, lib, ev = _run(gate, seed, tmp_path)
    sp = _one(ev, "split")
    learn, probe, test = map(set, (sp["learn_tasks"], sp["probe_tasks"], sp["test_tasks"]))
    assert not (learn & probe) and not (learn & test) and not (probe & test)
    rows = [e for e in ev if e["event"] == "eval.task"]
    assert [r["task_id"] for r in rows] == [task_id(t) for t in sp["test_tasks"]]  # identical, paired
    assert sc.flips_w2c == sum((not r["baseline"]) and r["with_lib"] for r in rows)
    assert sc.flips_c2w == sum(r["baseline"] and (not r["with_lib"]) for r in rows)
    learn_ids = {task_id(t) for t in learn}
    held = {task_id(t) for t in probe | test}
    for s in lib.skills:
        assert s.provenance and set(s.provenance) <= learn_ids and not set(s.provenance) & held
    assert sc.auditability == 1.0
    if not lib.skills:  # empty library => with-lib eval equals baseline
        assert sc.flips_w2c == sc.flips_c2w == 0


def test_baseline_actually_executes(tmp_path, monkeypatch):
    """Every test task must hit the model once WITHOUT skills injected (step 5)."""
    seen = []
    orig = StubClient.complete

    def spy(self, prompt, *, seed, max_tokens=1024):
        seen.append(prompt)
        return orig(self, prompt, seed=seed, max_tokens=max_tokens)
    monkeypatch.setattr(StubClient, "complete", spy)
    sc, lib, ev = _run("noop", 3, tmp_path)
    assert lib.skills
    for t in _one(ev, "split")["test_tasks"]:
        plain = [p for p in seen if f"Task: {t}\n" in p and "# Relevant skills" not in p]
        skilled = [p for p in seen if f"Task: {t}\n" in p and "# Relevant skills" in p]
        assert plain and skilled, t


def test_auditability_penalises_bad_provenance(tmp_path, monkeypatch):
    from harness.learners.trivial import TrivialLearner
    orig = TrivialLearner.induce

    def leaky(self, traces, current):
        sk = orig(self, traces, current)
        sk[0].provenance = []
        return sk
    monkeypatch.setattr(TrivialLearner, "induce", leaky)
    sc, lib, _ = _run("noop", 0, tmp_path)
    assert sc.library_size == 1 and sc.auditability == 0.0


def test_gate_compares_same_probe_same_seed_and_rejects(tmp_path):
    rejected = []
    for s in SEEDS:
        sc, lib, ev = _run("non_regression", s, tmp_path)
        tr = _one(ev, "train.traces")
        for d in [e for e in ev if e["event"] == "gate.decision"]:
            # gate's fresh replay of the current (empty) library == the step-1 probe traces
            assert d["base_rate"] == tr["probe_pass"] / tr["n_probe"]
            if not d["admitted"]:
                rejected.append(s)
                assert d["cand_rate"] < d["base_rate"] or d["category_drops"]
    assert rejected
