"""Scaled-runner gepa_valset carve (bind_optimizer_split): disjointness, leak guards, non-optimizer unchanged."""
import pytest

from harness.seeding import task_id
from harness.splits import ProbeLeakError, bind_optimizer_split

LEARN = [f"learn {c} {i}" for c in "abc" for i in range(4)]
PROBE = [f"probe {c}" for c in "abc"]
TEST = [f"test {c}" for c in "abc"]


class Bench:
    def category(self, t): return t.split()[1]


class Opt:
    uses_valset = True
    def __init__(self): self.bound = None
    def bind_split(self, train, valset, forbidden_ids):
        self.bound = (train, valset, forbidden_ids)
        for t in train + valset:
            if task_id(t) in forbidden_ids:
                raise ProbeLeakError(t)
    @property
    def internal_valset(self): return self.bound[1] if self.bound else []


class Plain:
    def bind_split(self, *a, **k): raise AssertionError("non-optimizer must not be bound")


def test_valset_carved_from_learn_disjoint_seeded():
    l = Opt()
    tr, val = bind_optimizer_split(Bench(), l, LEARN, PROBE, TEST, seed=0)
    assert len(val) == 6 and {Bench().category(t) for t in val} == set("abc")       # 2 per category
    assert set(tr) | set(val) == set(LEARN) and not set(tr) & set(val)
    assert not (set(val) | set(tr)) & set(PROBE + TEST)
    assert l.bound[2] == {task_id(t) for t in PROBE + TEST}                          # probe AND test forbidden
    assert bind_optimizer_split(Bench(), Opt(), LEARN, PROBE, TEST, seed=0) == (tr, val)  # pinned by seed


def test_non_optimizer_unchanged():
    assert bind_optimizer_split(Bench(), Plain(), LEARN, PROBE, TEST, seed=0) == (LEARN, [])


def test_learn_overlapping_probe_or_test_raises():
    with pytest.raises(ProbeLeakError):
        bind_optimizer_split(Bench(), Opt(), LEARN + [PROBE[0]], PROBE, TEST, seed=0)
    with pytest.raises(ProbeLeakError):
        bind_optimizer_split(Bench(), Opt(), LEARN + [TEST[1]], PROBE, TEST, seed=0)
    with pytest.raises(ProbeLeakError):
        bind_optimizer_split(Bench(), Plain(), LEARN + [TEST[1]], PROBE, TEST, seed=0)


def test_internal_valset_touching_heldout_raises():
    class Bad(Opt):
        @property
        def internal_valset(self): return [TEST[0]]
    with pytest.raises(ProbeLeakError):
        bind_optimizer_split(Bench(), Bad(), LEARN, PROBE, TEST, seed=0)


def test_real_gepa_learner_binds_on_clean_split():
    import json
    from pathlib import Path
    from harness.learners.gepa import GEPALearner
    s = json.loads((Path(__file__).parent.parent / "manifests/alfworld_scale_clean/split.json").read_text())
    # task texts are not needed for disjointness: use ids-as-text with the real category labels
    learn = [e["task_id"] for e in s["learn_corpus"]]; cat = {e["task_id"]: e["type"] for k in ("learn_corpus", "probe", "test")
                                                            for e in s[k]}
    B = type("B", (), {"category": lambda self, t: cat[t]})()
    g = GEPALearner(None, None, None, B, 0)
    tr, val = bind_optimizer_split(B, g, learn, [e["task_id"] for e in s["probe"]], [e["task_id"] for e in s["test"]], 0)
    assert len(val) == 12 and len(tr) == 12 and g.internal_valset == val
