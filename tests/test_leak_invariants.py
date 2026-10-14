"""Frozen/transfer origin disjointness invariant (hard error, analogous to ProbeLeakError)."""
import csv

import pytest

from harness.learners.evoskill_frozen import FrozenEvoSkillLearner
from harness.splits import FrozenOriginLeakError, ProbeLeakError, check_disjoint, check_frozen_origin

M3_ORIGIN = ["t_f1750d1af9", "t_5ce171db07", "t_5d572fbd5d", "t_b90a99fbeb"]


@pytest.fixture
def m3_project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    (root / ".evoskill").mkdir()
    with open(root / "data" / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "question", "answer", "category"])
        w.writeheader()
        for u in M3_ORIGIN:
            w.writerow({"uid": u, "question": "q", "answer": "a", "category": "c"})
    monkeypatch.setenv("EVOSKILL_FROZEN_PROJECT", str(root))
    return root


def test_a_origin_overlapping_test_raises_naming_ids(m3_project):
    l = FrozenEvoSkillLearner()
    with pytest.raises(FrozenOriginLeakError) as e:
        check_frozen_origin(l, {"t_probe1"}, {"t_cfbd947ce0", "t_5ce171db07"}, "evoskill_frozen")
    assert "t_5ce171db07" in str(e.value) and "test" in str(e.value)
    assert isinstance(e.value, ProbeLeakError)


def test_origin_overlapping_probe_raises(m3_project):
    with pytest.raises(FrozenOriginLeakError, match="t_f1750d1af9"):
        check_frozen_origin(FrozenEvoSkillLearner(), {"t_f1750d1af9"}, set(), "evoskill_frozen")


def test_disjoint_origin_passes_and_returns_ids(m3_project):
    assert check_frozen_origin(FrozenEvoSkillLearner(), {"t_p"}, {"t_t"}, "evoskill_frozen") == M3_ORIGIN


def test_unresolvable_origin_is_a_hard_error(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("EVOSKILL_FROZEN_PROJECT", str(tmp_path / "empty"))
    with pytest.raises(FrozenOriginLeakError, match="unavailable"):
        check_frozen_origin(FrozenEvoSkillLearner(), set(), set(), "evoskill_frozen")


def test_future_frozen_learner_without_method_fails_loudly():
    class TransferThing:
        pass
    with pytest.raises(FrozenOriginLeakError, match="no origin_task_ids"):
        check_frozen_origin(TransferThing(), set(), set(), "trace2skill_transfer")
    assert check_frozen_origin(TransferThing(), set(), set(), "trace2skill") is None


def test_four_way_split_still_enforced():
    with pytest.raises(ProbeLeakError):
        check_disjoint(learn_corpus=["a"], gepa_valset=[], gate_probe=["b"], test=["a"])
