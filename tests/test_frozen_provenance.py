"""Frozen EvoSkill provenance: real origin ids carried forward, external state flagged."""
import csv
import json

import pytest

from harness.learners import evoskill_frozen as ef
from harness.provenance import ORIGIN_UNAVAILABLE, audit_states, classify
from harness.types import Skill, Trace


def _project(tmp_path, uids=None, explicit=None):
    root = tmp_path / f"proj{len(list(tmp_path.iterdir()))}"
    (root / ".evoskill").mkdir(parents=True)
    (root / "data").mkdir()
    if uids is not None:
        with open(root / "data" / "dataset.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["uid", "question", "answer", "category"])
            w.writeheader()
            for u in uids:
                w.writerow({"uid": u, "question": "q", "answer": "a", "category": "c"})
    if explicit is not None:
        (root / ".evoskill" / "origin_provenance.json").write_text(json.dumps(explicit))
    return root


class _Log:
    def __init__(self):
        self.recs = []

    def log(self, event, **kw):
        self.recs.append({"event": event, **kw})


def _tr(tid):
    return Trace(task_id=tid, steps=[], outcome=True, failure_mode=None, final_output="", tokens=0)


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    def make(uids=None, explicit=None):
        root = _project(tmp_path, uids, explicit)
        monkeypatch.setenv("EVOSKILL_FROZEN_PROJECT", str(root))
        monkeypatch.setattr(ef, "export_skills", lambda r: ("base+iter-skill-1", {"ctx-skill": "# body"}))
        log = _Log()
        return ef.FrozenEvoSkillLearner(log), log
    return make


def test_a_recovered_origin_ids_not_frozen_marker(frozen):
    l, log = frozen(uids=["t_o1", "t_o2", "t_o1", "t_o3"])
    [s] = l.induce([_tr("t_run1")], None)
    assert s.provenance == ["t_o1", "t_o2", "t_o3"]
    assert not any(p.startswith("frozen:") for p in s.provenance)
    st = log.recs[-1]["stats"]
    assert st["origin_provenance_source"] == "data/dataset.csv:uid" and st["external_provenance"] is True
    # explicit origin file wins over the csv
    l2, _ = frozen(uids=["t_o1"], explicit=["t_x9"])
    assert l2.induce([], None)[0].provenance == ["t_x9"]


def test_unrecoverable_is_labelled_not_masquerading(frozen):
    l, log = frozen(uids=None)
    [s] = l.induce([_tr("t_run1")], None)
    assert s.provenance == [ORIGIN_UNAVAILABLE]
    assert l.external_provenance == {} and log.recs[-1]["stats"]["external_provenance"] is False
    assert classify(s, {"t_run1"}, set(), l.external_provenance) == "unauditable"


def test_b_external_state_not_zero_by_default(frozen):
    l, _ = frozen(uids=["t_o1", "t_o2"])
    skills = l.induce([_tr("t_run1"), _tr("t_run2")], None)
    a = audit_states(skills, {"t_run1", "t_run2"}, {"t_probe", "t_test"}, l.external_provenance)
    assert a["n_external"] == 1 and a["per_skill"][0]["external_provenance"] is True
    assert a["auditability_local"] == 0.0 and a["traceable_fraction"] == 1.0
    # origin id that is a held-out task of THIS run -> unauditable (leak), never external
    a2 = audit_states(skills, {"t_run1"}, {"t_o2"}, l.external_provenance)
    assert a2["per_skill"][0]["state"] == "unauditable"
    # ids that are this run's own traces -> local auditable, not flagged external
    l3, _ = frozen(uids=["t_run1"])
    s3 = l3.induce([_tr("t_run1")], None)
    assert l3.external_provenance == {}
    assert audit_states(s3, {"t_run1"}, set(), l3.external_provenance)["per_skill"][0]["state"] == "auditable"


def test_undeclared_foreign_ids_are_unauditable():
    s = Skill(id="x", content="", representation="skill_md", provenance=["t_elsewhere"], version=1)
    assert classify(s, {"t_run1"}, set(), None) == "unauditable"
    s.provenance = ["frozen:proj"]
    assert classify(s, {"t_run1"}, set(), {"x": {}}) == "unauditable"
