"""Four-way split: learn_corpus / gepa_valset / gate_probe / test, pairwise disjoint."""
import json
import pytest
from harness import registry
from harness.context import Context
from harness.runner import _split_probe, run
from harness.seeding import task_id
from harness.splits import ProbeLeakError, carve_valset, check_disjoint
from harness.types import RunConfig


def _bench():
    cfg = RunConfig("trivial", "noop", "skill_md", "synthetic", "stub", 0, 3)
    return registry.build("benchmark", "synthetic", Context(cfg=cfg, llm=None, embedder=None))


@pytest.mark.parametrize("seed", range(5))
def test_four_slices_pairwise_disjoint_and_stratified(seed):
    b = _bench()
    learn2, probe = _split_probe(b, b.train_tasks(), seed)
    learn, val = carve_valset(b, learn2, seed)
    check_disjoint(learn_corpus=learn, gepa_valset=val, gate_probe=probe, test=b.test_tasks())
    assert sorted(learn + val + probe) == sorted(b.train_tasks())
    cats = {b.category(t) for t in b.train_tasks()}
    assert val and all(sum(b.category(t) == c for t in val) == 2 for c in cats)
    assert carve_valset(b, learn2, seed) == (learn, val)            # pinned by seed


def test_check_disjoint_raises():
    with pytest.raises(ProbeLeakError):
        check_disjoint(gepa_valset=["a", "b"], gate_probe=["b"])


def test_gepa_run_valset_vs_probe_and_auditability(tmp_path, monkeypatch):
    monkeypatch.setenv("GEPA_MAX_METRIC_CALLS", "20")
    log = tmp_path / "g.jsonl"
    sc = run(RunConfig("gepa", "non_regression", "skill_md", "synthetic", "stub", 0, 3), log_path=str(log))
    ev = [json.loads(l) for l in log.read_text().splitlines()]
    split = next(e for e in ev if e.get("event", e.get("type")) == "split" or "gepa_valset" in e)
    val_ids = {task_id(t) for t in split["gepa_valset"]}
    probe_ids = {task_id(t) for t in split["probe_tasks"]}
    assert val_ids and not val_ids & probe_ids
    lib = run.last_library
    for s in lib.skills:
        assert set(s.provenance) <= val_ids and not set(s.provenance) & probe_ids
    if lib.skills:
        assert sc.auditability > 0
    assert sc.tokens_build > 0
