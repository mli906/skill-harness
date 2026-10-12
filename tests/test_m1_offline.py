"""Offline M1 tests (no network LLM calls)."""
import subprocess
from pathlib import Path
import pytest
from harness.seeding import task_id
from harness.types import Trace, SkillLibrary

oqa = pytest.importorskip("harness.benchmarks.officeqa")


@pytest.fixture(scope="module")
def bench():
    return oqa.OfficeQAAdapter(source="sample")


def test_officeqa_splits_and_official_verifier(bench):
    tr, te = bench.train_tasks(), bench.test_tasks()
    assert tr and te and not set(tr) & set(te)
    assert len(tr) + len(te) == len(bench.rows) == 10
    for t in tr + te:
        assert bench.verify(t, bench.answer(t))
        assert not bench.verify(t, "not a number")
        assert "Treasury Bulletin excerpts" in t
    # uses OfficeQA's own reward module (pinned checkout)
    assert bench._reward.__file__.endswith("reward.py") and "7b9a3c154ef9" in bench._reward.__file__


def test_officeqa_limit():
    b = oqa.OfficeQAAdapter(source="sample", limit=6)
    assert len(b.rows) == 6 and len(b.train_tasks()) + len(b.test_tasks()) == 6


def test_probe_disjoint_on_officeqa(bench):
    from harness.runner import _split_probe
    learn, probe = _split_probe(bench, bench.train_tasks(), 0)
    assert not set(learn) & set(probe)
    assert not set(probe) & set(bench.test_tasks()) and not set(learn) & set(bench.test_tasks())


def _fake_evoskill_repo(tmp_path: Path) -> Path:
    from harness.learners import evoskill as E
    root = E.write_project(tmp_path / "proj", [{"uid": "u", "question": "q", "answer": "1", "category": "easy"}],
                           [], "openai/gpt-4o-mini", "openhands")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "checkout", "-qb", "program/iter-skill-1"], cwd=root, check=True)
    d = root / ".claude" / "skills" / "table-lookup"; d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: table-lookup\ndescription: x\n---\nRead tables.\n")
    (d / "helper.py").write_text("print(1)\n")
    sc = root / ".claude" / "skills" / "skill-creator"; sc.mkdir()
    (sc / "SKILL.md").write_text("---\nname: skill-creator\n---\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(g + ["commit", "-qm", "skill"], cwd=root, check=True)
    return root


def test_evoskill_output_mapping(tmp_path, monkeypatch, bench):
    pytest.importorskip("src.registry")
    from harness.learners import evoskill as E
    root = _fake_evoskill_repo(tmp_path)
    sk = E.read_skills(root, "iter-skill-1")
    assert list(sk) == ["table-lookup"] and sk["table-lookup"].startswith("---")
    assert "## file: helper.py" in sk["table-lookup"]
    # end-to-end induce with the EvoSkill subprocess replaced (only the LLM loop is faked)
    train = bench.train_tasks()
    lookup = {task_id(t): t for t in train}
    traces = [Trace(task_id(t), [], False, "wrong_answer", "", 0) for t in train]
    L = E.EvoSkillLearner("gpt-4o-mini", bench, lookup, 0, workdir=str(tmp_path / "w"))
    rows = L.traces_to_rows(traces)
    assert [r["uid"] for r in rows] == list(lookup) and all(r["answer"] for r in rows)
    assert all(r["question"] == bench.question(lookup[r["uid"]]) for r in rows)
    monkeypatch.setattr(E, "run_evoskill", lambda r, lf: (lf.write_text("(1 val samples)"), 0)[1])
    monkeypatch.setattr(E, "best_program", lambda r: "iter-skill-1")
    monkeypatch.setattr(E, "read_skills", lambda r, p: sk)
    out = L.induce(traces, SkillLibrary([]))
    assert len(out) == 1 and out[0].representation == "skill_md" and out[0].version == 1
    assert out[0].provenance == list(lookup)
    assert L.last_stats["traces_in"] == L.last_stats["rows_written"] == len(train)
    cfg = (Path(L.last_stats["project"]) / ".evoskill" / "config.toml").read_text()
    assert 'model = "openai/gpt-4o-mini"' in cfg and 'name = "openhands"' in cfg


def test_evoskill_pinned_install():
    import importlib.metadata as md
    d = md.distribution("evoskill")
    direct = d.read_text("direct_url.json") or ""
    assert "36f6f04952293d7054145550c2b9f0b0411bff1c" in direct
