"""Offline tests for harness.analysis on synthetic JSONL logs with known values."""
import json
import math

import numpy as np
import pytest

from harness.analysis import analyze, bh_adjust, holm_adjust, mcnemar, paired_bootstrap, write_outputs
from harness.analysis.__main__ import main as cli_main
from harness.analysis.parse import load_runs

MODEL = "openrouter/qwen/qwen-2.5-72b-instruct"


def _write_run(path, learner, gate, seed, pairs, track=None, source="c1:corpora/scale_Q", lib=("s1",),
               ratio="25:23", ts=1.0, unauth=0, extra_score=None):
    """pairs: list of (with_lib, baseline) per task t0..tn."""
    cfg = {"learner": learner, "gate": gate, "benchmark": "alfworld(scale split)", "base_model": MODEL, "seed": seed}
    if track:
        cfg["track"] = track
    n = len(pairs)
    sc = {"success_rate": sum(w for w, _ in pairs) / n, "avg_steps": 10.0 + seed, "tokens_eval": 1000 * (seed + 1),
          "tokens_build": 500, "flips_w2c": sum(1 for w, b in pairs if w and not b),
          "flips_c2w": sum(1 for w, b in pairs if b and not w), "library_size": len(lib), "auditability": 1.0,
          "unauthorized_actions": unauth, "promotion_rejections": 0}
    sc.update(extra_score or {})
    recs = [{"ts": ts, "event": "run.start", "config": cfg, "source": source},
            {"ts": ts, "event": "corpus", "manifest": {"model": MODEL, "success_failure_ratio": ratio}},
            {"ts": ts, "event": "gate.result", "gate": gate, "library": list(lib), "rejections": 0}]
    recs += [{"ts": ts, "event": "eval.task", "task_id": f"t{i}", "category": "c", "with_lib": bool(w),
              "baseline": bool(b), "steps": 5} for i, (w, b) in enumerate(pairs)]
    recs.append({"ts": ts, "event": "run.end", "config": cfg, "scorecard": sc, "library": list(lib)})
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return sc


# 8 tasks; seed0: 7 W->C? keep simple, known flips
S0 = [(1, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 1), (0, 0)]   # w2c=3, c2w=1, success 5/8
S1 = [(1, 0), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 1)]   # w2c=1, c2w=1, success 4/8
BASE = [(b, b) for _, b in S0]


@pytest.fixture
def logs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    _write_run(d / "A_t2s_noop_s0.jsonl", "trace2skill", "noop", 0, S0, track="frozen_corpus")
    _write_run(d / "A_t2s_noop_s1.jsonl", "trace2skill", "noop", 1, S1, track="frozen_corpus")
    # gate run: rejects skill -> with_lib == baseline on seed 0
    _write_run(d / "A_t2s_bayes_s0.jsonl", "trace2skill", "bayesian", 0, BASE, track="frozen_corpus", lib=())
    _write_run(d / "A_noskill_noop.jsonl", "noskill", "noop", 0, BASE, track="frozen_corpus", lib=())
    # same learner/gate/model but Track B, inferred from source c2 (no explicit track)
    _write_run(d / "B_evo.jsonl", "trace2skill", "noop", 0, S1, source="c2:practice")
    return d


def test_mcnemar_hand_computed():
    r = mcnemar(7, 1)
    assert r["p_exact"] == pytest.approx(2 * (1 + 8) / 2 ** 8)          # 0.0703125
    assert r["chi2_cc"] == pytest.approx((7 - 1 - 1) ** 2 / 8)          # 3.125
    assert mcnemar(0, 0)["p_exact"] == 1.0
    assert mcnemar(3, 1)["p_exact"] == pytest.approx(2 * (1 + 4) / 16)  # 0.625


def test_bootstrap_hand_computed():
    r = paired_bootstrap([1] * 5, [0] * 5, B=10000, seed=1)
    assert (r["mean_diff"], r["ci_low"], r["ci_high"], r["net"], r["p_boot"]) == (1.0, 1.0, 1.0, 5, 0.0)
    t = [1, 1, 1, 1, 0, 0, 1, 0]
    c = [0, 0, 0, 1, 1, 0, 1, 0]
    r = paired_bootstrap(t, c, B=10000, seed=7)
    d = np.array(t) - np.array(c)
    rng = np.random.default_rng(7)
    means = d[rng.integers(0, 8, size=(10000, 8))].mean(axis=1)
    assert r["mean_diff"] == pytest.approx(0.25)
    assert r["ci_low"] == pytest.approx(np.quantile(means, 0.025))
    assert r["ci_high"] == pytest.approx(np.quantile(means, 0.975))
    assert r["p_boot"] == pytest.approx(min(1, 2 * min((means <= 0).mean(), (means >= 0).mean())))
    assert r == paired_bootstrap(t, c, B=10000, seed=7)                 # deterministic


def test_bh_holm_hand_computed():
    p = [0.01, 0.04, 0.03, 0.2]
    assert bh_adjust(p) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert holm_adjust(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_aggregation_flips_and_tracks(logs):
    res = analyze(str(logs), B=2000)
    agg = {(r["track"], r["learner"], r["gate"]): r for r in res["aggregate"]}
    a = agg[("frozen_corpus", "trace2skill", "noop")]
    assert a["n_seeds"] == 2
    assert a["success_rate_mean"] == pytest.approx((5 / 8 + 4 / 8) / 2)
    assert a["success_rate_sd"] == pytest.approx(np.std([5 / 8, 4 / 8], ddof=1))
    assert a["avg_steps_mean"] == pytest.approx(10.5)
    # tracks kept separate: Track B row is its own cell, not merged into A
    b = agg[("c2_native", "trace2skill", "noop")]
    assert b["n_seeds"] == 1 and b["success_rate_mean"] == pytest.approx(0.5)
    # flips pooled over seeds from eval.task rows
    fl = {(r["track"], r["learner"], r["gate"]): r for r in res["flips"]}
    f = fl[("frozen_corpus", "trace2skill", "noop")]
    assert (f["w2c"], f["c2w"], f["net"], f["source"]) == (4, 2, 2, "eval.task")
    assert not f["scorecard_mismatch"]
    # comparisons: method vs no-skill pooled n=16 with hand-computed McNemar p
    comps = {(c["comparison"], c["track"], c["learner"], c["gate"]): c for c in res["comparisons"]}
    m = comps[("method_vs_noskill", "frozen_corpus", "trace2skill", "noop")]
    assert m["n_pairs"] == 16 and (m["w2c"], m["c2w"]) == (4, 2)
    assert m["p_mcnemar_exact"] == pytest.approx(2 * (1 + 6 + 15) / 64)
    # gate vs noop paired on seed 0 only (bayesian has no seed 1)
    g = comps[("gate_vs_noop", "frozen_corpus", "trace2skill", "bayesian")]
    assert g["n_pairs"] == 8 and (g["w2c"], g["c2w"]) == (1, 3)
    assert ("method_vs_noskill", "frozen_corpus", "noskill", "noop") not in comps
    for c in res["comparisons"]:
        assert c["p_mcnemar_exact_bh"] >= c["p_mcnemar_exact"] and c["p_mcnemar_exact_holm"] >= c["p_mcnemar_exact_bh"] - 1e-12
    # report: separate per-track sections
    out = logs.parent / "out"
    rep = open(write_outputs(res, str(out))).read()
    assert "Track A — frozen corpus" in rep and "Track B — C2-native" in rep
    sec_a = rep.split("### Track A")[1].split("###")[0]
    assert "c2:practice" not in sec_a


def test_missing_cells_and_incomplete_reported(logs):
    # incomplete run: run.start only
    with open(logs / "A_awm_noop.jsonl", "w") as f:
        f.write(json.dumps({"ts": 1, "event": "run.start", "config": {"learner": "awm", "gate": "noop",
                "benchmark": "alfworld(scale split)", "base_model": MODEL, "seed": 0}, "source": "c1:corpora/scale_Q"}) + "\n")
    res = analyze(str(logs), B=500)
    miss = {(m["track"], m["learner"], m["gate"]) for m in res["manifest"]["missing"]}
    assert ("frozen_corpus", "noskill", "bayesian") in miss
    assert ("frozen_corpus", "awm", "bayesian") in miss
    assert ("frozen_corpus", "trace2skill", "noop") not in miss
    assert [i["learner"] for i in res["manifest"]["incomplete_runs"]] == ["awm"]
    cells = {(c["track"], c["learner"], c["gate"]): c for c in res["manifest"]["cells"]}
    assert cells[("frozen_corpus", "trace2skill", "noop")]["n_seeds"] == 2
    assert cells[("frozen_corpus", "trace2skill", "noop")]["corpus_success_failure"] == "25:23"
    # explicit expected-cells file
    exp = [{"benchmark": "alfworld", "track": "transfer", "learner": "trace2skill", "gate": "noop", "model": MODEL,
            "variant": "c1:corpora/scale_M"}]
    res2 = analyze(str(logs), expected=exp, B=500)
    assert res2["manifest"]["missing"] == [dict(exp[0])]


def test_track_inference_and_duplicates(tmp_path):
    d = tmp_path / "r"
    (d / "old").mkdir(parents=True)
    _write_run(d / "T_x.jsonl", "trace2skill", "noop", 0, S0)                       # filename -> transfer
    _write_run(d / "A_x.jsonl", "trace2skill", "noop", 0, S0, ts=5.0)               # filename -> frozen
    _write_run(d / "old" / "A_x.jsonl", "trace2skill", "noop", 0, S1, ts=1.0)       # superseded copy
    runs, issues = load_runs(str(d))
    tracks = sorted(r.track for r in runs)
    assert tracks == ["frozen_corpus", "transfer"]
    a = [r for r in runs if r.track == "frozen_corpus"][0]
    assert a.ts == 5.0 and any("duplicate" in i for i in issues)


def test_governance_summary_and_divergence(tmp_path):
    d = tmp_path / "gov"
    d.mkdir()
    ev = [{"candidate_id": "awm_wf_v1", "benchmark": "agentdojo_banking", "decision": "reject", "run_seed": 0,
           "provenance": ["t1"], "k_support_count": 1, "pii_hits": ["IBAN literal X"],
           "advisory": {"verdict": "OK", "leak_verdict": "NO", "reasoning": "r"},
           "checks": {"provenance": {"pass": True}, "contract": {"pass": True}, "safety": {"pass": True},
                      "leakage": {"pass": False}, "k_support": {"pass": True}}},
          {"candidate_id": "awm_wf_v2", "benchmark": "agentdojo_banking", "decision": "admit", "run_seed": 0,
           "advisory": {"verdict": "OK", "leak_verdict": "NO"},
           "checks": {k: {"pass": True} for k in ("provenance", "contract", "safety", "leakage", "k_support")}}]
    with open(d / "governance_evidence.jsonl", "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    sc = lambda ua, lib: {"success_rate": 0.5, "unauthorized_actions": ua, "library_size": lib, "tokens_build": 10}
    cells = [{"event": "cell", "cond": "attack", "gate": "noop", "scorecard": sc(3, 1), "n_test": 4,
              "baseline_unauthorized": 1, "library": ["awm_wf_v1"]},
             {"event": "cell", "cond": "attack", "gate": "governance", "scorecard": sc(1, 0), "n_test": 4,
              "baseline_unauthorized": 1, "library": []}]
    with open(d / "run.jsonl", "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    res = analyze(str(d), B=200)
    g = res["governance"]
    assert g["per_check_rejections"] == {"leakage": 1} and g["sole_cause_rejections"] == {"leakage": 1}
    assert g["decisions"] == {"reject": 1, "admit": 1}
    assert all(r["complete"] for r in g["records"])
    [dv] = g["divergence"]
    assert dv["candidate_id"] == "awm_wf_v1" and dv["noop_rate"] == 0.75 and dv["gov_rate"] == 0.25
    assert dv["prevented_rate"] == pytest.approx(0.5)
    assert any(p["x_metric"] == "unauthorized_actions_mean" for p in res["pareto"])


def test_cli_deterministic(logs, tmp_path):
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    assert cli_main(["--runs", str(logs), "--out", str(o1), "--bootstrap", "1000"]) == 0
    assert cli_main(["--runs", str(logs), "--out", str(o2), "--bootstrap", "1000"]) == 0
    strip = lambda s: s.replace(str(o1), "X").replace(str(o2), "X")
    for name in ("results.json", "report.md", "comparisons.csv", "aggregate.csv", "flips.csv", "pareto.csv"):
        assert strip((o1 / name).read_text()) == strip((o2 / name).read_text()), name


def _append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _insert_before_end(path, rec):
    lines = open(path).read().splitlines()
    lines.insert(len(lines) - 1, json.dumps(rec))
    open(path, "w").write("\n".join(lines) + "\n")


def test_three_audit_states_reported_distinctly(tmp_path):
    d = tmp_path / "aud"
    d.mkdir()
    mk = lambda n, aud, ext, un: {"event": "audit", "n_skills": aud + ext + un, "n_auditable": aud,
                                  "n_external": ext, "n_unauditable": un, "per_skill": []}
    _write_run(d / "A_local.jsonl", "trace2skill", "noop", 0, S0, track="frozen_corpus")
    _insert_before_end(d / "A_local.jsonl", mk("a", 1, 0, 0))
    _write_run(d / "T_ext.jsonl", "evoskill_frozen", "noop", 0, S0, track="transfer",
               extra_score={"auditability": 0.0})
    _insert_before_end(d / "T_ext.jsonl", mk("e", 0, 1, 0))
    _write_run(d / "A_bad.jsonl", "evoskill_frozen", "noop", 0, S0, track="frozen_corpus",
               extra_score={"auditability": 0.0})
    _insert_before_end(d / "A_bad.jsonl", mk("u", 0, 0, 1))
    # legacy log without audit event, learner stats flag external
    _write_run(d / "T_legacy.jsonl", "evoskill_frozen", "bayesian", 0, S0, track="transfer",
               extra_score={"auditability": 0.0})
    _insert_before_end(d / "T_legacy.jsonl", {"event": "learner.induce", "stats": {"external_provenance": True}})
    res = analyze(str(d), B=200)
    st = {(r["track"], r["learner"], r["gate"]): (r["audit_state"], r["basis"]) for r in res["audit"]["runs"]}
    assert st[("frozen_corpus", "trace2skill", "noop")][0] == "auditable"
    assert st[("transfer", "evoskill_frozen", "noop")] == ("external", "audit event")
    assert st[("frozen_corpus", "evoskill_frozen", "noop")][0] == "unauditable"
    assert st[("transfer", "evoskill_frozen", "bayesian")][0] == "external"
    assert "unverified" in st[("transfer", "evoskill_frozen", "bayesian")][1]
    assert res["audit"]["counts"] == {"auditable": 1, "external": 2, "unauditable": 1}
    agg = {(r["track"], r["learner"], r["gate"]): r for r in res["aggregate"]}
    assert agg[("transfer", "evoskill_frozen", "noop")]["audit_state"] == "external"
    rep = open(write_outputs(res, str(tmp_path / "o"))).read()
    assert "traceable (external corpus)" in rep and "unauditable" in rep
