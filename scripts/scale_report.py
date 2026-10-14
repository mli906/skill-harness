"""Assemble RESULTS.md + run_manifest.json from <SCALE_OUT>/*.result.json (n=1, no SD).

Degrades gracefully: writes a report for WHATEVER cells/tracks exist. Absent tracks and absent
corpora/practice manifests are skipped with a "section skipped: <reason>" note (printed and kept
in the manifest's "skipped" list) instead of crashing.

Env: SCALE_OUT (logs, default logs/scale), SCALE_MANIFEST (split + practice, default
manifests/alfworld_scale), CORPORA_DIR (default corpora), REPORT_OUT (where RESULTS.md and
run_manifest.json are written; default SCALE_MANIFEST).
"""
import hashlib
import json
import os
from pathlib import Path

L = Path(os.environ.get("SCALE_OUT", "logs/scale"))
MAN = Path(os.environ.get("SCALE_MANIFEST", "manifests/alfworld_scale"))
CORP = Path(os.environ.get("CORPORA_DIR", "corpora"))
DEST = Path(os.environ.get("REPORT_OUT", str(MAN)))
R = {p.name[:-12]: json.loads(p.read_text()) for p in sorted(L.glob("*.result.json"))}
SKIPPED = []
GATES = ["noop", "non_regression", "bayesian"]


def skip(section, reason):
    msg = f"section skipped: {section}: {reason}"
    SKIPPED.append(msg)
    print(msg)
    return f"\n_{msg}_\n"


def row(n, label):
    r = R.get(n)
    if not r: return f"| {label} | _not run / incomplete_ |" + " |" * 8
    s = r["scorecard"]
    return (f"| {label} | {s['success_rate']:.3f} (base {r['baseline_success_rate']:.3f}) | +{s['flips_w2c']}/-{s['flips_c2w']} | "
            f"{r['n_candidates']} | {s['library_size']} | {s['auditability']:.2f} | {s['tokens_build']:,} | {s['tokens_eval']:,} | {s['promotion_rejections']} |")


H = ("| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |\n"
     "|---|---|---|---|---|---|---|---|---|")
_sp = MAN / "split.json"
_sj = json.loads(_sp.read_text()) if _sp.exists() else None
n_test = len(_sj["test"]) if _sj else None
o = ["# Scaled ALFWorld results (option 3: single seed, n=1 — no SD reported)\n",
     (f"Split: `{_sp}` (seed {_sj.get('seed')}): " if _sj else "Split: _split.json not found_ — ")
     + "test 24 / probe 24 / learn 24 / practice pool 48, 6 task types x 4. "
     "Flips are vs the empty-library run of the same consumer on the same 24 test tasks, same seed (in-cell baseline). "
     "Learner builds (EvoSkill, Trace2Skill, AWM) use `openai/gpt-4o` on the OpenAI key (flagged; not one of the two named models). "
     "Note: `unauthorized_actions` is not meaningful on ALFWorld and is omitted.\n"]
if _sj and _sj.get("exclude"):
    o.append(f"Clean split: task_ids {', '.join(_sj['exclude'])} (frozen M3 skill origin) excluded from every set; "
             f"skipped: {', '.join(e['task_id'] for e in _sj.get('excluded_skipped', []))}.\n")

# ---- summary computed from the data (never asserted)
from scipy.stats import binomtest  # noqa: E402
fam = [(n, r["scorecard"]["flips_w2c"], r["scorecard"]["flips_c2w"]) for n, r in sorted(R.items())
       if r["scorecard"]["flips_w2c"] + r["scorecard"]["flips_c2w"] > 0]
ps = [binomtest(b, b + c, 0.5).pvalue for _, b, c in fam]
m = len(ps)
order = sorted(range(m), key=lambda i: ps[i])
holm, run = [0.0] * m, 0.0
for k, i in enumerate(order):
    run = max(run, (m - k) * ps[i]); holm[i] = min(1.0, run)
bh, prev = [0.0] * m, 1.0
for rank in range(m, 0, -1):
    i = order[rank - 1]; prev = min(prev, ps[i] * m / rank); bh[i] = prev
sig = [fam[i][0] for i in range(m) if holm[i] <= 0.05 or bh[i] <= 0.05]
gated = {n: r for n, r in R.items() if any(n.endswith("_" + g) for g in ("non_regression", "bayesian"))}
admitted = sorted(n for n, r in gated.items() if r["scorecard"]["library_size"] > 0)
noskill_equal = sorted(n for n, r in gated.items() if r["scorecard"]["library_size"] == 0
                       and r["scorecard"]["flips_w2c"] == 0 and r["scorecard"]["flips_c2w"] == 0)
o += ["## Summary (computed from the logs)\n",
      f"- n = {n_test if n_test else '?'} test tasks, 1 seed, {len(R)} cells present.",
      f"- Significance: per-cell exact McNemar on flips vs the in-cell no-skill baseline ({m} cells with any flip), "
      f"Holm and BH across that family: " + ("**nothing significant after correction** "
      f"(smallest raw p = {min(ps):.3f}; Holm-adjusted {holm[order[0]]:.3f})." if not sig and m else
      ("no cell has any flip." if not m else f"significant after correction: {', '.join(sig)}.")),
      (f"- Gated cells (non_regression / bayesian): {len(gated)}; "
       + ("**all equal no-skill** — every gate rejected every candidate (library_size 0, 0/0 flips)."
          if not admitted and len(noskill_equal) == len(gated) else
          f"admitted a skill: {', '.join(admitted) or 'none'}; equal to no-skill: {len(noskill_equal)}/{len(gated)}.")),
      "- `unauthorized_actions` counts actions outside the admissible-command list; not a safety signal on ALFWorld.\n"]

tracks = [("A", "## Track A — frozen-corpus C1 (corpus: Qwen-72B, consumer: Qwen-72B, retrieval_k 3)\n"),
          ("B", "## Track B — C2-native EvoSkill self-collection\n"),
          ("T", "## Transfer 2x2 — trace2skill (noop gate), corpus x consumer\n")]
present = {t: any(n.startswith(t + "_") for n in R) for t, _ in tracks}
if present["A"]:
    o += [tracks[0][1], H]
    for l in ["noskill", "raw_trace_rag", "awm", "trace2skill", "frozen_evoskill"]:
        for g in GATES:
            if l == "raw_trace_rag" and g != "noop": continue
            o.append(row(f"A_{l}_{g}", f"{l} / {g}"))
else:
    o.append(skip("Track A", "no A_* cells in logs"))
if present["B"]:
    o += ["\n" + tracks[1][1],
          "**agent-and-difficulty-coupled; not directly comparable to Track A.** Practice sets prefiltered per self-collector "
          "(target ~50% pass). EvoSkill caveat: its ALFWorld use is open-loop (it proposes skills from collected traces; it does not "
          "act in the env during build). Skills built = candidates; 0-skill runs are a finding, not a failure.\n", H]
    for mm in "QM":
        for g in GATES:
            o.append(row(f"B_{mm}_evoskill_{g}", f"self-collector {'Qwen-72B' if mm=='Q' else 'gpt-4o-mini'} / {g}"))
else:
    o.append(skip("Track B", "no B_* cells in logs"))
if present["T"]:
    o += ["\n" + tracks[2][1], H,
          row("A_noskill_noop", "no-skill, consumer Qwen"), row("T_noskill_M", "no-skill, consumer gpt-4o-mini"),
          row("A_trace2skill_noop", "Q corpus -> Qwen (same as Track A cell)"), row("T_Mcorpus_on_Q", "M corpus -> Qwen"),
          row("T_Mcorpus_on_M", "M corpus -> gpt-4o-mini"), row("T_Qcorpus_on_M", "Q corpus -> gpt-4o-mini")]
else:
    o.append(skip("Transfer 2x2", "no T_* cells in logs"))

man = {"split": str(_sp), "corpora": {}, "practice": {}, "skill_counts": {}, "skipped": SKIPPED}
if _sj:
    man.update({"split_sha256": hashlib.sha256(_sp.read_bytes()).hexdigest(), "split_seed": _sj.get("seed"),
                "split_exclude": _sj.get("exclude", []), "split_excluded_skipped": _sj.get("excluded_skipped", []),
                "pinned_test_ids": [e["task_id"] for e in _sj["test"]], "pinned_probe_ids": [e["task_id"] for e in _sj["probe"]]})
else:
    skip("split manifest", f"{_sp} not found")
used = {"Q": present["A"] or any(n.startswith("B_Q") or "Qcorpus" in n for n in R),
        "M": any(n.startswith("B_M") or n.startswith("T_") for n in R)}
for mm in "QM":
    cp = CORP / f"scale_{mm}" / "manifest.json"
    if not used[mm]:
        skip(f"corpus {mm}", "no cell in logs uses it")
    elif not cp.exists():
        skip(f"corpus {mm}", f"{cp} not found")
    else:
        c = json.loads(cp.read_text())
        man["corpora"][mm] = {"model": c["model"], "success_failure": c["success_failure_ratio"], "n_traces": c["n_traces"], "sha256": c["sha256"]}
    pp = MAN / f"practice_{mm}.json"
    if not any(n.startswith(f"B_{mm}") for n in R):
        skip(f"practice {mm}", "no Track B cell for this self-collector")
    elif not pp.exists():
        skip(f"practice {mm}", f"{pp} not found")
    else:
        p = json.loads(pp.read_text())
        man["practice"][mm] = {"model": p["model"], "pool_pass_rate": p["pool_pass_rate"], "selected_pass_rate": p["selected_pass_rate"]}
for n, r in sorted(R.items()):
    man["skill_counts"][n] = {"candidates": r["n_candidates"], "library": len(r["library"]), "learn_pass": r["learn_pass"], "probe_pass": r["probe_pass"]}
_gv = MAN / "gepa_valset.json"
if _gv.exists():
    man["gepa_valset"] = json.loads(_gv.read_text())
man["skipped"] = SKIPPED
o += ["\n## Manifest\n", "```json", json.dumps(man, indent=1), "```"]
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "run_manifest.json").write_text(json.dumps(man, indent=1))
(DEST / "RESULTS.md").write_text("\n".join(o) + "\n")
print("\n".join(o))
