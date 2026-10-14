"""Assemble manifests/alfworld_scale/RESULTS.md + run_manifest.json from logs/scale/*.result.json (n=1, no SD)."""
import json
from pathlib import Path
import os
L, MAN = Path(os.environ.get("SCALE_OUT", "logs/scale")), Path(os.environ.get("SCALE_MANIFEST", "manifests/alfworld_scale"))
R = {p.name[:-12]: json.loads(p.read_text()) for p in L.glob("*.result.json")}
def row(n, label):
    r = R.get(n)
    if not r: return f"| {label} | _not run / incomplete_ |" + " |" * 8
    s = r["scorecard"]
    return (f"| {label} | {s['success_rate']:.3f} (base {r['baseline_success_rate']:.3f}) | +{s['flips_w2c']}/-{s['flips_c2w']} | "
            f"{r['n_candidates']} | {s['library_size']} | {s['auditability']:.2f} | {s['tokens_build']:,} | {s['tokens_eval']:,} | {s['promotion_rejections']} |")
H = ("| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |\n"
     "|---|---|---|---|---|---|---|---|---|")
o = ["# Scaled ALFWorld results (option 3: single seed, n=1 — no SD reported)\n",
     "Split: `split.json` (seed 20261013): test 24 / probe 24 / learn 24 / practice pool 48, 6 task types x 4. "
     "Flips are vs the empty-library run of the same consumer on the same 24 test tasks, same seed (in-cell baseline). "
     "Learner builds (EvoSkill, Trace2Skill, AWM) use `openai/gpt-4o` on the OpenAI key (flagged; not one of the two named models). "
     "Note: `unauthorized_actions` is not meaningful on ALFWorld and is omitted.\n",
     "## Track A — frozen-corpus C1 (corpus: Qwen-72B, consumer: Qwen-72B, retrieval_k 3)\n", H]
for l in ["noskill", "raw_trace_rag", "awm", "trace2skill", "frozen_evoskill"]:
    for g in ["noop", "non_regression", "bayesian"]:
        if l == "raw_trace_rag" and g != "noop": continue
        o.append(row(f"A_{l}_{g}", f"{l} / {g}"))
o += ["\n## Track B — C2-native EvoSkill self-collection\n",
      "**agent-and-difficulty-coupled; not directly comparable to Track A.** Practice sets prefiltered per self-collector "
      "(target ~50% pass). EvoSkill caveat: its ALFWorld use is open-loop (it proposes skills from collected traces; it does not "
      "act in the env during build). Skills built = candidates; 0-skill runs are a finding, not a failure.\n", H]
for m in "QM":
    for g in ["noop", "non_regression", "bayesian"]:
        o.append(row(f"B_{m}_evoskill_{g}", f"self-collector {'Qwen-72B' if m=='Q' else 'gpt-4o-mini'} / {g}"))
o += ["\n## Transfer 2x2 — trace2skill (noop gate), corpus x consumer\n", H,
      row("A_noskill_noop", "no-skill, consumer Qwen"), row("T_noskill_M", "no-skill, consumer gpt-4o-mini"),
      row("A_trace2skill_noop", "Q corpus -> Qwen (same as Track A cell)"), row("T_Mcorpus_on_Q", "M corpus -> Qwen"),
      row("T_Mcorpus_on_M", "M corpus -> gpt-4o-mini"), row("T_Qcorpus_on_M", "Q corpus -> gpt-4o-mini")]
import hashlib
_sp = MAN / "split.json"
_sj = json.loads(_sp.read_text())
man = {"split": str(_sp), "split_sha256": hashlib.sha256(_sp.read_bytes()).hexdigest(), "split_seed": _sj.get("seed"),
       "split_exclude": _sj.get("exclude", []), "split_excluded_skipped": _sj.get("excluded_skipped", []),
       "pinned_test_ids": [e["task_id"] for e in _sj["test"]], "pinned_probe_ids": [e["task_id"] for e in _sj["probe"]],
       "corpora": {}, "practice": {}, "skill_counts": {}}
for m in "QM":
    c = json.loads(Path(f"corpora/scale_{m}/manifest.json").read_text())
    man["corpora"][m] = {"model": c["model"], "success_failure": c["success_failure_ratio"], "n_traces": c["n_traces"], "sha256": c["sha256"]}
    if not (MAN / f"practice_{m}.json").exists():
        continue
    p = json.loads((MAN / f"practice_{m}.json").read_text())
    man["practice"][m] = {"model": p["model"], "pool_pass_rate": p["pool_pass_rate"], "selected_pass_rate": p["selected_pass_rate"]}
for n, r in sorted(R.items()):
    man["skill_counts"][n] = {"candidates": r["n_candidates"], "library": len(r["library"]), "learn_pass": r["learn_pass"], "probe_pass": r["probe_pass"]}
o += ["\n## Manifest\n", "```json", json.dumps(man, indent=1), "```"]
(MAN / "run_manifest.json").write_text(json.dumps(man, indent=1))
(MAN / "RESULTS.md").write_text("\n".join(o) + "\n")
print("\n".join(o))
