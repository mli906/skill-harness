"""Aggregation, comparisons, Pareto, governance summary, manifest; report writer."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from collections import Counter, defaultdict

from .parse import SCORE_FIELDS, TRACK_LABEL, TRACKS, load_evidence, load_runs
from .stats import bh_adjust, holm_adjust, mcnemar, mean_sd, paired_bootstrap

KEY_FIELDS = ("benchmark", "track", "learner", "gate", "model", "variant")
NO_SKILL = {"noskill", "no_skill", "none"}


def _kd(key):
    return dict(zip(KEY_FIELDS, key))


def _label(key):
    d = _kd(key)
    s = f"{d['learner']} / {d['gate']} / {d['model']}"
    return s + (f" [{d['variant']}]" if d["variant"] else "")


# ---------------------------------------------------------------- aggregation
def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        if r.complete and r.scorecard is not None:
            groups[r.key].append(r)
    rows = []
    for key in sorted(groups):
        rs = sorted(groups[key], key=lambda r: r.seed)
        row = _kd(key)
        row["seeds"] = [r.seed for r in rs]
        row["n_seeds"] = len(rs)
        for f in SCORE_FIELDS:
            m, sd, n = mean_sd([r.scorecard.get(f) for r in rs])
            row[f + "_mean"], row[f + "_sd"] = m, sd
        rows.append(row)
    return rows


# ---------------------------------------------------------------- comparisons
def _outcome(v):
    return 1 if bool(v) else 0


def comparisons(runs, B, seed):
    """Paired task-level comparisons, pooled over seeds within a config.

    method_vs_noskill: within each run, eval.task with_lib vs baseline (same tasks,
    same seed — both logged on the row). Skipped for no-skill learners.
    gate_vs_noop: run(learner, gate G) vs run(learner, noop) at identical
    (benchmark, track, model, variant, seed), matched on task_id, comparing with_lib.
    """
    out = []
    by_cfg = defaultdict(list)
    for r in runs:
        if r.complete and r.tasks:
            by_cfg[r.key].append(r)
    for key in sorted(by_cfg):
        d = _kd(key)
        if d["learner"] in NO_SKILL:
            continue
        t, c = [], []
        for r in sorted(by_cfg[key], key=lambda r: r.seed):
            for row in r.tasks:
                t.append(_outcome(row.get("with_lib")))
                c.append(_outcome(row.get("baseline")))
        out.append(_compare("method_vs_noskill", key, None, t, c, B, seed))
    for key in sorted(by_cfg):
        d = _kd(key)
        if d["gate"] == "noop":
            continue
        ref = (d["benchmark"], d["track"], d["learner"], "noop", d["model"], d["variant"])
        if ref not in by_cfg:
            continue
        ref_by_seed = {r.seed: r for r in by_cfg[ref]}
        t, c = [], []
        for r in sorted(by_cfg[key], key=lambda r: r.seed):
            nr = ref_by_seed.get(r.seed)
            if nr is None:
                continue
            nmap = {row.get("task_id"): row for row in nr.tasks}
            for row in r.tasks:
                if row.get("task_id") in nmap:
                    t.append(_outcome(row.get("with_lib")))
                    c.append(_outcome(nmap[row.get("task_id")].get("with_lib")))
        if t:
            out.append(_compare("gate_vs_noop", key, ref, t, c, B, seed))
    return out


def _compare(kind, key, ref, t, c, B, seed):
    w2c = sum(1 for a, b in zip(t, c) if a == 1 and b == 0)
    c2w = sum(1 for a, b in zip(t, c) if a == 0 and b == 1)
    mc = mcnemar(w2c, c2w)
    bs = paired_bootstrap(t, c, B=B, seed=seed)
    row = {"comparison": kind, **_kd(key), "reference": _label(ref) if ref else "no-skill baseline (same run)",
           "n_pairs": len(t), "treat_rate": sum(t) / len(t) if t else float("nan"),
           "ctrl_rate": sum(c) / len(c) if c else float("nan"), "w2c": w2c, "c2w": c2w, "net": w2c - c2w,
           "p_mcnemar_exact": mc["p_exact"], "p_mcnemar_chi2cc": mc["p_chi2_cc"],
           "mean_diff": bs["mean_diff"], "boot_ci_low": bs["ci_low"], "boot_ci_high": bs["ci_high"],
           "net_ci_low": bs["net_ci_low"], "net_ci_high": bs["net_ci_high"], "p_boot": bs["p_boot"]}
    return row


def correct(rows, alpha):
    """BH-FDR and Holm across the whole family, applied to each test's p-values."""
    for col in ("p_mcnemar_exact", "p_boot"):
        ps = [r[col] for r in rows]
        for r, q, h in zip(rows, bh_adjust(ps), holm_adjust(ps)):
            r[col + "_bh"], r[col + "_holm"] = q, h
            r[col + "_sig_bh"], r[col + "_sig_holm"] = q <= alpha, h <= alpha
    return rows


# ---------------------------------------------------------------- flips
def flip_summary(runs, comps):
    """Per-config W->C / C->W against the no-skill baseline from eval.task rows,
    cross-checked against the Scorecard flips (mismatch flagged)."""
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in runs:
        if r.complete and r.scorecard is not None:
            a = agg[r.key]
            a[2] += int(r.scorecard.get("flips_w2c") or 0)
            a[3] += int(r.scorecard.get("flips_c2w") or 0)
    ci = {tuple(c[k] for k in KEY_FIELDS): c for c in comps if c["comparison"] == "method_vs_noskill"}
    rows = []
    for key in sorted(agg):
        c = ci.get(key)
        sw, sc = agg[key][2], agg[key][3]
        row = {**_kd(key), "w2c": c["w2c"] if c else sw, "c2w": c["c2w"] if c else sc}
        row["net"] = row["w2c"] - row["c2w"]
        row["net_ci_low"] = c["net_ci_low"] if c else float("nan")
        row["net_ci_high"] = c["net_ci_high"] if c else float("nan")
        row["source"] = "eval.task" if c else "scorecard"
        row["scorecard_mismatch"] = bool(c) and (sw, sc) != (c["w2c"], c["c2w"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- pareto
def pareto(rows, x, y="success_rate_mean", minimize_x=True):
    """Mark non-dominated configs: maximize y, minimize x (or maximize x)."""
    pts = [r for r in rows if not math.isnan(r.get(x, float("nan"))) and not math.isnan(r.get(y, float("nan")))]
    out = []
    for r in pts:
        dom = False
        for o in pts:
            if o is r:
                continue
            bx = o[x] <= r[x] if minimize_x else o[x] >= r[x]
            sx = o[x] < r[x] if minimize_x else o[x] > r[x]
            if bx and o[y] >= r[y] and (sx or o[y] > r[y]):
                dom = True
                break
        out.append({**{k: r[k] for k in KEY_FIELDS}, "x_metric": x, "x": r[x], "y": r[y], "on_frontier": not dom})
    return out


# ---------------------------------------------------------------- governance
CHECKS = ("provenance", "contract", "safety", "leakage", "k_support")


def governance_summary(runs, evidence):
    per_check = Counter()
    sole = Counter()
    decisions = Counter()
    recs = []
    for e in evidence:
        checks = e.get("checks") or {}
        failed = [k for k in CHECKS if (checks.get(k) or {}).get("pass") is False]
        failed += [k for k, v in checks.items() if k not in CHECKS and isinstance(v, dict) and v.get("pass") is False]
        for k in failed:
            per_check[k] += 1
        if len(failed) == 1:
            sole[failed[0]] += 1
        decisions[e.get("decision", "unknown")] += 1
        adv = e.get("advisory") or {}
        recs.append({"file": e.get("_file"), "candidate_id": e.get("candidate_id"), "benchmark": e.get("benchmark"),
                     "decision": e.get("decision"), "failed_checks": failed,
                     "advisory_contract": adv.get("verdict"), "advisory_leak": adv.get("leak_verdict"),
                     "pii_hits": len(e.get("pii_hits") or []), "k_support_count": e.get("k_support_count"),
                     "complete": all(k in checks for k in CHECKS) and "verdict" in adv and "leak_verdict" in adv})
    # divergence: governance rejected, noop admitted (same benchmark, candidate in a noop library)
    div = []
    for rec in recs:
        if rec["decision"] != "reject":
            continue
        cid = rec["candidate_id"]
        noops = [r for r in runs if r.gate == "noop" and cid in r.library and r.complete]
        if not noops:
            continue
        for nr in sorted(noops, key=lambda r: (r.file, r.variant, r.seed)):
            gov = [r for r in runs if r.gate == "governance" and r.complete and r.benchmark == nr.benchmark
                   and r.learner == nr.learner and r.model == nr.model and r.variant == nr.variant
                   and r.track == nr.track and r.seed == nr.seed]
            n = nr.n_test or 0
            ua = (nr.scorecard or {}).get("unauthorized_actions")
            g = gov[0] if gov else None
            gua = (g.scorecard or {}).get("unauthorized_actions") if g else None
            div.append({"candidate_id": cid, "failed_checks": "+".join(rec["failed_checks"]),
                        "variant": nr.variant, "seed": nr.seed, "n_test": n,
                        "noop_unauthorized": ua, "noop_rate": ua / n if (ua is not None and n) else float("nan"),
                        "gov_unauthorized": gua,
                        "gov_rate": gua / (g.n_test or n) if (gua is not None and (g.n_test or n)) else float("nan"),
                        "baseline_unauthorized": nr.baseline_unauthorized,
                        "noop_success": (nr.scorecard or {}).get("success_rate"),
                        "gov_success": (g.scorecard or {}).get("success_rate") if g else None})
    for d in div:
        d["prevented_rate"] = d["noop_rate"] - d["gov_rate"]
    return {"per_check_rejections": dict(per_check), "sole_cause_rejections": dict(sole),
            "decisions": dict(decisions), "records": recs, "divergence": div, "n_records": len(recs)}


# ---------------------------------------------------------------- manifest
def manifest(runs, expected=None):
    seeds = defaultdict(set)
    ratio = {}
    for r in runs:
        if r.complete:
            seeds[r.key].add(r.seed)
        if r.corpus_ratio or r.learn_pass:
            ratio.setdefault(r.key, r.corpus_ratio or f"learn_pass {r.learn_pass}")
    present = set(seeds)
    if expected:
        exp = set()
        for e in expected:
            exp.add(tuple(str(e.get(k, "")) for k in KEY_FIELDS))
        missing = sorted(exp - present)
    else:
        cells = defaultdict(lambda: (set(), set()))
        for k in {r.key for r in runs}:
            b, t, l, g, m, v = k
            cells[(b, t)][0].add(l)
            cells[(b, t)][1].add(g)
        bmv = defaultdict(set)
        for (b, t, l, g, m, v) in {r.key for r in runs}:
            bmv[(b, t)].add((m, v))
        missing = []
        for (b, t), (ls, gs) in sorted(cells.items()):
            for (m, v) in sorted(bmv[(b, t)]):
                for l, g in itertools.product(sorted(ls), sorted(gs)):
                    if (b, t, l, g, m, v) not in present:
                        missing.append((b, t, l, g, m, v))
    incomplete = [{"file": r.file, **_kd(r.key), "seed": r.seed, "n_eval_task": len(r.tasks)}
                  for r in runs if not r.complete]
    return {"missing": [_kd(k) for k in missing],
            "cells": [{**_kd(k), "n_seeds": len(seeds[k]), "seeds": sorted(seeds[k]),
                       "corpus_success_failure": ratio.get(k)} for k in sorted(seeds)],
            "incomplete_runs": incomplete,
            "expected_source": "user-supplied" if expected else "observed learners x gates per (benchmark, track, model, variant)"}


# ---------------------------------------------------------------- driver
def analyze(runs_dir, expected=None, B=10000, seed=20261013, alpha=0.05):
    runs, issues = load_runs(runs_dir)
    evidence = load_evidence(runs_dir)
    if isinstance(expected, str):
        with open(expected, encoding="utf-8") as f:
            expected = json.load(f)
    agg = aggregate(runs)
    comps = correct(comparisons(runs, B, seed), alpha)
    pareto_rows = []
    for (b, t) in sorted({(r["benchmark"], r["track"]) for r in agg}):
        sub = [r for r in agg if r["benchmark"] == b and r["track"] == t]
        pareto_rows += pareto(sub, "tokens_build_mean")
        if "agentdojo" in b:
            pareto_rows += pareto(sub, "unauthorized_actions_mean")
    notes = sorted({f"{os.path.basename(r.file)}: {n}" for r in runs for n in r.notes})
    for r in runs:
        if r.complete and r.tasks and r.scorecard:
            w = sum(1 for x in r.tasks if x.get("with_lib") and not x.get("baseline"))
            c = sum(1 for x in r.tasks if not x.get("with_lib") and x.get("baseline"))
            if (w, c) != (r.scorecard.get("flips_w2c"), r.scorecard.get("flips_c2w")):
                notes.append(f"{os.path.basename(r.file)}: eval.task flips ({w},{c}) != scorecard "
                             f"({r.scorecard.get('flips_w2c')},{r.scorecard.get('flips_c2w')})")
    return {"runs_dir": os.path.abspath(runs_dir), "n_runs": len(runs),
            "n_complete": sum(r.complete for r in runs), "aggregate": agg, "comparisons": comps,
            "flips": flip_summary(runs, comps), "pareto": pareto_rows,
            "governance": governance_summary(runs, evidence), "manifest": manifest(runs, expected),
            "parse_issues": issues, "parse_notes": notes,
            "params": {"B": B, "seed": seed, "alpha": alpha}}


def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:,.0f}"
    return str(x)


def _ms(r, f):
    m, sd = r[f + "_mean"], r[f + "_sd"]
    return f"{_f(m)} ± {_f(sd)}" if not math.isnan(sd) else f"{_f(m)} ± —"


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def _plots(res, out):
    paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional; data tables are still written
        return paths, f"matplotlib unavailable ({e.__class__.__name__}); plots skipped"
    groups = defaultdict(list)
    for p in res["pareto"]:
        groups[(p["benchmark"], p["track"], p["x_metric"])].append(p)
    for (b, t, x), pts in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for p in pts:
            ax.scatter(p["x"], p["y"], marker="*" if p["on_frontier"] else "o",
                       s=120 if p["on_frontier"] else 40, c="C3" if p["on_frontier"] else "C0")
            ax.annotate(f"{p['learner']}/{p['gate']}", (p["x"], p["y"]), fontsize=7)
        fr = sorted([p for p in pts if p["on_frontier"]], key=lambda p: p["x"])
        ax.plot([p["x"] for p in fr], [p["y"] for p in fr], "C3--", lw=1)
        ax.set_xlabel(x.replace("_mean", "")); ax.set_ylabel("success_rate")
        ax.set_title(f"{b} — {t}", fontsize=9)
        fig.tight_layout()
        name = f"pareto_{b}_{t}_{x.replace('_mean', '')}.png".replace("/", "_")
        fig.savefig(os.path.join(out, name), dpi=120, metadata={"Software": None})
        plt.close(fig)
        paths.append(name)
    return paths, None


def write_outputs(res, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
    _write_csv(os.path.join(out, "aggregate.csv"), res["aggregate"])
    _write_csv(os.path.join(out, "comparisons.csv"), res["comparisons"])
    _write_csv(os.path.join(out, "flips.csv"), res["flips"])
    _write_csv(os.path.join(out, "pareto.csv"), res["pareto"])
    _write_csv(os.path.join(out, "governance_divergence.csv"), res["governance"]["divergence"])
    _write_csv(os.path.join(out, "governance_records.csv"), res["governance"]["records"])
    _write_csv(os.path.join(out, "manifest_missing.csv"), res["manifest"]["missing"])
    plots, plot_note = _plots(res, out)
    L = [f"# Results analysis", "", f"Runs dir: `{res['runs_dir']}` — {res['n_runs']} runs parsed, "
         f"{res['n_complete']} complete. Bootstrap B={res['params']['B']}, seed={res['params']['seed']}, "
         f"alpha={res['params']['alpha']}.", ""]
    L.append("## 1. Comparison tables (one per track; never merged)\n")
    cols = ["config", "seeds"] + list(SCORE_FIELDS)
    for t in list(TRACKS) + ["unassigned"]:
        sub = [r for r in res["aggregate"] if r["track"] == t]
        if not sub:
            continue
        for b in sorted({r["benchmark"] for r in sub}):
            L.append(f"### {TRACK_LABEL[t]} — {b}\n")
            L.append(_md_table(cols, [[_label(tuple(r[k] for k in KEY_FIELDS)), r["n_seeds"]] +
                                      [_ms(r, f) for f in SCORE_FIELDS] for r in sub if r["benchmark"] == b]))
            L.append("")
    L.append("## 2. Paired significance (McNemar exact + paired bootstrap; BH-FDR and Holm across the family)\n")
    if res["comparisons"]:
        L.append(_md_table(["type", "track", "config", "vs", "n", "W→C", "C→W", "Δ (95% boot CI)",
                            "p McNemar exact", "BH q", "Holm p", "p boot", "BH q (boot)", "Holm (boot)"],
                           [[c["comparison"], c["track"], _label(tuple(c[k] for k in KEY_FIELDS)), c["reference"],
                             c["n_pairs"], c["w2c"], c["c2w"],
                             f"{_f(c['mean_diff'])} [{_f(c['boot_ci_low'])}, {_f(c['boot_ci_high'])}]",
                             _f(c["p_mcnemar_exact"], 4),
                             _f(c["p_mcnemar_exact_bh"], 4) + (" ✓" if c["p_mcnemar_exact_sig_bh"] else ""),
                             _f(c["p_mcnemar_exact_holm"], 4) + (" ✓" if c["p_mcnemar_exact_sig_holm"] else ""),
                             _f(c["p_boot"], 4),
                             _f(c["p_boot_bh"], 4) + (" ✓" if c["p_boot_sig_bh"] else ""),
                             _f(c["p_boot_holm"], 4) + (" ✓" if c["p_boot_sig_holm"] else "")]
                            for c in res["comparisons"]]))
        L.append(f"\n✓ = survives correction at alpha={res['params']['alpha']}. Family = all "
                 f"{len(res['comparisons'])} comparisons above (corrections applied separately to the McNemar "
                 "and bootstrap p-values). McNemar χ² (continuity-corrected) p is in comparisons.csv.\n")
    else:
        L.append("_No paired comparisons (no eval.task rows)._\n")
    L.append("## 3. Flip summary (vs no-skill baseline)\n")
    L.append(_md_table(["track", "config", "W→C", "C→W", "net", "net 95% paired boot CI", "source", "scorecard mismatch"],
                       [[r["track"], _label(tuple(r[k] for k in KEY_FIELDS)), r["w2c"], r["c2w"], r["net"],
                         f"[{_f(r['net_ci_low'], 1)}, {_f(r['net_ci_high'], 1)}]", r["source"],
                         "YES" if r["scorecard_mismatch"] else ""] for r in res["flips"]]))
    L.append("\n## 4. Pareto frontier\n")
    L.append(_md_table(["benchmark", "track", "config", "x metric", "x", "success", "frontier"],
                       [[p["benchmark"], p["track"], _label(tuple(p[k] for k in KEY_FIELDS)),
                         p["x_metric"].replace("_mean", ""), _f(p["x"]), _f(p["y"]), "★" if p["on_frontier"] else ""]
                        for p in res["pareto"]]))
    L.append("")
    for p in plots:
        L.append(f"![{p}]({p})")
    if plot_note:
        L.append(f"_{plot_note}_")
    g = res["governance"]
    L.append("\n## 5. Governance summary\n")
    if g["n_records"]:
        L.append(f"Evidence records: {g['n_records']}; decisions: {json.dumps(g['decisions'], sort_keys=True)}\n")
        L.append(_md_table(["check", "rejections (failed)", "sole cause"],
                           [[k, g["per_check_rejections"].get(k, 0), g["sole_cause_rejections"].get(k, 0)]
                            for k in CHECKS]))
        L.append("\n" + _md_table(["candidate", "decision", "failed checks", "advisory (contract/leak)", "PII hits",
                                   "k-support", "record complete"],
                                  [[r["candidate_id"], r["decision"], "+".join(r["failed_checks"]) or "—",
                                    f"{r['advisory_contract']}/{r['advisory_leak']}", r["pii_hits"],
                                    r["k_support_count"], r["complete"]] for r in g["records"]]))
        L.append("\n### Divergence: governance rejected, noop admitted\n")
        if g["divergence"]:
            L.append(_md_table(["candidate", "failed", "condition", "n_test", "noop unauth (rate)", "gov unauth (rate)",
                                "prevented rate", "noop success", "gov success"],
                               [[d["candidate_id"], d["failed_checks"], d["variant"], d["n_test"],
                                 f"{_f(d['noop_unauthorized'])} ({_f(d['noop_rate'])})",
                                 f"{_f(d['gov_unauthorized'])} ({_f(d['gov_rate'])})", _f(d["prevented_rate"]),
                                 _f(d["noop_success"]), _f(d["gov_success"])] for d in g["divergence"]]))
        else:
            L.append("_No governance-rejected candidate appears in a noop library._")
    else:
        L.append("_No governance evidence logs found._")
    m = res["manifest"]
    L.append("\n## 6. Run manifest\n")
    L.append(f"Expected cells: {m['expected_source']}.\n")
    L.append(f"**Missing cells ({len(m['missing'])})**\n")
    L.append(_md_table(list(KEY_FIELDS), [[d[k] for k in KEY_FIELDS] for d in m["missing"]]) if m["missing"] else "_none_")
    L.append(f"\n**Incomplete runs (run.start without run.end): {len(m['incomplete_runs'])}**\n")
    if m["incomplete_runs"]:
        L.append(_md_table(["file", "config", "seed", "eval.task rows"],
                           [[os.path.basename(i["file"]), _label(tuple(i[k] for k in KEY_FIELDS)), i["seed"],
                             i["n_eval_task"]] for i in m["incomplete_runs"]]))
    L.append("\n**Seeds per cell and corpus success:failure**\n")
    L.append(_md_table(["track", "config", "n_seeds", "seeds", "corpus S:F"],
                       [[c["track"], _label(tuple(c[k] for k in KEY_FIELDS)), c["n_seeds"], c["seeds"],
                         c["corpus_success_failure"] or "—"] for c in m["cells"]]))
    if res["parse_issues"] or res["parse_notes"]:
        L.append("\n## Parse notes\n")
        L += [f"- {x}" for x in res["parse_issues"] + res["parse_notes"]]
    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
