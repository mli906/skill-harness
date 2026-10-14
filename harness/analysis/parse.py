"""Parse harness JSONL run logs into Run records (read-only)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

TRACKS = ("frozen_corpus", "c2_native", "transfer")
TRACK_LABEL = {"frozen_corpus": "Track A — frozen corpus (C1)",
               "c2_native": "Track B — C2-native (self-collecting)",
               "transfer": "Track T — transfer (cross-model corpus)",
               "unassigned": "Unassigned track (no run.start track/source/filename prefix)"}
_PREFIX_TRACK = {"A": "frozen_corpus", "B": "c2_native", "T": "transfer"}
_TRACK_ALIASES = {"a": "frozen_corpus", "c1": "frozen_corpus", "frozen": "frozen_corpus",
                  "frozen_corpus": "frozen_corpus", "frozen-corpus": "frozen_corpus",
                  "b": "c2_native", "c2": "c2_native", "c2_native": "c2_native", "c2-native": "c2_native",
                  "t": "transfer", "transfer": "transfer"}
SCORE_FIELDS = ("success_rate", "avg_steps", "tokens_eval", "tokens_build", "flips_w2c", "flips_c2w",
                "library_size", "auditability", "unauthorized_actions", "promotion_rejections")


@dataclass
class Run:
    file: str
    benchmark: str
    track: str
    learner: str
    gate: str
    model: str
    seed: int
    variant: str = ""            # corpus source and/or attack condition (disambiguates cells)
    scorecard: dict | None = None
    tasks: list = field(default_factory=list)   # eval.task rows
    library: list = field(default_factory=list)
    candidate_ids: list = field(default_factory=list)
    gate_rejections: int | None = None
    corpus_ratio: str | None = None
    learn_pass: str | None = None
    n_test: int | None = None
    baseline_unauthorized: float | None = None
    complete: bool = False
    ts: float = 0.0
    audit: dict | None = None          # runner 'audit' event (three-state provenance)
    induce_stats: dict | None = None
    notes: list = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.benchmark, self.track, self.learner, self.gate, self.model, self.variant)

    @property
    def is_agentdojo(self) -> bool:
        return "agentdojo" in self.benchmark.lower()


def _norm_track(v):
    return _TRACK_ALIASES.get(str(v).strip().lower()) if v else None


def infer_track(config: dict, rec: dict, source: str | None, corpus_model: str | None, path: str):
    """Return (track, how). Precedence:
    1. explicit ``track`` in run.start config/record;
    2. source ``c2:*`` -> c2_native; ``c1:*`` whose corpus model != base model -> transfer;
    3. filename prefix A_/B_/T_ (needed for same-model c1 runs: A_* vs T_*_on_M / T_noskill_M);
    4. ``c1:*`` -> frozen_corpus; else unassigned.
    """
    t = _norm_track(config.get("track") or rec.get("track"))
    if t:
        return t, "run.start"
    base = config.get("base_model")
    if source and source.startswith("c2"):
        return "c2_native", "source"
    if source and source.startswith("c1") and corpus_model and base and corpus_model != base:
        return "transfer", "source+corpus_model"
    m = re.match(r"([ABT])_", os.path.basename(path))
    if m:
        return _PREFIX_TRACK[m.group(1)], "filename"
    if source and source.startswith("c1"):
        return "frozen_corpus", "source"
    return "unassigned", "none"


def _bench(name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(name or "unknown")).strip().lower() or "unknown"


def _read_jsonl(path):
    out, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


def is_evidence_file(path: str) -> bool:
    return "evidence" in os.path.basename(path)


def _iter_files(root):
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".jsonl"):
                yield os.path.join(d, f)


def _finish(run: Run, recs: list):
    pending = [r for r in recs]
    for r in pending:
        ev = r.get("event")
        if ev == "corpus":
            run.corpus_ratio = (r.get("manifest") or {}).get("success_failure_ratio")
        elif ev == "train.traces":
            if r.get("n_learn") is not None:
                run.learn_pass = f"{r.get('learn_pass')}/{r.get('n_learn')}"
        elif ev == "gate.result":
            run.library = list(r.get("library") or [])
            run.gate_rejections = r.get("rejections")
        elif ev == "audit":
            run.audit = {k: v for k, v in r.items() if k not in ("ts", "event")}
        elif ev == "learner.induce":
            run.induce_stats = r.get("stats") if isinstance(r.get("stats"), dict) else None
            run.candidate_ids = list(r.get("skill_ids") or run.candidate_ids)
        elif ev == "eval.task":
            run.tasks.append(r)
        elif ev == "run.end":
            run.scorecard = dict(r.get("scorecard") or {})
            run.library = list(r.get("library", run.library) or [])
            run.candidate_ids = list(r.get("candidate_ids") or run.candidate_ids)
            run.complete = True
            if run.learn_pass is None and r.get("learn_pass"):
                run.learn_pass = r.get("learn_pass")
    run.n_test = len(run.tasks) if run.tasks else run.n_test


def _runs_from_starts(path, recs):
    runs, cur, cur_recs = [], None, []

    def close():
        if cur is not None:
            _finish(cur, cur_recs)
            runs.append(cur)

    for r in recs:
        if r.get("event") == "run.start":
            close()
            cfg = r.get("config") or {}
            source = r.get("source")
            corpus_model = None
            cur_recs = []
            cur = Run(file=path, benchmark=_bench(cfg.get("benchmark")), track="", learner=str(cfg.get("learner")),
                      gate=str(cfg.get("gate")), model=str(cfg.get("base_model") or cfg.get("model")),
                      seed=int(cfg.get("seed", 0)), variant=str(source or ""),
                      ts=float(r.get("ts") or 0.0))
            cur._cfg, cur._rec, cur._source = cfg, r, source  # resolved after corpus event is seen
            continue
        if cur is None:
            continue
        if r.get("event") == "corpus":
            cur._corpus_model = (r.get("manifest") or {}).get("model")
        cur_recs.append(r)
    close()
    for run in runs:
        t, how = infer_track(run._cfg, run._rec, run._source, getattr(run, "_corpus_model", None), path)
        run.track = t
        if how in ("filename", "none"):
            run.notes.append(f"track inferred from {how}")
        for a in ("_cfg", "_rec", "_source", "_corpus_model"):
            if hasattr(run, a):
                delattr(run, a)
    return runs


def _runs_from_cells(path, recs, evidence_dir_records):
    """Governance-smoke format: no run.start; one ``cell`` event per (condition, gate)."""
    bench, learner, seed = "unknown", "unknown", 0
    for e in evidence_dir_records:
        bench = _bench(e.get("benchmark") or bench)
        cid = str(e.get("candidate_id") or "")
        if cid and learner == "unknown":
            learner = cid.split("_")[0]
        seed = int(e.get("run_seed", seed) or 0)
    runs = []
    for r in recs:
        if r.get("event") != "cell":
            continue
        run = Run(file=path, benchmark=bench, track="unassigned", learner=learner, gate=str(r.get("gate")),
                  model="unknown", seed=seed, variant=f"cond={r.get('cond')}",
                  scorecard=dict(r.get("scorecard") or {}), library=list(r.get("library") or []),
                  n_test=r.get("n_test"), baseline_unauthorized=r.get("baseline_unauthorized"), complete=True)
        run.notes.append("cell-format log (no run.start/eval.task): scorecard-only; learner inferred from "
                         "evidence candidate_id prefix; model/track unknown")
        runs.append(run)
    return runs


def load_evidence(root) -> list:
    out = []
    for p in _iter_files(root):
        if is_evidence_file(p):
            recs, _ = _read_jsonl(p)
            for r in recs:
                r = dict(r)
                r["_file"] = p
                out.append(r)
    return out


def load_runs(root) -> tuple:
    """Return (runs, issues). Issues lists unparseable lines / files with no runs."""
    runs, issues = [], []
    evidence = load_evidence(root)
    for p in _iter_files(root):
        if is_evidence_file(p):
            continue
        recs, bad = _read_jsonl(p)
        if bad:
            issues.append(f"{p}: {bad} unparseable line(s) skipped")
        if any(r.get("event") == "run.start" for r in recs):
            got = _runs_from_starts(p, recs)
        elif any(r.get("event") == "cell" for r in recs):
            d = os.path.dirname(p)
            got = _runs_from_cells(p, recs, [e for e in evidence if os.path.dirname(e["_file"]) == d])
        else:
            issues.append(f"{p}: no run.start or cell events; ignored")
            got = []
        runs.extend(got)
    return dedupe(runs, issues), issues


def dedupe(runs, issues):
    """One run per (config key, seed): keep the latest complete run (by run.start ts,
    then path); fall back to the latest incomplete one. Dropped runs are reported, so
    reruns / superseded copies are never silently averaged in as extra seeds."""
    best = {}
    for r in runs:
        k = (r.key, r.seed)
        cur = best.get(k)
        rank = (r.complete, r.ts, r.file)
        if cur is None or rank > (cur.complete, cur.ts, cur.file):
            if cur is not None:
                issues.append(f"duplicate run for {k[0]} seed {k[1]}: kept {r.file}, dropped {cur.file}")
            best[k] = r
        else:
            issues.append(f"duplicate run for {k[0]} seed {k[1]}: kept {cur.file}, dropped {r.file}")
    return [r for r in runs if best[(r.key, r.seed)] is r]


AUDIT_STATES = ("auditable", "external", "unauditable")


def audit_state(run: Run) -> tuple[str, str]:
    """(state, basis) for a run's admitted library.

    * runner ``audit`` event present: auditable if every skill is auditable,
      external if some are external and none unauditable, else unauditable;
    * else (older logs): learner stats ``external_provenance: true`` with local
      auditability < 1 -> ``external`` (unverified: no held-out check logged);
      otherwise from Scorecard.auditability (1.0 -> auditable, else unauditable).
    An empty library reads auditable (Scorecard convention), basis notes it.
    """
    a = run.audit
    if a is not None:
        if not a.get("n_skills"):
            return "auditable", "audit event (empty library)"
        if a.get("n_unauditable"):
            return "unauditable", "audit event"
        if a.get("n_external"):
            return "external", "audit event"
        return "auditable", "audit event"
    sc = run.scorecard or {}
    aud = sc.get("auditability")
    if sc.get("library_size") == 0:
        return "auditable", "scorecard (empty library)"
    if (run.induce_stats or {}).get("external_provenance") is True and (aud is None or aud < 1.0):
        return "external", "learner stats (unverified: no audit event)"
    if aud is None:
        return "unknown", "no auditability logged"
    return ("auditable" if aud >= 1.0 else "unauditable"), "scorecard auditability"
