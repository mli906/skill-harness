"""GO/NO-GO self-consistency check: EvoSkill standalone vs harness-wrapped.

Standalone: build an EvoSkill project directly from the OfficeQA subset's
training rows and run `evoskill run` (EvoSkill CLI, no harness types).
Wrapped: produce harness Traces for the same rows (empty library, same seed)
and call EvoSkillLearner.induce(). Compare rows in, native val count,
skills out, and skill structure.

python scripts/validate_evoskill.py --limit 10 --seed 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default="logs/evoskill_validation.json")
    a = ap.parse_args()
    os.environ["OFFICEQA_LIMIT"] = str(a.limit)
    from harness.benchmarks.officeqa import OfficeQAAdapter
    from harness.learners import evoskill as E
    from harness.agents.react import ReActAgent
    from harness.representations.skill_md import SkillMdRepresentation
    from harness.llm import make_client, make_embedder
    from harness.seeding import task_id
    from harness.types import SkillLibrary

    bench = OfficeQAAdapter()
    from harness.runner import _split_probe
    train, _probe = _split_probe(bench, bench.train_tasks(), a.seed)  # same learner split as run()
    lookup = {task_id(t): t for t in train}
    learner = E.EvoSkillLearner(a.model, bench, lookup, a.seed,
                                workdir="logs/evoskill_validation")
    # standalone
    rows = [{"uid": task_id(t), "question": bench.question(t), "answer": bench.answer(t),
             "category": bench.category(t)} for t in train]
    root = Path(f"logs/evoskill_validation/standalone_{int(time.time())}").resolve()
    E.write_project(root, rows, [str(Path(bench.corpus_dir).resolve())],
                    os.environ.get("EVOSKILL_MODEL", f"openai/{a.model}"),
                    os.environ.get("EVOSKILL_HARNESS", "openhands"))
    rc = E.run_evoskill(root, root / "evoskill_stdout.log")
    log = (root / "evoskill_stdout.log").read_text(errors="ignore")
    prog, sa_sk = E.export_skills(root) if rc == 0 else (None, {})
    standalone = {"rc": rc, "rows_in": len(rows), **E.native_split_counts(log), "best_program": prog,
                  "skills_out": len(sa_sk), "skill_names": sorted(sa_sk),
                  "skill_summaries": {k: v[:300] for k, v in sa_sk.items()}, "tokens": E.cache_tokens(root)}
    # wrapped
    agent = ReActAgent(make_client(a.model), bench, a.seed)
    rep = SkillMdRepresentation(make_embedder(a.model))
    traces = [agent.run_task(t, SkillLibrary([]), rep, 0) for t in train]
    skills = learner.induce(traces, SkillLibrary([]))
    wrapped = {**learner.last_stats, "tokens": learner.last_build_tokens,
               "skill_ids": [s.id for s in skills],
               "skill_summaries": {s.id: s.content[:300] for s in skills},
               "provenance_ok": all(set(s.provenance) == set(lookup) for s in skills),
               "all_skill_md": all(s.content.lstrip().startswith("---") for s in skills)}
    same_rows = standalone["rows_in"] == wrapped["rows_written"] == wrapped["traces_in"]
    same_val = standalone["val"] == wrapped["val"]
    both_ok = standalone["rc"] == 0
    go = both_ok and same_rows and same_val and wrapped["provenance_ok"] and \
        (standalone["skills_out"] > 0) == (wrapped["skills_out"] > 0)
    report = {"standalone": standalone, "wrapped": wrapped,
              "checks": {"same_rows": same_rows, "same_native_val": same_val, "standalone_ok": both_ok},
              "verdict": "GO" if go else "NO-GO"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
