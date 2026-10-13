"""Frozen trace corpus for the C1 track (Part A).

A fixed reference agent (ReActMultiStepAgent, the same ConsumingAgent used for
evaluation) with a pinned model and FIXED temperature runs K rollouts on each
ALFWorld TRAIN task (the learner share of the train tasks; the gate's probe
tasks and all test tasks are excluded). Every rollout records the full steps,
output artifacts, the env outcome label, gold info (ALFWorld expert
walkthrough + goal + task type, for analysis only, never shown to the agent),
and tokens. The corpus is serialized to <dir>/corpus.jsonl with a
<dir>/manifest.json (counts, success:failure ratio, generation params, sha256).

    python -m harness.corpus --config examples/c1_alfworld_noskill.yaml --dir corpora/alfworld_c1 --regenerate
    python -m harness.corpus --config ... --dir ... --load        # validate a frozen corpus, no LLM calls

Load-frozen (default) never calls an LLM or env: it reads and validates.
Validation: counts (= tasks x k), manifest ratio == recomputed, sha256 matches,
every task is a TRAIN task, none is a probe or test task (no leakage).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from harness.seeding import derive_seed, task_id
from harness.types import Step, Trace

CORPUS_VERSION = 1


class _NoSkills:
    def retrieve(self, task, lib, k): return []
    def inject(self, skills, prompt): return prompt


def gold_info(bench, task: str) -> dict:
    gold = {"goal": bench.goal(task), "task_type": bench.category(task)}
    try:
        gf = bench.gamefile(task)
        gold["expert_walkthrough"] = json.loads(Path(gf).read_text()).get("walkthrough")
        gold["game"] = gf.split("json_2.1.1/")[-1]
    except Exception as e:  # pragma: no cover
        gold["error"] = repr(e)
    return gold


def generate(bench, tasks: list[str], *, k: int, model: str, temperature: float, seed: int,
             out_dir: str, probe_tasks=(), test_tasks=(), llm=None) -> dict:
    from harness.agents.react_multistep import ReActMultiStepAgent
    from harness.llm.sampling import SamplingOpenAIClient
    from harness.types import SkillLibrary
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # separate LLM cache dir: corpus calls are temperature>0 and must not mix with eval cache
    os.environ.setdefault("HARNESS_LLM_CACHE", str(Path.home() / ".cache" / "skill-harness" / "llm_cache_corpus"))
    llm = llm or SamplingOpenAIClient(model, temperature)
    recs = []
    for task in tasks:
        for r in range(k):
            rseed = derive_seed(seed, "corpus-rollout", r)
            agent = ReActMultiStepAgent(llm, bench, rseed)
            t0 = time.time()
            tr = agent.run_task(task, SkillLibrary(skills=[]), _NoSkills(), 0)
            recs.append({
                "task_id": tr.task_id, "rollout": r, "rollout_seed": rseed, "task": task,
                "category": bench.category(task), "outcome": bool(tr.outcome),
                "failure_mode": tr.failure_mode, "final_output": tr.final_output, "tokens": tr.tokens,
                "steps": [asdict(s) for s in tr.steps],
                "artifacts": {"final_output": tr.final_output, "actions": [s.action for s in tr.steps],
                              "final_observation": tr.steps[-1].tool_result if tr.steps else None},
                "gold": gold_info(bench, task), "wall_s": round(time.time() - t0, 1)})
    body = "".join(json.dumps(x, sort_keys=True) + "\n" for x in recs)
    (out / "corpus.jsonl").write_text(body)
    n_s = sum(x["outcome"] for x in recs)
    manifest = {
        "corpus_version": CORPUS_VERSION, "benchmark": type(bench).__name__,
        "train_split": getattr(bench, "train_split", None),
        "reference_agent": "react_multistep (ReActMultiStepAgent)", "model": model,
        "temperature": temperature, "k": k, "seed": seed,
        "n_tasks": len(tasks), "n_traces": len(recs), "n_success": n_s, "n_failure": len(recs) - n_s,
        "success_failure_ratio": f"{n_s}:{len(recs) - n_s}",
        "tokens_total": sum(x["tokens"] for x in recs),
        "task_ids": sorted({x["task_id"] for x in recs}),
        "excluded_probe_ids": sorted(task_id(t) for t in probe_tasks),
        "excluded_test_ids": sorted(task_id(t) for t in test_tasks),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load(corpus_dir: str) -> tuple[list[dict], dict]:
    d = Path(corpus_dir)
    body = (d / "corpus.jsonl").read_text()
    return [json.loads(l) for l in body.splitlines() if l.strip()], json.loads((d / "manifest.json").read_text())


def to_traces(recs: list[dict]) -> list[Trace]:
    return [Trace(task_id=r["task_id"], steps=[Step(**s) for s in r["steps"]], outcome=r["outcome"],
                  failure_mode=r["failure_mode"], final_output=r["final_output"], tokens=r["tokens"])
            for r in recs]


def validate(corpus_dir: str, train_tasks=None, probe_tasks=(), test_tasks=()) -> dict:
    recs, m = load(corpus_dir)
    body = (Path(corpus_dir) / "corpus.jsonl").read_text()
    ids = {r["task_id"] for r in recs}
    train_ids = {task_id(t) for t in train_tasks} if train_tasks is not None else None
    probe_ids = {task_id(t) for t in probe_tasks} | set(m.get("excluded_probe_ids", []))
    test_ids = {task_id(t) for t in test_tasks} | set(m.get("excluded_test_ids", []))
    n_s = sum(r["outcome"] for r in recs)
    per_task = {}
    for r in recs:
        per_task[r["task_id"]] = per_task.get(r["task_id"], 0) + 1
    checks = {
        "sha256_ok": hashlib.sha256(body.encode()).hexdigest() == m["sha256"],
        "count_ok": len(recs) == m["n_traces"] == m["n_tasks"] * m["k"],
        "k_per_task_ok": all(v == m["k"] for v in per_task.values()) and len(per_task) == m["n_tasks"],
        "ratio_ok": m["success_failure_ratio"] == f"{n_s}:{len(recs) - n_s}",
        "has_success_and_failure": 0 < n_s < len(recs),
        "fields_ok": all({"steps", "artifacts", "gold", "outcome", "tokens"} <= set(r) and r["steps"] for r in recs),
        "train_only_ok": train_ids is None or ids <= train_ids,
        "no_probe_leak": not (ids & probe_ids),
        "no_test_leak": not (ids & test_ids),
    }
    checks["all_ok"] = all(v for k, v in checks.items() if k != "has_success_and_failure")
    return checks


def _bench_and_split(config: str):
    from harness import registry
    from harness.context import Context
    from harness.run import load_config
    from harness.runner import _split_probe
    from harness.llm.stub import StubClient  # never called: only builds the benchmark
    cfg, _ = load_config(config)
    ctx = Context(cfg=cfg, llm=StubClient("stub"), embedder=None)
    bench = registry.build("benchmark", cfg.benchmark, ctx)
    learn, probe = _split_probe(bench, bench.train_tasks(), cfg.seed)
    return cfg, bench, learn, probe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--regenerate", action="store_true")
    g.add_argument("--load", action="store_true", help="(default) load the frozen corpus and validate")
    ap.add_argument("--k", type=int, default=int(os.environ.get("CORPUS_K", 2)))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("CORPUS_TEMPERATURE", 0.7)))
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    cfg, bench, learn, probe = _bench_and_split(a.config)
    test = bench.test_tasks()
    if a.regenerate:
        m = generate(bench, learn, k=a.k, model=a.model or cfg.base_model, temperature=a.temperature,
                     seed=cfg.seed, out_dir=a.dir, probe_tasks=probe, test_tasks=test)
        print(json.dumps({k: v for k, v in m.items() if k not in ("task_ids", "excluded_probe_ids", "excluded_test_ids")}, indent=2))
    elif not (Path(a.dir) / "manifest.json").exists():
        raise SystemExit(f"no frozen corpus at {a.dir}; pass --regenerate")
    print(json.dumps({"validation": validate(a.dir, bench.train_tasks(), probe, test)}, indent=2))


if __name__ == "__main__":
    main()
