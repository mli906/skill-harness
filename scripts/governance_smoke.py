"""Governance-gate end-to-end smoke on AgentDojo-Banking (resumable).

One C1 learner (awm by default; $GOV_SMOKE_LEARNER), induced ONCE on the clean learn split; the same candidates go
through gates {noop, governance}; each library is evaluated on the clean and the attack test
split (same train/probe split in both conditions: the adapter only injects TEST tasks).
Steps mirror harness.runner.run (same _split_probe/_eval, same Scorecard fields); the only
difference is that induction + gating are shared across the 4 cells instead of re-run.
Agent = react_tools (as harness/agentdojo_run.py). LLM calls go through CachedClient, and
every stage is checkpointed to OUT/, so a rerun resumes. Aborts if the OpenRouter key usage
exceeds $GOV_SMOKE_USAGE_CAP (default 9.0) before a stage.
"""
from __future__ import annotations
import json, os, sys, urllib.request
from dataclasses import asdict
from pathlib import Path

OUT = Path(os.environ.get("GOV_SMOKE_OUT", "logs/gov_smoke"))
MODEL = os.environ.get("GOV_SMOKE_MODEL", "openrouter/qwen/qwen-2.5-72b-instruct")
LIMIT = int(os.environ.get("GOV_SMOKE_LIMIT", 8))
LEARNER = os.environ.get("GOV_SMOKE_LEARNER", "awm")
SEED, K = 0, 3
CAP = float(os.environ.get("GOV_SMOKE_USAGE_CAP", 9.0))


def usage() -> float:
    r = urllib.request.Request("https://openrouter.ai/api/v1/key",
                               headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    return float(json.load(urllib.request.urlopen(r, timeout=30))["data"]["usage"])


def guard(stage):
    import time
    for i in range(6):
        try:
            u = usage(); break
        except Exception as e:  # usage endpoint rate-limited: back off, never crash the run
            print(f"[{stage}] usage check failed ({e}); retry", flush=True); time.sleep(10 * (i + 1))
    else:
        print(f"[{stage}] usage unavailable; continuing (watchdog in live run still applies)", flush=True); return
    print(f"[{stage}] openrouter key usage ${u:.3f}", flush=True)
    if u > CAP:
        sys.exit(f"abort: key usage ${u:.2f} > cap ${CAP}")


def jdump(p, obj): p.write_text(json.dumps(obj, indent=1, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))


def main():
    os.environ["AGENTDOJO_LIMIT"] = str(LIMIT)
    os.environ.setdefault("HARNESS_EMBEDDING", "hashed")  # OpenAI credits exhausted; 1-skill libs retrieve it anyway
    os.environ.setdefault("TRACE2SKILL_REUSE", "1")
    os.environ.setdefault("GOVERNANCE_EVIDENCE_LOG", str(OUT / "governance_evidence.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    from harness import registry
    registry._load_all()
    from harness.agents.react_tools import ReActToolAgent
    registry._REG["agent"]["react"] = ReActToolAgent
    from harness.benchmarks.agentdojo_banking import AgentDojoBankingAdapter
    from harness.context import Context
    from harness.llm import make_client, make_embedder
    from harness.llm.cache import CachedClient
    from harness.logging_utils import JsonlLogger
    from harness.runner import _split_probe, _eval
    from harness.seeding import task_id
    from harness.splits import is_auditable
    from harness.types import RunConfig, Scorecard, Skill, SkillLibrary, Trace, Step

    cfg = RunConfig(LEARNER, "noop", "skill_md", "agentdojo_banking", MODEL, SEED, K)
    logger = JsonlLogger(str(OUT / "run.jsonl"))
    llm = CachedClient(make_client(MODEL))
    ctx = Context(cfg=cfg, llm=llm, embedder=make_embedder(MODEL), logger=logger)
    ctx.benchmark = AgentDojoBankingAdapter(seed=SEED, condition="clean", limit=LIMIT)
    ctx.representation = registry.build("representation", "skill_md", ctx)
    ctx.agent = ReActToolAgent(llm, ctx.benchmark, SEED, logger=logger)
    bench, agent, rep = ctx.benchmark, ctx.agent, ctx.representation
    train = bench.train_tasks()
    learn, probe = _split_probe(bench, train, SEED)
    atk_bench = AgentDojoBankingAdapter(seed=SEED, condition="attack", limit=LIMIT)
    tests = {"clean": bench.test_tasks(), "attack": atk_bench.test_tasks()}
    agents = {"clean": agent, "attack": ReActToolAgent(llm, atk_bench, SEED, logger=logger)}
    ctx.task_lookup.update({task_id(t): t for t in train + tests["clean"] + tests["attack"]})
    logger.log("split", learn_tasks=learn, probe_tasks=probe, test_clean=tests["clean"], test_attack=tests["attack"])
    jdump(OUT / "split.json", {"learn": learn, "probe": probe, **{f"test_{c}": v for c, v in tests.items()}})
    empty = SkillLibrary([])

    def traces(name, ag, tasks, lib):
        p = OUT / f"traces_{name}.json"
        if p.exists():
            return [Trace(**{**d, "steps": [Step(**s) for s in d["steps"]]}) for d in json.loads(p.read_text())]
        guard(name)
        tr = _eval(ag, rep, K, tasks, lib)
        jdump(p, tr)
        return tr

    learn_tr = traces("learn", agent, learn, empty)
    probe_tr = traces("probe", agent, probe, empty)

    cp = OUT / "candidates.json"
    if cp.exists():
        c = json.loads(cp.read_text()); cands = [Skill(**s) for s in c["skills"]]; build_tok = c["tokens"]
    else:
        guard("induce")
        learner = registry.build("learner", LEARNER, ctx)
        cands = learner.induce(learn_tr, empty); build_tok = int(learner.last_build_tokens)
        jdump(cp, {"skills": cands, "tokens": build_tok})

    libs = {}
    for g in ("noop", "governance"):
        gp = OUT / f"gate_{g}.json"
        if gp.exists():
            d = json.loads(gp.read_text()); libs[g] = (SkillLibrary([Skill(**s) for s in d["skills"]]), d["rej"], d["tok"])
            continue
        guard(f"gate_{g}")
        ctx.cfg = RunConfig(LEARNER, g, "skill_md", "agentdojo_banking", MODEL, SEED, K)
        gate = registry.build("gate", g, ctx)
        lib = gate.admit(cands, empty, probe_tr)
        libs[g] = (lib, int(gate.last_rejections), int(gate.last_tokens))
        jdump(gp, {"skills": lib.skills, "rej": libs[g][1], "tok": libs[g][2]})

    learner_ids = {t.task_id for t in learn_tr}
    results = {}
    for cond in ("clean", "attack"):
        base = traces(f"{cond}_base", agents[cond], tests[cond], empty)
        for g, (lib, rej, gtok) in libs.items():
            ev = base if not lib.skills else traces(f"{cond}_{g}", agents[cond], tests[cond], lib)
            heldout = {task_id(t) for t in probe + tests[cond]}
            aud = [s for s in lib.skills if is_auditable(s, learner_ids, heldout)]
            sc = Scorecard(
                success_rate=sum(t.outcome for t in ev) / len(ev),
                avg_steps=sum(len(t.steps) for t in ev) / len(ev),
                tokens_eval=sum(t.tokens for t in ev), tokens_build=build_tok + gtok,
                flips_w2c=sum((not b.outcome) and e.outcome for e, b in zip(ev, base)),
                flips_c2w=sum(b.outcome and (not e.outcome) for e, b in zip(ev, base)),
                library_size=len(lib.skills),
                auditability=(len(aud) / len(lib.skills)) if lib.skills else 1.0,
                unauthorized_actions=sum(s.meta.get("unauthorized", 0) for t in ev for s in t.steps),
                promotion_rejections=rej)
            results[f"{cond}/{g}"] = {"scorecard": asdict(sc), "n_test": len(ev),
                                      "baseline_success_rate": sum(t.outcome for t in base) / len(base),
                                      "baseline_unauthorized": sum(s.meta.get("unauthorized", 0) for t in base for s in t.steps),
                                      "library": [s.id for s in lib.skills]}
            logger.log("cell", cond=cond, gate=g, **results[f"{cond}/{g}"])
    jdump(OUT / "results.json", results)
    guard("done")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
