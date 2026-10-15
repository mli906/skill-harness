"""Scaled ALFWorld orchestration (run orchestration + reporting only; no interface changes).

Uses the pinned split in manifests/alfworld_scale/split.json. Subcommands:
  render                         render task texts once -> manifests/alfworld_scale/tasks.json
  corpus  <Q|M>                  frozen corpus (24 learn tasks x k=2, temp 0.7) -> corpora/scale_<m>
  prefilter <Q|M>                one empty-library rollout per practice-pool task; pick 24 (~50% pass)
  cell <name> <learner> <gate> <rep> <k> <Q|M> <c1:corpus_dir | c2:practice>
A cell mirrors harness.c1_runner.run_c1 step for step, except the probe set is the PINNED probe
(not _split_probe) and, for c2, learn traces are fresh agent rollouts on the practice set.
Every cell writes logs/scale/<name>.jsonl and logs/scale/<name>.result.json (skip if present).
"""
from __future__ import annotations
import json, os, sys, urllib.request
from dataclasses import asdict
from pathlib import Path

MAN = Path(os.environ.get("SCALE_MANIFEST", "manifests/alfworld_scale"))
OUT = Path(os.environ.get("SCALE_OUT", "logs/scale"))
MODELS = {"Q": "openrouter/qwen/qwen-2.5-72b-instruct", "M": "openrouter/openai/gpt-4o-mini"}
ROOT = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld")) / "json_2.1.1"
BUDGET = float(os.environ.get("SCALE_BUDGET_USD", 9.3))


def or_usage() -> float:
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    return float(json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]["total_usage"])


def split():
    return json.loads((MAN / "split.json").read_text())


def render():
    from harness.benchmarks.alfworld import ALFWorldAdapter
    r = ALFWorldAdapter.__new__(ALFWorldAdapter); r.max_steps = 50
    out = {}
    for name in ("test", "probe", "learn_corpus", "practice_pool"):
        out[name] = [{**e, "task": r._render(str(ROOT / e["game"]))} for e in split()[name]]
    (MAN / "tasks.json").write_text(json.dumps(out, indent=1))


def bench():
    from harness.benchmarks.alfworld import ALFWorldAdapter, TASK_TYPES
    T = json.loads((MAN / "tasks.json").read_text())
    b = ALFWorldAdapter.__new__(ALFWorldAdapter)
    b.max_steps = int(os.environ.get("ALFWORLD_MAX_EPISODE_STEPS", 50)); b.types = list(TASK_TYPES)
    b._game, b._cat, b._won, b._envs, b.train_split = {}, {}, {}, {}, "train"
    for es in T.values():
        for e in es:
            b._game[e["task"]], b._cat[e["task"]] = str(ROOT / e["game"]), e["type"]
    sets = {k: [e["task"] for e in v] for k, v in T.items()}
    b._train = sets["learn_corpus"] + sets["probe"] + sets["practice_pool"]
    b._test = sets["test"]
    return b, sets


def _setup():
    from harness import registry
    registry._load_all()
    from harness.agents.react_multistep import ReActMultiStepAgent
    registry._REG["agent"]["react"] = ReActMultiStepAgent
    return registry


def corpus_cmd(m):
    from harness import corpus
    from harness.llm.openrouter import ProviderClient
    _setup(); b, s = bench()
    mid = MODELS[m][len("openrouter/"):]
    llm = ProviderClient(MODELS[m] + "@t0.7", mid, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    llm.extra_body = {"temperature": 0.7}
    man = corpus.generate(b, s["learn_corpus"], k=2, model=MODELS[m], temperature=0.7, seed=0,
                          out_dir=f"corpora/scale_{m}", probe_tasks=s["probe"], test_tasks=s["test"], llm=llm)
    print(json.dumps({k: v for k, v in man.items() if not k.endswith("_ids")}, indent=1))


def prefilter_cmd(m):
    from harness.agents.react_multistep import ReActMultiStepAgent
    from harness.corpus import _NoSkills
    from harness.llm import make_client
    from harness.types import SkillLibrary
    _setup(); b, s = bench()
    ag = ReActMultiStepAgent(make_client(MODELS[m]), b, 0)
    T = json.loads((MAN / "tasks.json").read_text())["practice_pool"]
    res = []
    for e in T:
        tr = ag.run_task(e["task"], SkillLibrary(skills=[]), _NoSkills(), 0)
        res.append({"task_id": tr.task_id, "type": e["type"], "pass": bool(tr.outcome), "steps": len(tr.steps), "tokens": tr.tokens})
    chosen = []
    for t in sorted({e["type"] for e in T}):   # per type 4: prefer 2 pass + 2 fail
        g = [r for r in res if r["type"] == t]
        p, f = [r for r in g if r["pass"]], [r for r in g if not r["pass"]]
        pick = p[:2] + f[:2]
        rest = [r for r in g if r not in pick]
        chosen += pick + rest[:4 - len(pick)]
    out = {"model": MODELS[m], "pool_n": len(res), "pool_pass_rate": sum(r["pass"] for r in res) / len(res),
           "selected_ids": [r["task_id"] for r in chosen],
           "selected_pass_rate": sum(r["pass"] for r in chosen) / len(chosen), "rollouts": res}
    (MAN / f"practice_{m}.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("rollouts", "selected_ids")})


def cell_cmd(name, learner_name, gate_name, rep_name, k, m, src):
    out = OUT; out.mkdir(parents=True, exist_ok=True)
    resf = out / f"{name}.result.json"
    if resf.exists():
        print("skip", name); return
    used = or_usage()
    if used > BUDGET:
        raise SystemExit(f"BUDGET STOP before {name}: OpenRouter usage ${used:.3f} > ${BUDGET}")
    registry = _setup()
    from harness import corpus
    from harness.context import Context
    from harness.env_guard import forbid_env
    from harness.interfaces.learner import C1_FIXED_TRACES
    from harness.llm import make_client, make_embedder
    from harness.logging_utils import JsonlLogger
    from harness.runner import _eval
    from harness.seeding import task_id
    from harness.provenance import audit_states
    from harness.splits import bind_optimizer_split, check_frozen_origin, is_auditable
    from harness.types import RunConfig, Scorecard, SkillLibrary
    b, s = bench()
    cfg = RunConfig(learner=learner_name, gate=gate_name, representation=rep_name, benchmark="alfworld(scale split)",
                    base_model=MODELS[m], seed=0, retrieval_k=int(k))
    logger = JsonlLogger(str(out / f"{name}.jsonl"))
    logger.log("run.start", config=cfg, source=src, n_seeds=1)
    ctx = Context(cfg=cfg, llm=make_client(cfg.base_model), embedder=make_embedder(cfg.base_model), logger=logger)
    ctx.benchmark = b
    ctx.representation = registry.build("representation", rep_name, ctx)
    ctx.agent = registry.build("agent", "react", ctx)
    probe, test = s["probe"], s["test"]
    ctx.task_lookup.update({task_id(t): t for t in b._train + test})
    kind, arg = src.split(":", 1)
    if kind == "c1":
        os.environ["HARNESS_CORPUS_DIR"] = arg
        ctx.corpus_dir = arg
    learner = registry.build("learner", learner_name, ctx)
    ctx.learner = learner
    gate = registry.build("gate", gate_name, ctx)
    # invariant: frozen/transfer origin corpus disjoint from {probe, test} (hard error, before any rollout)
    origin = check_frozen_origin(learner, {task_id(t) for t in probe}, {task_id(t) for t in test}, learner_name)
    # optimizer-style learners (GEPA): seeded gepa_valset carved from the learn corpus; others unchanged
    opt_train, valset = bind_optimizer_split(b, learner, s["learn_corpus"], probe, test, cfg.seed, learner_name)
    if valset:
        vrec = {"seed": cfg.seed, "source": "learn_corpus", "per_category": int(os.environ.get("GEPA_VALSET_PER_CATEGORY", 2)),
                "gepa_valset_ids": [task_id(t) for t in valset], "gepa_train_ids": [task_id(t) for t in opt_train]}
        vp = MAN / "gepa_valset.json"
        if vp.exists():
            if json.loads(vp.read_text())["gepa_valset_ids"] != vrec["gepa_valset_ids"]:
                raise RuntimeError(f"gepa_valset carve differs from pinned {vp}")
        else:
            vp.write_text(json.dumps(vrec, indent=1))
        logger.log("gepa_valset", **vrec)
    if origin is not None:
        logger.log("frozen.origin", origin_task_ids=origin, disjoint_from_probe_test=True)
    agent, rep, kk = ctx.agent, ctx.representation, cfg.retrieval_k
    empty = SkillLibrary(skills=[])
    if kind == "c1":
        checks = corpus.validate(arg, b._train, probe, test)
        recs, manifest = corpus.load(arg)
        if not checks["all_ok"]:
            raise RuntimeError(f"corpus validation failed: {checks}")
        learn_traces = corpus.to_traces(recs)
        logger.log("corpus", manifest={a: v for a, v in manifest.items() if not a.endswith("_ids")}, validation=checks)
        learn_ids = sorted({r["task"] for r in recs})
    else:
        sel = set(json.loads((MAN / f"practice_{m}.json").read_text())["selected_ids"])
        learn_tasks = [t for t in s["practice_pool"] if task_id(t) in sel]
        learn_traces = _eval(agent, rep, kk, learn_tasks, empty)
        learn_ids = learn_tasks
    probe_traces = _eval(agent, rep, kk, probe, empty)
    if valset:
        logger.log("split", learn_tasks=learn_ids, gepa_valset=valset, probe_tasks=probe, test_tasks=test)
    else:
        logger.log("split", learn_tasks=learn_ids, probe_tasks=probe, test_tasks=test)
    logger.log("train.traces", n_learn=len(learn_traces), n_probe=len(probe_traces),
               learn_pass=sum(t.outcome for t in learn_traces), probe_pass=sum(t.outcome for t in probe_traces))
    induce_traces = learn_traces
    if type(learner).__name__ == "AWMLearner" and os.environ.get("AWM_SUCCESS_ONLY") == "1":
        # deviation: Qwen-72B provider ctx is 32,767; AWM offline induction uses successful examples only
        induce_traces = [t for t in learn_traces if t.outcome]
        logger.log("awm.success_only", n_in=len(learn_traces), n_used=len(induce_traces))
    if learner.contract == C1_FIXED_TRACES:
        with forbid_env(b):
            cands = learner.induce(induce_traces, empty)
    else:
        cands = learner.induce(learn_traces, empty)
    lib = gate.admit(cands, empty, probe_traces)
    logger.log("gate.result", gate=gate_name, n_candidates=len(cands), library=[x.id for x in lib.skills],
               rejections=gate.last_rejections, probe_tokens=gate.last_tokens)
    ev = _eval(agent, rep, kk, test, lib)
    base = _eval(agent, rep, kk, test, empty)
    for i, (a_, b_) in enumerate(zip(ev, base)):
        logger.log("eval.task", task_id=a_.task_id, category=b.category(test[i]), with_lib=a_.outcome,
                   baseline=b_.outcome, failure_mode=a_.failure_mode, steps=len(a_.steps))
    lids = {t.task_id for t in learn_traces}; held = {task_id(t) for t in probe + test}
    aud = [x for x in lib.skills if is_auditable(x, lids, held)]
    logger.log("audit", **audit_states(lib.skills, lids, held, getattr(learner, "external_provenance", None)))
    sc = Scorecard(success_rate=sum(t.outcome for t in ev) / len(ev), avg_steps=sum(len(t.steps) for t in ev) / len(ev),
                   tokens_eval=sum(t.tokens for t in ev), tokens_build=int(learner.last_build_tokens) + int(gate.last_tokens),
                   flips_w2c=sum((not y.outcome) and x.outcome for x, y in zip(ev, base)),
                   flips_c2w=sum(y.outcome and (not x.outcome) for x, y in zip(ev, base)),
                   library_size=len(lib.skills), auditability=(len(aud) / len(lib.skills)) if lib.skills else 1.0,
                   unauthorized_actions=sum(st.meta.get("unauthorized", 0) for t in ev for st in t.steps),
                   promotion_rejections=int(gate.last_rejections))
    res = {"name": name, "config": asdict(cfg), "source": src, "n_seeds": 1, "scorecard": asdict(sc),
           "baseline_success_rate": sum(t.outcome for t in base) / len(base), "n_candidates": len(cands),
           "candidate_ids": [x.id for x in cands], "library": [x.id for x in lib.skills],
           "learner_build_tokens": int(learner.last_build_tokens), "gate_tokens": int(gate.last_tokens),
           "learn_pass": f"{sum(t.outcome for t in learn_traces)}/{len(learn_traces)}",
           "probe_pass": f"{sum(t.outcome for t in probe_traces)}/{len(probe_traces)}",
           "learner_stats": getattr(learner, "last_stats", None), "openrouter_usage_at_start": used}
    logger.log("run.end", **{k_: v for k_, v in res.items() if k_ != "name"})
    resf.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["scorecard"]))


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"render": render, "corpus": corpus_cmd, "prefilter": prefilter_cmd, "cell": cell_cmd}[cmd](*a)
