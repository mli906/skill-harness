"""run_c1(cfg, corpus_dir): the frozen-corpus (C1) track runner.

Same 6 steps and Scorecard as harness.runner.run (unchanged), except:
  1. the learner's input traces are the FROZEN corpus (validated: counts,
     ratio, sha256, train-only, no probe/test leakage) instead of fresh
     rollouts; probe traces for the gate are still collected (empty library);
  2. a learner whose contract is C1_fixed_traces induces inside
     env_guard.forbid_env(benchmark) -- any env access raises.
C2_self_collect learners (EvoSkill) get the same corpus traces as seed input
and are not guarded (they run their own rollouts).
"""
from __future__ import annotations
import os
from harness.provenance import audit_states
import time
from harness import registry, corpus
from harness.context import Context
from harness.env_guard import forbid_env
from harness.interfaces.learner import C1_FIXED_TRACES
from harness.llm import make_client, make_embedder
from harness.logging_utils import JsonlLogger
from harness.runner import _split_probe, _eval
from harness.seeding import task_id
from harness.types import RunConfig, Scorecard, SkillLibrary


def run_c1(cfg: RunConfig, corpus_dir: str, log_path: str | None = None) -> Scorecard:
    log_path = log_path or os.path.join("logs", f"c1_{cfg.learner}_{cfg.gate}_s{cfg.seed}_{int(time.time())}.jsonl")
    logger = JsonlLogger(log_path)
    logger.log("run.start", config=cfg, corpus_dir=corpus_dir)
    ctx = Context(cfg=cfg, llm=make_client(cfg.base_model), embedder=make_embedder(cfg.base_model), logger=logger)
    ctx.benchmark = registry.build("benchmark", cfg.benchmark, ctx)
    ctx.representation = registry.build("representation", cfg.representation, ctx)
    ctx.agent = registry.build("agent", "react", ctx)
    bench = ctx.benchmark
    train, test = bench.train_tasks(), bench.test_tasks()
    ctx.task_lookup.update({task_id(t): t for t in train + test})
    learner = registry.build("learner", cfg.learner, ctx)
    gate = registry.build("gate", cfg.gate, ctx)
    agent, rep, k = ctx.agent, ctx.representation, cfg.retrieval_k
    empty = SkillLibrary(skills=[])

    learn_tasks, probe_tasks = _split_probe(bench, train, cfg.seed)
    checks = corpus.validate(corpus_dir, train, probe_tasks, test)
    recs, manifest = corpus.load(corpus_dir)
    if not checks["all_ok"]:
        raise RuntimeError(f"frozen corpus failed validation: {checks}")
    learn_traces = corpus.to_traces(recs)
    probe_traces = _eval(agent, rep, k, probe_tasks, empty)
    logger.log("split", learn_tasks=sorted({r["task"] for r in recs}), probe_tasks=probe_tasks, test_tasks=test)
    logger.log("corpus", manifest={k_: v for k_, v in manifest.items() if not k_.endswith("_ids")},
               validation=checks, contract=learner.contract)
    if learner.contract == C1_FIXED_TRACES:
        with forbid_env(bench):
            candidates = learner.induce(learn_traces, empty)
    else:
        candidates = learner.induce(learn_traces, empty)
    lib = gate.admit(candidates, empty, probe_traces)
    logger.log("gate.result", gate=cfg.gate, library=[s.id for s in lib.skills],
               rejections=gate.last_rejections, probe_tokens=gate.last_tokens)
    ev = _eval(agent, rep, k, test, lib)
    base = _eval(agent, rep, k, test, empty)
    for t_lib, t_base in zip(ev, base):
        logger.log("eval.task", task_id=t_lib.task_id, with_lib=t_lib.outcome, baseline=t_base.outcome,
                   output=t_lib.final_output, failure_mode=t_lib.failure_mode, steps=len(t_lib.steps))
    learner_ids = {t.task_id for t in learn_traces}
    heldout_ids = {task_id(t) for t in probe_tasks + test}
    audited = [s for s in lib.skills if s.provenance
               and set(s.provenance) <= learner_ids and not set(s.provenance) & heldout_ids]
    logger.log("audit", **audit_states(lib.skills, learner_ids, heldout_ids,
                                        getattr(learner, "external_provenance", None)))
    sc = Scorecard(
        success_rate=sum(t.outcome for t in ev) / len(ev),
        avg_steps=sum(len(t.steps) for t in ev) / len(ev),
        tokens_eval=sum(t.tokens for t in ev),
        tokens_build=int(learner.last_build_tokens) + int(gate.last_tokens),
        flips_w2c=sum((not b.outcome) and e.outcome for e, b in zip(ev, base)),
        flips_c2w=sum(b.outcome and (not e.outcome) for e, b in zip(ev, base)),
        library_size=len(lib.skills),
        auditability=(len(audited) / len(lib.skills)) if lib.skills else 1.0,
        unauthorized_actions=sum(s.meta.get("unauthorized", 0) for t in ev for s in t.steps),
        promotion_rejections=int(gate.last_rejections),
    )
    logger.log("run.end", scorecard=sc, baseline_success_rate=sum(t.outcome for t in base) / len(base),
               learner_build_tokens=int(learner.last_build_tokens), gate_tokens=int(gate.last_tokens))
    run_c1.last_library = lib
    return sc
