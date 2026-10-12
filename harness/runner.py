"""run(cfg) -> Scorecard: the only thing a run invokes."""
from __future__ import annotations
import os
import time
from collections import defaultdict
from harness import registry
from harness.context import Context
from harness.llm import make_client, make_embedder
from harness.logging_utils import JsonlLogger
from harness.seeding import rng, task_id
from harness.types import RunConfig, Scorecard, SkillLibrary, Trace

PROBE_PER_CATEGORY = 2


def _split_probe(bench, tasks: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Seeded, category-stratified hold-out of the train tasks for the gate's probe."""
    cat = getattr(bench, "category", None)
    groups = defaultdict(list)
    for t in tasks:
        groups[cat(t) if callable(cat) else "_all"].append(t)
    learn, probe = [], []
    for c in sorted(groups):
        g = list(groups[c])
        rng(seed, "probe-split", c).shuffle(g)
        n = PROBE_PER_CATEGORY if callable(cat) else max(1, len(g) * 3 // 10)
        probe += g[:n]; learn += g[n:]
    return learn, probe


def _eval(agent, rep, k, tasks, lib) -> list[Trace]:
    return [agent.run_task(t, lib, rep, k) for t in tasks]


def run(cfg: RunConfig, log_path: str | None = None) -> Scorecard:
    if log_path is None:
        d = os.environ.get("HARNESS_LOG_DIR", "logs")
        log_path = os.path.join(d, f"run_{cfg.learner}_{cfg.gate}_s{cfg.seed}_{int(time.time()*1000)}.jsonl")
    logger = JsonlLogger(log_path)
    logger.log("run.start", config=cfg)
    ctx = Context(cfg=cfg, llm=make_client(cfg.base_model), embedder=make_embedder(cfg.base_model), logger=logger)
    ctx.benchmark = registry.build("benchmark", cfg.benchmark, ctx)
    ctx.representation = registry.build("representation", cfg.representation, ctx)
    ctx.agent = registry.build("agent", "react", ctx)  # fixed consuming agent
    train, test = ctx.benchmark.train_tasks(), ctx.benchmark.test_tasks()
    ctx.task_lookup.update({task_id(t): t for t in train + test})
    learner = registry.build("learner", cfg.learner, ctx)
    gate = registry.build("gate", cfg.gate, ctx)
    agent, rep, k = ctx.agent, ctx.representation, cfg.retrieval_k
    empty = SkillLibrary(skills=[])

    # 1. collect training traces with empty library
    learn_tasks, probe_tasks = _split_probe(ctx.benchmark, train, cfg.seed)
    learn_traces = _eval(agent, rep, k, learn_tasks, empty)
    probe_traces = _eval(agent, rep, k, probe_tasks, empty)
    logger.log("split", learn_tasks=learn_tasks, probe_tasks=probe_tasks, test_tasks=test)
    logger.log("train.traces", n_learn=len(learn_traces), n_probe=len(probe_traces),
               learn_pass=sum(t.outcome for t in learn_traces), probe_pass=sum(t.outcome for t in probe_traces))
    # 2. learner.induce(traces) -> candidates
    candidates = learner.induce(learn_traces, empty)
    # 3. gate.admit(candidates, lib, probe) -> library
    lib = gate.admit(candidates, empty, probe_traces)
    logger.log("gate.result", gate=cfg.gate, library=[s.id for s in lib.skills],
               rejections=gate.last_rejections, probe_tokens=gate.last_tokens)
    # 4. evaluate on test_tasks with library injected
    ev = _eval(agent, rep, k, test, lib)
    # 5. no-skill baseline, same seed
    base = _eval(agent, rep, k, test, empty)
    for t_lib, t_base in zip(ev, base):
        logger.log("eval.task", task_id=t_lib.task_id, with_lib=t_lib.outcome, baseline=t_base.outcome,
                   output=t_lib.final_output, failure_mode=t_lib.failure_mode, steps=len(t_lib.steps))
    # 6. assemble Scorecard
    # Auditable = non-empty provenance, every id is a train trace the learner actually
    # saw, and none leak from the held-out probe or the test set.
    learner_ids = {t.task_id for t in learn_traces}
    heldout_ids = {task_id(t) for t in probe_tasks + test}
    audited = [s for s in lib.skills if s.provenance
               and set(s.provenance) <= learner_ids and not set(s.provenance) & heldout_ids]
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
    logger.log("run.end", scorecard=sc, baseline_success_rate=sum(t.outcome for t in base) / len(base))
    run.last_library = lib  # convenience for tests/inspection
    return sc
