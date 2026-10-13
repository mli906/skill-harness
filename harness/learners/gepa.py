"""GEPALearner (`learner: gepa`, contract C2_self_collect).

GEPA via gepa==0.1.4 (optimize + GEPAAdapter). Evolutionary reflective-mutation + Pareto
selection; adapter wires evaluation to the harness consuming agent. Contract C2 (self-collects
by running candidates); seeds from a trivial draft, same as AWM.

Wrapping notes (verified against installed gepa==0.1.4 source):
  * gepa.optimize asserts task_lm is None when a custom adapter is passed, so task_lm is NOT
    passed; the "task LM" is the harness consuming agent (which already uses ctx.llm).
  * reflection_lm is a LanguageModel protocol: __call__(prompt: str | list[dict]) -> str.
    _CountingReflectionLM adapts ctx.llm.complete (seeded, temperature 0) and counts tokens.
  * EvaluationBatch(outputs, scores, trajectories, objective_scores=None, num_metric_calls=None).
  * the engine dereferences adapter.propose_new_texts without hasattr, so the adapter
    sets it to None (selects GEPA's default reflective proposer).
  * make_reflective_dataset -> {component: [ {"Inputs", "Generated Outputs", "Feedback"} ]}
    (the record keys GEPA's default instruction proposer renders; "Feedback" carries the
    failure summary).
Build tokens = reflection-LM tokens + consuming-agent tokens (Trace.tokens) of every rollout
GEPA triggers, so all model calls made during induction are counted once.
trainset = learn split, valset = probe split (runner._split_probe, same seed); the test set is
never handed to GEPA. NOTE: the probe split is also what the gates use -> GEPA optimises on
the gate's probe (leakage into gate decisions); flagged, not fixed, per spec.
"""
from __future__ import annotations
from typing import Any
from harness.interfaces.learner import Learner, C2_SELF_COLLECT
from harness.registry import register
from harness.seeding import derive_seed, task_id
from harness.types import Trace, Skill, SkillLibrary

COMPONENT = "skill_md"
DEFAULT_MAX_METRIC_CALLS = 50
OBS_CHARS = 120
MAX_STEPS_SHOWN = 30

# Trivial draft in the AWM workflow style; GEPA evolves it.
SEED_SKILL = """## solve_household_task
Given an ALFWorld household task, this workflow finds the object and completes the goal.
go to {receptacle}
take {object} from {receptacle}
go to {target-receptacle}
put {object} in/on {target-receptacle}"""


class _CountingReflectionLM:
    """gepa LanguageModel protocol over ctx.llm (seeded); accumulates tokens."""
    def __init__(self, llm, seed: int):
        self.llm, self.seed, self.tokens, self.calls = llm, seed, 0, 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):
            prompt = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt)
        resp = self.llm.complete(prompt, seed=derive_seed(self.seed, "gepa-reflect", self.calls), max_tokens=2048)
        self.calls += 1
        self.tokens += resp.total_tokens
        return resp.text


def _summarize(trace: Trace, task: str) -> str:
    lines = [f"Task: {task}", f"Outcome: {'SUCCESS' if trace.outcome else 'FAILURE'}"
             + (f" (failure_mode={trace.failure_mode})" if trace.failure_mode else "")]
    for s in trace.steps[:MAX_STEPS_SHOWN]:
        obs = (s.tool_result or "").replace("\n", " ").strip()[:OBS_CHARS]
        lines.append(f"{s.action} -> {obs}" if obs else s.action)
    if len(trace.steps) > MAX_STEPS_SHOWN:
        lines.append(f"... ({len(trace.steps) - MAX_STEPS_SHOWN} more steps)")
    return "\n".join(lines)


class HarnessGEPAAdapter:
    """GEPAAdapter: DataInst = {"id", "text"} ALFWorld task; candidate = {"skill_md": ...}."""
    # gepa 0.1.4 reads adapter.propose_new_texts unguarded; None -> use the default proposer.
    propose_new_texts = None
    def __init__(self, agent, representation, benchmark, k: int):
        self.agent, self.rep, self.bench, self.k = agent, representation, benchmark, k
        self.agent_tokens = 0
        self.rollouts = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa import EvaluationBatch  # lazy: registry auto-imports learners
        lib = SkillLibrary(skills=[Skill(id="gepa_candidate", content=candidate[COMPONENT],
                                         representation="skill_md", provenance=[], version=0)])
        outputs, scores, trajs = [], [], []
        for inst in batch:
            tr = self.agent.run_task(inst["text"], lib, self.rep, self.k)
            self.agent_tokens += int(tr.tokens); self.rollouts += 1
            ok = bool(self.bench.verify(inst["text"], tr.final_output))
            outputs.append(tr.final_output); scores.append(1.0 if ok else 0.0)
            trajs.append({"task": inst, "trace": tr, "verified": ok})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        recs = []
        for tj, score in zip(eval_batch.trajectories or [], eval_batch.scores):
            tr = tj["trace"]
            fb = ("Task solved with this skill." if score >= 1.0 else
                  f"Task FAILED (verifier rejected final output {tr.final_output!r}"
                  + (f", failure_mode={tr.failure_mode}" if tr.failure_mode else "")
                  + "). Revise the skill so the agent avoids the mistakes in the trajectory.")
            recs.append({"Inputs": tj["task"]["text"],
                         "Generated Outputs": _summarize(tr, tj["task"]["text"]),
                         "Feedback": fb})
        return {c: recs for c in components_to_update}


@register("learner", "gepa")
class GEPALearner(Learner):
    contract = C2_SELF_COLLECT

    def __init__(self, llm, agent, representation, benchmark, seed: int, k: int = 1,
                 train_tasks: list[str] | None = None, val_tasks: list[str] | None = None,
                 max_metric_calls: int = DEFAULT_MAX_METRIC_CALLS, logger=None):
        self.llm, self.agent, self.rep, self.bench = llm, agent, representation, benchmark
        self.seed, self.k, self.logger = seed, k, logger
        self.train_tasks, self.val_tasks = train_tasks, val_tasks
        self.max_metric_calls = max_metric_calls
        self.last_build_tokens = 0
        self.last_result = None

    @classmethod
    def from_context(cls, ctx):
        import os
        from harness.runner import _split_probe
        learn, probe = _split_probe(ctx.benchmark, ctx.benchmark.train_tasks(), ctx.cfg.seed)
        budget = int(os.environ.get("GEPA_MAX_METRIC_CALLS", DEFAULT_MAX_METRIC_CALLS))
        return cls(ctx.llm, ctx.agent, ctx.representation, ctx.benchmark, ctx.cfg.seed,
                   ctx.cfg.retrieval_k, learn, probe, budget, ctx.logger)

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        import gepa  # lazy: registry auto-imports learners
        # C2: `traces` are ignored for collection; GEPA runs its own rollouts.
        mk = lambda ts: [{"id": task_id(t), "text": t} for t in ts]
        trainset, valset = mk(self.train_tasks), mk(self.val_tasks)
        adapter = HarnessGEPAAdapter(self.agent, self.rep, self.bench, self.k)
        rlm = _CountingReflectionLM(self.llm, self.seed)
        res = gepa.optimize(seed_candidate={COMPONENT: SEED_SKILL}, trainset=trainset, valset=valset,
                            adapter=adapter, reflection_lm=rlm, candidate_selection_strategy="pareto",
                            max_metric_calls=self.max_metric_calls, seed=self.seed,
                            use_wandb=False, use_mlflow=False, display_progress_bar=False)
        self.last_result = res
        self.last_build_tokens = rlm.tokens + adapter.agent_tokens
        v = 1 + max((s.version for s in current.skills), default=0)
        skill = Skill(id=f"gepa_skill_v{v}", content=res.best_candidate[COMPONENT], representation="skill_md",
                      provenance=sorted({d["id"] for d in valset}), version=v)
        if self.logger:
            self.logger.log("learner.induce", learner="gepa", tokens=self.last_build_tokens,
                            reflection_tokens=rlm.tokens, agent_tokens=adapter.agent_tokens,
                            rollouts=adapter.rollouts, n_candidates=len(res.candidates),
                            best_val=res.val_aggregate_scores[res.best_idx], skill_id=skill.id,
                            content=skill.content)
        return [skill]
