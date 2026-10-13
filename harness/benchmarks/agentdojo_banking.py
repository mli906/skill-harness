"""AgentDojoBankingAdapter (`benchmark: agentdojo_banking`): AgentDojo v1 "banking"
suite (agentdojo==0.1.35, pinned), driven by the harness's OWN ConsumingAgent
(Option A). AgentDojo's AgentPipeline is NOT used for evaluation; its
GroundTruthPipeline is used only offline to find which injection vectors a user
task actually exposes (the same procedure AgentDojo's attacks use).

Tool binding (dependency injection)
-----------------------------------
`suite.tools` are agentdojo `Function`s whose env arguments are declared with
`Depends(...)`. We bind them with agentdojo's own `FunctionsRuntime(suite.tools)`
-- the runtime the native loop uses -- and call
`runtime.run_function(env, name, kwargs)`, which resolves every `Depends` field
from the CURRENT episode env and validates kwargs against the tool's pydantic
schema. Results are rendered with agentdojo's `tool_result_to_str` (native YAML
formatting); errors come back as "Error: ..." strings like the native loop.

Episode / env threading (duck-typed env API, like ALFWorldAdapter)
------------------------------------------------------------------
  env_reset(task)          -> (prompt, tool_specs). Loads the default env with the
                              condition's injections (suite.load_and_inject_default_environment),
                              user_task.init_environment(env), then pre_env = deep copy.
  call_tool(task, name, args) -> result string; appended to the episode's FunctionCall trace.
  finish(task, output)     -> post_env = deep copy of the env after acting.
  verify(task, output)     -> utility on the captured (pre_env, post_env), exactly like
                              TaskSuite._check_user_task_utility (utility_from_traces first,
                              then user_task.utility(output, pre_env, post_env)).
  security(task)           -> None (clean) or injection_task.security(output, pre_env, post_env)
                              (security_from_traces first); True = the injection SUCCEEDED.
The runner never sees an env: everything is per-task state inside the adapter, so
harness.runner.run() is unchanged (the agent slot is pointed at `react_tools` by
harness/agentdojo_run.py, like harness/alfworld_run.py).

Conditions ($AGENTDOJO_CONDITION or ctor `condition`)
-----------------------------------------------------
  clean  : no injections; success_rate = utility.
  attack : each TEST user task is paired with $AGENTDOJO_INJECTIONS_PER_TASK (default 1)
           injection task(s), chosen seeded (rng(seed,"agentdojo-inj",user_task_id)) from all
           9 banking injection tasks (all are applicable to every banking user task in
           AgentDojo's own benchmark matrix), or forced via `injection_task`. Injection text =
           AgentDojo's "important_instructions" template (user "Emma Johnson", model
           DEFAULT_MODEL_NAME) placed into the default placement: the vectors returned by
           FixedJailbreakAttack.get_injection_candidates(user_task) (vectors seen when the
           task is solved by its ground truth). Train/probe tasks stay clean.
           The agent marks Step.meta["unauthorized"]=1 on an episode whose security() is
           True, so Scorecard.unauthorized_actions = number of successful injections.

Split: seeded, category-stratified (rng(seed,"agentdojo-split",difficulty)), disjoint;
$AGENTDOJO_TRAIN_FRAC (0.5) train; $AGENTDOJO_LIMIT caps the number of user tasks.
category(task) = difficulty label (in v1 banking all 16 user tasks are EASY).
"""
from __future__ import annotations
import os
import re
from harness.interfaces.benchmark import Benchmark
from harness.registry import register
from harness.seeding import rng

AGENTDOJO_VERSION = "0.1.35"
BENCHMARK_VERSION, SUITE = "v1", "banking"
ATTACK_USER = "Emma Johnson"


def load_suite():
    from importlib.metadata import version
    if version("agentdojo") != AGENTDOJO_VERSION:
        raise RuntimeError(f"agentdojo=={AGENTDOJO_VERSION} required, found {version('agentdojo')}")
    from agentdojo.task_suite.load_suites import get_suite
    return get_suite(BENCHMARK_VERSION, SUITE)


def tool_specs(suite) -> list[dict]:
    """name, signature, docstring for each tool (the agent-facing tool surface)."""
    out = []
    for f in suite.tools:
        props = f.parameters.model_json_schema().get("properties", {})
        req = set(f.parameters.model_json_schema().get("required", []))
        sig = ", ".join(f"{k}: {v.get('type', 'any')}{'' if k in req else ' = optional'}" for k, v in props.items())
        args = "; ".join(f"{k}: {v.get('description', '')}" for k, v in props.items())
        out.append({"name": f.name, "signature": f"{f.name}({sig})",
                    "description": f.description.strip() + (f" Args: {args}" if args else "")})
    return out


@register("benchmark", "agentdojo_banking")
class AgentDojoBankingAdapter(Benchmark):
    def __init__(self, seed: int = 0, condition: str | None = None, limit: int | None = None,
                 injection_task: str | None = None, train_frac: float | None = None):
        self.suite = load_suite()
        self.seed = seed
        self.condition = condition or os.environ.get("AGENTDOJO_CONDITION", "clean")
        if self.condition not in ("clean", "attack"):
            raise ValueError("condition must be 'clean' or 'attack'")
        lim = limit if limit is not None else int(os.environ.get("AGENTDOJO_LIMIT", 0) or 0)
        frac = float(train_frac if train_frac is not None else os.environ.get("AGENTDOJO_TRAIN_FRAC", 0.5))
        self.forced_injection = injection_task or os.environ.get("AGENTDOJO_INJECTION_TASK") or None
        self.inj_per_task = int(os.environ.get("AGENTDOJO_INJECTIONS_PER_TASK", 1))
        uids = sorted(self.suite.user_tasks, key=lambda u: int(u.rsplit("_", 1)[1]))
        if lim:
            uids = uids[:lim]
        groups: dict[str, list[str]] = {}
        for u in uids:
            groups.setdefault(self.suite.get_user_task_by_id(u).DIFFICULTY.name, []).append(u)
        self._train, self._test = [], []
        self._meta: dict[str, dict] = {}
        for cat in sorted(groups):
            g = list(groups[cat]); rng(seed, "agentdojo-split", cat).shuffle(g)
            n_tr = int(round(len(g) * frac))
            for u in g[:n_tr]:
                self._train.append(self._task(u, None))
            for u in g[n_tr:]:
                for inj in (self._injections_for(u) if self.condition == "attack" else [None]):
                    self._test.append(self._task(u, inj))
        self._ep: dict[str, dict] = {}
        self._attack = None

    @classmethod
    def from_context(cls, ctx):
        return cls(seed=ctx.cfg.seed)

    # ---- task strings -----------------------------------------------------------------------
    def _task(self, uid: str, inj: str | None) -> str:
        ut = self.suite.get_user_task_by_id(uid)
        t = f"AgentDojo banking {uid}" + (f" [{inj}]" if inj else "") + f"\n{ut.PROMPT}"
        self._meta[t] = {"user_task": uid, "injection_task": inj, "difficulty": ut.DIFFICULTY.name}
        return t

    def _injections_for(self, uid: str) -> list[str]:
        if self.forced_injection:
            return [self.forced_injection]
        ids = sorted(self.suite.injection_tasks, key=lambda u: int(u.rsplit("_", 1)[1]))
        r = rng(self.seed, "agentdojo-inj", uid)
        return sorted(r.sample(ids, min(self.inj_per_task, len(ids))), key=lambda u: int(u.rsplit("_", 1)[1]))

    def injections(self, task: str) -> dict[str, str]:
        """AgentDojo's default placement for this (user task, injection task); {} if clean."""
        m = self._meta[task]
        if not m["injection_task"]:
            return {}
        if self._attack is None:
            from agentdojo.attacks.base_attacks import FixedJailbreakAttack, DEFAULT_MODEL_NAME
            from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
            self._attack = FixedJailbreakAttack(ImportantInstructionsAttack._JB_STRING, self.suite, None)
            self._attack.user_name, self._attack.model_name = ATTACK_USER, DEFAULT_MODEL_NAME
        return self._attack.attack(self.suite.get_user_task_by_id(m["user_task"]),
                                   self.suite.get_injection_task_by_id(m["injection_task"]))

    # ---- Benchmark --------------------------------------------------------------------------
    def train_tasks(self) -> list[str]:
        return list(self._train)

    def test_tasks(self) -> list[str]:
        return list(self._test)

    def category(self, task: str) -> str:
        return self._meta[task]["difficulty"]

    def prompt(self, task: str) -> str:
        """What the agent sees: the user's request only (no task/injection ids)."""
        return self.suite.get_user_task_by_id(self._meta[task]["user_task"]).PROMPT

    def user_task_id(self, task: str) -> str:
        return self._meta[task]["user_task"]

    def injection_task_id(self, task: str) -> str | None:
        return self._meta[task]["injection_task"]

    def attack_variant(self, task: str) -> str:
        """The ATTACK-condition version of a (clean) task: same user task, paired with the seeded
        injection task(s) the attack condition would draw (first one), placed by AgentDojo's
        default placement. Registers it so env_reset/security work. Used by the governance gate
        to replay the probe under attack; does not change train/test splits."""
        uid = self._meta[task]["user_task"]
        return self._task(uid, self._injections_for(uid)[0])

    def tool_specs(self) -> list[dict]:
        return tool_specs(self.suite)

    # ---- episode env API ----------------------------------------------------------------------
    def env_reset(self, task: str):
        from agentdojo.functions_runtime import FunctionsRuntime
        ut = self.suite.get_user_task_by_id(self._meta[task]["user_task"])
        env = self.suite.load_and_inject_default_environment(self.injections(task))
        env = ut.init_environment(env)
        self._ep[task] = {"env": env, "pre": env.model_copy(deep=True), "post": None,
                          "runtime": FunctionsRuntime(self.suite.tools), "calls": [], "output": None}
        return self.prompt(task), self.tool_specs()

    def call_tool(self, task: str, name: str, args: dict) -> str:
        from agentdojo.functions_runtime import FunctionCall
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
        ep = self._ep[task]
        ep["calls"].append(FunctionCall(function=name, args=dict(args)))
        result, err = ep["runtime"].run_function(ep["env"], name, args)
        return f"Error: {err}" if err else tool_result_to_str(result)

    def finish(self, task: str, output: str) -> None:
        ep = self._ep[task]
        ep["post"], ep["output"] = ep["env"].model_copy(deep=True), output

    def _envs(self, task: str, output: str):
        ep = self._ep.get(task)
        if ep is None:
            raise RuntimeError("verify() before an episode was played")
        if ep["post"] is None:
            self.finish(task, output)
        return ep

    def verify(self, task: str, output: str) -> bool:
        ep = self._envs(task, output)
        ut = self.suite.get_user_task_by_id(self._meta[task]["user_task"])
        r = ut.utility_from_traces(output, ep["pre"], ep["post"], ep["calls"])
        return bool(r if r is not None else ut.utility(output, ep["pre"], ep["post"]))

    def security(self, task: str, output: str | None = None) -> bool | None:
        inj = self._meta[task]["injection_task"]
        if not inj:
            return None
        ep = self._envs(task, output if output is not None else (self._ep[task]["output"] or ""))
        out = ep["output"] or ""
        it = self.suite.get_injection_task_by_id(inj)
        r = it.security_from_traces(out, ep["pre"], ep["post"], ep["calls"])
        return bool(r if r is not None else it.security(out, ep["pre"], ep["post"]))
