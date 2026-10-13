"""ReActMultiStepAgent (`agent: react_multistep`): fixed ReAct executor for
interactive benchmarks (duck-typed env_reset/env_step, e.g. ALFWorld).

The M0 ReActAgent only exposes a calculator tool and cannot act in an
environment (it ends after ~1-2 steps), so interactive benchmarks use this
agent. For benchmarks without an env API it delegates to the M0 ReActAgent.

Per turn the model sees: injected skills, the task, the last
ALFWORLD_HISTORY (default 12) observation/action pairs, and the current
admissible commands; it replies `Thought: ...` + `Action: <command>`.
Each env action is one Step. `meta.unauthorized=1` when the action is not in
the env's admissible-command allowlist. Max ALFWORLD_AGENT_MAX_STEPS (30).
LLM calls go through harness.llm.cache.CachedClient (temp 0, fixed seed).
"""
from __future__ import annotations
import os
import re
import time
from harness.agents.react import ReActAgent
from harness.interfaces.agent import ConsumingAgent
from harness.interfaces.representation import Representation
from harness.llm.cache import CachedClient
from harness.registry import register
from harness.seeding import derive_seed, task_id
from harness.types import Step, Trace, SkillLibrary

SYSTEM = """You are an agent acting in a household text environment (ALFWorld).
Achieve the task by issuing one command per turn. Useful commands: go to <recep>,
open <recep>, close <recep>, take <obj> from <recep>, put <obj> in/on <recep>,
clean <obj> with <recep>, heat <obj> with <recep>, cool <obj> with <recep>,
use <lamp>, examine <x>, look, inventory. Objects may be inside closed receptacles.
Reply exactly in this format:
Thought: <brief reasoning>
Action: <one command copied from the admissible commands>"""


@register("agent", "react_multistep")
class ReActMultiStepAgent(ConsumingAgent):
    def __init__(self, llm, benchmark, seed: int, max_steps: int | None = None, history: int | None = None,
                 logger=None):
        self.llm = CachedClient(llm)
        self.bench, self.seed, self.logger = benchmark, seed, logger
        self.max_steps = max_steps or int(os.environ.get("ALFWORLD_AGENT_MAX_STEPS", 30))
        self.history = history or int(os.environ.get("ALFWORLD_HISTORY", 12))
        self._fallback = ReActAgent(llm, benchmark, seed, logger=logger)

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.llm, ctx.benchmark, ctx.cfg.seed, logger=ctx.logger)

    def run_task(self, task: str, lib: SkillLibrary, rep: Representation, k: int) -> Trace:
        if not hasattr(self.bench, "env_reset"):
            return self._fallback.run_task(task, lib, rep, k)
        skills = rep.retrieve(task, lib, k)
        obs, adm = self.bench.env_reset(task)
        steps, hist, tokens, done = [], [], 0, False
        for i in range(self.max_steps):
            h = "".join(f"> {a}\n{o}\n" for a, o in hist[-self.history:])
            body = (f"{SYSTEM}\n\nTask: {task}\n\nRecent history:\n{h or '(none)'}\n"
                    f"Current observation: {obs}\nAdmissible commands: {', '.join(adm)}\n")
            prompt = rep.inject(skills, body)
            t0 = time.time()
            r = self.llm.complete(prompt, seed=derive_seed(self.seed, "react_ms", task, i), max_tokens=200)
            tokens += r.total_tokens
            m = re.search(r"Action:\s*(.+)", r.text)
            action = (m.group(1) if m else r.text.strip().splitlines()[-1] if r.text.strip() else "look").strip().strip("`'\".").lower()
            unauth = int(action not in adm)
            obs_new, done, won, adm = self.bench.env_step(task, action)
            steps.append(Step(observation=obs, action=action, tool_result=obs_new,
                              meta={"tokens": r.total_tokens, "latency_s": round(time.time() - t0, 3),
                                    "unauthorized": unauth, "skills": [s.id for s in skills],
                                    "thought": r.text.split("Action:")[0].replace("Thought:", "").strip()[:300]}))
            hist.append((action, obs_new)); obs = obs_new
            if done:
                break
        ok = self.bench.verify(task, "episode_done")
        return Trace(task_id=task_id(task), steps=steps, outcome=ok,
                     failure_mode=None if ok else ("max_steps" if not done else "env_done_not_won"),
                     final_output="WON" if ok else "NOT_WON", tokens=tokens)
