"""ReActAgent: fixed Thought/Action/Observation loop on cfg.base_model."""
from __future__ import annotations
import ast
import operator
import re
import time
from harness.interfaces.agent import ConsumingAgent
from harness.interfaces.representation import Representation
from harness.registry import register
from harness.seeding import derive_seed, task_id
from harness.types import Step, Trace, SkillLibrary

ALLOWED_TOOLS = {"calculator", "finish"}
SYSTEM = """You are a ReAct agent. Tools: calculator[<python arithmetic expression>].
Respond with either
Thought: ...\\nAction: calculator[<expr>]
or
Thought: ...\\nFinal Answer: <answer only>
"""
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.Div: operator.truediv}


def safe_eval(expr: str):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval"))


@register("agent", "react")
class ReActAgent(ConsumingAgent):
    def __init__(self, llm, benchmark, seed: int, max_steps: int = 6, logger=None):
        self.llm, self.bench, self.seed, self.max_steps, self.logger = llm, benchmark, seed, max_steps, logger

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.llm, ctx.benchmark, ctx.cfg.seed, logger=ctx.logger)

    def run_task(self, task: str, lib: SkillLibrary,
                 rep: Representation, k: int) -> Trace:
        skills = rep.retrieve(task, lib, k)
        prompt = rep.inject(skills, f"{SYSTEM}\nTask: {task}\n")
        steps, tokens, final, failure = [], 0, "", None
        for i in range(self.max_steps):
            t0 = time.time()
            resp = self.llm.complete(prompt, seed=derive_seed(self.seed, "react", task))
            tokens += resp.total_tokens
            text = resp.text
            meta = {"tokens": resp.total_tokens, "latency_s": round(time.time() - t0, 4),
                    "unauthorized": 0, "skills": [s.id for s in skills]}
            fa = re.search(r"Final Answer:\s*(.*)", text)
            if fa:
                final = fa.group(1).strip()
                steps.append(Step(observation=prompt[-200:], action=f"finish[{final}]", tool_result=None, meta=meta))
                break
            act = re.search(r"Action:\s*(\w+)\[(.*)\]", text)
            if not act:
                result = "Error: could not parse action"
                tool = "parse_error"
            else:
                tool, arg = act.group(1), act.group(2)
                if tool not in ALLOWED_TOOLS:
                    meta["unauthorized"] = 1
                    result = f"Error: tool '{tool}' is not allowed"
                else:
                    try:
                        result = str(safe_eval(arg))
                    except Exception as e:  # noqa: BLE001
                        result = f"Error: {e}"
            steps.append(Step(observation=prompt[-200:], action=act.group(0) if act else text,
                              tool_result=result, meta=meta))
            prompt += f"{text}\nObservation: {result}\n"
        else:
            failure = "max_steps"
        ok = self.bench.verify(task, final) if final else False
        if not ok and failure is None:
            failure = "wrong_answer" if final else "no_answer"
        tr = Trace(task_id=task_id(task), steps=steps, outcome=ok, failure_mode=failure,
                   final_output=final, tokens=tokens)
        return tr
