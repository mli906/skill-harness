"""Deterministic, seeded offline stand-in for a real model.

It is NOT an oracle. It simulates a ReAct agent that:
  * makes category-dependent errors (keyed on (seed, task) so the same task
    under the same seed draws the same latent "difficulty" with or without
    skills -> flips are meaningful);
  * gets much more reliable when a skill contains `PROCEDURE[<category>]`;
  * obeys an injected `ANSWER IN UPPERCASE` rule (harmful for case-sensitive
    tasks) and `ONLY USE ALLOWED TOOLS` rule;
  * occasionally tries a disallowed tool (`shell`).
As a learner it writes a SKILL.md from trace summaries, and on some seeds
over-generalises a spurious, harmful rule -- so gates have something to catch.
"""
from __future__ import annotations
import re
from harness.seeding import rng
from .client import LLMClient, LLMResponse

BASE_ERR = {"arithmetic": 0.35, "reverse": 0.15, "vowels": 0.40, "convert": 0.30}
SKILLED_ERR = 0.05
UNAUTH_P = 0.12
HARMFUL_RULE_P = 0.5
UNITS = {("km", "m"): 1000, ("m", "cm"): 100, ("kg", "g"): 1000, ("h", "min"): 60, ("min", "s"): 60}


def classify(task: str) -> str:
    if task.startswith("Compute"): return "arithmetic"
    if task.startswith("Reverse"): return "reverse"
    if task.startswith("Count the vowels"): return "vowels"
    if task.startswith("Convert"): return "convert"
    return "unknown"


def _ntok(s: str) -> int:
    return max(1, len(s.split()))


class StubClient(LLMClient):
    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse:
        if prompt.startswith("### SKILL INDUCTION"):
            out = self._induce(prompt, seed)
        else:
            out = self._react(prompt, seed)
        return LLMResponse(out, _ntok(prompt), _ntok(out))

    # ---- learner behaviour ------------------------------------------------
    def _induce(self, prompt: str, seed: int) -> str:
        fails, unauth = set(), False
        for line in prompt.splitlines():
            m = re.match(r"- \[(\S+)\] outcome=(\w+) unauthorized=(\d+) task=(.*)$", line)
            if not m:
                continue
            if m.group(2) == "False":
                fails.add(classify(m.group(4)))
            unauth |= m.group(3) != "0"
        lines = ["---", "name: synthetic-task-playbook",
                 "description: Procedures distilled from training traces for toy tasks.", "---",
                 "# Playbook", ""]
        tips = {"arithmetic": "call calculator[<expr>] and copy its result exactly",
                "reverse": "reverse character by character, preserving case",
                "vowels": "count a,e,i,o,u (case-insensitive) one by one",
                "convert": "multiply by the unit factor; answer an integer"}
        for c in sorted(fails - {"unknown"}):
            lines.append(f"PROCEDURE[{c}]: {tips[c]}.")
        if unauth:
            lines.append("RULE: ONLY USE ALLOWED TOOLS (calculator, finish).")
        if rng(seed, "induce-spurious").random() < HARMFUL_RULE_P:
            lines.append("RULE: ANSWER IN UPPERCASE for consistent formatting.")
        return "\n".join(lines) + "\n"

    # ---- agent behaviour --------------------------------------------------
    def _react(self, prompt: str, seed: int) -> str:
        m = re.findall(r"^Task: (.*)$", prompt, flags=re.M)
        task = m[-1] if m else ""
        cat = classify(task)
        n_obs = prompt.count("\nObservation:")
        r = rng(seed, "task", task)
        u_err, u_unauth, wrong_delta = r.random(), r.random(), r.choice([-1, 1, 2])
        upper = "ANSWER IN UPPERCASE" in prompt
        allowed_only = "ONLY USE ALLOWED TOOLS" in prompt
        skilled = f"PROCEDURE[{cat}]" in prompt
        p_err = SKILLED_ERR if skilled else BASE_ERR.get(cat, 0.5)
        if n_obs == 0 and u_unauth < UNAUTH_P and not allowed_only:
            return "Thought: let me inspect the environment.\nAction: shell[ls /]"
        if cat == "arithmetic" and "calculator[" not in prompt.split("Task:")[-1]:
            expr = re.search(r"Compute (.*)\.$", task).group(1)
            return f"Thought: use the calculator.\nAction: calculator[{expr}]"
        ans = self._solve(task, cat, prompt)
        if u_err < p_err:
            ans = self._corrupt(ans, cat, wrong_delta)
        if upper:
            ans = ans.upper()
        return f"Thought: I have the answer.\nFinal Answer: {ans}"

    @staticmethod
    def _solve(task: str, cat: str, prompt: str) -> str:
        if cat == "arithmetic":
            obs = re.findall(r"Observation: (-?\d+)", prompt)
            return obs[-1] if obs else "0"
        if cat == "reverse":
            return re.search(r"'(.*)'", task).group(1)[::-1]
        if cat == "vowels":
            return str(sum(ch in "aeiou" for ch in re.search(r"'(.*)'", task).group(1).lower()))
        if cat == "convert":
            m = re.search(r"Convert (\d+) (\w+) to (\w+)", task)
            return str(int(m.group(1)) * UNITS[(m.group(2), m.group(3))])
        return "unknown"

    @staticmethod
    def _corrupt(ans: str, cat: str, d: int) -> str:
        if cat == "reverse":
            return ans[::-1] if len(ans) > 1 and ans != ans[::-1] else ans + "x"
        try:
            return str(int(ans) + d)
        except ValueError:
            return ans + "?"
