"""ALFWorldAdapter (`benchmark: alfworld`): ALFWorld text games (alfworld==0.4.2,
textworld==1.7.0, data from `alfworld-download`), success = the environment's
own `won` flag.

How verify() works with an interactive env
------------------------------------------
Benchmark.verify(task, output) only sees the final output, so the adapter also
exposes a duck-typed env API used by an interactive ConsumingAgent
(`react_multistep`):
    env_reset(task) -> (observation, admissible_commands)
    env_step(task, action) -> (observation, done, won, admissible_commands)
Each episode's `won` is recorded per task; verify(task, output) returns the
env-recorded success of the most recent episode of `task` (output is ignored
except that an episode must have been played). The agent calls verify right
after its episode, so the success signal is ALFWorld's, not a string match.

Selection: games from $ALFWORLD_SPLIT (default valid_seen), grouped by task
type (= category: pick_and_place_simple, look_at_obj_in_light, ...), ordered by
a stable hash; per category the first ALFWORLD_TRAIN_PER_CAT (default 4) go to
train and the next ones (up to limit/6 total per category) to test.
Env: ALFWORLD_DATA (default ~/.cache/alfworld), OFFICE-style --limit via
ALFWORLD_LIMIT, ALFWORLD_MAX_EPISODE_STEPS (default 50), ALFWORLD_TYPES (comma list; default all 6).
"""
from __future__ import annotations
import glob
import hashlib
import os
import re
from pathlib import Path
from harness.interfaces.benchmark import Benchmark
from harness.registry import register
from harness.benchmarks import _alfworld_compat
from harness.llm import embedding_safe

embedding_safe.apply()

ALFWORLD_VERSION = "0.4.2"
TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def game_type(path: str) -> str:
    d = Path(path).parent.parent.name
    return next((t for t in TASK_TYPES if d.startswith(t)), "other")


def make_env(gamefile: str, max_steps: int):
    _alfworld_compat.apply()
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    ri = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    eid = textworld.gym.register_games([gamefile], ri, batch_size=1, asynchronous=False,
                                       max_episode_steps=max_steps,
                                       wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    return textworld.gym.make(eid)


def list_games(split: str | None = None) -> list[str]:
    root = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld"))
    split = split or os.environ.get("ALFWORLD_SPLIT", "valid_seen")
    games = glob.glob(str(root / "json_2.1.1" / split / "*" / "*" / "game.tw-pddl"))
    return sorted(games, key=lambda g: _h(g.split("json_2.1.1/")[-1]))


@register("benchmark", "alfworld")
class ALFWorldAdapter(Benchmark):
    def __init__(self, limit: int | None = None, train_per_cat: int | None = None):
        lim = limit if limit is not None else os.environ.get("ALFWORLD_LIMIT")
        limit = int(lim) if lim not in (None, "", "0") else 48
        tpc = int(train_per_cat or os.environ.get("ALFWORLD_TRAIN_PER_CAT", 4))
        self.max_steps = int(os.environ.get("ALFWORLD_MAX_EPISODE_STEPS", 50))
        types = [t for t in os.environ.get("ALFWORLD_TYPES", ",".join(TASK_TYPES)).split(",") if t]
        self.types = types
        per_cat = max(tpc + 1, limit // len(types))
        by_type: dict[str, list[str]] = {t: [] for t in types}
        for g in list_games():
            t = game_type(g)
            if t in by_type and len(by_type[t]) < per_cat:
                by_type[t].append(g)
        self._game: dict[str, str] = {}
        self._cat: dict[str, str] = {}
        self._won: dict[str, bool] = {}
        self._envs: dict[str, object] = {}
        self._train, self._test = [], []
        for t in types:
            for i, g in enumerate(by_type[t]):
                task = self._render(g)
                self._game[task], self._cat[task] = g, t
                (self._train if i < tpc else self._test).append(task)

    def _render(self, gamefile: str) -> str:
        env = make_env(gamefile, self.max_steps)
        obs, _ = env.reset()
        env.close()
        intro = obs[0].split("-= Welcome to TextWorld, ALFRED! =-")[-1].strip()
        gid = _h(gamefile.split("json_2.1.1/")[-1])[:8]
        return f"ALFWorld game {gid}\n{intro}"

    # ---- Benchmark ----------------------------------------------------------
    def train_tasks(self) -> list[str]:
        return list(self._train)

    def test_tasks(self) -> list[str]:
        return list(self._test)

    def verify(self, task: str, output: str) -> bool:
        return bool(self._won.get(task, False))

    # ---- optional extensions ------------------------------------------------
    def category(self, task: str) -> str:
        return self._cat[task]

    def gamefile(self, task: str) -> str:
        return self._game[task]

    def goal(self, task: str) -> str:
        m = re.search(r"Your task is to: (.*)", task)
        return m.group(1).strip() if m else task

    def env_reset(self, task: str):
        if task in self._envs:
            self._envs[task].close()
        env = make_env(self._game[task], self.max_steps)
        self._envs[task] = env
        self._won[task] = False
        obs, info = env.reset()
        return obs[0], list(info["admissible_commands"][0])

    def env_step(self, task: str, action: str):
        obs, _score, done, info = self._envs[task].step([action])
        won = bool(info["won"][0])
        self._won[task] = self._won[task] or won
        if done[0]:
            self._envs.pop(task).close()
        return obs[0], bool(done[0]), won, list(info["admissible_commands"][0])

    # ---- EvoSkill native-input mapping (used by EvoSkillLearner) -------------
    EVOSKILL_TASK_MD = """# Task

You solve ALFWorld household text-game tasks. You are given the task (initial
room description + goal) and the trajectory of a previous attempt by another
agent, which reveals where objects are. Output the complete sequence of game
commands that solves the task, in order, separated by ' ; '. Your final answer
is executed verbatim in the real environment and scored by its success flag.

---

# Constraints

- Final answer: only commands separated by ' ; ' (e.g. go to desk 1 ; take pen 1 from desk 1 ; ...)
- Never use quotes or apostrophes in the final answer
- Use command forms: go to X, open X, close X, take O from R, put O in/on R, clean O with R, heat O with R, cool O with R, use L, examine X
"""

    def gid(self, task: str) -> str:
        return task.split()[2]

    def evoskill_row(self, task: str, trace) -> dict:
        traj = "\n".join(f"> {s.action} -> {(s.tool_result or '').replace(chr(10), ' ')[:140]}"
                         for s in trace.steps[:25])
        q = (f"{task}\n\nPrevious attempt ({'SUCCESS' if trace.outcome else 'FAILURE'}, "
             f"{len(trace.steps)} steps):\n{traj}\n\nGive the full command sequence that solves the task, "
             f"separated by ' ; '.")
        return {"question": q, "answer": self.gid(task), "category": self.category(task)}

    def evoskill_scorer_toml(self, root) -> str:
        import json, sys
        mp = Path(root) / "alfworld_games.json"
        mp.write_text(json.dumps({self.gid(t): g for t, g in self._game.items()}))
        cmd = f"{sys.executable} -m harness.benchmarks.alfworld_score --map {mp} --game {{expected}} --plan {{predicted}}"
        return f'[scorer]\ntype = "script"\ncommand = "{cmd}"\n'
