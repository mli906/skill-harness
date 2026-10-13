"""GEPALearner: offline, stub LM + scripted agent (no spend)."""
from harness import registry
from harness.interfaces.learner import C2_SELF_COLLECT
from harness.learners.gepa import GEPALearner, HarnessGEPAAdapter, SEED_SKILL
from harness.llm.client import LLMResponse
from harness.seeding import task_id
from harness.types import Step, Trace, SkillLibrary

TRAIN = [f"ALFWorld game {i}\nYour task is to: put a pen in shelf." for i in range(3)]
VAL = [f"ALFWorld game v{i}\nYour task is to: put a mug in desk." for i in range(3)]
TEST = ["ALFWorld game TEST\nYour task is to: heat a mug."]


class StubLM:
    def __init__(self): self.prompts = []
    def complete(self, prompt, *, seed, max_tokens=1024):
        self.prompts.append(prompt)
        return LLMResponse("```\n## improved_skill\nWhen the target is closed, open it first.\n```", 200, 30)


class Bench:
    def train_tasks(self): return TRAIN
    def test_tasks(self): return TEST
    def verify(self, task, output): return output == "WON" and "game 1" not in task


class ScriptedAgent:
    """Wins iff the skill mentions 'open'; except game 1 is always failed by the verifier."""
    def __init__(self): self.seen = []
    def run_task(self, task, lib, rep, k):
        self.seen.append(task)
        won = any("open" in s.content for s in lib.skills) or "v0" in task
        steps = [Step("obs", "go to shelf 1", "The shelf 1 is closed.", {})]
        return Trace(task_id(task), steps, won, None if won else "max_steps", "WON" if won else "NOT_WON", 7)


def test_contract_is_c2():
    assert GEPALearner.contract == C2_SELF_COLLECT
    assert "gepa" in registry.available("learner")


def test_adapter_scores_match_verify():
    b, ag = Bench(), ScriptedAgent()
    ad = HarnessGEPAAdapter(ag, None, b, 1)
    batch = [{"id": task_id(t), "text": t} for t in TRAIN + VAL]
    eb = ad.evaluate(batch, {"skill_md": "open the receptacle"}, capture_traces=True)
    expected = [1.0 if b.verify(d["text"], o) else 0.0 for d, o in zip(batch, eb.outputs)]
    assert eb.scores == expected and 0.0 in eb.scores and 1.0 in eb.scores
    assert len(eb.trajectories) == len(batch)
    assert ad.evaluate(batch, {"skill_md": "x"}).trajectories is None
    ds = ad.make_reflective_dataset({"skill_md": "x"}, eb, ["skill_md"])
    assert set(ds) == {"skill_md"} and all("Feedback" in r for r in ds["skill_md"])


def test_induce_one_skill_valset_provenance_and_tokens():
    lm, ag = StubLM(), ScriptedAgent()
    L = GEPALearner(lm, ag, None, Bench(), seed=0, train_tasks=TRAIN, val_tasks=VAL, max_metric_calls=12)
    out = L.induce([], SkillLibrary(skills=[]))
    assert len(out) == 1
    sk = out[0]
    assert sk.representation == "skill_md" and sk.version == 1
    assert sk.provenance == sorted(task_id(t) for t in VAL)
    assert TEST[0] not in ag.seen                       # never touches the test set
    assert lm.prompts, "reflection LM was called"
    assert "improved_skill" in sk.content and sk.content != SEED_SKILL
    assert L.last_build_tokens > 0
    assert L.last_build_tokens == 230 * len(lm.prompts) + 7 * len(ag.seen)
