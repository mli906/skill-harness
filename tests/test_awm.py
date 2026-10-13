"""AWMLearner: offline, fake LLM (no spend)."""
from harness.env_guard import forbid_env
from harness.interfaces.learner import C1_FIXED_TRACES
from harness.learners.awm import AWMLearner, filter_workflows
from harness.llm.client import LLMResponse
from harness.types import Step, Trace, SkillLibrary

RESPONSE = """Summary Workflows:

## find_and_take_object
Given that you need an object, this workflow locates and picks it up.
go to {receptacle}
take {object} from {receptacle}

## place_object
go to {target}
put {object} in/on {target}

## take_object_from_closed_receptacle
Given that you are in the toyhouse, this is the one-shot echo.
open {receptacle}
take {object} from {receptacle}"""


class FakeLLM:
    def __init__(self): self.prompts = []
    def complete(self, prompt, *, seed, max_tokens=1024):
        self.prompts.append(prompt)
        return LLMResponse(RESPONSE, 100, 20)


class FakeBench:
    def env_reset(self, task): raise AssertionError("unguarded env access")
    def env_step(self, task, a): raise AssertionError("unguarded env access")


def _traces():
    s = lambda a, o: Step("obs", a, o, {})
    return [Trace("t_a", [s("go to desk 1", "On the desk 1, you see a pen 1."), s("take pen 1 from desk 1", "You pick up the pen 1.")], True, None, "WON", 5),
            Trace("t_b", [s("go to shelf 1", "You see nothing."), s("put pen 1 in/on shelf 1", "You put it.")], False, "max_steps", "NOT_WON", 5),
            Trace("t_a", [s("look", "You are in the middle of a room.")], False, "max_steps", "NOT_WON", 2)]


LOOKUP = {"t_a": "ALFWorld game a\nYour task is to: put a pen in shelf.", "t_b": "ALFWorld game b\nYour task is to: put a pen in shelf."}


def test_contract_is_c1():
    assert AWMLearner.contract == C1_FIXED_TRACES


def test_one_skill_md_with_input_provenance():
    llm = FakeLLM()
    L = AWMLearner(llm, 0, LOOKUP)
    out = L.induce(_traces(), SkillLibrary(skills=[]))
    assert len(out) == 1 and len(llm.prompts) == 1
    sk = out[0]
    assert sk.representation == "skill_md" and sk.version == 1
    assert sk.provenance == ["t_a", "t_b"]
    assert "## find_and_take_object" in sk.content and "## place_object" in sk.content
    assert "toyhouse" not in sk.content and "take_object_from_closed_receptacle" not in sk.content
    assert L.last_build_tokens == 120


def test_runs_under_forbid_env():
    b = FakeBench()
    with forbid_env(b):
        out = AWMLearner(FakeLLM(), 0, LOOKUP).induce(_traces(), SkillLibrary(skills=[]))
    assert len(out) == 1


def test_prompt_is_alfworld_domain():
    llm = FakeLLM()
    AWMLearner(llm, 0, LOOKUP).induce(_traces(), SkillLibrary(skills=[]))
    p = llm.prompts[0]
    assert p.startswith("Given a list of household text-game tasks, extract the common workflows")
    assert "Query #1: ALFWorld game a" in p and "go to desk 1 -> On the desk 1" in p
    assert p.rstrip().endswith("# Summary Workflows")
    low = p.lower()
    for w in ("spreadsheet", "xlsx", "excel", "web", "website", "click", "delta", "mind2web", "browser"):
        assert w not in low, w


def test_filter_workflows_port():
    txt = "# alfworld\n\n## w1\ngo to {x}\nopen {x}\n\n# Environment: toyhouse\n\n## w2\nmore"
    assert filter_workflows(txt) == "## w1\ngo to {x}\nopen {x}"
