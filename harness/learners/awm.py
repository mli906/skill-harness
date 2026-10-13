"""AWMLearner (`learner: awm`, contract C1_fixed_traces): Agent Workflow Memory, offline induction.

AWM reimplemented natively (zorazrw/agent-workflow-memory @ 8c0ff8c is Mind2Web-bound). Prompt structure, format_examples and filter_workflows ported faithfully; INSTRUCTION re-domained and ONE_SHOT authored for ALFWorld to avoid an off-domain-prompt confound.

Port map (reference @ 8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1):
  mind2web/offline_induction.py L54-68  llm_generate: header line + format_examples, joined
      as '\\n\\n'.join([INSTRUCTION, ONE_SHOT, prompt]); one call, temperature 0, max_tokens 1024
      -> build_prompt() + induce() (one ctx.llm call: temperature 0 + derived seed, max_tokens 1024).
  mind2web/offline_induction.py L148    --suffix default "# Summary Workflows" -> SUFFIX.
  mind2web/utils/data.py L48-60         format_examples -> format_examples() ("Query #i:",
      "Actions and Environments:", one line per action, blank line; prefix/suffix).
  mind2web/utils/data.py L65-71         is_website_header -> is_env_header().
  mind2web/utils/data.py L73-86         filter_workflows -> filter_workflows(): cut before the
      target header, cut at the one-shot header, drop blocks mentioning the one-shot marker
      ("toyhouse" here, "delta" in the reference).
  mind2web/prompt/instruction_action.txt -> INSTRUCTION (re-domained to household text games).
  mind2web/prompt/one_shot_action.txt    -> ONE_SHOT (authored ALFWorld example, same structure:
      header line, '## Query k' + 'Actions:' blocks, 'Summary Workflows:', '## workflow_name'
      blocks with a 'Given that ...' description and {abstracted-slots}).
Not ported (non-goals): Mind2Web data loading / domain hierarchy, add_scores + scores_all_data.pkl,
online mode.
"""
from __future__ import annotations
from harness.interfaces.learner import Learner, C1_FIXED_TRACES
from harness.registry import register
from harness.seeding import derive_seed
from harness.types import Trace, Skill, SkillLibrary

ENV_TAG = "alfworld"            # analogue of the website tag (reference L56: "Website: " + tags)
ONE_SHOT_MARKER = "toyhouse"    # analogue of "delta" stripped in reference data.py L80-85
SUFFIX = "# Summary Workflows"  # reference offline_induction.py L148
OBS_CHARS = 100

INSTRUCTION = (
    "Given a list of household text-game tasks, extract the common workflows that solve them. "
    "Each task has an instruction and a series of actions. Find the repetitive subset of actions "
    "across tasks and extract each as a workflow. Each workflow is a commonly-reused sub-routine; "
    "do not generate overlapping workflows; each has at least two steps."
)

ONE_SHOT = """Environment: Household, ALFWorld, toyhouse
## Query 1: Your task is to: put a mug in coffeemachine.
Actions:
go to countertop 1 -> On the countertop 1, you see a mug 1, and a knife 1.
take mug 1 from countertop 1 -> You pick up the mug 1 from the countertop 1.
go to coffeemachine 1 -> On the coffeemachine 1, you see nothing.
put mug 1 in/on coffeemachine 1 -> You put the mug 1 in/on the coffeemachine 1.

## Query 2: Your task is to: examine the book with the desklamp.
Actions:
go to drawer 1 -> The drawer 1 is closed.
open drawer 1 -> You open the drawer 1. The drawer 1 is open. In it, you see a book 2.
take book 2 from drawer 1 -> You pick up the book 2 from the drawer 1.
go to sidetable 1 -> On the sidetable 1, you see a desklamp 1.
use desklamp 1 -> You turn on the desklamp 1.

## Query 3: Your task is to: put a book in shelf.
Actions:
go to drawer 2 -> The drawer 2 is closed.
open drawer 2 -> You open the drawer 2. The drawer 2 is open. In it, you see a book 1.
take book 1 from drawer 2 -> You pick up the book 1 from the drawer 2.
go to shelf 1 -> On the shelf 1, you see nothing.
put book 1 in/on shelf 1 -> You put the book 1 in/on the shelf 1.


Summary Workflows:
## take_object_from_closed_receptacle
Given that you are in the toyhouse and the object is inside a closed receptacle, this workflow retrieves it.
go to {receptacle}
open {receptacle}
take {object} from {receptacle}

## place_object_in_receptacle
Given that you are in the toyhouse holding the object, this workflow places it at the target receptacle.
go to {target-receptacle}
put {object} in/on {target-receptacle}

## examine_object_under_lamp
Given that you are in the toyhouse holding the object, this workflow examines it with the light.
go to {lamp-location}
use {desklamp}"""


def format_examples(traces: list[Trace], task_lookup: dict, prefix: str | None = None,
                    suffix: str | None = None) -> str:
    """Port of data.py L48-60. confirmed_task -> task text; action_reprs -> one line per Step."""
    lines = []
    for i, t in enumerate(traces):
        task = task_lookup.get(t.task_id, t.task_id)
        lines.append(f"Query #{i+1}: {task}")
        lines.append("Actions and Environments:")
        for s in t.steps:
            obs = (s.tool_result or "").replace("\n", " ").strip()[:OBS_CHARS]
            lines.append(f"{s.action} -> {obs}" if obs else s.action)
        lines.append("")
    prompt = "\n".join(lines)
    if prefix is not None:
        prompt = prefix + "\n" + prompt
    if suffix is not None:
        prompt += "\n\n" + suffix
    return prompt


def build_prompt(traces: list[Trace], task_lookup: dict) -> str:
    """Port of offline_induction.py L56-58."""
    prompt = "Environment: Household, ALFWorld, " + ENV_TAG + "\n"
    prompt += format_examples(traces, task_lookup, None, SUFFIX)
    return "\n\n".join([INSTRUCTION, ONE_SHOT, prompt])


def is_env_header(block: str, tag: str) -> bool:
    """Port of data.py L65-71 (is_website_header)."""
    lines = block.strip().split("\n")
    if len(lines) > 1:
        return False
    text = lines[0].strip()
    return text.startswith("#") and text.lower().endswith(tag)


def filter_workflows(text: str, tag: str = ENV_TAG, marker: str = ONE_SHOT_MARKER) -> str:
    """Port of data.py L73-86 (website -> tag, "delta" -> one-shot marker)."""
    blocks = text.split("\n\n")
    for i, b in enumerate(blocks):
        if is_env_header(b, tag):
            blocks = blocks[i + 1:]
            break
    for i, b in enumerate(blocks):
        if is_env_header(b, marker):
            blocks = blocks[:i]
            break
    blocks = [b for b in blocks if marker not in b.lower()]
    return "\n\n".join(blocks)


@register("learner", "awm")
class AWMLearner(Learner):
    contract = C1_FIXED_TRACES

    def __init__(self, llm, seed: int, task_lookup: dict | None = None, logger=None):
        self.llm, self.seed, self.logger = llm, seed, logger
        self.task_lookup = task_lookup if task_lookup is not None else {}
        self.last_build_tokens = 0
        self.last_prompt = ""

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.llm, ctx.cfg.seed, ctx.task_lookup, ctx.logger)

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        self.last_prompt = build_prompt(traces, self.task_lookup)
        resp = self.llm.complete(self.last_prompt, seed=derive_seed(self.seed, "awm-induction"), max_tokens=1024)
        self.last_build_tokens = resp.total_tokens
        workflows = filter_workflows(resp.text).strip()
        v = 1 + max((s.version for s in current.skills), default=0)
        skill = Skill(id=f"awm_workflows_v{v}", content=workflows, representation="skill_md",
                      provenance=sorted({t.task_id for t in traces}), version=v)
        if self.logger:
            self.logger.log("learner.induce", learner="awm", tokens=self.last_build_tokens,
                            skill_id=skill.id, content=skill.content)
        return [skill]
