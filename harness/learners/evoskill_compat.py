"""Runtime compatibility shim: EvoSkill (pinned) + OpenHands/litellm + OpenAI direct.

Problem: EvoSkill's OpenHands fallback extraction sends
response_format={"type":"json_schema", "strict": True, "schema": <pydantic schema>}.
OpenAI strict mode rejects pydantic's schema (needs additionalProperties:false on
every object, every property listed in `required`, no `default`).

Fix (no edits to installed EvoSkill files): wrap openhands' LLM.completion in
THIS process and rewrite any strict json_schema to the OpenAI-strict subset.
Launched as `python -m harness.learners.evoskill_compat <evoskill CLI args>`,
which applies the patch then calls EvoSkill's own CLI entrypoint unchanged.
Both scripts/validate_evoskill.py (standalone) and EvoSkillLearner use it.
"""
from __future__ import annotations
import copy
import sys

_PATCHED = False


def strictify(schema: dict) -> dict:
    s = copy.deepcopy(schema)

    def walk(n):
        if isinstance(n, dict):
            n.pop("default", None)
            if n.get("type") == "object" or "properties" in n:
                props = n.setdefault("properties", {})
                n["additionalProperties"] = False
                n["required"] = list(props.keys())
            for v in list(n.values()):
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(s)
    return s


def _fix_kwargs(kwargs: dict) -> dict:
    rf = kwargs.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        if js.get("strict") and isinstance(js.get("schema"), dict):
            rf = dict(rf, json_schema=dict(js, schema=strictify(js["schema"])))
            kwargs = dict(kwargs, response_format=rf)
    return kwargs


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from openhands.sdk.llm import LLM
    orig = LLM.completion

    def completion(self, *args, **kwargs):
        return orig(self, *args, **_fix_kwargs(kwargs))
    LLM.completion = completion
    _PATCHED = True


def main(argv: list[str]) -> None:
    apply()
    from src.cli.main import cli  # EvoSkill's own CLI, unchanged
    cli.main(args=argv, prog_name="evoskill")


if __name__ == "__main__":
    main(sys.argv[1:])
