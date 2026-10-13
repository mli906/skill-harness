"""Runtime shim for the pinned Trace2Skill checkout (no edits to its files).

    python -m harness.learners.trace2skill_compat <t2s_root> <usage.json> (-m <module> | <script.py>) [args...]

* puts <t2s_root> first on sys.path (Trace2Skill's top-level `src`/`analysis`
  packages; EvoSkill also ships a top-level `src`, so Trace2Skill always runs in
  its own subprocess);
* binds `src` to the checkout's namespace package explicitly;
* wraps openai's Chat Completions `create` (sync + async) to sum token usage
  and writes {prompt_tokens, completion_tokens, calls} to <usage.json> at exit,
  so the harness can report Trace2Skill build tokens;
* then runs the Trace2Skill entrypoint unchanged via runpy.
"""
from __future__ import annotations
import atexit
import json
import runpy
import sys
import threading

_U = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
_L = threading.Lock()


def _add(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return
    with _L:
        _U["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _U["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        _U["calls"] += 1


def patch_openai():
    from openai.resources.chat import completions as cc
    orig, aorig = cc.Completions.create, cc.AsyncCompletions.create

    def create(self, *a, **k):
        r = orig(self, *a, **k); _add(r); return r

    async def acreate(self, *a, **k):
        r = await aorig(self, *a, **k); _add(r); return r
    cc.Completions.create, cc.AsyncCompletions.create = create, acreate


def _bind_toplevel(root: str, name: str) -> None:
    """Trace2Skill's `src` has no __init__.py (namespace package), so an installed
    regular package of the same name wins regardless of sys.path order. Bind the
    name to the checkout's directory explicitly."""
    import os
    import types
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(root, name)]
    sys.modules[name] = m


def main(argv):
    root, usage = argv[0], argv[1]
    rest = argv[2:]
    sys.path.insert(0, root)
    _bind_toplevel(root, "src")   # EvoSkill's installed regular `src` package would shadow
    patch_openai()
    atexit.register(lambda: open(usage, "w").write(json.dumps(_U)))
    if rest[0] == "-m":
        sys.argv = [rest[1], *rest[2:]]
        runpy.run_module(rest[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = [rest[0], *rest[1:]]
        runpy.run_path(rest[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
