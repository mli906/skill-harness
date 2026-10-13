"""Runtime shim for textworld (pinned 1.7.0) on Python >= 3.13 (no package edits).

textworld.envs.pddl.textgen.EvalSymbol.derive does `locals().update(vars);
eval(expr)`, which stopped working with PEP 667 (3.13): writes to locals() no
longer reach eval. We re-implement derive with an explicit namespace.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    import textworld.envs.pddl.textgen as tg

    def derive(self, context=None):
        context = context or self.context
        ns = dict(vars(tg))
        ns.update(context["variables"])
        return [tg.TerminalSymbol(eval(self.expression, ns, ns))]

    tg.EvalSymbol.derive = derive
    _APPLIED = True
