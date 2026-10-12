"""Plugin registry. New methods are added as NEW FILES in the matching
subpackage (learners/, gates/, representations/, agents/, benchmarks/) that
call @register(kind, name). Subpackages are auto-imported, so no existing
file needs editing."""
from __future__ import annotations
import importlib
import pkgutil
from typing import Any, Callable

KINDS = ("learner", "gate", "representation", "agent", "benchmark")
_PKG = {"learner": "learners", "gate": "gates", "representation": "representations",
        "agent": "agents", "benchmark": "benchmarks"}
_REG: dict[str, dict[str, Callable[[Any], Any]]] = {k: {} for k in KINDS}
_loaded = False


def register(kind: str, name: str):
    """Decorate a class exposing `from_context(ctx)` (or a no-arg ctor)."""
    def deco(cls):
        if name in _REG[kind]:
            raise ValueError(f"duplicate {kind} '{name}'")
        _REG[kind][name] = cls
        return cls
    return deco


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    for pkg in _PKG.values():
        mod = importlib.import_module(f"harness.{pkg}")
        for info in pkgutil.iter_modules(mod.__path__):
            importlib.import_module(f"harness.{pkg}.{info.name}")
    _loaded = True


def available(kind: str) -> list[str]:
    _load_all()
    return sorted(_REG[kind])


def build(kind: str, name: str, ctx: Any) -> Any:
    _load_all()
    if name not in _REG[kind]:
        raise KeyError(f"unknown {kind} '{name}'; available: {available(kind)}")
    cls = _REG[kind][name]
    return cls.from_context(ctx) if hasattr(cls, "from_context") else cls()
