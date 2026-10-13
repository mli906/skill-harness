"""Enforce the C1_fixed_traces contract: while a C1 learner induces, any
benchmark-environment access raises EnvAccessError.

In-process: the benchmark's env_reset / env_step are swapped for raisers.
Out-of-process: HARNESS_ENV_FORBIDDEN=1 is exported, which harness env
constructors (alfworld.make_env) check, so a learner's subprocess cannot open
an ALFWorld env either."""
from __future__ import annotations
import contextlib
import os

ENV_FLAG = "HARNESS_ENV_FORBIDDEN"
ENV_METHODS = ("env_reset", "env_step")


class EnvAccessError(RuntimeError):
    pass


def check_env_allowed(what: str = "environment") -> None:
    if os.environ.get(ENV_FLAG) == "1":
        raise EnvAccessError(f"C1_fixed_traces learner tried to access the {what}")


@contextlib.contextmanager
def forbid_env(bench):
    saved = {}
    for m in ENV_METHODS:
        if hasattr(bench, m):
            saved[m] = bench.__dict__.get(m, None)
            def _deny(*a, _m=m, **k):
                raise EnvAccessError(f"C1_fixed_traces learner called benchmark.{_m}")
            setattr(bench, m, _deny)
    old = os.environ.get(ENV_FLAG)
    os.environ[ENV_FLAG] = "1"
    try:
        yield
    finally:
        for m, v in saved.items():
            if v is None:
                delattr(bench, m)
            else:
                setattr(bench, m, v)
        if old is None:
            os.environ.pop(ENV_FLAG, None)
        else:
            os.environ[ENV_FLAG] = old
