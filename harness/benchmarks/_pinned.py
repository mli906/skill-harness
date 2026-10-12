"""Pinned external dependencies (fetched by commit SHA, never forked or copied
into this repo). No @register here; the registry import is a no-op."""
from __future__ import annotations
import importlib.util
import os
import subprocess
from pathlib import Path

OFFICEQA_REPO = "https://github.com/databricks/officeqa"
OFFICEQA_SHA = "7b9a3c154ef9fb40215bb67934afc43e6799de16"
EVOSKILL_REPO = "https://github.com/sentient-agi/EvoSkill"
EVOSKILL_SHA = "36f6f04952293d7054145550c2b9f0b0411bff1c"
CACHE = Path(os.environ.get("HARNESS_CACHE", Path.home() / ".cache" / "skill-harness"))


def checkout(repo: str, sha: str) -> Path:
    """Shallow-fetch `repo` at exactly `sha` into the cache; return its path."""
    dest = CACHE / f"{repo.rstrip('/').split('/')[-1]}-{sha[:12]}"
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", repo],
                    ["git", "fetch", "-q", "--depth", "1", "origin", sha],
                    ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=dest, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != sha:
        raise RuntimeError(f"{dest} is at {head}, expected pinned {sha}")
    return dest


def officeqa_reward():
    """Import OfficeQA's own reward.py from the pinned checkout."""
    path = checkout(OFFICEQA_REPO, OFFICEQA_SHA) / "reward.py"
    spec = importlib.util.spec_from_file_location("officeqa_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evoskill_officeqa_sample() -> Path:
    """The 10-question OfficeQA sample + 9 Treasury Bulletins shipped in the pinned
    EvoSkill repo (examples/officeqa/data). Used when HF access is unavailable."""
    return checkout(EVOSKILL_REPO, EVOSKILL_SHA) / "examples" / "officeqa" / "data"
