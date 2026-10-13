"""EvoSkillLearner: wraps sentient-agi/EvoSkill (pinned commit, installed as an
external package -- see requirements-evoskill.txt). EvoSkill's loop is invoked
UNCHANGED through its own CLI (`evoskill run`) in an isolated git project.

Mapping
  Trace -> native input: each training trace's task becomes one CSV row
      (question = benchmark.question(task) or task, ground_truth =
      benchmark.answer(task), category = benchmark.category(task)).
      EvoSkill then splits these rows into its own train pools / val set and
      re-executes them with its base agent -- that is its native algorithm
      (it consumes Q/A pairs, not pre-recorded traces).
  native output -> Skill: every `.claude/skills/<name>/SKILL.md` on the best
      frontier program branch becomes one Skill(representation="skill_md",
      provenance=<task_ids of input traces>, version=current+1). Auxiliary
      files in the skill folder are appended to `content` under headers.
  last_build_tokens: sum of input+output token usage recorded in EvoSkill's run
      cache (base-agent rollouts) + proposer/generator usage if reported.

Env knobs: EVOSKILL_ITERATIONS (2), EVOSKILL_CONCURRENCY (2),
EVOSKILL_FRONTIER (2), EVOSKILL_FAILURE_SAMPLES (2), EVOSKILL_TRAIN_RATIO (0.5),
EVOSKILL_VAL_RATIO (0.5), EVOSKILL_HARNESS (openhands), EVOSKILL_MODEL
(default "openai/<cfg.base_model>"), EVOSKILL_TIMEOUT (600), EVOSKILL_WORKDIR.
"""
from __future__ import annotations
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from harness.interfaces.learner import Learner
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary

TASK_MD = """# Task

You are an expert analyst. You are given a question about U.S. Treasury Bulletin
documents. Use the documents in the provided data directories to find the figures
and answer the question.

---

# Constraints

- Must answers include units? → yes
- Should the agent use external APIs? → no
"""


def _env(name, default):
    return type(default)(os.environ.get(name, default))


def write_project(root: Path, rows: list[dict], data_dirs: list[str], model: str,
                  harness_name: str, bench=None) -> Path:
    """Create an isolated EvoSkill project (same layout as its examples/officeqa)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".evoskill").mkdir(exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    with open(root / "data" / "dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "question", "answer", "category"])
        w.writeheader(); w.writerows(rows)
    dirs = ", ".join(json.dumps(str(d)) for d in data_dirs)
    (root / ".evoskill" / "config.toml").write_text(f"""[harness]
name = "{harness_name}"
model = "{model}"
data_dirs = [{dirs}]
timeout_seconds = {_env('EVOSKILL_TIMEOUT', 600)}
max_retries = 1

[evolution]
mode = "skill_only"
iterations = {_env('EVOSKILL_ITERATIONS', 2)}
frontier_size = {_env('EVOSKILL_FRONTIER', 2)}
concurrency = {_env('EVOSKILL_CONCURRENCY', 2)}
no_improvement_limit = 2
failure_samples = {_env('EVOSKILL_FAILURE_SAMPLES', 2)}

[dataset]
path = "data/dataset.csv"
question_column = "question"
ground_truth_column = "answer"
category_column = "category"
train_ratio = {_env('EVOSKILL_TRAIN_RATIO', 0.5)}
val_ratio = {_env('EVOSKILL_VAL_RATIO', 0.5)}

""" + (bench.evoskill_scorer_toml(root) if hasattr(bench, "evoskill_scorer_toml")
       else '[scorer]\ntype = "multi_tolerance"\n'))
    (root / ".evoskill" / "task.md").write_text(getattr(bench, "EVOSKILL_TASK_MD", TASK_MD))
    (root / ".evoskill" / "state.json").write_text('{"original_branch": "main"}')
    (root / ".gitignore").write_text(".cache/\n")
    # Pre-create the skills dir (EvoSkill's own examples ship one). Without it, an
    # agent's first file-editor "create .claude/skills" can make it a plain FILE,
    # after which every SKILL.md write fails with ENOTDIR.
    (root / ".claude" / "skills").mkdir(parents=True, exist_ok=True)
    (root / ".claude" / "skills" / ".gitkeep").write_text("")
    git = ["git", "-c", "user.name=skill-harness", "-c", "user.email=harness@localhost"]
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        # EvoSkill commits program branches; give the isolated repo a local identity.
        subprocess.run(["git", "config", "user.name", "skill-harness"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "harness@localhost"], cwd=root, check=True)
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(git + ["commit", "-qm", "init"], cwd=root, check=True)
    return root


def run_evoskill(root: Path, log_file: Path) -> int:
    """Invoke EvoSkill's own CLI, unchanged."""
    # Launch via the compat shim (patches OpenAI strict-schema issue in-process,
    # then calls EvoSkill's CLI unchanged). Used identically by standalone validation.
    cmd = [sys.executable, "-m", "harness.learners.evoskill_compat", "run", "--quiet",
           "--config", str(root / ".evoskill" / "config.toml")]
    repo = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, EVOSKILL_REMOTE="1",  # force local execution
               PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with open(log_file, "w") as lf:
        return subprocess.run(cmd, cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def best_program(root: Path) -> str:
    """Best frontier program as recorded by EvoSkill's ProgramManager."""
    from src.registry import ProgramManager  # EvoSkill package
    best = ProgramManager(cwd=root).get_best_from_frontier()
    return best or "base"


def export_skills(root: Path) -> tuple[str, dict[str, str]]:
    """EVOSKILL_EXPORT=best (default): skills of EvoSkill's best frontier program.
    EVOSKILL_EXPORT=frontier_union: union of skills over all frontier programs
    (lets the harness Gate, not EvoSkill's frontier, make the admission call)."""
    mode = os.environ.get("EVOSKILL_EXPORT", "best")
    if mode == "frontier_union":
        from src.registry import ProgramManager
        progs = ProgramManager(cwd=root).get_frontier() or ["base"]
        merged: dict[str, str] = {}
        for p in progs:
            for k, v in read_skills(root, p).items():
                merged.setdefault(k, v)
        return "+".join(progs), merged
    prog = best_program(root)
    return prog, read_skills(root, prog)


def read_skills(root: Path, program: str) -> dict[str, str]:
    """name -> concatenated content of .claude/skills/<name>/ on program/<program>."""
    ref = f"program/{program}"
    try:
        files = [f for f in _git(root, "ls-tree", "-r", "--name-only", ref, ".claude/skills").splitlines() if f]
    except subprocess.CalledProcessError:
        return {}
    out: dict[str, list[str]] = {}
    for f in sorted(files, key=lambda p: (not p.endswith("SKILL.md"), p)):
        parts = f.split("/")
        if len(parts) < 4:
            continue
        name = parts[2]
        if name == "skill-creator":  # EvoSkill's bundled meta-skill, not a learned skill
            continue
        body = _git(root, "show", f"{ref}:{f}")
        out.setdefault(name, []).append(body if f.endswith("SKILL.md") else f"\n\n## file: {'/'.join(parts[3:])}\n{body}")
    return {k: "".join(v) for k, v in out.items()}


def cache_tokens(root: Path) -> int:
    total = 0
    for p in (root / ".cache" / "runs").glob("*/*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        u = d.get("usage") or d.get("trace", {}).get("usage") or {}
        total += sum(int(u.get(k, 0) or 0) for k in
                     ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens"))
    return total


def data_dirs_for(bench, rows: list[dict], out: Path) -> list[str]:
    """Corpus dir handed to EvoSkill's agents. If the benchmark supports date
    narrowing (officeqa_v2), expose only the date-candidate bulletins for these
    rows (symlinks), mirroring the harness retrieval scope and bounding cost."""
    corpus = getattr(bench, "corpus_dir", None)
    if not corpus:
        return []
    corpus = Path(corpus).resolve()
    by_file = getattr(bench, "_by_file", None)
    if by_file is None:
        return [str(corpus)]
    from harness.benchmarks.officeqa_v2 import date_candidates
    names = set()
    for r in rows:
        names |= set(date_candidates(r["question"], list(by_file)))
    if not names:
        return [str(corpus)]
    out.mkdir(parents=True, exist_ok=True)
    for n in sorted(names):
        link = out / n
        if not link.exists():
            link.symlink_to(corpus / n)
    return [str(out.resolve())]


def _cache_key(rows: list[dict], model: str) -> str:
    import hashlib
    keys = sorted(k for k in os.environ if k.startswith("EVOSKILL_") and k not in ("EVOSKILL_WORKDIR", "EVOSKILL_REUSE"))
    blob = json.dumps([rows, model, [(k, os.environ[k]) for k in keys]], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def native_split_counts(log_text: str) -> dict:
    m = re.search(r"\((\d+) val samples\)", log_text)
    return {"val": int(m.group(1)) if m else None}


@register("learner", "evoskill")
class EvoSkillLearner(Learner):
    def __init__(self, base_model: str, benchmark, task_lookup: dict, seed: int, logger=None,
                 workdir: str | None = None):
        self.base_model, self.bench, self.task_lookup = base_model, benchmark, task_lookup
        self.seed, self.logger = seed, logger
        self.workdir = Path(workdir or os.environ.get("EVOSKILL_WORKDIR", "logs/evoskill_runs"))
        self.last_build_tokens = 0
        self.last_stats: dict = {}

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.cfg.base_model, ctx.benchmark, ctx.task_lookup, ctx.cfg.seed, ctx.logger)

    def traces_to_rows(self, traces: list[Trace]) -> list[dict]:
        rows = []
        for t in traces:
            task = self.task_lookup[t.task_id]
            if hasattr(self.bench, "evoskill_row"):  # benchmark-specific native mapping
                rows.append({"uid": t.task_id, **self.bench.evoskill_row(task, t)}); continue
            q = getattr(self.bench, "question", None)
            rows.append({"uid": t.task_id,
                         "question": q(task) if callable(q) else task,
                         "answer": self.bench.answer(task),
                         "category": self.bench.category(task) if hasattr(self.bench, "category") else "default"})
        return rows

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        rows = self.traces_to_rows(traces)
        model = os.environ.get("EVOSKILL_MODEL", f"openai/{self.base_model}")
        key = _cache_key(rows, model)
        # EVOSKILL_REUSE=1: reuse a finished EvoSkill project with identical input
        # rows + model + EVOSKILL_* settings (e.g. both gates share one induction).
        reuse = None
        if os.environ.get("EVOSKILL_REUSE") == "1":
            for d in sorted(self.workdir.glob(f"*_{key}"), reverse=True):
                if (d / ".evoskill" / "done").exists():
                    reuse = d.resolve(); break
        root = reuse or (self.workdir / f"seed{self.seed}_{int(time.time())}_{key}").resolve()
        log_file = root / "evoskill_stdout.log"
        if reuse is None:
            write_project(root, rows, data_dirs_for(self.bench, rows, root.parent / f"data_{key}"), model,
                          os.environ.get("EVOSKILL_HARNESS", "openhands"), bench=self.bench)
            rc = run_evoskill(root, log_file)
            if rc == 0:
                (root / ".evoskill" / "done").write_text("ok")
        else:
            rc = 0
        log_text = log_file.read_text(errors="ignore")
        if rc != 0:
            raise RuntimeError(f"EvoSkill exited {rc}; see {log_file}")
        prog, skills_raw = export_skills(root)
        version = 1 + max((s.version for s in current.skills), default=0)
        prov = [t.task_id for t in traces]
        skills = [Skill(id=f"evoskill_{name}_v{version}", content=content, representation="skill_md",
                        provenance=list(prov), version=version) for name, content in sorted(skills_raw.items())]
        self.last_build_tokens = cache_tokens(root)
        self.last_stats = {"project": str(root), "traces_in": len(traces), "rows_written": len(rows),
                           "best_program": prog, "skills_out": len(skills), "reused": reuse is not None, **native_split_counts(log_text)}
        if self.logger:
            self.logger.log("learner.induce", learner="evoskill", tokens=self.last_build_tokens,
                            stats=self.last_stats, skill_ids=[s.id for s in skills])
        return skills
