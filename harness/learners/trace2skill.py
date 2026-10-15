"""Trace2SkillLearner (`learner: trace2skill`, contract C1_fixed_traces).

Wraps github.com/Qwen-Applications/Trace2Skill pinned at TRACE2SKILL_SHA
(cloned to ~/.cache/skill-harness/trace2skill/<sha>, never forked/edited; run
through harness.learners.trace2skill_compat, a runtime shim that only adds
sys.path + token accounting).

Mapping (native input = per-trajectory agent logs named
`<prefix>_<instance>_<SUCCEED|FAILED>.md`, the format its analysts read):
  corpus trace -> one markdown agent log: task (first user message), each step
  (observation / thought / action / result), the final output artifacts, the
  env outcome label, and the gold info (expert walkthrough) in a clearly
  labelled reference section (Trace2Skill's own agentic error analysis compares
  against ground truth; its LLM-only analyst reads the log).
Pipeline, all Trace2Skill entrypoints unchanged:
  1. analysis/run_error_analysis_llm.py   (FAILED logs -> failure analyst reports)
  2. analysis/run_success_analysis_llm.py (SUCCEED logs -> success analyst reports)
  3. analysis/parse_{error,success}_analysis_outputs.py
  4. python -m skill_evolver.run_parallel_combined_skill_evolution  (parallel
     patch proposal + hierarchical merge) on a weak initial SKILL.md draft
     ("skill creation from scratch" mode).
Output skill dir -> one Skill (representation skill_md, content = SKILL.md +
any extra files, provenance = corpus task_ids analysed).
Env: TRACE2SKILL_MODEL (default gpt-4o), TRACE2SKILL_WORKERS (4),
TRACE2SKILL_MODEL=openrouter/<id> routes the build via OpenRouter (build_routing),
TRACE2SKILL_REUSE=1 (reuse a finished workdir keyed on corpus hash+seed+model),
TRACE2SKILL_WORKDIR (default logs/trace2skill).
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from harness.interfaces.learner import Learner, C1_FIXED_TRACES
from harness.registry import register
from harness.types import Trace, Skill, SkillLibrary

TRACE2SKILL_REPO = "https://github.com/Qwen-Applications/Trace2Skill"
TRACE2SKILL_SHA = "3d0b52a140f002a512930252b613c49048f7d5ac"
# Trace2Skill's evolver requires `skills/skill-creator/scripts/quick_validate.py`
# relative to its cwd (Anthropic's skill-creator skill, not shipped in the repo).
SKILL_CREATOR_REPO = "https://github.com/anthropics/skills"
SKILL_CREATOR_SHA = "dbd4588f9e1033efb41dad4bef2f7947c8993d44"
SKILL_NAME = "alfworld-household"
INITIAL_SKILL = f"""---
name: {SKILL_NAME}
description: Solve ALFWorld household text-game tasks (find, take, clean/heat/cool, place, examine objects).
---

# ALFWorld household tasks

Initial draft. Read the goal, find the needed object, and complete the task
with the admissible commands.
"""


def checkout() -> Path:
    root = Path(os.environ.get("TRACE2SKILL_HOME", Path.home() / ".cache" / "skill-harness" / "trace2skill")) / TRACE2SKILL_SHA
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", TRACE2SKILL_REPO, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", TRACE2SKILL_SHA], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != TRACE2SKILL_SHA:
        raise RuntimeError(f"Trace2Skill checkout at {head}, expected {TRACE2SKILL_SHA}")
    return root


def skill_creator() -> Path:
    root = Path(os.environ.get("HARNESS_CACHE", Path.home() / ".cache" / "skill-harness")) / "anthropic-skills" / SKILL_CREATOR_SHA
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", SKILL_CREATOR_REPO, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", SKILL_CREATOR_SHA], check=True)
    return root / "skills" / "skill-creator"


def instance_name(task_id: str, rollout: int) -> str:
    return f"{task_id.replace('_', '-')}-r{rollout}"   # no '_' : Trace2Skill parses the id after the last '_'


def render_log(task: str, trace: Trace, rollout: int = 0, gold: dict | None = None) -> str:
    out = [f"# Agent log: ALFWorld episode {instance_name(trace.task_id, rollout)}", "",
           "## User (task description)", "", task, ""]
    for i, s in enumerate(trace.steps, 1):
        th = (s.meta or {}).get("thought", "")
        out += [f"## Step {i}", f"Observation: {s.observation}", f"Assistant thought: {th}",
                f"Tool call (env command): {s.action}", f"Tool result: {s.tool_result}", ""]
    out += ["## Final result", f"Outcome (environment success flag): {'SUCCEED' if trace.outcome else 'FAILED'}",
            f"Failure mode: {trace.failure_mode}", f"Final output artifact: {trace.final_output}",
            f"Steps used: {len(trace.steps)}", ""]
    if gold:
        out += ["## Reference information (gold; not visible to the agent during the episode)",
                f"Goal: {gold.get('goal')}", f"Task type: {gold.get('task_type')}",
                f"Expert walkthrough: {' ; '.join(gold.get('expert_walkthrough') or [])}", ""]
    return "\n".join(out)


def read_skill_dir(d: Path) -> str:
    files = sorted((p for p in d.rglob("*") if p.is_file()), key=lambda p: (p.name != "SKILL.md", str(p)))
    parts = []
    for p in files:
        body = p.read_text(errors="replace")
        parts.append(body if p.name == "SKILL.md" else f"\n\n## file: {p.relative_to(d)}\n{body}")
    return "".join(parts)


@register("learner", "trace2skill")
class Trace2SkillLearner(Learner):
    contract = C1_FIXED_TRACES

    def __init__(self, seed: int, task_lookup: dict, logger=None, corpus_records=None):
        self.seed, self.task_lookup, self.logger = seed, task_lookup, logger
        self.corpus_records = corpus_records   # optional: rollout index + gold per trace
        self.last_build_tokens = 0
        self.model = os.environ.get("TRACE2SKILL_MODEL", "gpt-4o")
        from harness.learners.build_routing import resolve
        self.route = resolve(self.model)   # openrouter/<id> or $BUILD_BASE_URL; default OpenAI

    @classmethod
    def from_context(cls, ctx):
        recs = None
        cd = getattr(ctx, "corpus_dir", None) or os.environ.get("HARNESS_CORPUS_DIR")
        if cd:
            from harness import corpus
            recs = corpus.load(cd)[0]
        return cls(ctx.cfg.seed, ctx.task_lookup, ctx.logger, recs)

    # -- mapping -------------------------------------------------------------
    def write_logs(self, traces: list[Trace], logs_dir: Path) -> list[str]:
        logs_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, int] = {}
        meta = {}
        if self.corpus_records:
            for r in self.corpus_records:
                meta.setdefault(r["task_id"], []).append(r)
        names = []
        for t in traces:
            r_i = seen.get(t.task_id, 0); seen[t.task_id] = r_i + 1
            rec = (meta.get(t.task_id) or [None] * (r_i + 1))[r_i] if t.task_id in meta else None
            gold = rec["gold"] if rec else None
            name = instance_name(t.task_id, rec["rollout"] if rec else r_i)
            fn = logs_dir / f"alfworld_agent_{name}_{'SUCCEED' if t.outcome else 'FAILED'}.md"
            fn.write_text(render_log(self.task_lookup.get(t.task_id, t.task_id), t, r_i, gold))
            names.append(fn.name)
        return names

    def _run(self, root: Path, work: Path, tag: str, *args, cwd: Path | None = None) -> dict:
        usage = work / f"usage_{tag}.json"
        cmd = [sys.executable, "-m", "harness.learners.trace2skill_compat", str(root), str(usage), *args]
        if self.route.provider == "openai":   # unchanged default behaviour
            env = {**os.environ, "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")}
        else:
            env = self.route.subprocess_env()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        with open(work / f"{tag}.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=cwd or root, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"Trace2Skill step {tag} failed (rc={rc}); see {work / (tag + '.log')}")
        return json.loads(usage.read_text()) if usage.exists() else {}

    def induce(self, traces: list[Trace], current: SkillLibrary) -> list[Skill]:
        root = checkout()
        key = hashlib.sha256(json.dumps([[t.task_id, t.outcome, len(t.steps)] for t in traces]).encode()).hexdigest()[:12]
        work = Path(os.environ.get("TRACE2SKILL_WORKDIR", "logs/trace2skill")).resolve() / f"{key}_s{self.seed}_{self.model.replace('/', '-')}"
        final = work / "final_skill"
        stats_p = work / "stats.json"
        reuse = os.environ.get("TRACE2SKILL_REUSE") == "1" and (final / "SKILL.md").exists() and stats_p.exists()
        if reuse:
            stats = json.loads(stats_p.read_text()); stats["reused"] = True
        else:
            work.mkdir(parents=True, exist_ok=True)
            logs = work / "logs"
            names = self.write_logs(traces, logs)
            n_fail = sum(n.endswith("_FAILED.md") for n in names)
            n_succ = len(names) - n_fail
            api = ["--model", self.route.api_model, "--base_url", self.route.base_url, "--seed", str(self.seed)]
            w = os.environ.get("TRACE2SKILL_WORKERS", "4")
            usages = {}
            if n_fail:
                usages["error_analysis"] = self._run(root, work, "error_analysis", "analysis/run_error_analysis_llm.py",
                    "--logs_dir", str(logs), "--output_dir", str(work / "error_analysis"), "--max_workers", w, *api)
                self._run(root, work, "parse_error", "analysis/parse_error_analysis_outputs.py",
                          "--input_dir", str(work / "error_analysis"), "--output", str(work / "error_parsed.json"))
            else:
                (work / "error_parsed.json").write_text("[]")
            if n_succ:
                usages["success_analysis"] = self._run(root, work, "success_analysis", "analysis/run_success_analysis_llm.py",
                    "--logs_dir", str(logs), "--output_dir", str(work / "success_analysis"), "--max_workers", w, *api)
                self._run(root, work, "parse_success", "analysis/parse_success_analysis_outputs.py",
                          "--input_dir", str(work / "success_analysis"), "--output", str(work / "success_parsed.json"))
            else:
                (work / "success_parsed.json").write_text("[]")
            skill_dir = work / "skills" / SKILL_NAME
            if skill_dir.exists():
                import shutil; shutil.rmtree(work / "skills")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(INITIAL_SKILL)
            (work / "skills" / "skill-creator").symlink_to(skill_creator())
            if final.exists():
                import shutil; shutil.rmtree(final)
            usages["evolution"] = self._run(root, work, "evolution", "-m", "skill_evolver.run_parallel_combined_skill_evolution",
                "--error-json", str(work / "error_parsed.json"), "--success-json", str(work / "success_parsed.json"),
                "--skill-dir", str(skill_dir), "--model", self.route.api_model, "--base-url", self.route.base_url,
                "--seed", str(self.seed), "--max-workers", w, "--save-intermediates",
                "--intermediates-dir", str(work / "intermediates"), "--changelog", str(work / "change.log"),
                "--parse-failure-dir", str(work / "parse_failures"), "--output-dir", str(final), cwd=work)
            tok = sum(u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in usages.values())
            stats = {"workdir": str(work), "sha": TRACE2SKILL_SHA, "model": self.model, "logs": len(names),
                     "succeed_logs": n_succ, "failed_logs": n_fail, "usage": usages, "tokens": tok, "reused": False}
            stats_p.write_text(json.dumps(stats, indent=2))
        if not (final / "SKILL.md").exists():
            raise RuntimeError(f"Trace2Skill produced no SKILL.md in {final}")
        self.last_build_tokens = int(stats["tokens"])
        v = 1 + max((s.version for s in current.skills), default=0)
        skill = Skill(id=f"trace2skill_{SKILL_NAME}_v{v}", content=read_skill_dir(final), representation="skill_md",
                      provenance=sorted({t.task_id for t in traces}), version=v)
        if self.logger:
            self.logger.log("learner.induce", learner="trace2skill", tokens=self.last_build_tokens, stats=stats,
                            skill_ids=[skill.id], content=skill.content)
        return [skill]
