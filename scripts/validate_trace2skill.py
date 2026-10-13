"""Standalone vs wrapped structural validation for Trace2Skill.

Standalone: the pinned Trace2Skill checkout run with its OWN documented CLIs in
a clean venv (no harness, no shim) on the same agent-log directory the wrapped
learner produced. Wrapped: the harness Trace2SkillLearner output. Compares
structure (skill dir files, SKILL.md frontmatter, quick_validate verdict,
section counts) -- not text, since the LLM output is sampled.

    python scripts/validate_trace2skill.py --wrapped-work logs/trace2skill/<key> --python /workspace/.t2s_venv/bin/python
"""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.learners.trace2skill import checkout, skill_creator, INITIAL_SKILL, SKILL_NAME


def structure(d: Path, qv: Path, py: str) -> dict:
    sk = (d / "SKILL.md").read_text()
    fm = re.match(r"^---\n(.*?)\n---", sk, re.S)
    r = subprocess.run([py, str(qv), str(d)], capture_output=True, text=True)
    return {"files": sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()),
            "frontmatter_keys": sorted(re.findall(r"^(\w[\w-]*):", fm.group(1), re.M)) if fm else None,
            "name": (re.search(r"^name:\s*(.+)$", fm.group(1), re.M) or [None, None])[1] if fm else None,
            "quick_validate_ok": r.returncode == 0, "quick_validate_msg": (r.stdout or r.stderr).strip()[:200],
            "h2_sections": len(re.findall(r"^## ", sk, re.M)), "lines": sk.count("\n") + 1,
            "grew_from_draft": len(sk) > len(INITIAL_SKILL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapped-work", required=True)
    ap.add_argument("--python", required=True, help="clean venv python with Trace2Skill's README deps")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out", default="logs/trace2skill_standalone")
    a = ap.parse_args()
    root, wrapped = checkout(), Path(a.wrapped_work).resolve()
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True); (out / "skills" / SKILL_NAME).mkdir(parents=True)
    (out / "skills" / SKILL_NAME / "SKILL.md").write_text(INITIAL_SKILL)
    (out / "skills" / "skill-creator").symlink_to(skill_creator())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    api = ["--model", a.model, "--base_url", "https://api.openai.com/v1", "--seed", "0"]
    def run(args, cwd):
        subprocess.run([a.python, *args], cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    run(["analysis/run_error_analysis_llm.py", "--logs_dir", str(wrapped / "logs"), "--output_dir", str(out / "ea"), *api], root)
    run(["analysis/run_success_analysis_llm.py", "--logs_dir", str(wrapped / "logs"), "--output_dir", str(out / "sa"), *api], root)
    run(["analysis/parse_error_analysis_outputs.py", "--input_dir", str(out / "ea"), "--output", str(out / "e.json")], root)
    run(["analysis/parse_success_analysis_outputs.py", "--input_dir", str(out / "sa"), "--output", str(out / "s.json")], root)
    env["PYTHONPATH"] = str(root)
    run(["-m", "skill_evolver.run_parallel_combined_skill_evolution", "--error-json", str(out / "e.json"),
         "--success-json", str(out / "s.json"), "--skill-dir", str(out / "skills" / SKILL_NAME), "--model", a.model,
         "--base-url", "https://api.openai.com/v1", "--seed", "0", "--output-dir", str(out / "final_skill"),
         "--parse-failure-dir", str(out / "pf")], out)
    qv = skill_creator() / "scripts" / "quick_validate.py"
    res = {"standalone": structure(out / "final_skill", qv, a.python),
           "wrapped": structure(wrapped / "final_skill", qv, a.python),
           "n_records": {"standalone": [len(json.load(open(out / f))) for f in ("e.json", "s.json")],
                         "wrapped": [len(json.load(open(wrapped / f))) for f in ("error_parsed.json", "success_parsed.json")]}}
    s, w = res["standalone"], res["wrapped"]
    res["structurally_equivalent"] = (s["files"] == w["files"] and s["frontmatter_keys"] == w["frontmatter_keys"]
                                      and s["name"] == w["name"] and s["quick_validate_ok"] and w["quick_validate_ok"]
                                      and s["grew_from_draft"] and w["grew_from_draft"])
    print(json.dumps(res, indent=2))
    (out / "validation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
