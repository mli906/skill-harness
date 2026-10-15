#!/usr/bin/env bash
# bootstrap.sh - reconstruct the skill-harness environment on a bare machine. Idempotent; fails loudly.
#   ./bootstrap.sh                  full bootstrap (venv + lock + external repos + ALFWorld data)
#   SKIP_ALFWORLD_DATA=1 ./bootstrap.sh   skip the ~2-3 GB ALFWorld game-file download
#   SKIP_EXTERNAL=1 ./bootstrap.sh        skip git pre-clones of pinned external repos
# Env (all optional): PYTHON (default python3.13) VENV (default <repo>/.venv) HARNESS_CACHE (default ~/.cache/skill-harness)
#   TRACE2SKILL_HOME  ALFWORLD_DATA (default ~/.cache/alfworld)  PYTHON_ALLOW_OTHER=1 (accept a non-3.13 interpreter, >=3.12)
# Never prints secrets. Nothing here calls a model API.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
PYTHON=${PYTHON:-python3.13}
VENV=${VENV:-$ROOT/.venv}
export HARNESS_CACHE=${HARNESS_CACHE:-$HOME/.cache/skill-harness}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
say()  { printf '\n== %s\n' "$*"; }
die()  { printf 'BOOTSTRAP FAILED: %s\n' "$*" >&2; exit 1; }

say "1/5 system prerequisites"
missing=()
command -v git >/dev/null      || missing+=("git")
command -v "$PYTHON" >/dev/null || missing+=("$PYTHON (python 3.13; set PYTHON=/path/to/python3.13)")
# fast-downward-textworld (a textworld dependency) builds from source: needs a C/C++ toolchain + make.
{ command -v g++ >/dev/null || command -v c++ >/dev/null; } || missing+=("g++/c++ (apt: build-essential)")
command -v make >/dev/null     || missing+=("make (apt: build-essential)")
command -v curl >/dev/null || command -v wget >/dev/null || echo "note: neither curl nor wget found (not required; alfworld-download uses python requests)"
if [ ${#missing[@]} -gt 0 ]; then
  printf 'MISSING system prerequisites:\n'; printf '  - %s\n' "${missing[@]}"
  echo "Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y build-essential git python3.13 python3.13-venv"
  echo "No root/apt? Provide a wheel-only toolchain alternative: a prebuilt fast-downward-textworld==20.6.4 wheel, or conda-forge gxx_linux-64 + make."
  die "install the prerequisites above and re-run"
fi
pyver=$("$PYTHON" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')
case $pyver in
  3.13.*) echo "python $pyver (verified version family; dev env was 3.13.5)";;
  *) if [ "${PYTHON_ALLOW_OTHER:-0}" = 1 ] && "$PYTHON" -c 'import sys;sys.exit(sys.version_info<(3,12))'; then
       echo "WARNING: python $pyver is not the verified 3.13 family (lock requires >=3.12; untested)"
     else die "python $pyver found; the lock was verified on 3.13.x (>=3.12 needed). Set PYTHON=..., or PYTHON_ALLOW_OTHER=1 to try >=3.12"; fi;;
esac

say "2/5 virtualenv at $VENV"
if [ -x "$VENV/bin/python" ]; then
  have=$("$VENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  want=$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  [ "$have" = "$want" ] || die "existing $VENV is python $have, expected $want; remove it and re-run"
  echo "reusing existing venv"
else
  if command -v uv >/dev/null; then uv venv -q --python "$PYTHON" "$VENV"; else "$PYTHON" -m venv "$VENV" || die "python venv module missing (apt: python3.13-venv)"; fi
fi

# fast-downward-textworld's sdist build calls a bare `python`; put the venv first on PATH (a bare box may have only python3).
export PATH="$VENV/bin:$PATH"
say "3/5 install requirements.lock (pinned; evoskill is a git pin and needs github.com)"
if command -v uv >/dev/null; then uv pip install -q --python "$VENV/bin/python" -r requirements.lock
else "$VENV/bin/python" -m pip install -q -r requirements.lock; fi
"$VENV/bin/python" - <<'PY' || die "installed packages do not import / pins not satisfied"
import importlib.metadata as m, sys
want = {"agentdojo": "0.1.35", "gepa": "0.1.4", "alfworld": "0.4.2", "textworld": "1.7.0", "numpy": "2.3.5", "scipy": "1.18.1", "matplotlib": "3.11.2"}
bad = {k: m.version(k) for k, v in want.items() if m.version(k) != v}
assert not bad, bad
import harness, gepa, agentdojo, textworld, alfworld, scipy, numpy, yaml, openai  # noqa
m.version("evoskill"); import src.registry  # noqa  (evoskill dist installs the top-level package `src`)
print("key pins verified:", ", ".join(f"{k}=={v}" for k, v in want.items()))
PY

if [ "${SKIP_EXTERNAL:-0}" != 1 ]; then
say "4/5 pinned external repos (cache under HARNESS_CACHE=$HARNESS_CACHE)"
PYTHONPATH=$ROOT "$VENV/bin/python" - <<'PY' || die "could not fetch/verify a pinned external repo (needs github.com egress)"
import subprocess
from harness.benchmarks import _pinned as p
from harness.learners import trace2skill as t
def head(d): return subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
rows = []
d = p.checkout(p.EVOSKILL_REPO, p.EVOSKILL_SHA); rows.append(("EvoSkill", p.EVOSKILL_SHA, d))
d = p.checkout(p.OFFICEQA_REPO, p.OFFICEQA_SHA); rows.append(("databricks/officeqa", p.OFFICEQA_SHA, d))
d = t.checkout();                                rows.append(("Trace2Skill", t.TRACE2SKILL_SHA, d))
d = t.skill_creator().parents[1];                rows.append(("anthropics/skills", t.SKILL_CREATOR_SHA, d))
for name, sha, path in rows:
    h = head(path)
    assert h == sha, f"{name}: {path} is at {h}, expected {sha}"
    print(f"ok  {name:22s} {sha[:12]}  {path}")
assert (t.skill_creator() / "scripts" / "quick_validate.py").exists(), "skill-creator/scripts/quick_validate.py missing"
PY
else say "4/5 SKIP_EXTERNAL=1: external repos not pre-cloned (they will be fetched on first use)"; fi

if [ "${SKIP_ALFWORLD_DATA:-0}" != 1 ]; then
say "5/5 ALFWorld game files at ALFWORLD_DATA=$ALFWORLD_DATA"
n() { find "$ALFWORLD_DATA/json_2.1.1/$1" -name game.tw-pddl 2>/dev/null | wc -l; }
if [ "$(n train)" -gt 0 ] && [ "$(n valid_seen)" -gt 0 ] && [ -f "$ALFWORLD_DATA/logic/alfred.pddl" ]; then
  echo "already present (train=$(n train) valid_seen=$(n valid_seen) game files)"
else
  echo "downloading via alfworld-download (github.com release assets, ~2-3 GB; incl. a ~0.2 GB detector .pth the harness doesn't use)"
  "$VENV/bin/alfworld-download" --data-dir "$ALFWORLD_DATA" || die "alfworld-download failed (needs github.com + objects.githubusercontent.com egress)"
fi
[ "$(n train)" -gt 0 ] && [ "$(n valid_seen)" -gt 0 ] || die "ALFWorld game files still missing under $ALFWORLD_DATA/json_2.1.1/{train,valid_seen}"
echo "ok  ALFWorld games: train=$(n train) valid_seen=$(n valid_seen) valid_unseen=$(n valid_unseen)"
else say "5/5 SKIP_ALFWORLD_DATA=1: ALFWorld data not downloaded"; fi

cat <<MSG

Bootstrap complete.
  python:  $VENV/bin/python   (scripts honor PY=$VENV/bin/python; default is .venv/bin/python in the repo root)
  next:    set the env vars from CONFIG.md (keys are NOT handled here), then:  make smoke
MANUAL STEPS this script cannot automate: (1) API keys / HF_TOKEN (CONFIG.md); (2) the frozen EvoSkill project
  logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4 and corpora/scale_Q|scale_M are NOT in git (see PORTABILITY.md).
MSG
