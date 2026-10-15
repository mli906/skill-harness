#!/usr/bin/env bash
# One-command smoke test: M0 end-to-end on the deterministic stub model (no network, no keys, no spend), then the full pytest suite.
# Usage: scripts/smoke.sh   (or: make smoke).  PY=/path/to/python overrides the interpreter (default: <repo>/.venv/bin/python).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "SMOKE FAILED: no interpreter at $PY (run ./bootstrap.sh, or set PY=)" >&2; exit 1; }
export PYTHONPATH=.
echo "== python: $($PY --version)"
echo "== 1/2 M0 end-to-end (stub model) with exact scorecard assertion"
$PY scripts/smoke_m0.py
echo "== 2/2 full pytest suite"
out=$($PY -m pytest -q -p no:cacheprovider 2>&1) || { echo "$out" | tail -30; echo "SMOKE FAILED: pytest" >&2; exit 1; }
echo "$out" | tail -4
summary=$(echo "$out" | tail -1)
EXPECT="137 passed, 1 xfailed"
case $summary in
  *"$EXPECT"*) echo "SMOKE OK: matches documented expectation ($EXPECT)";;
  *) echo "SMOKE OK (pytest green) but summary differs from documented '$EXPECT' - tests were added/removed since the doc; check TRANSFER.md";;
esac
