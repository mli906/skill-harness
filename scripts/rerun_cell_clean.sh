#!/usr/bin/env bash
# Re-run ONE missing clean Track A cell (default A_awm_bayesian) on manifests/alfworld_scale_clean (split NOT regenerated).
# Detached-safe: skip-if-done; retries ONLY on network timeouts (CELL_RETRIES, default 3), removing that cell's partial
# log before each retry; OpenRouter hard stop at $HARDSTOP_USD kills this process group. No pgrep (no self-match).
# Launch: setsid nohup bash scripts/rerun_cell_clean.sh >/dev/null 2>&1 &
set -u; cd "$(dirname "$0")/.."
NAME=${CELL_NAME:-A_awm_bayesian}; ARGS=${CELL_ARGS:-"awm bayesian skill_md 3 Q c1:corpora/scale_Q"}
OUT=logs/scale_clean; LOG=$OUT/$NAME.rerun.out; mkdir -p $OUT
[ -f $OUT/$NAME.result.json ] && { echo "$(date) skip $NAME (done)" >> $LOG; exit 0; }
export PYTHONPATH=. SCALE_MANIFEST=manifests/alfworld_scale_clean SCALE_OUT=$OUT SCALE_BUDGET_USD=9.3
export HARNESS_EMBEDDING=openai:text-embedding-3-small AWM_SUCCESS_ONLY=1
export EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4 TRACE2SKILL_REUSE=1
PY=${PY:-/workspace/skill-harness/.venv/bin/python}; HARDSTOP_USD=${HARDSTOP_USD:-9.4}; PGID=$(ps -o pgid= $$ | tr -d ' ')
usage() { python3 -c "
import json,os,urllib.request
r=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
print(json.load(urllib.request.urlopen(r,timeout=30))['data']['usage'])" 2>/dev/null; }
( while true; do u=$(usage); echo "$(date +%T) usage=$u" >> $OUT/watchdog.out
    if python3 -c "import sys;sys.exit(0 if float('${u:-0}')>$HARDSTOP_USD else 1)"; then
      echo "$(date +%T) HARDSTOP usage=$u" | tee -a $OUT/watchdog.out >> $LOG; kill -TERM -- -$PGID; fi
    sleep 60; done ) &
WD=$!
echo "$(date) start $NAME pid=$$ pgid=$PGID usage=$(usage)" >> $LOG
for i in $(seq 1 ${CELL_RETRIES:-3}); do
  rm -f $OUT/$NAME.jsonl; T=$(mktemp)
  $PY scripts/scale_run.py cell $NAME $ARGS > $T 2>&1; rc=$?; cat $T >> $LOG
  if [ $rc -eq 0 ] && [ -f $OUT/$NAME.result.json ]; then echo "$(date) OK attempt $i" >> $LOG; rm -f $T; break; fi
  if grep -qE "TimeoutError|timed out|ConnectionResetError|RemoteDisconnected|URLError" $T; then
    echo "$(date) network timeout on attempt $i; retrying" >> $LOG; rm -f $T; sleep 30; continue; fi
  echo "$(date) FAIL $NAME (non-network error, no retry)" >> $LOG; rm -f $T; break
done
echo "$(date) END usage=$(usage)" >> $LOG; kill $WD 2>/dev/null
