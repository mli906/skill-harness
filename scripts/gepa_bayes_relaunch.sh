#!/usr/bin/env bash
# Relaunch ONLY A_gepa_bayesian (clean split), detached. Watchdog hard-stops at $9.4 OpenRouter usage (own pgid only).
set -u; cd "$(dirname "$0")/.."
OUT=logs/scale_clean_gepa; LOG=$OUT/gepa.out
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} SCALE_MANIFEST=manifests/alfworld_scale_clean SCALE_OUT=$OUT
export SCALE_BUDGET_USD=${SCALE_BUDGET_USD:-9.3} HARNESS_EMBEDDING=openai:text-embedding-3-small
export GEPA_MAX_METRIC_CALLS=${GEPA_MAX_METRIC_CALLS:-80} GEPA_VALSET_PER_CATEGORY=2
PY=${PY:-.venv/bin/python}; [ -x "$PY" ] || { echo "$(date) ABORT: no python at $PY" >> $LOG; exit 1; }; HARDSTOP_USD=9.4; PGID=$(ps -o pgid= $$ | tr -d ' ')
usage() { python3 -c "
import json,os,urllib.request
r=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
print(json.load(urllib.request.urlopen(r,timeout=30))['data']['usage'])" 2>/dev/null; }
( while true; do u=$(usage); echo "$(date +%T) usage=$u" >> $OUT/watchdog.out
    if python3 -c "import sys;sys.exit(0 if float('${u:-0}')>$HARDSTOP_USD else 1)"; then
      echo "$(date +%T) HARDSTOP usage=$u" | tee -a $OUT/watchdog.out >> $LOG; kill -TERM -- -$PGID; fi
    sleep 60; done ) &
WD=$!
echo "$(date) relaunch A_gepa_bayesian pid=$$ pgid=$PGID usage=$(usage)" >> $LOG
rm -f $OUT/A_gepa_bayesian.jsonl
$PY scripts/scale_run.py cell A_gepa_bayesian gepa bayesian skill_md 3 Q c1:corpora/scale_Q >> $LOG 2>&1 || echo "FAIL A_gepa_bayesian" >> $LOG
kill $WD 2>/dev/null
echo "$(date) END usage=$(usage)" >> $LOG
[ -f $OUT/A_gepa_bayesian.result.json ] || { echo "$(date) ABORT: no bayesian result, not committing" >> $LOG; exit 1; }
mkdir -p reports/gepa_smoke_clean
REPORT_OUT=reports/gepa_smoke_clean $PY scripts/scale_report.py >> $LOG 2>&1 || echo "REPORT FAIL" >> $LOG
git add scripts/gepa_smoke_clean.sh scripts/gepa_bayes_relaunch.sh manifests/alfworld_scale_clean/gepa_valset.json reports/gepa_smoke_clean >> $LOG 2>&1
git add -f $OUT/*.result.json $OUT/gepa.out $OUT/watchdog.out >> $LOG 2>&1
git -c user.name=mli906 -c user.email=mli906@users.noreply.github.com commit -m "GEPA live smoke (clean split): A_gepa_noop + A_gepa_bayesian results, summary report" >> $LOG 2>&1
git push origin feat/gepa-live >> $LOG 2>&1 || echo "PUSH FAIL" >> $LOG
echo GEPA_DONE >> $LOG
