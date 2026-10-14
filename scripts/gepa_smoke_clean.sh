#!/usr/bin/env bash
# GEPA live smoke on the CLEAN Track A split: gepa x {noop, bayesian}, Qwen-72B consumer + reflection LM,
# OpenAI embeddings, 1 seed, GEPA_MAX_METRIC_CALLS=80, gepa_valset = 2/type carved from the learn corpus (12).
# Detached-safe: skip-if-done per cell; OpenRouter hard stop (HARDSTOP_USD) kills this process group (no pgrep).
# Launch: setsid nohup bash scripts/gepa_smoke_clean.sh >/dev/null 2>&1 &
set -u; cd "$(dirname "$0")/.."
OUT=logs/scale_clean_gepa; LOG=$OUT/gepa.out; mkdir -p $OUT
export PYTHONPATH=.:/workspace/skill-harness-gepa/.gepa-pkgs SCALE_MANIFEST=manifests/alfworld_scale_clean SCALE_OUT=$OUT
export SCALE_BUDGET_USD=${SCALE_BUDGET_USD:-8.3} HARNESS_EMBEDDING=openai:text-embedding-3-small
export GEPA_MAX_METRIC_CALLS=${GEPA_MAX_METRIC_CALLS:-80} GEPA_VALSET_PER_CATEGORY=2
PY=${PY:-/workspace/skill-harness/.venv/bin/python}; HARDSTOP_USD=${HARDSTOP_USD:-8.5}; PGID=$(ps -o pgid= $$ | tr -d ' ')
usage() { python3 -c "
import json,os,urllib.request
r=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
print(json.load(urllib.request.urlopen(r,timeout=30))['data']['usage'])" 2>/dev/null; }
( while true; do u=$(usage); echo "$(date +%T) usage=$u" >> $OUT/watchdog.out
    if python3 -c "import sys;sys.exit(0 if float('${u:-0}')>$HARDSTOP_USD else 1)"; then
      echo "$(date +%T) HARDSTOP usage=$u" | tee -a $OUT/watchdog.out >> $LOG; kill -TERM -- -$PGID; fi
    sleep 60; done ) &
WD=$!
echo "$(date) start pid=$$ pgid=$PGID usage=$(usage)" >> $LOG
cell() { [ -f $OUT/$1.result.json ] && { echo "skip $1" >> $LOG; return; }
  rm -f $OUT/$1.jsonl; echo "$(date +%T) run $1" >> $LOG
  $PY scripts/scale_run.py cell "$@" >> $LOG 2>&1 || echo "FAIL $1" >> $LOG; }
cell A_gepa_noop gepa noop skill_md 3 Q c1:corpora/scale_Q
cell A_gepa_bayesian gepa bayesian skill_md 3 Q c1:corpora/scale_Q
echo "$(date) END usage=$(usage)" >> $LOG; kill $WD 2>/dev/null; echo GEPA_DONE >> $LOG
