#!/usr/bin/env bash
# Track A re-run on the CLEAN scaled split (manifests/alfworld_scale_clean; M3 origin ids excluded).
# Same config as the live scaled run: Qwen-72B consumer, OpenAI text-embedding-3-small, AWM_SUCCESS_ONLY=1,
# 1 seed, raw_trace_rag noop only. Detached-safe: skip-if-done per cell; waits for the live run to exit;
# OpenRouter hard stop at $HARDSTOP_USD (key usage, /api/v1/key) kills this process group.
# Launch: setsid nohup bash scripts/trackA_clean.sh >/dev/null 2>&1 &
set -u; cd "$(dirname "$0")/.."
OUT=logs/scale_clean; LOG=$OUT/trackA.out; mkdir -p $OUT
grep -q TRACKA_DONE $LOG 2>/dev/null && exit 0
export PYTHONPATH=. SCALE_MANIFEST=manifests/alfworld_scale_clean SCALE_OUT=$OUT SCALE_BUDGET_USD=9.3
export HARNESS_EMBEDDING=openai:text-embedding-3-small AWM_SUCCESS_ONLY=1
export EVOSKILL_MODEL=openai/gpt-4o EVOSKILL_ITERATIONS=1 EVOSKILL_FRONTIER=2 EVOSKILL_EXPORT=frontier_union EVOSKILL_REUSE=1
export EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4 TRACE2SKILL_REUSE=1
PY=${PY:-/workspace/skill-harness/.venv/bin/python}; HARDSTOP_USD=${HARDSTOP_USD:-9.4}; PGID=$(ps -o pgid= $$ | tr -d ' ')
echo "$(date) start pid=$$ pgid=$PGID; waiting for live run" >> $LOG
while pgrep -f "python.* scripts/scale_run.py cell|bash .*scripts/scale_(rerun_openai|finalize)\.sh" >/dev/null; do sleep 60; done
usage() { python3 -c "
import json,os,urllib.request
r=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
print(json.load(urllib.request.urlopen(r,timeout=30))['data']['usage'])" 2>/dev/null; }
( while true; do u=$(usage); echo "$(date +%T) usage=$u" >> $OUT/watchdog.out
    if python3 -c "import sys;sys.exit(0 if float('${u:-0}')>$HARDSTOP_USD else 1)"; then
      echo "$(date +%T) HARDSTOP usage=$u" | tee -a $OUT/watchdog.out >> $LOG; kill -TERM -- -$PGID; fi
    sleep 60; done ) &
WD=$!
echo "$(date) live run clear; usage at start $(usage)" >> $LOG
[ -f $SCALE_MANIFEST/tasks.json ] || { echo "$(date +%T) render" >> $LOG; $PY scripts/scale_run.py render >> $LOG 2>&1 || { echo FAIL render >> $LOG; kill $WD; exit 1; }; }
R="$PY scripts/scale_run.py cell"; QC=c1:corpora/scale_Q
cell() { [ -f $OUT/$1.result.json ] && { echo "skip $1" >> $LOG; return; }
  echo "$(date +%T) run $1" >> $LOG; $R "$@" >> $LOG 2>&1 || echo "FAIL $1" >> $LOG; }
cell A_noskill_noop noskill noop skill_md 3 Q $QC
( for g in noop non_regression bayesian; do cell A_trace2skill_$g trace2skill $g skill_md 3 Q $QC; done ) &
( for g in noop non_regression bayesian; do LLM_EXTRA_BODY='{"provider":{"ignore":["DeepInfra"]}}' cell A_awm_$g awm $g skill_md 3 Q $QC; done ) &
( for g in noop non_regression bayesian; do cell A_frozen_evoskill_$g evoskill_frozen $g skill_md 3 Q $QC; done ) &
( cell A_raw_trace_rag_noop raw_trace_rag noop raw_trace 3 Q $QC
  for g in non_regression bayesian; do cell A_noskill_$g noskill $g skill_md 3 Q $QC; done ) &
wait $(jobs -p | grep -v "^$WD$")
echo "$(date) cells finished; usage $(usage)" >> $LOG
$PY scripts/scale_report.py >> $LOG 2>&1
( cd /workspace/skill-harness-analysis && PYTHONPATH=/workspace/.analysis-pkgs:. $PY -m harness.analysis \
    --runs /workspace/skill-harness-leak/$OUT --out /workspace/analysis-out/trackA_clean ) >> $LOG 2>&1
git add $SCALE_MANIFEST >> $LOG 2>&1
git -c user.name=mli906 -c user.email=mli906@users.noreply.github.com commit -m "Track A re-run on clean split: RESULTS.md + run_manifest.json" >> $LOG 2>&1
git push origin feat/leak-invariants >> $LOG 2>&1
kill $WD 2>/dev/null; echo "$(date) TRACKA_DONE" >> $LOG
