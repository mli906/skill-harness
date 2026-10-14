#!/usr/bin/env bash
# Rerun affected cells after OpenAI top-up: real OpenAI embeddings everywhere; AWM on successful traces only (ctx fix).
cd "$(dirname "$0")/.."
export HARNESS_EMBEDDING=openai:text-embedding-3-small AWM_SUCCESS_ONLY=1
bash scripts/scale_all.sh & 
while pgrep -f scale_all.sh >/dev/null; do
  u=$(python3 -c "
import json,os,urllib.request
r=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']})
print(json.load(urllib.request.urlopen(r,timeout=30))['data']['usage'])" 2>/dev/null)
  echo "$(date +%T) $u" >> logs/scale/watchdog.out
  python3 -c "import sys;sys.exit(0 if float('${u:-0}')>9.4 else 1)" && { pkill -f scale_run.py; pkill -f scale_all.sh; echo HARDSTOP >> logs/scale/watchdog.out; }
  sleep 60
done; echo RERUN_DONE >> logs/scale/watchdog.out
