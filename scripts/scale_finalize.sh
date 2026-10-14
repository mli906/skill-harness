#!/usr/bin/env bash
# Follow-up: wait for scale_rerun_openai.sh, rerun any still-missing cells (skip-if-done) with the SAME
# embedder (OpenAI text-embedding-3-small) as every finished cell, then regenerate RESULTS.md/run_manifest.json and push main.
set -u; cd "$(dirname "$0")/.."; LOG=logs/scale/finalize.out
echo "$(date) waiting for scale_rerun_openai.sh" >> $LOG
while pgrep -f scale_rerun_openai.sh >/dev/null || pgrep -f "scale_run.py cell" >/dev/null; do sleep 60; done
export PYTHONPATH=. HARNESS_EMBEDDING=openai:text-embedding-3-small AWM_SUCCESS_ONLY=1 SCALE_BUDGET_USD=9.3
export EVOSKILL_MODEL=openai/gpt-4o EVOSKILL_ITERATIONS=1 EVOSKILL_FRONTIER=2 EVOSKILL_EXPORT=frontier_union EVOSKILL_REUSE=1
export EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4 TRACE2SKILL_REUSE=1
R=".venv/bin/python scripts/scale_run.py cell"; QC=c1:corpora/scale_Q; MC=c1:corpora/scale_M
cell() { [ -f logs/scale/$1.result.json ] && { echo "skip $1" >> $LOG; return; }
  grep -q HARDSTOP logs/scale/watchdog.out && [ "$(stat -c %Y logs/scale/watchdog.out)" -gt 0 ] && tail -1 logs/scale/watchdog.out | grep -q HARDSTOP && { echo "hardstop, skip $1" >> $LOG; return; }
  echo "$(date +%T) run $1" >> $LOG; $R "$@" >> $LOG 2>&1 || echo "FAIL $1" >> $LOG; }
# AWM ctx fix (fallback): keep Qwen-72B but steer OpenRouter away from DeepInfra's 32k-input endpoint
for g in noop non_regression bayesian; do LLM_EXTRA_BODY='{"provider":{"ignore":["DeepInfra"]}}' cell A_awm_$g awm $g skill_md 3 Q $QC; done
for g in noop non_regression bayesian; do cell A_trace2skill_$g trace2skill $g skill_md 3 Q $QC; cell A_frozen_evoskill_$g evoskill_frozen $g skill_md 3 Q $QC; cell A_noskill_$g noskill $g skill_md 3 Q $QC; done
cell A_raw_trace_rag_noop raw_trace_rag noop raw_trace 3 Q $QC
cell T_noskill_M noskill noop skill_md 3 M $MC
cell T_Mcorpus_on_Q trace2skill noop skill_md 3 Q $MC; cell T_Mcorpus_on_M trace2skill noop skill_md 3 M $MC; cell T_Qcorpus_on_M trace2skill noop skill_md 3 M $QC
for g in noop non_regression bayesian; do cell B_Q_evoskill_$g evoskill $g skill_md 3 Q c2:practice; cell B_M_evoskill_$g evoskill $g skill_md 3 M c2:practice; done
.venv/bin/python scripts/scale_report.py >> $LOG 2>&1
printf '\n\nEmbedder: all cells use OpenAI text-embedding-3-small (HARNESS_EMBEDDING=openai:...); hashed-embedder resume results superseded (logs/scale/superseded_hashed). AWM: AWM_SUCCESS_ONLY=1 (induce from successful traces only) to fit Qwen-72B 32k context.\n' >> manifests/alfworld_scale/RESULTS.md
git add manifests/alfworld_scale scripts/scale_*.sh scripts/scale_report.py >> $LOG 2>&1
git -c user.name=mli906 -c user.email=mli906@users.noreply.github.com commit -m "Scaled ALFWorld results (1 seed, OpenAI embeddings for all cells)" >> $LOG 2>&1
git push origin main >> $LOG 2>&1; echo "$(date) FINALIZE_DONE" >> $LOG
