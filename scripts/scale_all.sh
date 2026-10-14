#!/usr/bin/env bash
# Scaled ALFWorld matrix (option 3: 1 seed; raw_trace_rag with noop only). Resumable (cells skip if result exists).
# Prereqs: scale_run.py render, corpus Q|M, prefilter Q|M.
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=.
export EVOSKILL_MODEL=openai/gpt-4o EVOSKILL_ITERATIONS=1 EVOSKILL_FRONTIER=2 EVOSKILL_EXPORT=frontier_union EVOSKILL_REUSE=1
export EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4 TRACE2SKILL_REUSE=1
R=".venv/bin/python scripts/scale_run.py cell"
cell() { $R "$@" >> logs/scale/lanes.out 2>&1 || echo "FAIL $1" >> logs/scale/lanes.out; }
QC=c1:corpora/scale_Q; MC=c1:corpora/scale_M
# phase 1: shared probe-empty + no-skill baselines per consumer
cell A_noskill_noop noskill noop skill_md 3 Q $QC & cell T_noskill_M noskill noop skill_md 3 M $MC & wait
grep -q "BUDGET STOP" logs/scale/lanes.out && exit 1
# phase 2 lanes
( for g in noop non_regression bayesian; do cell A_trace2skill_$g trace2skill $g skill_md 3 Q $QC; done ) &
( for g in noop non_regression bayesian; do cell A_awm_$g awm $g skill_md 3 Q $QC; done ) &
( for g in noop non_regression bayesian; do cell A_frozen_evoskill_$g evoskill_frozen $g skill_md 3 Q $QC; done ) &
( cell A_raw_trace_rag_noop raw_trace_rag noop raw_trace 3 Q $QC
  for g in non_regression bayesian; do cell A_noskill_$g noskill $g skill_md 3 Q $QC; done ) &
( cell T_Mcorpus_on_Q trace2skill noop skill_md 3 Q $MC; cell T_Mcorpus_on_M trace2skill noop skill_md 3 M $MC
  cell T_Qcorpus_on_M trace2skill noop skill_md 3 M $QC ) &
( for g in noop non_regression bayesian; do cell B_Q_evoskill_$g evoskill $g skill_md 3 Q c2:practice; done ) &
( for g in noop non_regression bayesian; do cell B_M_evoskill_$g evoskill $g skill_md 3 M c2:practice; done ) &
wait; echo ALLDONE >> logs/scale/lanes.out
