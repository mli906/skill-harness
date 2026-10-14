#!/usr/bin/env bash
# Resume after OpenAI credit exhaustion: cells needing no OpenAI build, with hashed embeddings (flagged deviation).
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=. HARNESS_EMBEDDING=hashed
export EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4
R=".venv/bin/python scripts/scale_run.py cell"; QC=c1:corpora/scale_Q
cell() { $R "$@" >> logs/scale/resume.out 2>&1 || echo "FAIL $1" >> logs/scale/resume.out; }
( for g in noop non_regression bayesian; do cell A_frozen_evoskill_$g evoskill_frozen $g skill_md 3 Q $QC; done ) &
( cell A_raw_trace_rag_noop raw_trace_rag noop raw_trace 3 Q $QC ) &
wait; echo ALLDONE >> logs/scale/resume.out
