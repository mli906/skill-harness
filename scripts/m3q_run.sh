#!/usr/bin/env bash
# M3 ALFWorld rerun with the consuming agent on an OpenRouter Qwen model (learners unchanged: EvoSkill on gpt-4o).
set -u
cd "$(dirname "$0")/.."
export ALFWORLD_TYPES=pick_and_place_simple,look_at_obj_in_light ALFWORLD_TRAIN_PER_CAT=4 ALFWORLD_LIMIT=14
export EVOSKILL_FROZEN_PROJECT=${EVOSKILL_FROZEN_PROJECT:-logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4}  # M3 gpt-4o-mini-trace library
export EVOSKILL_MODEL=openai/gpt-4o EVOSKILL_ITERATIONS=1 EVOSKILL_FRONTIER=2 EVOSKILL_EXPORT=frontier_union EVOSKILL_REUSE=1
for c in ${@:-noskill evoskill_noop rawrag}; do
  .venv/bin/python -m harness.alfworld_run --config examples/m3q_alfworld_$c.yaml > logs/m3q_${c}_stdout.txt 2>&1; echo "$c EXIT $?" | tee -a logs/m3q_${c}_stdout.txt
done
