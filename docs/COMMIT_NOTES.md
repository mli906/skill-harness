# Commit notes (corrections to pushed commit messages)

History is not rewritten; these notes record what two already-pushed commits actually contain.

## 7671940 "Track A re-run on clean split: RESULTS.md + run_manifest.json"
The message is wrong. The commit contains ONLY `manifests/alfworld_scale_clean/tasks.json`.
`scripts/scale_report.py` crashed (FileNotFoundError on `corpora/scale_M/manifest.json`, a corpus that a
Track-A-only run never has), so neither RESULTS.md nor run_manifest.json was written. The real
`RESULTS.md` + `run_manifest.json` (13/13 cells, incl. the AWM x bayesian rerun) landed in **9ce4172**,
after the report writer was made to degrade gracefully in **df1870b**.

## 9da39aa "GEPA live smoke (clean split): A_gepa_noop + A_gepa_bayesian results, summary report"
The message is wrong. It has the A_gepa_noop result (`logs/scale_clean_gepa/A_gepa_noop.result.json`,
gepa.out, watchdog.out, gepa_valset.json, launch scripts) but NO A_gepa_bayesian result: that cell had not
completed and no bayesian test result exists anywhere. Partial bayesian logs (kept as an honest record, marked
incomplete) were added later in **d0ee54c** with `logs/scale_clean_gepa/README.md`. A false "GEPA_DONE"
marker was dropped in cb186f9.

## GEPA live smoke outcome (clean Track A split, Qwen-72B, 1 seed)
* `gepa x noop`: test success **0.458** (identical to the clean no-skill baseline), flips **+2 / -2**,
  **1.72M** build tokens. GEPA never evolved the seed skill (valset stayed 0.5; no candidate strictly better).
* `gepa x bayesian`: **incomplete**. First attempt finished induction (2 candidates, valset best 0.667) but
  logged no gate/test result; the relaunch was stopped by the user after a hang at GEPA iteration 6
  (valset 0.583 -> 0.667). No result is claimed.
* `corpora/scale_Q` (frozen Qwen corpus) is intentionally not tracked, like the other `corpora/scale_*`.
