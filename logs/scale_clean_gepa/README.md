# GEPA live smoke on the clean Track A split (partial record)

Config: gepa x {noop, bayesian}, Qwen-72B consumer + reflection LM, OpenAI embeddings, 1 seed,
GEPA_MAX_METRIC_CALLS=80, gepa_valset = 12 tasks (2/type) carved from the learn corpus
(`manifests/alfworld_scale_clean/gepa_valset.json`). Split hash unchanged (split.json sha256 5162e298...).

* `A_gepa_noop.jsonl` / `A_gepa_noop.result.json` - COMPLETE. success 0.458 (= clean no-skill 0.458),
  flips +2/-2, 1.72M build tokens (reflection 32k + agent rollouts 1.69M), 81 rollouts. GEPA's valset
  score never moved (0.5): all reflective proposals were rejected/skipped, best candidate = seed skill.
* `A_gepa_bayesian.jsonl.partial1` - first bayesian attempt: GEPA induction finished (2 candidates,
  best_val 0.667, ~1.89M build tokens) but the cell has no gate.result / run.end / test result.
* `A_gepa_bayesian.jsonl` - relaunch attempt: STOPPED BY THE USER after a hang at GEPA iteration 6
  (valset 0.583 -> 0.667 selected program 1). NO gate decision, NO test result. Do not read it as a result.
* `gepa.out`, `watchdog.out` - raw stdout and OpenRouter-usage watchdog for all attempts.

GEPA bayesian cell: **incomplete**. No secrets in these files (scanned for sk-, or-v1-, Bearer, hf_ patterns).
`corpora/scale_Q` (frozen Qwen corpus used by Track A, 488K) is deliberately NOT committed here, consistent with
how the repo handled corpora/scale_* (untracked in every earlier scaled-run commit).
