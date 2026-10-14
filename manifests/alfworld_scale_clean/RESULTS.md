# Scaled ALFWorld results (option 3: single seed, n=1 — no SD reported)

Split: `manifests/alfworld_scale_clean/split.json` (seed 20261013): test 24 / probe 24 / learn 24 / practice pool 48, 6 task types x 4. Flips are vs the empty-library run of the same consumer on the same 24 test tasks, same seed (in-cell baseline). Learner builds (EvoSkill, Trace2Skill, AWM) use `openai/gpt-4o` on the OpenAI key (flagged; not one of the two named models). Note: `unauthorized_actions` is not meaningful on ALFWorld and is omitted.

Clean split: task_ids t_5ce171db07, t_5d572fbd5d, t_b90a99fbeb, t_f1750d1af9 (frozen M3 skill origin) excluded from every set; skipped: t_5ce171db07.

## Summary (computed from the logs)

- n = 24 test tasks, 1 seed, 13 cells present.
- Significance: per-cell exact McNemar on flips vs the in-cell no-skill baseline (4 cells with any flip), Holm and BH across that family: **nothing significant after correction** (smallest raw p = 0.180; Holm-adjusted 0.719).
- Gated cells (non_regression / bayesian): 8; **all equal no-skill** — every gate rejected every candidate (library_size 0, 0/0 flips).
- `unauthorized_actions` counts actions outside the admissible-command list; not a safety signal on ALFWorld.

## Track A — frozen-corpus C1 (corpus: Qwen-72B, consumer: Qwen-72B, retrieval_k 3)

| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |
|---|---|---|---|---|---|---|---|---|
| noskill / noop | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 0 | 433,813 | 0 |
| noskill / non_regression | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 396,863 | 433,813 | 0 |
| noskill / bayesian | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 396,863 | 433,813 | 0 |
| raw_trace_rag / noop | 0.542 (base 0.458) | +4/-2 | 48 | 48 | 1.00 | 0 | 1,487,911 | 0 |
| awm / noop | 0.500 (base 0.458) | +4/-3 | 1 | 1 | 1.00 | 11,505 | 739,711 | 0 |
| awm / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,102,748 | 433,813 | 1 |
| awm / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 868,827 | 433,813 | 1 |
| trace2skill / noop | 0.667 (base 0.458) | +7/-2 | 1 | 1 | 1.00 | 493,781 | 950,021 | 0 |
| trace2skill / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,616,873 | 433,813 | 1 |
| trace2skill / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,616,873 | 433,813 | 1 |
| frozen_evoskill / noop | 0.375 (base 0.458) | +2/-4 | 1 | 1 | 0.00 | 0 | 624,742 | 0 |
| frozen_evoskill / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 888,983 | 433,813 | 1 |
| frozen_evoskill / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 888,983 | 433,813 | 1 |

_section skipped: Track B: no B_* cells in logs_


_section skipped: Transfer 2x2: no T_* cells in logs_


## Manifest

```json
{
 "split": "manifests/alfworld_scale_clean/split.json",
 "corpora": {
  "Q": {
   "model": "openrouter/qwen/qwen-2.5-72b-instruct",
   "success_failure": "25:23",
   "n_traces": 48,
   "sha256": "4dc1301b0a442a307fc58ba7f7ac02031c177f0605c63dd3b69ead6c7db7ae9a"
  }
 },
 "practice": {},
 "skill_counts": {
  "A_awm_bayesian": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_awm_non_regression": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_awm_noop": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_frozen_evoskill_bayesian": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_frozen_evoskill_non_regression": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_frozen_evoskill_noop": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_noskill_bayesian": {
   "candidates": 0,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_noskill_non_regression": {
   "candidates": 0,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_noskill_noop": {
   "candidates": 0,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_raw_trace_rag_noop": {
   "candidates": 48,
   "library": 48,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_trace2skill_bayesian": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_trace2skill_non_regression": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  },
  "A_trace2skill_noop": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "25/48",
   "probe_pass": "15/24"
  }
 },
 "skipped": [
  "section skipped: Track B: no B_* cells in logs",
  "section skipped: Transfer 2x2: no T_* cells in logs",
  "section skipped: practice Q: no Track B cell for this self-collector",
  "section skipped: corpus M: no cell in logs uses it",
  "section skipped: practice M: no Track B cell for this self-collector"
 ],
 "split_sha256": "5162e29841ae670ca43bdbd29336f10438d296d4dcba8846d4bb5fda6ab7db8f",
 "split_seed": 20261013,
 "split_exclude": [
  "t_5ce171db07",
  "t_5d572fbd5d",
  "t_b90a99fbeb",
  "t_f1750d1af9"
 ],
 "split_excluded_skipped": [
  {
   "game": "valid_seen/look_at_obj_in_light-Bowl-None-DeskLamp-316/trial_T20190908_230702_962704/game.tw-pddl",
   "task_id": "t_5ce171db07"
  }
 ],
 "pinned_test_ids": [
  "t_cfbd947ce0",
  "t_07b2f40abf",
  "t_4d61896076",
  "t_aca2bae211",
  "t_3e2562b5bb",
  "t_43b4af69e5",
  "t_fec77c954f",
  "t_02c9c1683e",
  "t_808bc1be80",
  "t_dadb03e9a1",
  "t_dab9bd6430",
  "t_16b39b44a5",
  "t_ab706e7a3d",
  "t_d5dc53999b",
  "t_7947984751",
  "t_2cac8c2718",
  "t_d1e2db9fec",
  "t_bb6b92a513",
  "t_bf5fdd7380",
  "t_ec3e21affa",
  "t_3d2b66e774",
  "t_266080ae64",
  "t_144ffd08b2",
  "t_373b96ca27"
 ],
 "pinned_probe_ids": [
  "t_bb8bf9939b",
  "t_b6f4ed1e7c",
  "t_c97993241d",
  "t_6350a51448",
  "t_6a90ad612e",
  "t_7bf0444e2b",
  "t_a18cdc37aa",
  "t_7d50bb2513",
  "t_5099e14c20",
  "t_81d0532444",
  "t_32015727d9",
  "t_0ff3fa60dc",
  "t_fa5c541a15",
  "t_9f074cdb84",
  "t_0d7c0f74cc",
  "t_0af9a5b84a",
  "t_1e8b41bd5f",
  "t_2298cb1b8c",
  "t_259a7297ba",
  "t_bba65f85ed",
  "t_e773d71f7b",
  "t_54820ae455",
  "t_8eefb50322",
  "t_b461aa1280"
 ]
}
```
