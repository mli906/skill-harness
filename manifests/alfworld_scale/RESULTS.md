# Scaled ALFWorld results (option 3: single seed, n=1 — no SD reported)

Split: `split.json` (seed 20261013): test 24 / probe 24 / learn 24 / practice pool 48, 6 task types x 4. Flips are vs the empty-library run of the same consumer on the same 24 test tasks, same seed (in-cell baseline). Learner builds (EvoSkill, Trace2Skill, AWM) use `openai/gpt-4o` on the OpenAI key (flagged; not one of the two named models). Note: `unauthorized_actions` is not meaningful on ALFWorld and is omitted.

## Track A — frozen-corpus C1 (corpus: Qwen-72B, consumer: Qwen-72B, retrieval_k 3)

| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |
|---|---|---|---|---|---|---|---|---|
| noskill / noop | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 0 | 424,459 | 0 |
| noskill / non_regression | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 396,863 | 424,459 | 0 |
| noskill / bayesian | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 396,863 | 424,459 | 0 |
| raw_trace_rag / noop | 0.542 (base 0.458) | +4/-2 | 48 | 48 | 1.00 | 0 | 1,459,005 | 0 |
| awm / noop | 0.417 (base 0.458) | +2/-3 | 1 | 1 | 1.00 | 11,505 | 748,984 | 0 |
| awm / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,062,444 | 424,459 | 1 |
| awm / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 996,541 | 424,459 | 1 |
| trace2skill / noop | 0.708 (base 0.458) | +7/-1 | 1 | 1 | 1.00 | 493,781 | 901,823 | 0 |
| trace2skill / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,616,873 | 424,459 | 1 |
| trace2skill / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,616,873 | 424,459 | 1 |
| frozen_evoskill / noop | 0.417 (base 0.458) | +2/-3 | 1 | 1 | 0.00 | 0 | 596,205 | 0 |
| frozen_evoskill / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 888,983 | 424,459 | 1 |
| frozen_evoskill / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 888,983 | 424,459 | 1 |

## Track B — C2-native EvoSkill self-collection

**agent-and-difficulty-coupled; not directly comparable to Track A.** Practice sets prefiltered per self-collector (target ~50% pass). EvoSkill caveat: its ALFWorld use is open-loop (it proposes skills from collected traces; it does not act in the env during build). Skills built = candidates; 0-skill runs are a finding, not a failure.

| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |
|---|---|---|---|---|---|---|---|---|
| self-collector Qwen-72B / noop | 0.542 (base 0.458) | +3/-1 | 1 | 1 | 1.00 | 200,153 | 522,372 | 0 |
| self-collector Qwen-72B / non_regression | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,181,057 | 424,459 | 1 |
| self-collector Qwen-72B / bayesian | 0.458 (base 0.458) | +0/-0 | 1 | 0 | 1.00 | 1,181,057 | 424,459 | 1 |
| self-collector gpt-4o-mini / noop | 0.208 (base 0.208) | +1/-1 | 1 | 1 | 1.00 | 211,470 | 651,365 | 0 |
| self-collector gpt-4o-mini / non_regression | 0.208 (base 0.208) | +0/-0 | 1 | 0 | 1.00 | 1,415,932 | 499,628 | 1 |
| self-collector gpt-4o-mini / bayesian | 0.208 (base 0.208) | +0/-0 | 1 | 0 | 1.00 | 1,415,932 | 499,628 | 1 |

## Transfer 2x2 — trace2skill (noop gate), corpus x consumer

| cell | success_rate (n=1, 24 test) | flips w2c/c2w vs no-skill | candidates | library_size | auditability | tokens_build | tokens_eval | promotion_rejections |
|---|---|---|---|---|---|---|---|---|
| no-skill, consumer Qwen | 0.458 (base 0.458) | +0/-0 | 0 | 0 | 1.00 | 0 | 424,459 | 0 |
| no-skill, consumer gpt-4o-mini | 0.208 (base 0.208) | +0/-0 | 0 | 0 | 1.00 | 0 | 499,628 | 0 |
| Q corpus -> Qwen (same as Track A cell) | 0.708 (base 0.458) | +7/-1 | 1 | 1 | 1.00 | 493,781 | 901,823 | 0 |
| M corpus -> Qwen | 0.542 (base 0.458) | +4/-2 | 1 | 1 | 1.00 | 518,895 | 936,374 | 0 |
| M corpus -> gpt-4o-mini | 0.208 (base 0.208) | +1/-1 | 1 | 1 | 1.00 | 518,895 | 1,227,245 | 0 |
| Q corpus -> gpt-4o-mini | 0.250 (base 0.208) | +2/-1 | 1 | 1 | 1.00 | 493,781 | 1,146,781 | 0 |

## Manifest

```json
{
 "split": "manifests/alfworld_scale/split.json",
 "corpora": {
  "Q": {
   "model": "openrouter/qwen/qwen-2.5-72b-instruct",
   "success_failure": "25:23",
   "n_traces": 48,
   "sha256": "4dc1301b0a442a307fc58ba7f7ac02031c177f0605c63dd3b69ead6c7db7ae9a"
  },
  "M": {
   "model": "openrouter/openai/gpt-4o-mini",
   "success_failure": "15:33",
   "n_traces": 48,
   "sha256": "da5cc89c9579f2dbb81723795056d1da9f7e9d04bdf25007baaefe9417c8412f"
  }
 },
 "practice": {
  "Q": {
   "model": "openrouter/qwen/qwen-2.5-72b-instruct",
   "pool_pass_rate": 0.5625,
   "selected_pass_rate": 0.5
  },
  "M": {
   "model": "openrouter/openai/gpt-4o-mini",
   "pool_pass_rate": 0.2708333333333333,
   "selected_pass_rate": 0.2916666666666667
  }
 },
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
  },
  "B_M_evoskill_bayesian": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "7/24",
   "probe_pass": "7/24"
  },
  "B_M_evoskill_non_regression": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "7/24",
   "probe_pass": "7/24"
  },
  "B_M_evoskill_noop": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "7/24",
   "probe_pass": "7/24"
  },
  "B_Q_evoskill_bayesian": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "12/24",
   "probe_pass": "15/24"
  },
  "B_Q_evoskill_non_regression": {
   "candidates": 1,
   "library": 0,
   "learn_pass": "12/24",
   "probe_pass": "15/24"
  },
  "B_Q_evoskill_noop": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "12/24",
   "probe_pass": "15/24"
  },
  "T_Mcorpus_on_M": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "15/48",
   "probe_pass": "7/24"
  },
  "T_Mcorpus_on_Q": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "15/48",
   "probe_pass": "15/24"
  },
  "T_Qcorpus_on_M": {
   "candidates": 1,
   "library": 1,
   "learn_pass": "25/48",
   "probe_pass": "7/24"
  },
  "T_noskill_M": {
   "candidates": 0,
   "library": 0,
   "learn_pass": "15/48",
   "probe_pass": "7/24"
  }
 }
}
```


Embedder: all cells use OpenAI text-embedding-3-small (HARNESS_EMBEDDING=openai:...); hashed-embedder resume results superseded (logs/scale/superseded_hashed). AWM: AWM_SUCCESS_ONLY=1 (induce from successful traces only) to fit Qwen-72B 32k context.
