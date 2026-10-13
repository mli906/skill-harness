# Scaled ALFWorld run: prep + cost gate (STOPPED before expensive runs; estimate > $5)

- Split: `split.json` (seed 20261013, `scripts/scale_split.py`, no LLM). 6 task types, 4/type each:
  test 24 (valid_seen), probe 24 (train split, held out), learn_corpus 24 (train), practice_pool 48 (train, 8/type, Track B prefilter). Pairwise disjoint (asserted). Train remainder: 3,529 games.
- Corpora (planned): Q = qwen/qwen-2.5-72b-instruct, M = openai/gpt-4o-mini (both via OpenRouter); 24 learn tasks x k=2, temp 0.7 => 48 traces each; ratio logged in the corpus manifest.
- Track A (C1): consumer Q, corpus Q; learners noskill, raw_trace_rag, awm, trace2skill, frozen_evoskill (`evoskill_frozen`, M3 library) x gates noop/non_regression/bayesian; retrieval_k 3; 3 seeds.
- Track B (C2, "agent-and-difficulty-coupled; not directly comparable to Track A"): prefilter practice_pool once per model, keep 24 tasks giving ~50% pass; EvoSkill x 3 gates x {Q, M}; EvoSkill build/inner agent model stays openai/gpt-4o (flagged).
- Transfer 2x2: trace2skill (noop gate) on {M corpus, Q corpus} x consumer {M, Q} + 2 no-skill baselines; 3 seeds.
- Cost: `cost_estimate_steps{18,22}.json` (`scripts/scale_estimate.py`).
