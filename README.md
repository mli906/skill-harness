# skill-harness (M0)

Modular harness for comparing methods that turn agent execution traces into reusable skills.
Every stage is a swappable plugin behind one of five ABCs; `harness.runner.run(cfg) -> Scorecard` is the only entrypoint.

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m harness.run --config examples/m0.yaml                  # NoOpGate
.venv/bin/python -m harness.run --config examples/m0_nonregression.yaml    # NonRegressionGate
.venv/bin/python -m harness.run --config examples/m0_nonregression.yaml --seed 3
.venv/bin/python -m pytest -q                                              # exit test
```

## Models
* `base_model: stub` – deterministic, seeded offline model (used by the exit test). It simulates category-dependent errors, benefits from `PROCEDURE[...]` skills, and on some seeds the learner over-generalises a harmful `ANSWER IN UPPERCASE` rule, so skills can help *or* hurt.
* any other name (e.g. `gpt-4o-mini`) – **OpenAI is the default real backend**: `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for compatible servers). Every call is pinned to `base_model`, temperature 0, with a derived seed.
* Embeddings: `HARNESS_EMBEDDING=auto` (default) uses OpenAI `text-embedding-3-small` for real runs with a key, else a deterministic hashed bag-of-words. Force with `hashed` or `openai:<model>`.

## Layout
```
harness/types.py            canonical dataclasses (verbatim)
harness/interfaces/         one ABC per module: learner, gate, representation, agent, benchmark
harness/registry.py         @register(kind, name); subpackages auto-imported
harness/runner.py           run(cfg) — the 6 spec steps
harness/run.py              CLI
harness/llm/                client (OpenAI + stub), embedding
harness/learners/trivial.py          TrivialLearner
harness/gates/noop.py                NoOpGate
harness/gates/non_regression.py      NonRegressionGate
harness/representations/skill_md.py  SkillMdRepresentation
harness/agents/react.py              ReActAgent (fixed)
harness/benchmarks/synthetic.py      SyntheticTasksAdapter (32 tasks, 4 categories)
```
Logs: JSON lines (`run.start`, `train.traces`, `learner.induce`, `gate.decision`, `gate.result`, `eval.task`, `run.end`).

## Where future methods plug in (add a new file, never edit existing ones)
| Future method | Kind | New file | Registry name / notes |
|---|---|---|---|
| AWM (Agent Workflow Memory) | Learner | `harness/learners/awm.py` | `awm`; induces workflow skills from successful traces |
| Trace2Skill | Learner | `harness/learners/trace2skill.py` | `trace2skill` |
| EvoSkill | Learner | `harness/learners/evoskill.py` | `evoskill`; may use `current` library for mutation |
| CoEvoSkills | Learner | `harness/learners/coevoskills.py` | `coevoskills` |
| SkillRL | Learner | `harness/learners/skillrl.py` | `skillrl` |
| Statistical / Bayesian gate | Gate | `harness/gates/bayesian.py` | `bayesian`; uses `probe` replay + posterior over pass rate |
| Verifier-governance gate | Gate | `harness/gates/verifier_governance.py` | `verifier_governance`; may repair/reject candidates |
| OfficeQA (next milestone) | Benchmark | `harness/benchmarks/officeqa.py` | `officeqa`; implement `category()` to enable per-category gating |
| FinToolBench, tau-bench | Benchmark | `harness/benchmarks/<name>.py` | extend tool allowlist via the agent for `unauthorized_actions` |
| NL-rule / code skills | Representation | `harness/representations/<name>.py` | `nl_rule`, `code` |

Each plugin class is decorated `@register("<kind>", "<name>")` and either has a no-arg constructor or `from_context(ctx)` (see `harness/context.py`: cfg, llm, embedder, logger, benchmark, representation, agent, task_lookup). Then set the name in YAML.

## Conventions beyond the verbatim spec
* Gates expose `last_rejections` and `last_tokens` (probe replay cost, counted in `tokens_build`).
* Benchmarks may expose `category(task)`; NonRegressionGate then requires no per-category drop.
* `Trace.task_id` = `t_` + sha1(task)[:10]; the runner keeps a `task_lookup` id→text.
* Probe set = seeded, category-stratified 2 tasks/category held out of train (learner sees the other 12).
* Scorecard: `auditability` = fraction of library skills whose provenance is non-empty and all real train ids (1.0 for an empty library); `unauthorized_actions` = tool calls outside `{calculator, finish}` in the with-library eval; flips compare with-library vs empty-library eval on the same tasks and seed.

## Notes from the audit
* `types.py` flip comments corrected (user-approved): `flips_w2c` = improvement (wrong→correct), `flips_c2w` = regression (correct→wrong).
* Flips are a paired, per-task comparison on the identical `test_tasks()` list: the with-library and empty-library evals both run live under the same seed. `eval.task` log lines record both outcomes.
* `split` log event records learn / probe / test task lists (pairwise disjoint; enforced by `tests/test_audit.py`).

## Milestone 1: OfficeQA + EvoSkill

New files only. The interface modules, `types.py` and `runner.run` are untouched.

| File | What |
|---|---|
| `harness/benchmarks/officeqa.py` | `OfficeQAAdapter` (`benchmark: officeqa`). Scored with OfficeQA's own `reward.score_answer`, taken from a pinned `databricks/officeqa@7b9a3c1`. BM25 retrieval over the Treasury Bulletin corpus is rendered into each task. Category = difficulty. |
| `harness/benchmarks/_pinned.py` | Fetches pinned external repos by SHA into `~/.cache/skill-harness` (no forks or copies in this repo). |
| `harness/learners/evoskill.py` | `EvoSkillLearner` (`learner: evoskill`). Runs EvoSkill's own CLI loop unchanged in an isolated git project. |
| `harness/officeqa_run.py` | CLI with `--limit N` / `--source {auto,hf,sample}`. |
| `scripts/validate_evoskill.py` | GO/NO-GO check: standalone EvoSkill vs harness-wrapped, on the same subset and seed. |
| `examples/m1_officeqa_*.yaml` | `empty` (retrieval_k 0: benchmark + ReActAgent alone), `evoskill_noop`, `evoskill_nonregression`. |

Install: `uv pip install --python .venv/bin/python -c constraints-evoskill.txt -r requirements.txt -r requirements-evoskill.txt`. EvoSkill is pinned to `sentient-agi/EvoSkill@36f6f04952293d7054145550c2b9f0b0411bff1c`. The constraints come from EvoSkill's own `uv.lock` at that SHA; it needs openhands-sdk 1.16.1, and newer releases break it.

Data:
* `OFFICEQA_SOURCE=hf` needs `HF_TOKEN` with access granted to the gated `databricks/officeqa` dataset.
* Without it, `sample` uses the official 10-question OfficeQA sample plus 9 bulletins shipped inside the pinned EvoSkill repo.

Models:
* EvoSkill runs through its OpenHands harness with `openai/<base_model>`, using `OPENAI_API_KEY`.
* Override with `EVOSKILL_HARNESS` / `EVOSKILL_MODEL`.

```
python -m harness.officeqa_run --config examples/m1_officeqa_empty.yaml --limit 10
python scripts/validate_evoskill.py --limit 10 --seed 0
python -m harness.officeqa_run --config examples/m1_officeqa_evoskill_noop.yaml --limit 10
python -m harness.officeqa_run --config examples/m1_officeqa_evoskill_nonregression.yaml --limit 10
```

Notes:
* EvoSkill consumes Q/A rows, not recorded traces. It re-executes them with its own base agent and splits them into its own train/val, using only the learner's traces. So probe and test tasks never reach it.
* EvoSkill has no seed parameter: its split is fixed at `random_state=42`, and its LLM calls are unseeded.
* `last_build_tokens` counts base-agent rollout usage from EvoSkill's run cache. Proposer and generator usage is not cached by EvoSkill, so it is not included.

### M1 deviations
* **OpenAI compatibility shim** (`harness/learners/evoskill_compat.py`):
  * The pinned EvoSkill's OpenHands fallback extraction sends a strict `json_schema` built from pydantic, and OpenAI rejects it (HTTP 400).
  * The shim wraps `openhands.sdk.llm.LLM.completion` in-process. Before each call it makes the schema strict-compatible: `additionalProperties:false`, every property required, `default` removed.
  * It then calls EvoSkill's own CLI unchanged. No installed EvoSkill file is edited.
  * The standalone validation and `EvoSkillLearner` launch EvoSkill the same way (`run_evoskill`), so the comparison stays fair.
* **Dependency pins:** EvoSkill's dependencies are pinned from its own `uv.lock` (`constraints-evoskill.txt`).
* **`EVOSKILL_EXPORT=frontier_union`** (smoke runs use it with `EVOSKILL_FRONTIER=2`):
  * It exports skills from every frontier program, not just EvoSkill's best.
  * On a 0-baseline subset, EvoSkill's own frontier discards every candidate on a 0-vs-0 tie. Without this, the harness Gate never sees a candidate.
  * The default (`best`) keeps EvoSkill's native selection.

### M1 status (real runs, 2026-10-12, gpt-4o-mini, HF OfficeQA)
* **Empty-library OfficeQA** (`--limit 20`): 0/6 test, 0/14 train. The result was the same with 8×2000-char context.
* **With the shim, EvoSkill completes its loop on OpenAI.** It ran the base eval, a failure, the skill proposer, the skill generator and the frontier update.
* **Smoke size** (`--limit 10`; 4 learner rows, 4 probe, 2 test; 1 iteration):
  * Standalone and wrapped agree: 4 rows in, 1 native val sample, 0 skills out. Verdict GO (structurally consistent).
  * gpt-4o-mini's skill **generator** never wrote a `.claude/skills/*/SKILL.md`. The kept child program only changed `program.yaml`, so no Skill reached the harness.
  * Both gates therefore got an empty library: success 0.0, 0 flips.
  * Learn/probe/test task sets are pairwise disjoint in the real logs.

### M2: retrieval fixes and EvoSkill with gpt-4o (2026-10-13)

**New files**
* `harness/benchmarks/officeqa_v2.py` (`benchmark: officeqa_v2`):
  * date-aware narrowing to `treasury_bulletin_YYYY_MM` files, with global fallback;
  * table-aware chunking (title and header repeated on each split);
  * up to 20 chunks within a 24k-char budget;
  * tolerance 0.01 by default.
  * The corpus is all date-candidate bulletins (283 files at limit 20), not only the gold files. v1's corpus was gold-only, which is an oracle.
* `scripts/retrieval_recall.py`: recall on the diag question set.
* `examples/m2_*.yaml`: configs for these runs.

**Changes to existing files**
* `EvoSkillLearner`:
  * pre-creates `.claude/skills/`, because gpt-4o otherwise created it as a plain file and every SKILL.md write failed;
  * gives EvoSkill's agents only the date-candidate bulletins (symlinks);
  * `EVOSKILL_REUSE=1` reuses a finished induction with identical inputs, so both gates share one EvoSkill run.
* `OpenAIEmbedder` truncates input to 12k chars, to stay under the 8192-token limit.
* ReAct `max_steps` was not raised. It is already 6, and runs average 1–1.3 steps, so it is not binding. Raising it would mean editing the M0 agent or the runner.

**Recall on the 10 diag questions (any gold-file chunk in top 20)**

| Corpus | Retrieval | Hit rate | Mean gold chunks |
|---|---|---|---|
| v1 (gold-only, oracle) | BM25 | 9/10 | 3.4 |
| v2 (283 files) | global BM25 | 3/10 | 0.7 |
| v2 (283 files) | date-narrowed | 8/10 | 3.4 |

The literal answer string appears in the context for 0/10 with v2 (1/10 with v1). Most answers are computed, not quoted.

**Empty-library baseline, `--limit 20`, gpt-4o-mini**
* v1: 0/6 test, 0/14 train.
* v2: 0/6 test, 1/14 train (1/10 learn, 0/4 probe).
* gpt-4o eval was skipped to stay within budget.

**EvoSkill, gpt-4o, `--limit 10`, 1 iteration** (4 learner rows, 4 probe, 2 test)
* Validation verdict: **NO-GO by outcome; structurally consistent.**
  * Same 4 rows in and 1 native val sample on both sides.
  * Standalone produced `information-inference-skill`.
  * The wrapped attempt's generator wrote `data-path-verification` as an empty file, not a SKILL.md, so the adapter correctly extracted 0 skills.
  * The divergence comes from LLM nondeterminism, not the adapter.
* The gate runs' fresh induction produced `historical-document-retrieval`.
* Both gates admitted it:
  * success 0.0 and 0 flips on 2 test questions;
  * NonRegressionGate probe was 0.0 before and after, with no per-category drop.
  * The gates are uninformative at a 0 baseline.
* Learn/probe/test sets are pairwise disjoint in both logs.

## M3 — ALFWorld (minimal-sample GO/NO-GO)

Run: `ALFWORLD_TYPES=pick_and_place_simple,look_at_obj_in_light ALFWORLD_TRAIN_PER_CAT=4 ALFWORLD_LIMIT=14 python -m harness.alfworld_run --config examples/m3_alfworld_<noskill|evoskill_noop|evoskill_nonregression|rawrag>.yaml`
(EvoSkill: `EVOSKILL_MODEL=openai/gpt-4o EVOSKILL_ITERATIONS=1 EVOSKILL_FRONTIER=2 EVOSKILL_EXPORT=frontier_union EVOSKILL_REUSE=1`).
Data: `alfworld-download` (alfworld==0.4.2 pinned, textworld 1.7.0), split json_2.1.1/valid_seen, seed 0, 4 learn / 4 probe / 6 test, disjoint.

| config | success | avg_steps | w2c | c2w | lib | tokens_eval | tokens_build | unauth |
|---|---|---|---|---|---|---|---|---|
| no-skill | 0.50 | 21.0 | 0 | 0 | 1 (k=0, not injected) | 89,416 | 950 | 10 |
| EvoSkill + NoOp | 0.67 | 16.7 | 1 | 0 | 1 | 91,804 | 33,947 | 9 |
| EvoSkill + NonRegression | 0.67 | 16.7 | 1 | 0 | 1 (admitted, probe 0.25→0.25) | 91,804 | 215,241 | 9 |
| raw-trace RAG (k=3) | 0.33 | 23.8 | 0 | 1 | 4 | 316,621 | 0 | 32 |

n=6 test tasks: this is a smoke test showing the pipeline works, not a statistical result.

Design notes and deviations:
- `verify` returns the ALFWorld env's own `won` flag, recorded while the agent ran (`env_reset`/`env_step` on the adapter).
- `run()` hard-codes agent "react". `harness/alfworld_run.py` points that registry slot at `ReActMultiStepAgent` without editing M0. That agent shows the admissible commands each turn and caps episodes at 30 steps. `unauthorized_actions` counts actions outside the admissible commands.
- `_alfworld_compat` patches textworld for py3.13: PEP 667 broke `locals().update`.
- `harness/llm/cache.py`: an on-disk LLM cache keyed on (model, prompt, seed, max_tokens), so identical no-skill episodes are shared across configs.
- Embedding truncation moved to `harness/llm/embedding_safe.py`. M0 `embedding.py` is restored.
- EvoSkill cannot drive the env interactively. Each row it gets is the task plus our agent's previous trajectory. Its agent outputs an open-loop command plan, and a script scorer (`harness.benchmarks.alfworld_score`) replays that plan in the real env (won → 1.0).
- `evoskill_compat` now also makes OpenHands `file_editor` create parent directories. Without it, the skill builder left an empty `.claude/skills/<name>` file (0-content skill; attempt 1 had library_size 0).
- Only 2 task categories are used, to keep the sample minimal.
