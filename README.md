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

Install: `pip install -r requirements.txt -r requirements-evoskill.txt`. EvoSkill is pinned to `sentient-agi/EvoSkill@36f6f04952293d7054145550c2b9f0b0411bff1c`.

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
