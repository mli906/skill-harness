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
