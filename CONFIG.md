# CONFIG.md - environment-variable surface of skill-harness

Derived from a grep of `os.environ` / `os.getenv` / `environ.get` across `harness/`, `scripts/`, `examples/` plus the
`scripts/*.sh` launchers (commit base: main @ 232bb57). Defaults are copied from the code (file:line given as `file`).
"Required" means: the code raises or the feature cannot work without it. **Nothing here is needed for `make smoke`**
(stub model, no network). Secrets are only ever read from the environment; the harness never writes them to logs.

## 1. Credentials (set these; never commit them)
| Var | Controls | Default | Required |
|---|---|---|---|
| `OPENAI_API_KEY` | Key for any non-stub `base_model` on the OpenAI path (`harness/llm/client.py`), OpenAI embeddings (`embedding.py`), Trace2Skill/EvoSkill/GEPA builds routed to OpenAI, and the default key for `LLM_BASE_URL` / `BUILD_BASE_URL` | none | Yes for real OpenAI runs and for `HARNESS_EMBEDDING=openai:*` / `auto`+key |
| `OPENROUTER_API_KEY` | Key for `base_model: openrouter/<id>` (`llm/openrouter.py`), `scripts/scale_run.py` (Track A/B consumer = `openrouter/qwen/qwen-2.5-72b-instruct` / `openrouter/openai/gpt-4o-mini`), spend watchdogs (`/api/v1/key`, `/api/v1/credits`) in `scale_run.py`, `governance_smoke.py` and the `*.sh` launchers | none | Yes for the scaled ALFWorld runs |
| `HF_TOKEN` | HuggingFace token for the gated `databricks/officeqa` dataset (`officeqa.py`, `officeqa_v2.py`). Its presence also flips `OFFICEQA_SOURCE=auto` from `sample` to `hf` | none | Only for OfficeQA with source `hf` |

## 2. LLM client / provider routing (`harness/llm/`)
| Var | Controls | Default | Required |
|---|---|---|---|
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint for chat + embeddings; also forwarded to the Trace2Skill subprocess (`client.py`, `embedding.py`, `trace2skill.py`) | `https://api.openai.com/v1` | No |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint (`openrouter.py`, `build_routing.py`) | `https://openrouter.ai/api/v1` | No |
| `LLM_BASE_URL` | If set, a plain-id `base_model` is sent to this OpenAI-compatible URL (generic provider, `openrouter.py`) | unset (=> OpenAI) | No |
| `LLM_API_KEY_ENV` | Name of the env var holding the key for `LLM_BASE_URL` | `OPENAI_API_KEY` | No |
| `LLM_EXTRA_BODY` | JSON merged into every provider request body (e.g. `{"provider":{"ignore":["X"]}}`) | `{}` | No |
| `LLM_MAX_RETRIES` | Retry count for provider calls (429/5xx backoff) | `8` | No |
| `HARNESS_EMBEDDING` | `auto` \| `hashed` \| `openai:<model>`. `auto` = OpenAI `text-embedding-3-small` only for non-stub models with `OPENAI_API_KEY`, else deterministic hashed bag-of-words | `auto` | No (scaled runs set `openai:text-embedding-3-small`) |
| `HARNESS_LLM_CACHE` | On-disk LLM response cache dir. `harness/corpus.py` defaults it (setdefault) to `~/.cache/skill-harness/llm_cache_corpus` for corpus generation | `~/.cache/skill-harness/llm_cache` | No (set to a writable path if `$HOME` is read-only) |

## 3. Paths, logging, external-repo cache
| Var | Controls | Default | Required |
|---|---|---|---|
| `HARNESS_CACHE` | Root cache for pinned external clones: EvoSkill, databricks/officeqa, anthropics/skills, OfficeQA HF corpus (`benchmarks/_pinned.py`, `learners/trace2skill.py`) | `~/.cache/skill-harness` | No |
| `TRACE2SKILL_HOME` | Where the pinned Trace2Skill repo is cloned (`<home>/<sha>`). NOT derived from `HARNESS_CACHE` | `~/.cache/skill-harness/trace2skill` | No |
| `HARNESS_LOG_DIR` | Run-log directory for `harness.runner.run` and default governance evidence log | `logs` | No |
| `HARNESS_CORPUS_DIR` | Frozen C1 corpus dir read by Trace2Skill when the context has none (`c1_run.py` / `scale_run.py` set it) | unset | Set by launchers for C1 learners |
| `ALFWORLD_DATA` | ALFWorld game-file root (`json_2.1.1/{train,valid_seen,...}`, `logic/`); also read by the `alfworld` package and `alfworld-download` | `~/.cache/alfworld` | Yes for any ALFWorld run (bootstrap downloads it) |
| `HARNESS_ENV_FORBIDDEN` | INTERNAL flag set by `env_guard.py` while a C1 learner runs (blocks env access). Do not set | unset | No |

## 4. Learners
| Var | Controls | Default | Required |
|---|---|---|---|
| `TRACE2SKILL_MODEL` | Model used for Trace2Skill analysis/merge (`trace2skill.py`) | `gpt-4o` (accepts `openrouter/<id>`) | No |
| `TRACE2SKILL_WORKDIR` | Work/output dir (per key+seed+model subdir) | `logs/trace2skill` | No |
| `TRACE2SKILL_REUSE` | `1` = reuse a finished build with identical key | off | No |
| `TRACE2SKILL_WORKERS` | `--max_workers` passed to Trace2Skill scripts | `4` | No |
| `BUILD_BASE_URL` | Route Trace2Skill/EvoSkill builds to a custom OpenAI-compatible URL (`build_routing.py`) | unset | No |
| `BUILD_API_KEY_ENV` | Env var name holding the key for `BUILD_BASE_URL` | `OPENAI_API_KEY` | No |
| `EVOSKILL_MODEL` | EvoSkill model string (litellm style) | `openai/<base_model>` (or `openrouter/...` if base_model is openrouter) | No |
| `EVOSKILL_HARNESS` | EvoSkill agent harness | `openhands` | No |
| `EVOSKILL_WORKDIR` | EvoSkill project dir root | `logs/evoskill_runs` | No |
| `EVOSKILL_REUSE` | `1` = reuse finished EvoSkill project with identical rows+model+EVOSKILL_* settings | off | No |
| `EVOSKILL_EXPORT` | `best` \| `frontier_union` (skills exported from the EvoSkill frontier) | `best` | No |
| `EVOSKILL_TIMEOUT` / `_ITERATIONS` / `_FRONTIER` / `_CONCURRENCY` / `_FAILURE_SAMPLES` / `_TRAIN_RATIO` / `_VAL_RATIO` | Values written into the generated EvoSkill `config.toml` | `600` / `2` / `2` / `2` / `2` / `0.5` / `0.5` | No |
| `EVOSKILL_FROZEN_PROJECT` | Path to a finished EvoSkill project; the `evoskill_frozen` learner reuses its skills + origin ids | none | **Yes** for `evoskill_frozen` (Track A: `logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4`, NOT in git - see PORTABILITY.md) |
| `EVOSKILL_REMOTE` | INTERNAL: harness sets `1` for the EvoSkill subprocess. Do not set | set by harness | No |
| `AWM_SUCCESS_ONLY` | `1` = AWM induces from successful traces only (read in `scripts/scale_run.py`, not in the learner) | off | Scaled runs set `1` |
| `GEPA_MAX_METRIC_CALLS` | GEPA metric-call budget (`learners/gepa.py`, `DEFAULT_MAX_METRIC_CALLS`) | `200` (the GEPA smoke scripts export `80`) | No |
| `GEPA_VALSET_PER_CATEGORY` | Tasks per category carved out of the learn corpus as GEPA's valset (`splits.py`, `scale_run.py`) | `2` | No |

Note: every `EVOSKILL_*` var except `EVOSKILL_WORKDIR`/`EVOSKILL_REUSE` is hashed into the EvoSkill reuse key - changing any invalidates reuse.

## 5. Gates
| Var | Controls | Default | Required |
|---|---|---|---|
| `BAYESIAN_GATE_ALPHA` | BayesianGate admission level (P(cand>base) >= 1-alpha) | `0.05` | No |
| `GOVERNANCE_K_SUPPORT` | Governance check 5: minimum distinct supporting traces | `1` | No |
| `GOVERNANCE_EVIDENCE_LOG` | Append-only governance evidence JSONL | `$HARNESS_LOG_DIR/governance_evidence.jsonl` (`logs/...`) | No |

## 6. Benchmarks and agents
| Var | Controls | Default | Required |
|---|---|---|---|
| `ALFWORLD_SPLIT` | Game split for `alfworld` | `valid_seen` | No |
| `ALFWORLD_TRAIN_SPLIT` | Train split for `alfworld_c1` | `train` | No |
| `ALFWORLD_LIMIT` | Max games (`alfworld` adapter) | `48` | No |
| `ALFWORLD_TRAIN_PER_CAT` | Train games per category | `4` | No |
| `ALFWORLD_TYPES` | Comma list of task types | all 6 `TASK_TYPES` | No |
| `ALFWORLD_MAX_EPISODE_STEPS` | Env step cap per episode | `50` | No |
| `ALFWORLD_AGENT_MAX_STEPS` / `ALFWORLD_HISTORY` | `react_multistep` agent step cap / history window | `30` / `12` | No |
| `AGENTDOJO_CONDITION` | `clean` \| `attack` | `clean` | No |
| `AGENTDOJO_LIMIT` | Cap on user tasks (0 = all) | `0` | No |
| `AGENTDOJO_TRAIN_FRAC` | Train fraction | `0.5` | No |
| `AGENTDOJO_INJECTION_TASK` / `AGENTDOJO_INJECTIONS_PER_TASK` | Force one injection task / injections per test task | none / `1` | No |
| `AGENT_MAX_STEPS` / `AGENT_HISTORY` | `react_tools` agent step cap / history window | `15` / `12` | No |
| `OFFICEQA_SOURCE` | `auto` \| `hf` \| `sample` | `auto` (`hf` if `HF_TOKEN` else `sample`) | No |
| `OFFICEQA_LIMIT` / `OFFICEQA_SPLIT` | Row cap / HF csv split | unset / `full` | No |
| `OFFICEQA_TOLERANCE` | Numeric tolerance for scoring | `0.0` (`officeqa_v2`: `0.01` when unset) | No |
| `OFFICEQA_CONTEXT_CHUNKS` / `OFFICEQA_CHUNK_CHARS` / `OFFICEQA_TEST_FRAC` | BM25 chunks per task / chunk size / test fraction | `3` / `1500` / `0.4` | No |
| `OFFICEQA_V2_K` / `OFFICEQA_V2_CTX_CHARS` / `OFFICEQA_V2_MAX_CHUNK` | v2 retrieval k / context chars / max chunk | `20` / `24000` / `6000` | No |
| `CORPUS_K` / `CORPUS_TEMPERATURE` | `python -m harness.corpus` defaults: rollouts per task / sampling temperature | `2` / `0.7` | No |

## 7. Scaled-run / analysis scripts (Python)
| Var | Controls | Default | Required |
|---|---|---|---|
| `SCALE_MANIFEST` | Pinned split manifest dir (`scale_run.py`, `scale_report.py`) | `manifests/alfworld_scale` (clean split: `manifests/alfworld_scale_clean`) | No |
| `SCALE_OUT` | Per-cell log/result dir | `logs/scale` | No |
| `SCALE_BUDGET_USD` | In-process OpenRouter spend cap for `scale_run.py` | `9.3` | No |
| `SCALE_SPLIT_EXCLUDE` | Comma list of task ids `scale_split.py` must exclude | empty | No |
| `CORPORA_DIR` / `REPORT_OUT` | `scale_report.py` corpora root / report destination | `corpora` / `$SCALE_MANIFEST` | No |
| `GOV_SMOKE_OUT` / `_MODEL` / `_LIMIT` / `_LEARNER` / `_USAGE_CAP` | `scripts/governance_smoke.py` | `logs/gov_smoke` / `openrouter/qwen/qwen-2.5-72b-instruct` / `8` / `awm` / `9.0` | No |
| `OFFICEQA_LIMIT` (set by `retrieval_recall.py`, `validate_evoskill.py`) | see section 6 | `20` in `retrieval_recall.py` | No |

## 8. Shell-script variables (scripts/*.sh, bootstrap.sh, smoke.sh)
| Var | Used by | Default |
|---|---|---|
| `PY` | `trackA_clean.sh`, `rerun_cell_clean.sh`, `gepa_smoke_clean.sh`, `gepa_bayes_relaunch.sh`, `smoke.sh`, Makefile | `.venv/bin/python` (was a hardcoded `/workspace/...` path; fixed in this branch). `scale_all.sh`, `scale_finalize.sh`, `scale_resume_hashed.sh`, `scale_rerun_openai.sh` call `.venv/bin/python` directly |
| `HARDSTOP_USD` | OpenRouter hard-stop watchdog (kills the process group) | `9.4` (`trackA_clean`, `rerun_cell_clean`), `8.5` (`gepa_smoke_clean`); hardcoded `9.4` in `gepa_bayes_relaunch.sh` |
| `SCALE_BUDGET_USD` | exported to `scale_run.py` | `9.3` (`gepa_smoke_clean.sh` default `8.3`) |
| `GEPA_MAX_METRIC_CALLS` | GEPA scripts | `80` |
| `CELL_NAME` / `CELL_ARGS` / `CELL_RETRIES` | `rerun_cell_clean.sh` single-cell rerun | `A_awm_bayesian` / `awm bayesian skill_md 3 Q c1:corpora/scale_Q` / `3` |
| `EVOSKILL_FROZEN_PROJECT` | `m3q_run.sh` (`${...:-logs/evoskill_alf/seed0_...}`); exported literally by the Track A scripts | `logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4` |
| `ANALYSIS_OUT` | `trackA_clean.sh` output of `python -m harness.analysis` | `analysis-out/trackA_clean` |
| `TRACKA_AUTOCOMMIT` | `trackA_clean.sh`: `1` = `git add/commit/push origin feat/leak-invariants` at the end | `1` (preserves old behavior). **Set `0` on a new platform.** |
| `PYTHONPATH` | all launchers export `.` (repo root); the GEPA scripts no longer append the dev-only `.gepa-pkgs` | `.` |
| bootstrap.sh: `PYTHON`, `VENV`, `PYTHON_ALLOW_OTHER`, `SKIP_EXTERNAL`, `SKIP_ALFWORLD_DATA` | see header of `bootstrap.sh` | `python3.13`, `<repo>/.venv`, 0, 0, 0 |

Scripts also hardcode (not env-overridable) `HARNESS_EMBEDDING=openai:text-embedding-3-small`, `EVOSKILL_MODEL=openai/gpt-4o`, `EVOSKILL_ITERATIONS=1`, `EVOSKILL_FRONTIER=2`, `EVOSKILL_EXPORT=frontier_union`, `EVOSKILL_REUSE=1`, `TRACE2SKILL_REUSE=1`, `AWM_SUCCESS_ONLY=1` - these reproduce the published scaled results; edit the script to change them.

## 9. Third-party variables you may need on a restricted platform (not read by this repo's code)
`HOME` (read-only home => set `HARNESS_CACHE`, `TRACE2SKILL_HOME`, `ALFWORLD_DATA`, `HARNESS_LLM_CACHE`, `HF_HOME`, `UV_CACHE_DIR`, `MPLCONFIGDIR` to writable paths), `HF_HOME` (HuggingFace cache), `MPLCONFIGDIR` (matplotlib), `HTTPS_PROXY`/`SSL_CERT_FILE` (egress).
