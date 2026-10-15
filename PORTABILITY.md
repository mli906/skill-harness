# PORTABILITY.md - path / network / git-tracking audit (main @ 232bb57 + this branch)

## 1. Hardcoded absolute paths (grep of `/workspace`, `/home/`, `/root/`, `/tmp/`, `/Users/` outside logs/manifests/docs)
| Location | What | Status |
|---|---|---|
| `scripts/trackA_clean.sh`, `rerun_cell_clean.sh`, `gepa_smoke_clean.sh`, `gepa_bayes_relaunch.sh` | `PY=${PY:-/workspace/skill-harness/.venv/bin/python}` | **Fixed**: default is now `.venv/bin/python` (scripts `cd` to the repo root; `PY=` still overrides) |
| `scripts/gepa_smoke_clean.sh`, `gepa_bayes_relaunch.sh` | `PYTHONPATH=.:/workspace/skill-harness-gepa/.gepa-pkgs` (dev-only GEPA dir) | **Fixed**: `PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}`; gepa is the PyPI pin `gepa==0.1.4` in the lock |
| `scripts/trackA_clean.sh` last step | `cd /workspace/skill-harness-analysis ... PYTHONPATH=/workspace/.analysis-pkgs ... --runs /workspace/skill-harness-leak/$OUT --out /workspace/analysis-out/...` | **Fixed**: `$PY -m harness.analysis --runs $OUT --out ${ANALYSIS_OUT:-analysis-out/trackA_clean}` (analysis package is on main; matplotlib is in the lock) |
| `scripts/trackA_clean.sh` last step | auto `git add/commit/push origin feat/leak-invariants` | **Guarded**: `TRACKA_AUTOCOMMIT` (default `1` = unchanged behavior). Set `0` on RockAI |
| `harness/learners/trace2skill.py` `skill_creator()` | clone path hardcoded to `~/.cache/skill-harness/anthropic-skills` (ignored `HARNESS_CACHE`) | **Fixed** (behavior-preserving): now `$HARNESS_CACHE/anthropic-skills/<sha>`, same default |
| `scripts/validate_trace2skill.py` docstring | example `--python /workspace/.t2s_venv/bin/python` | **Fixed** (doc only) |
| `scripts/retrieval_recall.py:14` | default input `/workspace/officeqa-diag/results.json` (arg 1 overrides) | **Only reported** (OfficeQA diagnostic, not part of scaled runs; pass a path) |
| `scripts/m3q_run.sh`, `scale_*.sh`, `trackA_clean.sh`, `rerun_cell_clean.sh` | `EVOSKILL_FROZEN_PROJECT=logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4` (relative) | **Only reported**: relative, fine, but the directory is NOT in git (section 3) |
| `scripts/scale_all.sh`, `scale_finalize.sh`, `scale_resume_hashed.sh` | call `.venv/bin/python` (relative to repo root) | OK if `bootstrap.sh` created `<repo>/.venv` (default). Not parameterized |
| `scripts/*.sh` | rely on `pgrep`, `ps`, `setsid`, `python3` (procps/util-linux on PATH) | **Only reported** (watchdog / wait-for-live-run logic) |
| `~/.cache/skill-harness/*`, `~/.cache/alfworld` | cache defaults under `$HOME` | Fine; all overridable (`HARNESS_CACHE`, `TRACE2SKILL_HOME`, `HARNESS_LLM_CACHE`, `ALFWORLD_DATA`). `harness/corpus.py:56` hardcodes `~/.cache/skill-harness/llm_cache_corpus` only via `setdefault(HARNESS_LLM_CACHE)`, so setting that var wins |
| `docs/`, `manifests/*/RESULTS.md`, `logs/scale_clean_gepa/*` | historical `/workspace/...` strings in recorded output | Data, not code; ignore |
No `/home/<user>` or other machine-specific paths exist in code. Relative defaults (`logs/`, `manifests/...`, `corpora/`) resolve against the CWD: always run from the repo root.

## 2. What the harness fetches at runtime (egress list)
| # | Destination | What | When | Credential |
|---|---|---|---|---|
| 1 | pypi.org, files.pythonhosted.org | `pip/uv install -r requirements.lock` (283 pkgs; `fast-downward-textworld` builds from source: g++ + make) | bootstrap | none |
| 2 | github.com | `evoskill @ git+https://github.com/sentient-agi/EvoSkill@36f6f04...` (pip git requirement) | bootstrap | none |
| 3 | github.com | `git fetch --depth 1 <sha>` of EvoSkill, databricks/officeqa (`harness/benchmarks/_pinned.py`); `git clone`+`checkout` of Qwen-Applications/Trace2Skill and anthropics/skills (`harness/learners/trace2skill.py`) | bootstrap pre-clones them; else first use | none (public repos) |
| 4 | github.com release assets -> objects.githubusercontent.com / release-assets.githubusercontent.com | `alfworld-download`: `json_2.1.1_json.zip`, `json_2.1.1_pddl.zip`, `json_2.1.2_tw-pddl.zip`, `mrcnn_alfred_objects_sep13_004.pth`, logic files (~2-3 GB total; on the dev box `~/.cache/alfworld` = 2.3 GB) | bootstrap step 5 | none |
| 5 | huggingface.co | dataset `databricks/officeqa` (csv via `datasets.load_dataset`, parsed bulletins via `hf_hub_download` / `snapshot_download`) | ONLY `OFFICEQA_SOURCE=hf` (or auto + `HF_TOKEN`). Not used by ALFWorld / AgentDojo / scaled runs | **`HF_TOKEN`** |
| 6 | api.openai.com (or `OPENAI_BASE_URL`) | chat completions, embeddings (`text-embedding-3-small`), Trace2Skill/EvoSkill/GEPA builds (gpt-4o) | real runs | `OPENAI_API_KEY` |
| 7 | openrouter.ai | `openrouter/<id>` chat (Qwen-72B consumer etc.); spend watchdog `GET /api/v1/key`, `/api/v1/credits` | scaled runs | `OPENROUTER_API_KEY` |
| 8 | third-party libs | OpenHands/litellm (EvoSkill build) talk to the model endpoints above; any telemetry they do was **not audited** | EvoSkill builds | - |
No runtime `pip install`, no `curl|sh`, no other downloads were found (`grep` for git/clone/fetch/urlopen/requests/snapshot_download/hf_hub_download/load_dataset). AgentDojo ships its suite data in the wheel. Only #5 needs the HF token. `make smoke` needs none of this after bootstrap.

## 3. External repos pinned in code (cache layout)
| Repo | SHA | Where pinned | Cache path (default) | Notes |
|---|---|---|---|---|
| sentient-agi/EvoSkill | `36f6f04952293d7054145550c2b9f0b0411bff1c` | `harness/benchmarks/_pinned.py:11-12` (checkout, used for the OfficeQA sample data); `requirements-evoskill.txt` and `requirements.lock` (pip git pin, the code that actually runs) | `$HARNESS_CACHE/EvoSkill-36f6f0495229` | shallow fetch by SHA |
| databricks/officeqa (GitHub) | `7b9a3c154ef9fb40215bb67934afc43e6799de16` | `_pinned.py:9-10` (reward.py) | `$HARNESS_CACHE/officeqa-7b9a3c154ef9` | |
| Qwen-Applications/Trace2Skill | `3d0b52a140f002a512930252b613c49048f7d5ac` | `harness/learners/trace2skill.py:40-41` | `$TRACE2SKILL_HOME/<sha>` = `~/.cache/skill-harness/trace2skill/3d0b52a1...` | full clone + checkout; HEAD verified |
| anthropics/skills (skill-creator) | `dbd4588f9e1033efb41dad4bef2f7947c8993d44` | `trace2skill.py:44-45` | `$HARNESS_CACHE/anthropic-skills/<sha>` | needs `skills/skill-creator/scripts/quick_validate.py` |
| GEPA | PyPI `gepa==0.1.4` | `requirements.txt`, `requirements.lock` | site-packages | **No repo clone.** Dev used a `PYTHONPATH=.gepa-pkgs` copy, but `gepa 0.1.4` is a normal PyPI wheel and is installed in the dev venv: no hack needed |
| zorazrw/agent-workflow-memory @ 8c0ff8c, DataArcTech/Bayesian-Agent @ 4b69b4e | - | docstrings in `learners/awm.py`, `gates/bayesian.py` | not fetched | AWM and the Beta-Bernoulli posterior were re-implemented/ported natively; reference only |
| databricks/officeqa (HF dataset) | no pin | `officeqa.py:28` | `$HARNESS_CACHE/officeqa-corpus` | unpinned dataset revision |
`bootstrap.sh` step 4 calls the code's own `checkout()` helpers and asserts every HEAD equals the pin.

## 4. Git-tracked vs ignored (what RockAI gets from `git clone` vs must supply)
`.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `logs/`, `*.jsonl`, `.pytest_cache/`. **`*.jsonl` is ignored everywhere** (including `corpora/`), tracked jsonl exist only because they were `git add -f`.
| Path | State | Regenerates? |
|---|---|---|
| `harness/`, `tests/`, `examples/`, `scripts/`, `docs/`, `README.md`, requirements*, `pyproject.toml` | tracked | - |
| `manifests/alfworld_scale/` (9 files: `split.json`, `tasks.json`, `practice_{Q,M}.json`, `PLAN.md`, `RESULTS.md`, `run_manifest.json`, 2 cost estimates) | tracked | Splits are pinned in git (sha256 of `split.json` = `e195fd80...`), so they reproduce |
| `manifests/alfworld_scale_clean/` (`split.json` sha256 `5162e298...`, `tasks.json`, `gepa_valset.json`, `RESULTS.md`, `run_manifest.json`) | tracked | pinned |
| `corpora/alfworld_c1/` (`corpus.jsonl`, `manifest.json`) | tracked (force-added) | yes, tracked |
| **`corpora/scale_Q/`, `corpora/scale_M/`** (frozen Qwen / gpt-4o-mini trajectory corpora, 488K / 600K) | **untracked, not in git** | **NOT reproducibly regenerable**: `scale_run.py corpus` samples at temperature 0.7 through OpenRouter (costs money; output differs run to run). Copy them from the dev box |
| **`logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4/`** (frozen EvoSkill project, 748K) | **ignored (`logs/`)** | Required by `evoskill_frozen` (Track A). Re-building = a paid EvoSkill run on gpt-4o and a different skill set. Copy it |
| `logs/scale_clean_gepa/` (6 files) | tracked (force-added) | record of the GEPA smoke run |
| other `logs/*` (run logs, `scale_clean/*.jsonl`, `.result.json`) | ignored | run outputs; cells skip if `<name>.result.json` exists, so copying `logs/scale_clean/` resumes/skips finished cells |
| `~/.cache/skill-harness/llm_cache` (the LLM response cache, 6.5 MB / 255 entries on the dev box) | outside repo | cache hit = free replay; losing it only costs money, not correctness (temp 0 runs) |
| `~/.cache/alfworld`, `~/.cache/skill-harness/{EvoSkill-*,officeqa-*,trace2skill,anthropic-skills}` | outside repo | rebuilt by `bootstrap.sh` |
| `.venv/` | ignored | rebuilt by `bootstrap.sh` |
| `.claude/skills/*` (3 files) | tracked | leftovers from an EvoSkill run; harmless |
A bundle of the three non-git artifacts (corpora/scale_Q, corpora/scale_M, the frozen EvoSkill project; 137 entries, 177 KB, secret-scanned) was created on the dev box at `/workspace/transfer-bundle/untracked-artifacts.tgz` (+ `.sha256`) - it is NOT in git. Unpack at the repo root: `tar xzf untracked-artifacts.tgz`.

## 5. M0 reference values: what "182ab911..." / "8e01cab2..." were
Searched the entire working tree, every git blob in history, commit messages, README/docs and logs: **no occurrence of either prefix anywhere**, and neither matches the md5 of any tracked file. Their definition is unknown, so they are not used. The smoke test instead asserts the exact M0 Scorecards (see `scripts/smoke_m0.py`), derived by running M0 on main @ 232bb57: md5 of `json.dumps(asdict(scorecard), sort_keys=True)` = `60a5bd42f0dd0b6479e2ec266cd96552` (examples/m0.yaml, NoOpGate) and `ed7af3a9aecbe49ffa1988f2418bd9e5` (examples/m0_nonregression.yaml). The same hashes were observed on origin/main (f19835b) before the consolidation, i.e. the merge did not change M0 behavior.

## 6. Things that can bite on RockAI (Google ADK-based platform)
* **Python version**: lock verified on CPython **3.13.5** only. Minimum is 3.12 (`scipy 1.18.1`, `openhands-sdk 1.16.1`, `evoskill` require >=3.12). 3.11 or older will not install.
* **No apt / no root**: `fast-downward-textworld==20.6.4` has no wheel and builds from source with `g++` + `make` (no cmake, no Python headers needed - verified). Without a toolchain, ALFWorld/TextWorld cannot be installed; supply a prebuilt wheel or conda-forge toolchain. `bootstrap.sh` detects and reports this.
* **Outbound git/github.com**: needed for the evoskill pip pin, 4 pinned repos, and the ALFWorld release zips. If egress is blocked, pre-populate the caches (copy `~/.cache/skill-harness`, `~/.cache/alfworld`) and run `SKIP_EXTERNAL=1 SKIP_ALFWORLD_DATA=1 ./bootstrap.sh`; a `pip download`/wheelhouse is needed for the evoskill git pin.
* **Read-only / small home**: redirect `HARNESS_CACHE`, `TRACE2SKILL_HOME`, `ALFWORLD_DATA`, `HARNESS_LLM_CACHE`, `HF_HOME`, `UV_CACHE_DIR`, `MPLCONFIGDIR` (CONFIG.md). The repo itself writes to `logs/`, `manifests/`, `corpora/`, `reports/` under CWD.
* **Disk**: ~2.5 GB ALFWorld + ~1.5 GB venv (OpenHands, litellm, datasets, matplotlib).
* **Process model**: the launchers assume a long-lived shell with `setsid`/`nohup`, `pgrep`, `ps`, background watchdog subshells that `kill -TERM -- -$PGID` the process group on spend hard-stop; an agent/sandbox runner that kills background processes or forbids signalling process groups will break them. Several launchers poll for another live run with `pgrep -f` (a stuck wait if unrelated matching processes exist).
* **Spend controls are OpenRouter-specific** (`/api/v1/key`, `/credits`); on another provider the hard-stop watchdog silently reads 0 (`usage()` swallows errors).
* **Secrets**: only via env vars. Do not bake into the clone; nothing in the repo contains a key (scanned).
