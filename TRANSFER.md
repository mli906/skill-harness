# TRANSFER.md - run skill-harness on a new box (e.g. RockAI)

```
git clone https://github.com/mli906/skill-harness && cd skill-harness
git checkout milestone-main-2026-10-14        # or the merged feat/transfer-readiness branch
./bootstrap.sh                                # venv + requirements.lock + pinned external repos + ALFWorld data (idempotent)
export OPENROUTER_API_KEY=... OPENAI_API_KEY=...   # see CONFIG.md; HF_TOKEN only for OfficeQA source=hf
make smoke                                    # M0 on the stub model + exact scorecard assertion + full pytest (no keys, no spend)
# scaled run, step A: put the non-git inputs in place (see below), then:
TRACKA_AUTOCOMMIT=0 setsid nohup bash scripts/trackA_clean.sh >/dev/null 2>&1 &
```

## Before the scaled run: three artifacts are NOT in git
`corpora/scale_Q/`, `corpora/scale_M/` and `logs/evoskill_alf/seed0_1791869166_c0dbb6101bf8f7c4/` (details: PORTABILITY.md section 4).
On the dev box they are bundled in `/workspace/transfer-bundle/untracked-artifacts.tgz` (177 KB; sha256 in the adjacent file). Unpack at the repo root.
They cannot be regenerated identically (temperature-0.7 sampling through a paid API), and without them the Track A cells `A_frozen_evoskill_*` and every `c1:corpora/scale_Q` cell cannot run.

## What `make smoke` prints (expected)
```
== python: Python 3.13.5
== 1/2 M0 end-to-end (stub model) with exact scorecard assertion
PASS examples/m0.yaml                   gate=noop           success=0.8333 md5=60a5bd42f0dd0b6479e2ec266cd96552
PASS examples/m0_nonregression.yaml     gate=non_regression success=0.8333 md5=ed7af3a9aecbe49ffa1988f2418bd9e5
M0 smoke: OK (scorecards reproduced exactly)
== 2/2 full pytest suite
137 passed, 1 xfailed in ~6s
SMOKE OK: matches documented expectation (137 passed, 1 xfailed)
```
The `1 xfailed` is `tests/test_exit.py::test_registry_only_milestone_plugins`, marked xfail because it was superseded by `tests/test_m1_registry.py` (it is an expected failure, not an error). All 137 other tests pass, including GEPA tests (gepa is pip-installed; no PYTHONPATH needed).
M0 expected values (trivial learner, synthetic benchmark, stub model, seed 0): success_rate 0.8333333333333334, avg_steps 1.25, tokens_eval 1342, tokens_build 219 (noop) / 1602 (non_regression), flips_w2c 2, flips_c2w 0, library_size 1, auditability 1.0, unauthorized_actions 0, promotion_rejections 0.

## Scaled-run entry points (all `cd` to the repo root themselves; run in a long-lived shell)
| Goal | Command | Needs |
|---|---|---|
| Track A on the clean split (13 cells; Qwen-72B via OpenRouter, OpenAI embeddings, hard-stop `HARDSTOP_USD`=9.4) | `TRACKA_AUTOCOMMIT=0 bash scripts/trackA_clean.sh` (detach with `setsid nohup`) | `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `corpora/scale_Q`, frozen EvoSkill project. Skips finished cells (`logs/scale_clean/<cell>.result.json`). Writes `manifests/alfworld_scale_clean/{RESULTS.md,run_manifest.json}` and `analysis-out/trackA_clean` |
| One cell | `PYTHONPATH=. SCALE_MANIFEST=manifests/alfworld_scale_clean SCALE_OUT=logs/scale_clean python scripts/scale_run.py cell <name> <learner> <gate> <skill_md\|raw_trace> <k> <Q\|M> c1:corpora/scale_Q` (+ the env block in `trackA_clean.sh`) | as above |
| Re-run one missing cell with retries | `CELL_NAME=A_awm_bayesian CELL_ARGS="awm bayesian skill_md 3 Q c1:corpora/scale_Q" bash scripts/rerun_cell_clean.sh` | as above |
| GEPA live smoke (clean split) | `bash scripts/gepa_smoke_clean.sh` (`gepa_bayes_relaunch.sh` = bayesian cell only) | as above; `GEPA_MAX_METRIC_CALLS` (80) |
| Original (unclean) scaled run | `scripts/scale_all.sh`, `scale_finalize.sh`, `scale_resume_hashed.sh`, `scale_rerun_openai.sh` | also `corpora/scale_M`; kept for provenance; these reference `.venv/bin/python` and the frozen project path directly |
| Rebuild the report | `python scripts/scale_report.py` (env: `SCALE_MANIFEST`, `SCALE_OUT`, `REPORT_OUT`) / `python -m harness.analysis --runs <dir> --out <dir>` | |
| Re-derive splits | `python scripts/scale_split.py` (pinned in `manifests/`; do not regenerate for comparable results) | ALFWorld data |
Caveats: the launchers hard-stop on OpenRouter key usage (a global key counter, not per-run); `trackA_clean.sh` first waits for any other `scale_run.py cell` process; hardcoded values (embedding model, EvoSkill build settings) are listed in CONFIG.md section 8.

## Verified vs not verified (packaging pass, 2026-10-15)
Verified, each from a fresh `git clone` of this branch with an empty `$HOME` and `env -i` (no API keys, no PYTHONPATH):
* `./bootstrap.sh` end-to-end with `uv` (new venv; 283 pinned packages incl. building `fast-downward-textworld` from source; the 4 external repos cloned and HEAD==pin asserted; full ALFWorld download: train=3553, valid_seen=140, valid_unseen=134 game files, identical file list and `valid_seen` content md5 to the dev box's `~/.cache/alfworld`; ALFWorldAdapter built and reset games on it). Re-run is idempotent.
* `./bootstrap.sh` without `uv` (stdlib `venv` + pip, `PATH=/usr/bin:/bin`): installs the same lock (found + fixed a bare-`python` PATH problem in the sdist build).
* `make smoke` in both fresh venvs: M0 scorecards reproduced exactly; pytest **137 passed, 1 xfailed**. `pip freeze` of the fresh venv == `requirements.lock` (0 differences, apart from pip itself in the non-uv venv).
Not verified: any real model call, a scaled run, Python other than 3.13.5, a machine with no `g++`/`make` or no network (bootstrap only detects/reports these), the actual RockAI platform, OfficeQA HF download (needs `HF_TOKEN`).
