"""M0 end-to-end smoke test: deterministic stub model, no network, no API keys, no spend.

Runs the two M0 configs (examples/m0.yaml = NoOpGate, examples/m0_nonregression.yaml = NonRegressionGate; trivial learner,
synthetic benchmark, seed 0) through harness.runner.run and asserts the exact Scorecards.

Derivation of the expected values: run once on a known-good checkout (main @ 232bb57) and recorded here. The md5 is
md5(json.dumps(dataclasses.asdict(scorecard), sort_keys=True)). Any change to M0 behavior flips these.
Exit code 0 = reproduced; non-zero = mismatch (prints a diff).
"""
import hashlib, json, os, sys, tempfile
from dataclasses import asdict
for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "LLM_BASE_URL", "HF_TOKEN"):
    os.environ.pop(k, None)           # prove no credentials are needed
os.environ["HARNESS_LOG_DIR"] = tempfile.mkdtemp(prefix="smoke_")
from harness.run import load_config
from harness.runner import run

EXPECTED = {
    "examples/m0.yaml": ("60a5bd42f0dd0b6479e2ec266cd96552", dict(
        success_rate=0.8333333333333334, avg_steps=1.25, tokens_eval=1342, tokens_build=219, flips_w2c=2, flips_c2w=0,
        library_size=1, auditability=1.0, unauthorized_actions=0, promotion_rejections=0)),
    "examples/m0_nonregression.yaml": ("ed7af3a9aecbe49ffa1988f2418bd9e5", dict(
        success_rate=0.8333333333333334, avg_steps=1.25, tokens_eval=1342, tokens_build=1602, flips_w2c=2, flips_c2w=0,
        library_size=1, auditability=1.0, unauthorized_actions=0, promotion_rejections=0)),
}
bad = 0
for path, (md5, fields) in EXPECTED.items():
    cfg, extra = load_config(path)
    assert cfg.base_model == "stub" and cfg.benchmark == "synthetic" and cfg.learner == "trivial" and cfg.seed == 0, cfg
    sc = asdict(run(cfg, log_path=os.path.join(os.environ["HARNESS_LOG_DIR"], os.path.basename(path) + ".jsonl")))
    got = hashlib.md5(json.dumps(sc, sort_keys=True).encode()).hexdigest()
    ok = sc == fields and got == md5
    bad += not ok
    print(("PASS " if ok else "FAIL ") + f"{path:34s} gate={cfg.gate:14s} success={sc['success_rate']:.4f} md5={got}")
    if not ok:
        print("  expected", fields, md5); print("  got     ", sc, got)
print("M0 smoke:", "OK (scorecards reproduced exactly)" if not bad else f"{bad} MISMATCH(ES)")
sys.exit(1 if bad else 0)
