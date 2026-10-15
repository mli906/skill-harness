import argparse
import sys

from .report import analyze, write_outputs


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m harness.analysis", description="Results-analysis pipeline (read-only over run logs).")
    p.add_argument("--runs", required=True, help="directory of per-run JSONL logs (searched recursively)")
    p.add_argument("--out", default="analysis_out", help="output directory (report.md, CSV, JSON, plots)")
    p.add_argument("--expected", default=None,
                   help="optional JSON list of expected cells, e.g. "
                        '[{"benchmark":..,"track":..,"learner":..,"gate":..,"model":..}]; '
                        "default: learners x gates observed per (benchmark, track, model, variant)")
    p.add_argument("--bootstrap", type=int, default=10000, help="paired bootstrap resamples (B)")
    p.add_argument("--seed", type=int, default=20261013, help="bootstrap RNG seed")
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args(argv)
    res = analyze(a.runs, expected=a.expected, B=a.bootstrap, seed=a.seed, alpha=a.alpha)
    path = write_outputs(res, a.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
