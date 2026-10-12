"""CLI with --limit for OfficeQA runs (new file; harness.run unchanged):
python -m harness.officeqa_run --config examples/m1_officeqa_noop.yaml --limit 10"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--source", choices=["auto", "hf", "sample"])
    a, rest = ap.parse_known_args(argv)
    if a.limit is not None:
        os.environ["OFFICEQA_LIMIT"] = str(a.limit)
    if a.source:
        os.environ["OFFICEQA_SOURCE"] = a.source
    from harness.run import main as run_main
    run_main(rest)


if __name__ == "__main__":
    main(sys.argv[1:])
