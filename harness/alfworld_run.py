"""CLI for interactive benchmarks: python -m harness.alfworld_run --config X.yaml [--limit N] [--seed S]

run() always builds the agent registered as "react" (M0, unchanged). For the
duration of this command the "react" registry slot is pointed at
ReActMultiStepAgent, which acts in the env (and delegates to the M0 ReActAgent
for non-interactive benchmarks). No M0 file is modified.
"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--limit", type=int)
    a, rest = ap.parse_known_args(argv)
    if a.limit is not None:
        os.environ["ALFWORLD_LIMIT"] = str(a.limit)
    from harness import registry
    registry._load_all()
    from harness.agents.react_multistep import ReActMultiStepAgent
    registry._REG["agent"]["react"] = ReActMultiStepAgent
    from harness.run import main as run_main
    run_main(rest)


if __name__ == "__main__":
    main(sys.argv[1:])
