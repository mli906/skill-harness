"""CLI: python -m harness.agentdojo_run --config X.yaml [--condition clean|attack] [--limit N]

Like harness/alfworld_run.py: run() always builds the agent registered as "react"
(M0, unchanged); for this process that slot is pointed at ReActToolAgent
(`react_tools`), the same ReAct loop with a tool-calling surface. No runner change."""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--condition", choices=["clean", "attack"])
    ap.add_argument("--limit", type=int)
    a, rest = ap.parse_known_args(argv)
    if a.condition:
        os.environ["AGENTDOJO_CONDITION"] = a.condition
    if a.limit is not None:
        os.environ["AGENTDOJO_LIMIT"] = str(a.limit)
    from harness import registry
    registry._load_all()
    from harness.agents.react_tools import ReActToolAgent
    registry._REG["agent"]["react"] = ReActToolAgent
    from harness.run import main as run_main
    run_main(rest)


if __name__ == "__main__":
    main(sys.argv[1:])
