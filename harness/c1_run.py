"""CLI: python -m harness.c1_run --config examples/c1_alfworld_X.yaml
Frozen-corpus track. Like harness.alfworld_run, points the fixed "react" agent
slot at ReActMultiStepAgent for this process (no M0 edit). Requires the frozen
corpus at `corpus_dir` (generate once with `python -m harness.corpus --regenerate`)."""
import argparse
import json
import sys
from dataclasses import asdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    from harness import registry
    registry._load_all()
    from harness.agents.react_multistep import ReActMultiStepAgent
    registry._REG["agent"]["react"] = ReActMultiStepAgent
    from harness.run import load_config
    from harness.c1_runner import run_c1
    cfg, extra = load_config(a.config, seed=a.seed)
    import os
    os.environ.setdefault("HARNESS_CORPUS_DIR", extra["corpus_dir"])
    sc = run_c1(cfg, extra["corpus_dir"], log_path=extra.get("log_path"))
    print(json.dumps({"config": asdict(cfg), "scorecard": asdict(sc),
                      "library": [s.id for s in run_c1.last_library.skills]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
