"""CLI: python -m harness.run --config examples/m0.yaml [--seed N]"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, fields
import yaml
from harness.runner import run
from harness.types import RunConfig


def load_config(path: str, **overrides) -> tuple[RunConfig, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    names = {f.name for f in fields(RunConfig)}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    cfg = RunConfig(**{k: raw[k] for k in names})
    return cfg, {k: v for k, v in raw.items() if k not in names}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    cfg, extra = load_config(a.config, seed=a.seed)
    sc = run(cfg, log_path=extra.get("log_path"))
    print(json.dumps({"config": asdict(cfg), "scorecard": asdict(sc),
                      "library": [s.id for s in run.last_library.skills]}, indent=2))


if __name__ == "__main__":
    main()
