"""Replay a command plan in an ALFWorld game; print 1.0 if the env reports won.
Used as EvoSkill's `script` scorer (see EvoSkill rows in alfworld_evoskill).

python -m harness.benchmarks.alfworld_score --map MAP.json --game KEY --plan cmd1 ; cmd2 ; ...
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--plan", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    try:
        gamefile = json.load(open(a.map))[a.game.strip().lower()]
        from harness.benchmarks.alfworld import make_env
        plan = " ".join(a.plan)
        cmds = [c.strip() for c in plan.replace("\n", ";").split(";") if c.strip()]
        env = make_env(gamefile, 60)
        env.reset()
        won = False
        for c in cmds[:60]:
            _o, _s, done, info = env.step([c])
            won = won or bool(info["won"][0])
            if done[0]:
                break
        print("1.0" if won else "0.0")
    except Exception:  # noqa: BLE001
        print("0.0")


if __name__ == "__main__":
    main(sys.argv[1:])
