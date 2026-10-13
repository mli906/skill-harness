"""Pin the scaled ALFWorld split (no LLM calls). Writes manifests/alfworld_scale/split.json.

test   = 24 valid_seen games, 4 per task type (seeded shuffle within type)
probe  = 24 train-split games, 4 per type (held out of train)
learn  = remainder of the train split; the frozen corpus uses the first 4/type (24 tasks, k=2)
practice_pool (Track B prefilter) = next 8/type (48) of the train remainder, disjoint from learn corpus
Each entry: game path (relative to json_2.1.1/), type, task_id (sha of rendered task text).
"""
import json, random, sys
from harness.benchmarks.alfworld import list_games, game_type, TASK_TYPES, ALFWorldAdapter
from harness.seeding import task_id

SEED = 20261013
PER = {"test": 4, "probe": 4, "learn_corpus": 4, "practice_pool": 8}


def pick(split):
    by = {t: [] for t in TASK_TYPES}
    for g in list_games(split):
        if game_type(g) in by: by[game_type(g)].append(g)
    for t in TASK_TYPES: random.Random(f"{SEED}:{split}:{t}").shuffle(by[t])
    return by


def main(out="manifests/alfworld_scale/split.json"):
    render = ALFWorldAdapter.__new__(ALFWorldAdapter); render.max_steps = 50
    vs, tr = pick("valid_seen"), pick("train")
    sets = {"test": [(t, g) for t in TASK_TYPES for g in vs[t][:4]]}
    off = 0
    for name in ("probe", "learn_corpus", "practice_pool"):
        n = PER[name]; sets[name] = [(t, g) for t in TASK_TYPES for g in tr[t][off:off + n]]; off += n
    res = {"seed": SEED, "types": TASK_TYPES, "sizes": {k: len(v) for k, v in sets.items()},
           "learn_remainder_size": sum(len(tr[t]) for t in TASK_TYPES) - 24}
    for name, items in sets.items():
        res[name] = [{"game": g.split("json_2.1.1/")[-1], "type": t, "task_id": task_id(render._render(g))}
                     for t, g in items]
    ids = [set(e["task_id"] for e in res[k]) for k in sets]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:]), "overlap"
    json.dump(res, open(out, "w"), indent=1); print(res["sizes"], "remainder", res["learn_remainder_size"])


if __name__ == "__main__":
    main(*sys.argv[1:])
