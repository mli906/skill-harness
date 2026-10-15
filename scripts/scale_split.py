"""Pin the scaled ALFWorld split (no LLM calls). Writes manifests/alfworld_scale/split.json.

test   = 24 valid_seen games, 4 per task type (seeded shuffle within type)
probe  = 24 train-split games, 4 per type (held out of train)
learn  = remainder of the train split; the frozen corpus uses the first 4/type (24 tasks, k=2)
practice_pool (Track B prefilter) = next 8/type (48) of the train remainder, disjoint from learn corpus
Each entry: game path (relative to json_2.1.1/), type, task_id (sha of rendered task text).

EXCLUDE (env SCALE_SPLIT_EXCLUDE, comma-separated task_ids, or argv[2]): task_ids that may not
appear in ANY set (used to keep frozen/transfer ORIGIN tasks out of test/probe). Excluded games
are skipped in the same seeded order, so with no hits the split is identical to the original.
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


def main(out="manifests/alfworld_scale/split.json", exclude=None):
    import os
    render = ALFWorldAdapter.__new__(ALFWorldAdapter); render.max_steps = 50
    excl = {x for x in (exclude or os.environ.get("SCALE_SPLIT_EXCLUDE", "")).split(",") if x}
    skipped = []
    vs, tr = pick("valid_seen"), pick("train")

    def take(lst, n, pos):
        got = []
        while len(got) < n:
            g = lst[pos]; pos += 1
            tid = task_id(render._render(g)) if excl else None
            if tid in excl:
                skipped.append({"game": g.split("json_2.1.1/")[-1], "task_id": tid}); continue
            got.append(g)
        return got, pos

    sets = {"test": []}
    for t in TASK_TYPES:
        got, _ = take(vs[t], 4, 0); sets["test"] += [(t, g) for g in got]
    pos = {t: 0 for t in TASK_TYPES}
    for name in ("probe", "learn_corpus", "practice_pool"):
        sets[name] = []
        for t in TASK_TYPES:
            got, pos[t] = take(tr[t], PER[name], pos[t]); sets[name] += [(t, g) for g in got]
    res = {"seed": SEED, "types": TASK_TYPES, "sizes": {k: len(v) for k, v in sets.items()},
           "learn_remainder_size": sum(len(tr[t]) for t in TASK_TYPES) - 24,
           "exclude": sorted(excl), "excluded_skipped": skipped}
    for name, items in sets.items():
        res[name] = [{"game": g.split("json_2.1.1/")[-1], "type": t, "task_id": task_id(render._render(g))}
                     for t, g in items]
    ids = [set(e["task_id"] for e in res[k]) for k in sets]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:]), "overlap"
    assert not (set().union(*ids) & excl), "excluded id present"
    json.dump(res, open(out, "w"), indent=1); print(res["sizes"], "remainder", res["learn_remainder_size"])


if __name__ == "__main__":
    main(*sys.argv[1:])
