"""Four-way disjoint task split + the gate-probe isolation invariant.

  learn_corpus -- C1 learners distill from this (unchanged: the runner's learn split)
  gepa_valset  -- optimizer-style C2 learners (GEPA) optimise/score on this
  gate_probe   -- the gate judges admission on this; NO learner may see it
  test         -- final eval (benchmark.test_tasks(), unchanged)

gate_probe is computed exactly as before (runner._split_probe), so it is unchanged for every
run. gepa_valset is carved out of the learn split ONLY when the learner opts in
(`uses_valset = True`); otherwise it is empty and the learn split is byte-identical to the old
two-way split (keeps M0 / test_exit outputs unchanged).
Size: VALSET_PER_CATEGORY tasks per category (env GEPA_VALSET_PER_CATEGORY, default 2 = the probe
size); for uncategorised benchmarks, 30% of the learn split.
"""
from __future__ import annotations
import os
from collections import defaultdict
from harness.seeding import rng, task_id

VALSET_PER_CATEGORY = 2


class ProbeLeakError(RuntimeError):
    """A learner tried to read / optimise on gate_probe tasks."""


def carve_valset(bench, learn: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Seeded, category-stratified hold-out of `learn` -> (learn_corpus, gepa_valset)."""
    n_per = int(os.environ.get("GEPA_VALSET_PER_CATEGORY", VALSET_PER_CATEGORY))
    cat = getattr(bench, "category", None)
    groups = defaultdict(list)
    for t in learn:
        groups[cat(t) if callable(cat) else "_all"].append(t)
    keep, val = [], []
    for c in sorted(groups):
        g = list(groups[c])
        rng(seed, "gepa-valset", c).shuffle(g)
        n = n_per if callable(cat) else max(1, len(g) * 3 // 10)
        n = min(n, max(0, len(g) - 1))      # never empty a category's learn corpus
        val += g[:n]; keep += g[n:]
    return keep, val


def check_disjoint(**slices: list[str]) -> None:
    """Assert all named slices are pairwise disjoint (by task text)."""
    names = list(slices)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(slices[a]) & set(slices[b])
            if both:
                raise ProbeLeakError(f"split slices {a} and {b} overlap on {len(both)} task(s)")


def guard_tasks(tasks, forbidden_ids: set[str], who: str) -> None:
    """Hard error if any task (text or {'id','text'} dict) is a gate_probe task."""
    for t in tasks:
        tid = t["id"] if isinstance(t, dict) else task_id(t)
        if tid in forbidden_ids:
            raise ProbeLeakError(f"{who} touched gate_probe task {tid}")


def is_auditable(skill, learner_ids: set[str], heldout_ids: set[str]) -> bool:
    """Runner's auditability predicate: non-empty provenance, every id a training-side task the
    learner saw (learn traces U gepa_valset), none from the held-out probe or test set."""
    return bool(skill.provenance) and set(skill.provenance) <= learner_ids \
        and not set(skill.provenance) & heldout_ids


class FrozenOriginLeakError(ProbeLeakError):
    """A frozen/transfer skill's ORIGIN task_ids overlap this run's probe or test set."""


def is_frozen_learner(learner, name: str = "") -> bool:
    """Frozen/transfer learners re-use skills distilled outside this run. Convention: they expose
    ``origin_task_ids()``; any learner whose registry name contains 'frozen' or 'transfer' is
    also treated as one (so a future learner that forgets the method fails loudly)."""
    return callable(getattr(learner, "origin_task_ids", None)) or any(
        w in (name or "").lower() for w in ("frozen", "transfer"))


def check_frozen_origin(learner, probe_ids: set[str], test_ids: set[str], name: str = "") -> list[str] | None:
    """Invariant: a frozen/transfer skill's origin corpus is disjoint from {probe, test}.

    Resolves origin ids BEFORE the run proceeds. Raises FrozenOriginLeakError naming the offending
    ids, or if a frozen learner's origin cannot be resolved (disjointness unverifiable). Returns
    the resolved origin ids (None for ordinary learners)."""
    if not is_frozen_learner(learner, name):
        return None
    fn = getattr(learner, "origin_task_ids", None)
    if not callable(fn):
        raise FrozenOriginLeakError(f"frozen/transfer learner {name!r} exposes no origin_task_ids(); "
                                    "origin/probe/test disjointness cannot be verified")
    origin = [str(i) for i in fn()]
    from harness.provenance import ORIGIN_UNAVAILABLE
    if not origin or ORIGIN_UNAVAILABLE in origin:
        raise FrozenOriginLeakError(f"frozen/transfer learner {name!r}: origin task_ids unavailable; "
                                    "origin/probe/test disjointness cannot be verified")
    bad_p, bad_t = sorted(set(origin) & set(probe_ids)), sorted(set(origin) & set(test_ids))
    if bad_p or bad_t:
        raise FrozenOriginLeakError(f"frozen/transfer learner {name!r}: origin task_ids overlap "
                                    f"probe {bad_p} / test {bad_t}")
    return origin
