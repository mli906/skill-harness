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
