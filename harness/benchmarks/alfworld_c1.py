"""ALFWorldC1Adapter (`benchmark: alfworld_c1`) for the frozen-corpus (C1) track.

Same env / verify / extensions as ALFWorldAdapter, but TRAIN tasks come from
ALFWorld's own `train` split ($ALFWORLD_TRAIN_SPLIT, default "train") while
TEST tasks are the ALFWorldAdapter test tasks from $ALFWORLD_SPLIT (default
valid_seen). The frozen corpus is generated on train-split games only, so it
cannot overlap the test games by construction (different ALFWorld splits).
"""
from __future__ import annotations
import os
from harness.registry import register
from harness.benchmarks.alfworld import ALFWorldAdapter, list_games, game_type


@register("benchmark", "alfworld_c1")
class ALFWorldC1Adapter(ALFWorldAdapter):
    def __init__(self, limit: int | None = None, train_per_cat: int | None = None):
        super().__init__(limit=limit, train_per_cat=train_per_cat)
        tpc = int(train_per_cat or os.environ.get("ALFWORLD_TRAIN_PER_CAT", 4))
        split = os.environ.get("ALFWORLD_TRAIN_SPLIT", "train")
        for t in list(self._train):           # drop valid_seen train tasks (unused here)
            self._game.pop(t, None); self._cat.pop(t, None)
        self._train = []
        by_type: dict[str, list[str]] = {t: [] for t in self.types}
        for g in list_games(split):
            t = game_type(g)
            if t in by_type and len(by_type[t]) < tpc:
                by_type[t].append(g)
        for t in self.types:
            for g in by_type[t]:
                task = self._render(g)
                self._game[task], self._cat[task] = g, t
                self._train.append(task)
        self.train_split = split
