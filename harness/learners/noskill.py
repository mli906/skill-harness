"""NullLearner (`learner: noskill`): the no-skill baseline. Returns no
candidates and spends 0 build tokens, so the Scorecard reports library_size 0
(the M3 no-skill config used `trivial` + retrieval_k 0, which built one skill
that was never injected and was therefore reported as library_size 1)."""
from __future__ import annotations
from harness.interfaces.learner import Learner, C1_FIXED_TRACES
from harness.registry import register


@register("learner", "noskill")
class NullLearner(Learner):
    contract = C1_FIXED_TRACES

    def __init__(self):
        self.last_build_tokens = 0

    def induce(self, traces, current):
        self.last_build_tokens = 0
        return []
