from abc import ABC, abstractmethod
from harness.types import Trace, Skill, SkillLibrary

# Data-access contract of a Learner.
#   C1_fixed_traces : induces ONLY from the traces it is handed (a frozen corpus);
#                     it must never touch the benchmark environment.
#   C2_self_collect : may collect its own rollouts (e.g. EvoSkill's own loop).
C1_FIXED_TRACES = "C1_fixed_traces"
C2_SELF_COLLECT = "C2_self_collect"
CONTRACTS = (C1_FIXED_TRACES, C2_SELF_COLLECT)


class Learner(ABC):
    """Search operator: traces -> skill candidates."""
    last_build_tokens: int = 0
    contract: str = C2_SELF_COLLECT

    @abstractmethod
    def induce(self, traces: list[Trace],
               current: SkillLibrary) -> list[Skill]:
        """Return candidate skills (NOT yet admitted).
        Build-cost tokens must be recorded on self.last_build_tokens."""
