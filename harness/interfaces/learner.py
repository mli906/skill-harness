from abc import ABC, abstractmethod
from harness.types import Trace, Skill, SkillLibrary


class Learner(ABC):
    """Search operator: traces -> skill candidates."""
    last_build_tokens: int = 0

    @abstractmethod
    def induce(self, traces: list[Trace],
               current: SkillLibrary) -> list[Skill]:
        """Return candidate skills (NOT yet admitted).
        Build-cost tokens must be recorded on self.last_build_tokens."""
