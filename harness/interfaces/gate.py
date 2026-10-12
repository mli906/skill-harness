from abc import ABC, abstractmethod
from harness.types import Trace, Skill, SkillLibrary


class Gate(ABC):
    """Promotion: decide which candidates enter the library.

    Harness convention (not part of the abstract signature): gates expose
    `last_rejections` (int) and `last_tokens` (int, probe-replay cost, counted
    toward build cost) after admit()."""
    last_rejections: int = 0
    last_tokens: int = 0

    @abstractmethod
    def admit(self, candidates: list[Skill], current: SkillLibrary,
              probe: list[Trace]) -> SkillLibrary:
        """Return the updated library. May admit, reject, or repair.
        `probe` is a held-out replay set for regression/statistical gates."""
