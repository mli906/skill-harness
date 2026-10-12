from abc import ABC, abstractmethod
from harness.types import Trace, SkillLibrary
from harness.interfaces.representation import Representation


class ConsumingAgent(ABC):
    """Fixed executor. Same across all runs."""
    @abstractmethod
    def run_task(self, task: str, lib: SkillLibrary,
                 rep: Representation, k: int) -> Trace: ...
