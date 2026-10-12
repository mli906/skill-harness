from abc import ABC, abstractmethod
from harness.types import Skill, SkillLibrary


class Representation(ABC):
    """How skills are stored, retrieved, and injected."""
    @abstractmethod
    def retrieve(self, task: str, lib: SkillLibrary, k: int) -> list[Skill]: ...
    @abstractmethod
    def inject(self, skills: list[Skill], prompt: str) -> str: ...
