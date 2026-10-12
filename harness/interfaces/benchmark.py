from abc import ABC, abstractmethod


class Benchmark(ABC):
    """Task source + verifier. An adapter, not a research axis.

    Optional (duck-typed) extension: `category(task) -> str` lets
    category-aware gates enforce per-category constraints."""
    @abstractmethod
    def train_tasks(self) -> list[str]: ...
    @abstractmethod
    def test_tasks(self) -> list[str]: ...
    @abstractmethod
    def verify(self, task: str, output: str) -> bool: ...
