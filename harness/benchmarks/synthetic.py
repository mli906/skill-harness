"""SyntheticTasksAdapter: 32 hand-written toy tasks, exact deterministic verifier."""
from __future__ import annotations
from harness.interfaces.benchmark import Benchmark
from harness.registry import register

# (task, category, answer) -- answers are exact strings
_TASKS: list[tuple[str, str, str]] = [
    ("Compute 17 * 23 + 4.", "arithmetic", "395"),
    ("Compute 128 - 47 * 2.", "arithmetic", "34"),
    ("Compute 9 * 9 * 9.", "arithmetic", "729"),
    ("Compute 1000 // 7.", "arithmetic", "142"),
    ("Compute 56 + 78 - 13.", "arithmetic", "121"),
    ("Compute 12 * 12 - 44.", "arithmetic", "100"),
    ("Compute 345 % 17.", "arithmetic", "5"),
    ("Compute 2 ** 10 + 1.", "arithmetic", "1025"),
    ("Reverse the string 'HelloWorld'.", "reverse", "dlroWolleH"),
    ("Reverse the string 'Python3'.", "reverse", "3nohtyP"),
    ("Reverse the string 'aBcDeF'.", "reverse", "FeDcBa"),
    ("Reverse the string 'OpenAgent'.", "reverse", "tnegAnepO"),
    ("Reverse the string 'SkillLib'.", "reverse", "biLllikS"),
    ("Reverse the string 'TraceLog'.", "reverse", "goLecarT"),
    ("Reverse the string 'Gatekeeper'.", "reverse", "repeeketaG"),
    ("Reverse the string 'ReAct'.", "reverse", "tcAeR"),
    ("Count the vowels in 'banana'.", "vowels", "3"),
    ("Count the vowels in 'Encyclopedia'.", "vowels", "5"),
    ("Count the vowels in 'rhythm'.", "vowels", "0"),
    ("Count the vowels in 'Queueing'.", "vowels", "5"),
    ("Count the vowels in 'harness'.", "vowels", "2"),
    ("Count the vowels in 'AUDITABILITY'.", "vowels", "6"),
    ("Count the vowels in 'retrieval'.", "vowels", "4"),
    ("Count the vowels in 'strengths'.", "vowels", "1"),
    ("Convert 3 km to m.", "convert", "3000"),
    ("Convert 7 m to cm.", "convert", "700"),
    ("Convert 5 kg to g.", "convert", "5000"),
    ("Convert 4 h to min.", "convert", "240"),
    ("Convert 9 min to s.", "convert", "540"),
    ("Convert 12 km to m.", "convert", "12000"),
    ("Convert 2 h to min.", "convert", "120"),
    ("Convert 15 kg to g.", "convert", "15000"),
]
_ANS = {t: a for t, _, a in _TASKS}
_CAT = {t: c for t, c, _ in _TASKS}


@register("benchmark", "synthetic")
class SyntheticTasksAdapter(Benchmark):
    """Per category: first 5 tasks train, last 3 test (20 train / 12 test)."""

    def __init__(self):
        self._train, self._test = [], []
        cats = sorted(set(_CAT.values()))
        for c in cats:
            ts = [t for t, cc, _ in _TASKS if cc == c]
            self._train += ts[:5]
            self._test += ts[5:]

    def train_tasks(self) -> list[str]:
        return list(self._train)

    def test_tasks(self) -> list[str]:
        return list(self._test)

    def verify(self, task: str, output: str) -> bool:
        return output.strip() == _ANS[task]

    def category(self, task: str) -> str:
        return _CAT[task]
