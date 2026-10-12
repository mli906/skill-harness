"""SKILL.md representation: top-k embedding retrieval, prepend injection."""
from __future__ import annotations
from harness.interfaces.representation import Representation
from harness.registry import register
from harness.types import Skill, SkillLibrary


@register("representation", "skill_md")
class SkillMdRepresentation(Representation):
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx.embedder)

    def _emb(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed(text)
        return self._cache[text]

    def retrieve(self, task: str, lib: SkillLibrary, k: int) -> list[Skill]:
        if k <= 0 or not lib.skills:
            return []
        q = self._emb(task)
        scored = sorted(lib.skills, key=lambda s: (-self.embedder.cosine(q, self._emb(s.content)), s.id))
        return scored[:k]

    def inject(self, skills: list[Skill], prompt: str) -> str:
        if not skills:
            return prompt
        block = "\n\n".join(f"<skill id=\"{s.id}\">\n{s.content}\n</skill>" for s in skills)
        return f"# Relevant skills\n{block}\n\n{prompt}"
