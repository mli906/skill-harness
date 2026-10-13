"""Input-length guard for OpenAIEmbedder, kept out of the M0 embedding module.

The OpenAI embeddings endpoint rejects inputs over 8192 tokens (HTTP 400); long
OfficeQA/ALFWorld task strings can exceed that. `apply()` wraps
OpenAIEmbedder.embed to truncate to MAX_CHARS. Benchmarks that produce long
tasks (officeqa_v2, alfworld) call apply() at import.
"""
from __future__ import annotations

MAX_CHARS = 12000
_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from harness.llm.embedding import OpenAIEmbedder
    orig = OpenAIEmbedder.embed

    def embed(self, text: str):
        return orig(self, text[:MAX_CHARS])
    OpenAIEmbedder.embed = embed
    _APPLIED = True
