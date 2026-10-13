"""Disk cache for LLM completions keyed on (model, prompt, seed, max_tokens).

All harness calls use temperature 0 + a fixed derived seed, so an identical
request is treated as having an identical response; caching lets the no-skill
parts of different runs (train traces, empty-library baseline) be shared
across runs instead of re-billed. Recorded token counts are the original ones.
Cache dir: $HARNESS_LLM_CACHE (default ~/.cache/skill-harness/llm_cache), outside the repo.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from harness.llm.client import LLMClient, LLMResponse


class CachedClient(LLMClient):
    def __init__(self, inner: LLMClient):
        super().__init__(inner.model)
        self.inner = inner
        self.dir = Path(os.environ.get("HARNESS_LLM_CACHE", Path.home() / ".cache" / "skill-harness" / "llm_cache"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse:
        key = hashlib.sha256(json.dumps([self.model, prompt, seed, max_tokens]).encode()).hexdigest()
        p = self.dir / key[:2] / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text()); self.hits += 1
            return LLMResponse(d["text"], d["prompt_tokens"], d["completion_tokens"])
        r = self.inner.complete(prompt, seed=seed, max_tokens=max_tokens)
        self.misses += 1
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({"text": r.text, "prompt_tokens": r.prompt_tokens,
                                 "completion_tokens": r.completion_tokens}))
        return r
