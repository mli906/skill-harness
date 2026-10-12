"""LLM client abstraction. All calls are pinned to cfg.base_model and carry a seed."""
from __future__ import annotations
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(ABC):
    def __init__(self, model: str):
        self.model = model  # pinned

    @abstractmethod
    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse: ...


class OpenAICompatibleClient(LLMClient):
    """Any OpenAI-compatible /chat/completions endpoint.
    Env: OPENAI_API_KEY (required), OPENAI_BASE_URL (default api.openai.com/v1)."""

    def __init__(self, model: str):
        super().__init__(model)
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set (use base_model: stub for offline runs)")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "seed": seed, "max_tokens": max_tokens}
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        u = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def make_client(base_model: str) -> LLMClient:
    """'stub' -> offline deterministic stub; anything else -> OpenAI (default real backend)."""
    if base_model == "stub" or base_model.startswith("stub:"):
        from .stub import StubClient
        return StubClient(base_model)
    return OpenAICompatibleClient(base_model)
