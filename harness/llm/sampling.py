"""Sampling OpenAI client for the frozen-corpus generator: same endpoint as the
M0 OpenAICompatibleClient but with a FIXED, configurable temperature (the M0
client hard-codes temperature 0, which would make k rollouts near-identical)."""
from __future__ import annotations
import json
import urllib.request
from harness.llm.client import LLMResponse, OpenAICompatibleClient


class SamplingOpenAIClient(OpenAICompatibleClient):
    def __init__(self, model: str, temperature: float):
        super().__init__(model)
        self.temperature = float(temperature)

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "seed": seed, "max_tokens": max_tokens}
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        u = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
