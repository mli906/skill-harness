"""Configurable OpenAI-compatible provider client (OpenRouter et al.).

Selected by make_client when base_model is "openrouter/<model-id>", e.g.
  base_model: openrouter/google/gemma-4-31b-it:free
or for any other OpenAI-compatible endpoint via env:
  LLM_BASE_URL=https://host/v1  LLM_API_KEY_ENV=MY_KEY_VAR  (base_model = plain model id)
self.model keeps the provider-qualified name (prefix / base_url), so the disk
cache key (model, prompt, seed, max_tokens) never collides with OpenAI results.
LLM_EXTRA_BODY (JSON) is merged into every request body.
Retries 429 / 5xx / transient errors with exponential backoff (free tiers).
"""
from __future__ import annotations
import json
import os
import random
import time
import urllib.error
import urllib.request
from harness.llm.client import LLMClient, LLMResponse

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class ProviderClient(LLMClient):
    def __init__(self, cache_model: str, api_model: str, base_url: str, key_env: str,
                 max_retries: int | None = None):
        super().__init__(cache_model)
        self.api_model = api_model
        self.base = base_url.rstrip("/")
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"{key_env} not set")
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", max_retries or 8))
        # optional JSON merged into each request, e.g. '{"reasoning":{"enabled":false}}'
        self.extra_body = json.loads(os.environ.get("LLM_EXTRA_BODY", "{}"))
        self.stats = {"calls": 0, "retries": 0, "rate_limited": 0, "latency_s": 0.0}

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json",
                                              "X-Title": "skill-harness"})
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read())
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError(f"no choices: {str(data.get('error', data))[:200]}")
                self.stats["latency_s"] += time.time() - t0
                return data
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self.stats["rate_limited"] += 1
                elif e.code < 500:
                    raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
                err = e
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
                err = e
            if attempt == self.max_retries:
                raise RuntimeError(f"provider call failed after retries: {err}")
            self.stats["retries"] += 1
            time.sleep(delay + random.random())
            delay = min(delay * 2, 90.0)

    def complete(self, prompt: str, *, seed: int, max_tokens: int = 1024) -> LLMResponse:
        body = {"model": self.api_model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "seed": seed, "max_tokens": max_tokens}
        body.update(self.extra_body)
        data = self._post(body)
        self.stats["calls"] += 1
        u = data.get("usage") or {}
        return LLMResponse(data["choices"][0]["message"].get("content") or "",
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def make_provider_client(base_model: str) -> LLMClient | None:
    """Return a provider client if base_model/env select one, else None (=> default OpenAI)."""
    if base_model.startswith("openrouter/"):
        mid = base_model[len("openrouter/"):]
        return ProviderClient(base_model, mid, os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE),
                              "OPENROUTER_API_KEY")
    base = os.environ.get("LLM_BASE_URL")
    if base:
        return ProviderClient(f"{base.rstrip('/')}|{base_model}", base_model, base,
                              os.environ.get("LLM_API_KEY_ENV", "OPENAI_API_KEY"))
    return None
