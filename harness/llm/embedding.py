"""Embeddings: deterministic hashed bag-of-words, with an optional real-embedding hook.
OpenAI embeddings are the default real hook (see make_embedder)."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import urllib.request


class Embedder:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
        return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


class HashedBowEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v


class OpenAIEmbedder(Embedder):
    MAX_CHARS = 12000  # stay under the 8192-token input limit for numeric/table text
    def __init__(self, model: str):
        self.model = model
        self.key = os.environ["OPENAI_API_KEY"]
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def embed(self, text: str) -> list[float]:
        req = urllib.request.Request(f"{self.base}/embeddings",
                                     data=json.dumps({"model": self.model, "input": text[:self.MAX_CHARS]}).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["data"][0]["embedding"]


DEFAULT_OPENAI_EMBEDDING = "text-embedding-3-small"


def make_embedder(base_model: str = "stub") -> Embedder:
    """HARNESS_EMBEDDING: 'auto' (default) | 'hashed' | 'openai:<model>'.
    auto -> OpenAI embeddings when running a real (non-stub) model with
    OPENAI_API_KEY set; otherwise the deterministic hashed embedder."""
    spec = os.environ.get("HARNESS_EMBEDDING", "auto")
    if spec.startswith("openai:"):
        return OpenAIEmbedder(spec.split(":", 1)[1])
    if spec == "auto" and not base_model.startswith("stub") and os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder(DEFAULT_OPENAI_EMBEDDING)
    return HashedBowEmbedder()
