"""Offline: provider routing in make_client; default OpenAI path unchanged."""
from harness.llm.client import make_client, OpenAICompatibleClient
from harness.llm.openrouter import ProviderClient


def test_openrouter_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    c = make_client("openrouter/qwen/qwen3-32b:free")
    assert isinstance(c, ProviderClient)
    assert c.api_model == "qwen/qwen3-32b:free" and c.model.startswith("openrouter/")
    assert c.base == "https://openrouter.ai/api/v1"


def test_generic_base_url(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://h:8000/v1/"); monkeypatch.setenv("LLM_API_KEY_ENV", "K"); monkeypatch.setenv("K", "y")
    c = make_client("m")
    assert c.model == "http://h:8000/v1|m" and c.api_model == "m"  # cache key distinct from OpenAI "m"


def test_default_unchanged(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False); monkeypatch.setenv("OPENAI_API_KEY", "z")
    c = make_client("gpt-4o-mini")
    assert type(c) is OpenAICompatibleClient and c.model == "gpt-4o-mini"
