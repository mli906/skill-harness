"""Endpoint routing for wrapped learners' OWN build-time LLM calls (Trace2Skill, EvoSkill).

Env/config-driven; OpenAI direct stays the default. Keys are only ever passed through
subprocess environment variables (never argv, never logged).

  model "openrouter/<id>"  -> <id> on $OPENROUTER_BASE_URL (https://openrouter.ai/api/v1),
                              key $OPENROUTER_API_KEY. e.g. TRACE2SKILL_MODEL=openrouter/openai/gpt-4o
  else $BUILD_BASE_URL set -> <model> on $BUILD_BASE_URL, key ${BUILD_API_KEY_ENV:-OPENAI_API_KEY}
  else                     -> <model> on https://api.openai.com/v1, key $OPENAI_API_KEY (default)
"""
from __future__ import annotations
import os
from dataclasses import dataclass

OPENAI_BASE = "https://api.openai.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class BuildRoute:
    api_model: str   # model id as the endpoint expects it
    base_url: str
    key_env: str     # NAME of the env var holding the key
    provider: str    # "openai" | "openrouter" | "custom"

    def subprocess_env(self, base: dict | None = None) -> dict:
        """Env for an OpenAI-SDK subprocess: OPENAI_API_KEY/OPENAI_BASE_URL point at the route."""
        env = dict(os.environ if base is None else base)
        if self.key_env in env:
            env["OPENAI_API_KEY"] = env[self.key_env]
        env["OPENAI_BASE_URL"] = self.base_url
        return env


def resolve(model: str) -> BuildRoute:
    if model.startswith("openrouter/"):
        return BuildRoute(model[len("openrouter/"):], os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE),
                          "OPENROUTER_API_KEY", "openrouter")
    if os.environ.get("BUILD_BASE_URL"):
        return BuildRoute(model, os.environ["BUILD_BASE_URL"], os.environ.get("BUILD_API_KEY_ENV", "OPENAI_API_KEY"),
                          "custom")
    return BuildRoute(model, OPENAI_BASE, "OPENAI_API_KEY", "openai")


def evoskill_model_env(model: str) -> tuple[str, dict]:
    """EvoSkill runs OpenHands (litellm). litellm already routes "openrouter/<id>" natively with
    $OPENROUTER_API_KEY, so the model string is kept. With $BUILD_BASE_URL (non-openrouter), the
    OpenAI-compatible base + key are exported for litellm's openai/ provider."""
    env = dict(os.environ)
    if not model.startswith("openrouter/") and os.environ.get("BUILD_BASE_URL"):
        r = resolve(model[len("openai/"):] if model.startswith("openai/") else model)
        env = r.subprocess_env(env)
        env["OPENAI_API_BASE"] = r.base_url
    return model, env
