"""Build-endpoint routing for wrapped learners (offline; no network)."""
from pathlib import Path
from harness.learners import build_routing as R
from harness.types import Step, Trace, SkillLibrary


def test_default_is_openai(monkeypatch):
    monkeypatch.delenv("BUILD_BASE_URL", raising=False)
    r = R.resolve("gpt-4o")
    assert (r.api_model, r.base_url, r.key_env, r.provider) == ("gpt-4o", R.OPENAI_BASE, "OPENAI_API_KEY", "openai")


def test_openrouter_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key"); monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    r = R.resolve("openrouter/openai/gpt-4o")
    assert (r.api_model, r.base_url, r.provider) == ("openai/gpt-4o", R.OPENROUTER_BASE, "openrouter")
    env = r.subprocess_env()
    assert env["OPENAI_API_KEY"] == "or-key" and env["OPENAI_BASE_URL"] == R.OPENROUTER_BASE


def test_build_base_url(monkeypatch):
    monkeypatch.setenv("BUILD_BASE_URL", "http://vllm:8000/v1"); monkeypatch.setenv("BUILD_API_KEY_ENV", "VLLM_KEY")
    monkeypatch.setenv("VLLM_KEY", "v")
    r = R.resolve("Qwen2.5-72B")
    assert (r.base_url, r.key_env, r.provider) == ("http://vllm:8000/v1", "VLLM_KEY", "custom")
    assert r.subprocess_env()["OPENAI_API_KEY"] == "v"
    m, env = R.evoskill_model_env("openai/Qwen2.5-72B")
    assert m == "openai/Qwen2.5-72B" and env["OPENAI_API_BASE"] == "http://vllm:8000/v1"


def test_evoskill_openrouter_kept_for_litellm(monkeypatch):
    monkeypatch.delenv("BUILD_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    m, env = R.evoskill_model_env("openrouter/openai/gpt-4o")
    assert m == "openrouter/openai/gpt-4o" and env["OPENAI_API_KEY"] == "oa-key"   # untouched


def test_trace2skill_subprocess_args_and_env(monkeypatch, tmp_path):
    from harness.learners import trace2skill as T
    monkeypatch.setenv("TRACE2SKILL_MODEL", "openrouter/openai/gpt-4o")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key"); monkeypatch.setenv("TRACE2SKILL_WORKDIR", str(tmp_path))
    calls = []

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None):
        calls.append((cmd, env))
        class P: returncode = 1
        return P()
    monkeypatch.setattr(T, "checkout", lambda: tmp_path)
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    L = T.Trace2SkillLearner(0, {"t_a": "task a"})
    tr = [Trace("t_a", [Step("o", "go to desk 1", "ok", {})], False, "max_steps", "", 1)]
    try:
        L.induce(tr, SkillLibrary([]))
    except RuntimeError:
        pass
    cmd, env = calls[0]
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-4o" and cmd[cmd.index("--base_url") + 1] == R.OPENROUTER_BASE
    assert env["OPENAI_API_KEY"] == "or-key" and env["OPENAI_BASE_URL"] == R.OPENROUTER_BASE
    assert "or-key" not in " ".join(cmd)                                  # key never on argv


def test_evoskill_default_model_from_openrouter_base():
    from harness.learners.evoskill import EvoSkillLearner
    import inspect
    assert "openrouter/" in inspect.getsource(EvoSkillLearner.induce)
