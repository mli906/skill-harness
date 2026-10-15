from harness import registry


def test_registry_contains_milestone_plugins():
    assert {"evoskill", "evoskill_frozen", "trivial", "raw_trace_rag", "trace2skill", "noskill", "awm", "gepa"} <= set(registry.available("learner"))
    assert registry.available("gate") == ["bayesian", "governance", "non_regression", "noop"]
    assert {"officeqa", "officeqa_v2", "alfworld", "alfworld_c1", "agentdojo_banking", "synthetic"} <= set(registry.available("benchmark"))
    assert {"skill_md", "raw_trace"} <= set(registry.available("representation"))
    assert {"react", "react_multistep"} <= set(registry.available("agent"))
