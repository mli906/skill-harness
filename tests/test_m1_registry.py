from harness import registry


def test_registry_contains_milestone_plugins():
    assert {"evoskill", "trivial", "raw_trace_rag", "trace2skill", "noskill"} <= set(registry.available("learner"))
    assert registry.available("gate") == ["non_regression", "noop"]  # no new gates
    assert {"officeqa", "officeqa_v2", "alfworld", "alfworld_c1", "synthetic"} <= set(registry.available("benchmark"))
    assert {"skill_md", "raw_trace"} <= set(registry.available("representation"))
    assert {"react", "react_multistep"} <= set(registry.available("agent"))
