from harness import registry


def test_registry_m1_plugins():
    assert registry.available("learner") == ["evoskill", "trivial"]
    assert registry.available("gate") == ["non_regression", "noop"]
    assert registry.available("benchmark") == ["officeqa", "synthetic"]
    assert registry.available("representation") == ["skill_md"]
    assert registry.available("agent") == ["react"]
