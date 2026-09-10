import pytest


def test_mitigation_strategies_are_deterministic_and_add_overhead() -> None:
    from statefuzz.mitigation import MITIGATION_STRATEGIES, apply_mitigation

    outputs = [apply_mitigation("The answer is", " red", strategy, query_suffix="The answer is") for strategy in MITIGATION_STRATEGIES]
    assert len(set(outputs)) == 3
    assert all("red" in output for output in outputs)
    with pytest.raises(ValueError, match="策略"):
        apply_mitigation("prompt", " red", "unknown", query_suffix="answer")
