import importlib.util

import pytest


ANALYZER_SPEC = importlib.util.find_spec("statefuzz.analyzer.hidden_state")
requires_analyzer = pytest.mark.skipif(
    ANALYZER_SPEC is None, reason="analyzer尚未实现"
)


def test_analyzer_modules_exist() -> None:
    assert ANALYZER_SPEC is not None
    assert importlib.util.find_spec("statefuzz.analyzer.failure_classifier") is not None


@requires_analyzer
def test_state_similarity_is_cosine_and_deterministic() -> None:
    from statefuzz.analyzer.hidden_state import compute_state_similarity

    assert compute_state_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert compute_state_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


@requires_analyzer
def test_state_collapse_detects_repeated_hidden_states() -> None:
    from statefuzz.analyzer.hidden_state import detect_state_collapse

    assert detect_state_collapse([[1.0, 2.0], [1.0, 2.0]])
    assert not detect_state_collapse([[1.0, 0.0], [0.0, 1.0]])


@requires_analyzer
def test_failure_classifier_exposes_mechanism_category() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert classify_failure("expected", "") == "state_forgetting"
    assert (
        classify_failure("expected", "wrong", [[1.0, 2.0], [1.0, 2.0]])
        == "state_collapse"
    )
    assert classify_failure("expected", "wrong") == "state_pollution"

