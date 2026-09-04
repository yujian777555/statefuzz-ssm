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
def test_state_norm_and_relative_change_are_explicit_evidence() -> None:
    from statefuzz.analyzer.hidden_state import (
        compute_state_norm,
        compute_state_norm_change,
    )

    assert compute_state_norm([3.0, 4.0]) == 5.0
    assert compute_state_norm_change([2.0, 0.0], [1.0, 0.0]) == 0.5


@requires_analyzer
def test_layer_statistics_and_retention_are_machine_readable() -> None:
    from statefuzz.analyzer.hidden_state import (
        compute_state_retention,
        summarize_layer_states,
    )

    summary = summarize_layer_states(
        {"layer0": [3.0, 4.0], "layer1": [1.0, 0.0]}
    )
    assert summary["layer0"]["norm"] == 5.0
    assert summary["layer1"]["norm"] == 1.0
    assert compute_state_retention([2.0, 0.0], [1.0, 0.0]) == 0.5


@requires_analyzer
def test_failure_classifier_exposes_mechanism_category() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert classify_failure("expected", "") == "state_forgetting"
    assert (
        classify_failure("expected", "wrong", [[1.0, 2.0], [1.0, 2.0]])
        == "state_collapse"
    )
    assert classify_failure("expected", "wrong") == "state_pollution"


@requires_analyzer
def test_failure_classifier_uses_state_norm_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert (
        classify_failure("expected", "wrong", state_norms=[10.0, 1.0])
        == "state_forgetting"
    )


@requires_analyzer
def test_failure_diagnosis_contains_mechanism_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import diagnose_failure

    report = diagnose_failure(
        "expected", "wrong", states=[[1.0, 0.0], [1.0, 0.0]]
    )
    assert report["category"] == "state_collapse"
    assert report["evidence"]["state_similarity"] == 1.0

