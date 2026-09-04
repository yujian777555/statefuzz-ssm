import json

import pytest


def test_search_boundary_finds_smallest_degrading_configuration() -> None:
    from statefuzz.search.engine import search_boundary

    def evaluate(config):
        score = 1.0 if config.context_tokens < 2048 else 0.0
        return {"score": score, "evidence": {"state_similarity": 0.2}}

    result = search_boundary(
        evaluate,
        context_lengths=[2048, 512, 1024],
        target_positions=[0.0, 0.5],
        interference_strengths=[0.0, 0.5],
    )
    assert result["boundary"]["context_tokens"] == 2048
    assert result["boundary"]["target_position"] == 0.0
    assert len(result["cases"]) == 12
    json.dumps(result)


def test_search_boundary_is_reproducible_and_ranks_failures() -> None:
    from statefuzz.search.engine import rank_failure_cases, search_boundary

    def evaluate(config):
        return 0.1 * config.interference_strength

    kwargs = {
        "context_lengths": [512, 1024],
        "target_positions": [0.5],
        "interference_strengths": [0.2, 0.8],
    }
    left = search_boundary(evaluate, **kwargs)
    right = search_boundary(evaluate, **kwargs)
    assert left == right
    ranked = rank_failure_cases(left["cases"])
    assert ranked[0]["score"] <= ranked[-1]["score"]


def test_search_boundary_rejects_invalid_space_or_scores() -> None:
    from statefuzz.search.engine import search_boundary

    with pytest.raises(ValueError, match="context_lengths"):
        search_boundary(lambda _: 1.0, context_lengths=[])
    with pytest.raises(ValueError, match="score"):
        search_boundary(lambda _: 2.0, context_lengths=[512])


def test_failure_artifact_contains_trigger_and_hidden_state_evidence() -> None:
    from statefuzz.search.engine import build_failure_artifact, search_boundary

    result = search_boundary(
        lambda config: {"score": 0.0, "evidence": {"state_similarity": 1.0}},
        context_lengths=[2048],
        target_positions=[0.25],
        interference_strengths=[0.5],
    )
    artifact = build_failure_artifact(
        "fake/mamba", result, "state_collapse", {"state_similarity": 1.0}
    )
    assert artifact["model_id"] == "fake/mamba"
    assert artifact["effective_memory_boundary"] == 2048
    assert artifact["trigger_configuration"]["target_position"] == 0.25
    assert "state_similarity" in artifact["hidden_state_evidence"]
    json.dumps(artifact)


def test_adaptive_search_refines_observed_failure_interval() -> None:
    from statefuzz.search.engine import adaptive_search_boundary

    def evaluate(config):
        return {"score": 1.0 if config.context_tokens < 1500 else 0.0}

    result = adaptive_search_boundary(
        evaluate,
        min_context=512,
        max_context=4096,
        target_position=0.25,
        interference_strength=0.5,
        tolerance=64,
        seed=9,
    )
    assert result["observed_failure_context_tokens"] is not None
    assert result["estimated_capability_boundary"] < result["observed_failure_context_tokens"]
    low, high = result["confidence_interval"]
    assert high - low <= 64
    assert result["capability_curve"]
    json.dumps(result)


def test_calibrated_search_refuses_invalid_baseline_and_classifies_reason() -> None:
    from statefuzz.search.engine import search_calibrated_boundary

    result = search_calibrated_boundary(
        lambda config: {"score": 0.0, "evidence": {"reason": "mismatch"}},
        min_context=64,
        max_context=512,
        threshold=0.5,
        baseline_threshold=0.5,
    )
    assert result["valid_baseline"] is False
    assert result["boundary"] is None
    assert result["failure_reason"] == "baseline_failure"

