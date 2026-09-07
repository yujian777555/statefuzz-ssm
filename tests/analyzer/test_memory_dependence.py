def _scores(a_on_a, b_on_a, a_on_b, b_on_b):
    return {
        "prompt_a": {
            "candidates": [
                {"token_id": 1, "probability": a_on_a},
                {"token_id": 2, "probability": b_on_a},
            ]
        },
        "prompt_b": {
            "candidates": [
                {"token_id": 1, "probability": a_on_b},
                {"token_id": 2, "probability": b_on_b},
            ]
        },
        "candidate_token_ids": [1, 2],
        "prompt_token_counts": [64, 64],
        "matched": True,
        "seed": 7,
        "template_id": 0,
        "candidate_values": [" red", " blue"],
    }


def test_counterfactual_memory_score_requires_both_directions() -> None:
    from statefuzz.analyzer.memory_dependence import compute_counterfactual_memory_score

    result = compute_counterfactual_memory_score(_scores(0.8, 0.2, 0.2, 0.8))
    assert result["direction_a_contrast"] == 0.6
    assert result["direction_b_contrast"] == 0.6
    assert result["memory_dependence_score"] == 0.8


def test_validate_remote_memory_task_rejects_no_counterfactual_dependence() -> None:
    from statefuzz.analyzer.memory_dependence import validate_remote_memory_task

    artifact = validate_remote_memory_task(
        [_scores(0.6, 0.4, 0.6, 0.4), _scores(0.6, 0.4, 0.6, 0.4)],
        threshold=0.55,
    )
    assert artifact["valid_short_context_task"] is False
    assert artifact["seed_count"] == 2


def test_validate_remote_memory_task_accepts_symmetric_dependence() -> None:
    from statefuzz.analyzer.memory_dependence import validate_remote_memory_task

    artifact = validate_remote_memory_task(
        [_scores(0.8, 0.2, 0.2, 0.8), _scores(0.7, 0.3, 0.3, 0.7)],
        threshold=0.55,
    )
    assert artifact["valid_short_context_task"] is True
    assert artifact["memory_dependence_score"] > 0.55


def _logit_record(logit_a_on_a, logit_b_on_a, logit_a_on_b, logit_b_on_b):
    return {
        "prompt_a": {
            "candidates": [
                {"token_id": 1, "probability": 0.001, "logit": logit_a_on_a},
                {"token_id": 2, "probability": 0.0005, "logit": logit_b_on_a},
            ]
        },
        "prompt_b": {
            "candidates": [
                {"token_id": 1, "probability": 0.00001, "logit": logit_a_on_b},
                {"token_id": 2, "probability": 0.000005, "logit": logit_b_on_b},
            ]
        },
        "candidate_token_ids": [1, 2],
        "prompt_token_counts": [64, 64],
        "matched": True,
        "candidate_valid": True,
        "seed": 7,
        "template_id": 1,
        "candidate_values": [" one", " two"],
    }


def test_pairwise_memory_metrics_use_within_prompt_signed_margins() -> None:
    import pytest

    from statefuzz.analyzer.memory_dependence import compute_pairwise_memory_metrics

    metrics = compute_pairwise_memory_metrics(_logit_record(4.0, 1.0, 1.0, 3.0))
    assert metrics["direction_a_margin"] == pytest.approx(3.0)
    assert metrics["direction_b_margin"] == pytest.approx(2.0)
    assert metrics["min_signed_margin"] == pytest.approx(2.0)
    assert metrics["pairwise_preference_valid"] is True


def test_pairwise_metric_survives_absolute_probability_shrinkage() -> None:
    from statefuzz.analyzer.memory_dependence import compute_pairwise_memory_metrics

    metrics = compute_pairwise_memory_metrics(_logit_record(2.0, 0.0, 0.0, 2.0))
    assert metrics["direction_a_pair_probability"] > 0.5
    assert metrics["direction_b_pair_probability"] > 0.5
    assert metrics["pairwise_preference_valid"] is True


def test_pairwise_score_sensitivity_is_secondary_and_threshold_labeled() -> None:
    from statefuzz.analyzer.memory_dependence import compute_pairwise_score_sensitivity

    sensitivity = compute_pairwise_score_sensitivity(
        [
            {"context_tokens": 64, "pairwise_memory_score": 0.95},
            {"context_tokens": 128, "pairwise_memory_score": 0.75},
            {"context_tokens": 256, "pairwise_memory_score": 0.55},
        ],
        thresholds=[0.60, 0.70, 0.80, 0.90],
    )
    assert sensitivity["0.6"] == 256
    assert sensitivity["0.7"] == 256
    assert sensitivity["0.8"] == 128
    assert sensitivity["0.9"] == 128


def test_decompose_pairwise_preference_separates_memory_signal_and_bias() -> None:
    from statefuzz.analyzer.memory_dependence import decompose_pairwise_preference

    strong = decompose_pairwise_preference(
        {"direction_a_margin": 4.0, "direction_b_margin": 2.0}
    )
    assert strong["memory_signal"] == 3.0
    assert strong["lexical_bias"] == 1.0
    assert strong["bias_dominance_margin"] == 2.0
    assert strong["bias_dominated"] is False

    biased = decompose_pairwise_preference(
        {"direction_a_margin": 3.0, "direction_b_margin": -1.0}
    )
    assert biased["memory_signal"] == 1.0
    assert biased["lexical_bias"] == 2.0
    assert biased["bias_dominance_margin"] == -1.0
    assert biased["memory_signal"] > 0.0
    assert biased["bias_dominated"] is True
