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
