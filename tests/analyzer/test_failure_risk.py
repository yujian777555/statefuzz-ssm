import pytest


def test_wilson_interval_handles_extreme_binomial_counts() -> None:
    from statefuzz.analyzer.failure_risk import wilson_interval

    for failures, total in ((0, 8), (6, 8), (8, 8)):
        low, high = wilson_interval(failures, total)
        assert 0.0 <= low <= high <= 1.0
        assert low <= failures / total <= high

    with pytest.raises(ValueError):
        wilson_interval(9, 8)


def test_failure_risk_curve_requires_complete_seed_sets() -> None:
    from statefuzz.analyzer.failure_risk import summarize_failure_risk_curve

    records = [
        {"seed": seed, "target_budget_tokens": 1000, "actual_input_tokens": 1000, "min_signed_margin": -1.0 if seed < 3 else 1.0, "matched": True, "candidate_valid": True}
        for seed in (1, 2, 3, 4)
    ]
    records.append(
        {"seed": 1, "target_budget_tokens": 1500, "actual_input_tokens": 1500, "min_signed_margin": -1.0, "matched": True, "candidate_valid": True}
    )
    result = summarize_failure_risk_curve(records, expected_seeds=[1, 2, 3, 4])
    complete = next(case for case in result["curve"] if case["target_budget_tokens"] == 1000)
    incomplete = next(case for case in result["curve"] if case["target_budget_tokens"] == 1500)
    assert complete["complete_seed_set"] is True
    assert complete["failure_count"] == 2
    assert 0.0 < complete["wilson_95"][0] < complete["wilson_95"][1] < 1.0
    assert incomplete["complete_seed_set"] is False


def test_seed_transition_intervals_report_censoring_and_nonmonotonicity() -> None:
    from statefuzz.analyzer.failure_risk import summarize_seed_transition_intervals

    records = [
        {"seed": 1, "actual_input_tokens": 100, "min_signed_margin": 1.0},
        {"seed": 1, "actual_input_tokens": 200, "min_signed_margin": -1.0},
        {"seed": 2, "actual_input_tokens": 100, "min_signed_margin": 1.0},
        {"seed": 2, "actual_input_tokens": 200, "min_signed_margin": 1.0},
        {"seed": 3, "actual_input_tokens": -1, "min_signed_margin": -1.0},
        {"seed": 3, "actual_input_tokens": 100, "min_signed_margin": 1.0},
    ]
    result = summarize_seed_transition_intervals(records, expected_seeds=[1, 2, 3])
    by_seed = {case["seed"]: case for case in result["per_seed"]}
    assert by_seed[1]["transition_interval_actual"] == [100, 200]
    assert by_seed[2]["right_censored"] is True
    assert by_seed[3]["nonmonotonic"] is True


def test_exact_paired_mcnemar_uses_only_discordant_pairs() -> None:
    from statefuzz.analyzer.failure_risk import exact_paired_mcnemar

    result = exact_paired_mcnemar(
        {seed: seed <= 6 for seed in range(1, 9)},
        {seed: False for seed in range(1, 9)},
    )
    assert result["mamba_only_fail"] == 6
    assert result["transformer_only_fail"] == 0
    assert result["exact_two_sided_p"] == pytest.approx(0.03125)
    assert result["direction"] == "mamba_higher"
    assert exact_paired_mcnemar(dict.fromkeys(range(8), True), dict.fromkeys(range(8), False))["exact_two_sided_p"] == pytest.approx(0.0078125)


def test_paired_comparison_rejects_duplicates_nan_and_budget_error():
    from statefuzz.analyzer.failure_risk import compare_architecture_failure_risk

    good = {"seed": 1, "target_budget_tokens": 1792, "actual_input_tokens": 1792,
            "min_signed_margin": 1.0, "matched": True, "candidate_valid": True}
    for bad in ([good, good], [{**good, "min_signed_margin": float("nan")}],
                [{**good, "actual_input_tokens": 1900}], [{**good, "candidate_valid": False}]):
        assert not compare_architecture_failure_risk(bad, [good], [1], 1792)["valid"]


def test_shared_paired_failures_do_not_establish_architecture_gap():
    from statefuzz.analyzer.failure_risk import compare_architecture_failure_risk, classify_prospective_architecture_risk

    rows = [{"seed": i, "target_budget_tokens": 1792, "actual_input_tokens": 1792,
             "min_signed_margin": -1.0, "matched": True, "candidate_valid": True} for i in range(8)]
    result = compare_architecture_failure_risk(rows, rows, range(8), 1792)
    assert result["mcnemar"]["discordant_pairs"] == 0
    assert result["mcnemar"]["exact_two_sided_p"] == 1.0
    assert classify_prospective_architecture_risk(result, {}, {}) == "shared_failure_risk"


def test_round018_validation_wrappers_preserve_risk_and_transition_outputs():
    from statefuzz.analyzer.failure_risk import (
        compare_paired_architecture_risk,
        summarize_seed_transition,
    )

    rows = [
        {"seed": seed, "target_budget_tokens": 1792, "actual_input_tokens": 1792,
         "min_signed_margin": -1.0 if seed == 1 else 1.0, "matched": True, "candidate_valid": True}
        for seed in (1, 2)
    ]
    transition = summarize_seed_transition(rows)
    assert transition["transition"]["per_seed"]
    comparison = compare_paired_architecture_risk(rows, rows)
    assert comparison["valid"] is True
    assert comparison["mcnemar"]["discordant_pairs"] == 0


def test_architecture_failure_risk_rejects_incomplete_or_mismatched_records() -> None:
    from statefuzz.analyzer.failure_risk import compare_architecture_failure_risk

    records = [
        {"seed": 1, "target_budget_tokens": 1792, "actual_input_tokens": 1792, "min_signed_margin": -1.0, "matched": True, "candidate_valid": True},
        {"seed": 2, "target_budget_tokens": 1792, "actual_input_tokens": 1792, "min_signed_margin": 1.0, "matched": True, "candidate_valid": True},
    ]
    result = compare_architecture_failure_risk(records, records, [1, 2, 3], 1792)
    assert result["valid"] is False


def test_architecture_failure_risk_and_claim_classifier_are_conservative() -> None:
    from statefuzz.analyzer.failure_risk import (
        classify_prospective_architecture_risk,
        compare_architecture_failure_risk,
    )

    mamba = [
        {"seed": seed, "target_budget_tokens": 1792, "actual_input_tokens": 1792, "min_signed_margin": -1.0 if seed <= 10 else 1.0, "matched": True, "candidate_valid": True}
        for seed in range(1, 17)
    ]
    transformer = [
        {"seed": seed, "target_budget_tokens": 1792, "actual_input_tokens": 1792, "min_signed_margin": 1.0, "matched": True, "candidate_valid": True}
        for seed in range(1, 17)
    ]
    comparison = compare_architecture_failure_risk(mamba, transformer, range(1, 17), 1792)
    assert comparison["valid"] is True
    assert comparison["mamba"]["failure_count"] == 10
    assert comparison["mcnemar"]["direction"] == "mamba_higher"
    assert (
        classify_prospective_architecture_risk(
            comparison,
            {"failure_count": 10, "seed_count": 16},
            {"classification": "structured_repetition_risk_specific"},
        )
        == "architecture_risk_gap_confirmed"
    )


def test_memory_specificity_metrics_and_classifier_are_conservative() -> None:
    from statefuzz.analyzer.failure_risk import (
        behavior_recovery_gap,
        classify_recurrent_state_causality,
        memory_restoration_effect,
        state_specificity_ratio,
    )

    assert memory_restoration_effect(2.0, 0.5) == pytest.approx(1.5)
    assert behavior_recovery_gap(2.0, 0.5) == pytest.approx(1.5)
    assert state_specificity_ratio(2.0, 0.5) == pytest.approx(0.8)
    assert classify_recurrent_state_causality(
        memory_consistent_recovers=True,
        randomized_recovers_equally=False,
        replicated=True,
    ) == "recurrent_state_causal_candidate_confirmed"
    assert classify_recurrent_state_causality(
        memory_consistent_recovers=True,
        randomized_recovers_equally=True,
        replicated=True,
    ) == "recurrent_state_intervention_effect_but_not_memory_specific"


def test_compare_memory_state_interventions_reports_specificity_gap() -> None:
    from statefuzz.analyzer.failure_risk import compare_memory_state_interventions

    result = compare_memory_state_interventions(
        [{"margin_b_minus_a": -1.0}] * 4,
        [{"margin_b_minus_a": -1.0}] * 4,
        [{"margin_b_minus_a": 2.0}] * 4,
        [{"margin_b_minus_a": -0.5}] * 4,
    )
    assert result["correct_memory_recovery_rate"] == 1.0
    assert result["wrong_memory_recovery_rate"] == 0.0
    assert result["randomized_recovery_rate"] == 0.0
    assert result["specificity_gap"] == 1.0
    with pytest.raises(ValueError, match="相同"):
        compare_memory_state_interventions([1.0], [1.0, 2.0], [1.0], [1.0])
