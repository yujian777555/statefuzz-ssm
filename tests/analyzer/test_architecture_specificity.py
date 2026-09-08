import pytest


def _model(boundary, *, max_actual=4096, availability="available"):
    return {
        "availability": availability,
        "boundary_kind": boundary,
        "actual_boundary_interval": [1115, 1388] if boundary == "replicated_zero_crossing" else None,
        "max_actual_tokens": max_actual,
        "min_actual_tokens": 256,
    }


def test_architecture_specificity_requires_transformer_coverage() -> None:
    from statefuzz.analyzer.architecture_specificity import compare_architecture_results

    result = compare_architecture_results(
        _model("replicated_zero_crossing"),
        _model("lower_bound", max_actual=1024),
    )
    assert result["specificity_classification"] == "inconclusive_control"
    assert result["coverage_sufficient"] is False


def test_architecture_specificity_identifies_ssm_specific_candidate() -> None:
    from statefuzz.analyzer.architecture_specificity import compare_architecture_results

    result = compare_architecture_results(
        _model("replicated_zero_crossing"),
        _model("lower_bound", max_actual=4096),
    )
    assert result["specificity_classification"] == "ssm_specific_candidate"
    assert result["coverage_sufficient"] is True


def test_architecture_specificity_identifies_shared_decay_and_differential() -> None:
    from statefuzz.analyzer.architecture_specificity import compare_architecture_results

    shared = compare_architecture_results(
        _model("replicated_zero_crossing"),
        _model("replicated_zero_crossing"),
    )
    assert shared["specificity_classification"] == "shared_base_lm_decay"

    differential = compare_architecture_results(
        _model("replicated_zero_crossing"),
        {
            **_model("replicated_zero_crossing"),
            "actual_boundary_interval": [3000, 3500],
        },
    )
    assert differential["specificity_classification"] == "architecture_differential"


def test_architecture_specificity_rejects_nonmonotonic_curves() -> None:
    from statefuzz.analyzer.architecture_specificity import compare_architecture_results

    result = compare_architecture_results(
        {**_model("nonmonotonic"), "boundary_kind": "nonmonotonic"},
        _model("replicated_zero_crossing"),
    )
    assert result["specificity_classification"] == "inconclusive_nonmonotonic"


def _stressor_result(seeds, boundary_kind, interval=None, max_actual=1920):
    return {
        "seed_set": seeds,
        "boundary_kind": boundary_kind,
        "actual_boundary_interval": interval,
        "max_actual_tokens": max_actual,
    }


def test_confirm_architecture_stressor_covers_confirmation_outcomes() -> None:
    from statefuzz.analyzer.architecture_specificity import confirm_architecture_stressor

    discovery = _stressor_result([17, 18, 19, 20], "replicated_zero_crossing", [1531, 1791])
    confirmation = _stressor_result(
        [21, 22, 23, 24, 25, 26, 27, 28],
        "replicated_zero_crossing",
        [1500, 1800],
    )
    transformer = _stressor_result(
        [21, 22, 23, 24, 25, 26, 27, 28], "lower_bound", max_actual=1920
    )
    confirmed = confirm_architecture_stressor(discovery, confirmation, transformer)
    assert confirmed["confirmation_classification"] == "confirmed_ssm_stressor"
    assert confirmed["coverage_sufficient"] is True

    shared = confirm_architecture_stressor(
        discovery,
        confirmation,
        _stressor_result(
            [21, 22, 23, 24, 25, 26, 27, 28],
            "replicated_zero_crossing",
            [1600, 1850],
        ),
    )
    assert shared["confirmation_classification"] == "shared_stressor_decay"


def test_confirm_architecture_stressor_handles_nonreplication_and_censoring() -> None:
    from statefuzz.analyzer.architecture_specificity import confirm_architecture_stressor

    discovery = _stressor_result([17, 18, 19, 20], "replicated_zero_crossing", [1531, 1791])
    no_boundary = _stressor_result([21, 22, 23, 24], "lower_bound")
    transformer = _stressor_result([21, 22, 23, 24], "lower_bound", max_actual=1920)
    assert (
        confirm_architecture_stressor(discovery, no_boundary, transformer)[
            "confirmation_classification"
        ]
        == "candidate_not_replicated"
    )

    confirmation = _stressor_result(
        [21, 22, 23, 24], "replicated_zero_crossing", [1500, 1800]
    )
    censored = _stressor_result([21, 22, 23, 24], "lower_bound", max_actual=1700)
    assert (
        confirm_architecture_stressor(discovery, confirmation, censored)[
            "confirmation_classification"
        ]
        == "control_censored"
    )


def test_confirm_architecture_stressor_rejects_overlapping_seed_sets() -> None:
    from statefuzz.analyzer.architecture_specificity import confirm_architecture_stressor

    with pytest.raises(ValueError, match="seed"):
        confirm_architecture_stressor(
            _stressor_result([17, 18], "replicated_zero_crossing", [1531, 1791]),
            _stressor_result([18, 21], "replicated_zero_crossing", [1500, 1800]),
            _stressor_result([18, 21], "lower_bound", max_actual=1920),
        )


def test_condition_dependence_summary_does_not_average_incompatible_conditions() -> None:
    from statefuzz.analyzer.architecture_specificity import summarize_condition_dependence

    summary = summarize_condition_dependence(
        [
            {
                "value_pair": [" one", " two"],
                "filler_style": "structured_repetitive",
                "seed_set": [13, 14, 15, 16],
                "protocol_version": "nominal_context_v1",
                "actual_boundary_interval": [1115, 1388],
            },
            {
                "value_pair": [" red", " blue"],
                "filler_style": "structured_repetitive",
                "seed_set": [17, 18, 19, 20],
                "protocol_version": "actual_budget_v1",
                "actual_boundary_interval": [1531, 1791],
            },
        ]
    )
    assert summary["condition_count"] == 2
    assert summary["universal_scalar_valid"] is False
    assert "condition-dependent response surface" in summary["interpretation"]
