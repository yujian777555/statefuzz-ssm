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
