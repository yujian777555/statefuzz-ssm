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
def test_temporal_retention_and_layer_similarity_are_explicit() -> None:
    from statefuzz.analyzer.hidden_state import (
        compute_layer_similarity,
        compute_temporal_retention,
    )

    assert compute_temporal_retention([[2.0, 0.0], [1.0, 0.0]]) == [1.0, 0.5]
    similarity = compute_layer_similarity(
        {"layer0": [1.0, 0.0], "layer1": [0.0, 1.0]},
        {"layer0": [1.0, 0.0], "layer1": [1.0, 0.0]},
    )
    assert similarity["layer0"] == 1.0
    assert similarity["layer1"] == 0.0


@requires_analyzer
def test_failure_diagnosis_includes_layer_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import diagnose_failure

    report = diagnose_failure(
        "expected",
        "wrong",
        layer_states={
            "layer0": ([1.0, 0.0], [1.0, 0.0]),
            "layer1": ([1.0, 0.0], [0.0, 1.0]),
        },
    )
    assert report["evidence"]["layer_similarity"] == {
        "layer0": 1.0,
        "layer1": 0.0,
    }


def test_compare_recurrent_states_separates_ssm_and_conv_evidence() -> None:
    from statefuzz.analyzer.hidden_state import compare_recurrent_states

    reference = {
        "ssm_states": [[1.0, 0.0], [1.0, 1.0]],
        "conv_states": [[0.0, 1.0], [1.0, 0.0]],
    }
    counterfactual = {
        "ssm_states": [[1.0, 0.0], [0.0, 1.0]],
        "conv_states": [[0.0, 1.0], [1.0, 0.0]],
    }
    result = compare_recurrent_states(reference, counterfactual)
    assert result["state_source"] == "direct_recurrent_cache"
    assert result["ssm_states"]["strongest_divergent_layer"] == 1
    assert result["conv_states"]["minimum_similarity"] == 1.0


def test_compare_recurrent_states_reports_relative_l2_distance() -> None:
    from statefuzz.analyzer.hidden_state import compare_recurrent_states

    result = compare_recurrent_states(
        {"ssm_states": [[1.0, 0.0]], "conv_states": [[1.0, 1.0]]},
        {"ssm_states": [[2.0, 0.0]], "conv_states": [[1.0, 1.0]]},
    )
    assert result["ssm_states"]["minimum_relative_l2_distance"] > 0.0
    assert result["conv_states"]["maximum_relative_l2_distance"] == 0.0


def test_behavior_state_alignment_keeps_mechanism_descriptive() -> None:
    from statefuzz.analyzer.hidden_state import summarize_behavior_state_alignment

    summary = summarize_behavior_state_alignment(
        [
            {
                "min_signed_margin": 1.0,
                "recurrent_state_comparison": {
                    "ssm_states": {
                        "minimum_similarity": 0.8,
                        "maximum_relative_l2_distance": 0.3,
                    }
                },
            },
            {
                "min_signed_margin": -1.0,
                "recurrent_state_comparison": {
                    "ssm_states": {
                        "minimum_similarity": 0.6,
                        "maximum_relative_l2_distance": 0.5,
                    }
                },
            },
        ]
    )
    assert summary["behavior"]["median_min_signed_margin"] == 0.0
    assert summary["behavior"]["failure_seed_count"] == 1
    assert summary["state"]["median_min_ssm_cosine"] == 0.7
    assert summary["state"]["median_max_ssm_relative_l2"] == 0.4


def test_counterfactual_state_convergence_aggregates_layers_and_seeds() -> None:
    from statefuzz.analyzer.hidden_state import summarize_counterfactual_state_convergence

    summary = summarize_counterfactual_state_convergence(
        [
            {
                "seed": 13,
                "recurrent_state_comparison": {
                    "ssm_states": {
                        "minimum_similarity": 0.8,
                        "maximum_relative_l2_distance": 0.4,
                        "strongest_divergent_layer": 3,
                    },
                    "conv_states": {
                        "minimum_similarity": 0.9,
                        "maximum_relative_l2_distance": 0.2,
                        "strongest_divergent_layer": 2,
                    },
                },
            },
            {
                "seed": 14,
                "recurrent_state_comparison": {
                    "ssm_states": {
                        "minimum_similarity": 0.6,
                        "maximum_relative_l2_distance": 0.8,
                        "strongest_divergent_layer": 3,
                    },
                    "conv_states": {
                        "minimum_similarity": 0.7,
                        "maximum_relative_l2_distance": 0.4,
                        "strongest_divergent_layer": 2,
                    },
                },
            },
        ]
    )
    assert summary["ssm_states"]["median_minimum_similarity"] == 0.7
    assert summary["ssm_states"]["median_maximum_relative_l2_distance"] == 0.6
    assert summary["ssm_states"]["most_frequent_strongest_divergent_layer"] == 3
    assert summary["conv_states"]["inter_seed_similarity_range"] == [0.7, 0.9]


@requires_analyzer
def test_failure_classifier_exposes_mechanism_category() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert classify_failure("expected", "") == "behavioral_interference"
    assert (
        classify_failure("expected", "wrong", [[1.0, 2.0], [1.0, 2.0]])
        == "state_collapse"
    )
    assert classify_failure("expected", "wrong") == "behavioral_interference"


@requires_analyzer
def test_failure_classifier_uses_state_norm_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert (
        classify_failure("expected", "wrong", state_norms=[10.0, 1.0])
        == "state_forgetting"
    )


@requires_analyzer
def test_failure_classifier_requires_explicit_collision_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import classify_failure

    assert (
        classify_failure(
            "expected",
            "wrong",
            evidence={"state_collision": True},
        )
        == "state_collision"
    )
    assert classify_failure("expected", "expected-prefix") == "behavioral_interference"


@requires_analyzer
def test_failure_diagnosis_contains_mechanism_evidence() -> None:
    from statefuzz.analyzer.failure_classifier import diagnose_failure

    report = diagnose_failure(
        "expected", "wrong", states=[[1.0, 0.0], [1.0, 0.0]]
    )
    assert report["category"] == "state_collapse"
    assert report["evidence"]["state_similarity"] == 1.0
