import json

import pytest


def test_capability_measurement_reports_effective_memory_boundary() -> None:
    from statefuzz.analyzer.capability import (
        CapabilityObservation,
        build_capability_artifact,
    )

    observations = [
        CapabilityObservation(context_tokens=512, score=1.0),
        CapabilityObservation(context_tokens=1024, score=0.75),
        CapabilityObservation(context_tokens=2048, score=0.25),
    ]
    artifact = build_capability_artifact("fake/mamba", observations)
    assert artifact["effective_memory_tokens"] == 1024
    assert artifact["degradation_context_tokens"] == 2048
    assert artifact["estimated_boundary_context_tokens"] == 1536
    json.dumps(artifact)


def test_capability_measurement_handles_no_degradation() -> None:
    from statefuzz.analyzer.capability import (
        CapabilityObservation,
        measure_effective_memory,
    )

    result = measure_effective_memory(
        [CapabilityObservation(context_tokens=512, score=1.0)]
    )
    assert result["effective_memory_tokens"] == 512
    assert result["degradation_context_tokens"] is None
    assert result["estimated_boundary_context_tokens"] == 512


def test_capability_measurement_rejects_invalid_observations() -> None:
    from statefuzz.analyzer.capability import (
        CapabilityObservation,
        measure_effective_memory,
    )

    with pytest.raises(ValueError, match="context_tokens"):
        measure_effective_memory([CapabilityObservation(context_tokens=0, score=1.0)])
    with pytest.raises(ValueError, match="score"):
        measure_effective_memory([CapabilityObservation(context_tokens=512, score=2.0)])


def test_calibrated_report_separates_task_validity_from_boundary_confidence() -> None:
    from statefuzz.analyzer.capability import build_calibrated_report

    report = build_calibrated_report(
        "fake/mamba",
        baseline_score=1.0,
        search_result={
            "valid_baseline": True,
            "failure_reason": "no_degradation_observed",
            "estimated_capability_boundary": 512,
            "observed_failure_context_tokens": None,
            "confidence": 1.0,
        },
        failure_evidence={"state_norm_change": 0.1},
    )
    assert report["task_validity"]["valid"] is True
    assert report["capability_boundary"]["kind"] == "lower_bound"
    assert report["capability_boundary"]["confidence"] == 1.0
    assert report["failure_mechanism_evidence"]["state_norm_change"] == 0.1

