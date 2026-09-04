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


def test_capability_measurement_rejects_invalid_observations() -> None:
    from statefuzz.analyzer.capability import (
        CapabilityObservation,
        measure_effective_memory,
    )

    with pytest.raises(ValueError, match="context_tokens"):
        measure_effective_memory([CapabilityObservation(context_tokens=0, score=1.0)])
    with pytest.raises(ValueError, match="score"):
        measure_effective_memory([CapabilityObservation(context_tokens=512, score=2.0)])

