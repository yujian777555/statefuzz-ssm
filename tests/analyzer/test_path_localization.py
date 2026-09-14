import pytest

from statefuzz.analyzer.path_localization import (
    compute_transfer_ratio,
    summarize_path_localization,
)


@pytest.mark.parametrize(
    "recipient,donor,intervention,expected",
    [(2, -2, 2, 0), (2, -2, 0, 0.5), (2, -2, -2, 1), (-2, 2, 0, 0.5)],
)
def test_transfer_ratio_sanity(recipient, donor, intervention, expected):
    assert compute_transfer_ratio(recipient, donor, intervention) == expected


def test_separation_threshold_is_deterministic():
    with pytest.raises(ValueError, match="separation"):
        compute_transfer_ratio(0.0, 0.1, 0.05)


def test_summary_uses_seed_after_averaging_directions():
    records = []
    for seed in range(69, 77):
        for direction in ("a_from_b", "b_from_a"):
            records.append(
                {
                    "stress_family": "structured_repetitive",
                    "target_budget": 256,
                    "path": "ssm",
                    "seed": seed,
                    "direction": direction,
                    "transfer_ratio": 0.5,
                    "toward_donor": True,
                    "protocol_valid": True,
                }
            )
    result = summarize_path_localization(records, bootstrap_samples=100)
    group = result["groups"][0]
    assert group["valid_seed_count"] == 8
    assert group["donor_movement_seed_count"] == 8
    assert group["bootstrap_95_ci"] == [0.5, 0.5]
    assert result["classification"] == "ssm_path_causal_influence_candidate"


def test_protocol_invalid_direction_is_excluded():
    records = [
        {
            "stress_family": "structured_repetitive",
            "target_budget": 256,
            "path": "attention",
            "seed": 69,
            "direction": "a_from_b",
            "transfer_ratio": 1.0,
            "toward_donor": True,
            "protocol_valid": False,
        }
    ]
    assert summarize_path_localization(records, bootstrap_samples=10)["classification"] == "protocol_invalid"
