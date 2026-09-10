import pytest


def test_stress_family_interface_is_reproducible_and_reports_family() -> None:
    from statefuzz.generator.remote_memory import STRESS_FAMILIES, generate_stress_family_pair

    assert len(STRESS_FAMILIES) == 4
    for family in STRESS_FAMILIES:
        first = generate_stress_family_pair(family, seed=7, context_tokens=64)
        second = generate_stress_family_pair(family, seed=7, context_tokens=64)
        assert first == second
        assert first.filler_style == family
        assert first.context_tokens == 64
    with pytest.raises(ValueError, match="stress family"):
        generate_stress_family_pair("unknown")
