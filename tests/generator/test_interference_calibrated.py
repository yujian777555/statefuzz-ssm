def test_calibrated_interference_prompt_is_deterministic() -> None:
    from statefuzz.generator.calibrated import generate_interference_prompt

    left = generate_interference_prompt(128, seed=8, interference_strength=0.5)
    right = generate_interference_prompt(128, seed=8, interference_strength=0.5)
    assert left == right
    assert left.endswith("The next symbol is")
    assert len(left) > len(generate_interference_prompt(128, seed=8, interference_strength=0.0))


def test_length_matched_pair_replaces_slots_without_appending() -> None:
    from statefuzz.generator.calibrated import (
        generate_length_matched_interference_pair,
    )

    control, stressed = generate_length_matched_interference_pair(
        128, seed=8, interference_strength=0.5
    )
    assert control != stressed
    assert len(control.split()) == len(stressed.split())
    assert control.endswith("The next symbol is")
    assert stressed.endswith("The next symbol is")

