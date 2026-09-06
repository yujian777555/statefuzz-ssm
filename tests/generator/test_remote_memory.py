def test_remote_memory_pair_is_immutable_and_changes_only_remote_value() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    from statefuzz.generator.remote_memory import generate_remote_memory_pair

    pair = generate_remote_memory_pair(
        context_tokens=64,
        seed=7,
        template_id=0,
        value_a=" red",
        value_b=" blue",
        target_position=0.0,
    )
    assert pair.prompt_a != pair.prompt_b
    assert pair.local_suffix == "The stored color is"
    assert pair.value_a == " red"
    assert pair.value_b == " blue"
    assert len(pair.prompt_a.split()) == len(pair.prompt_b.split())
    with pytest.raises(FrozenInstanceError):
        pair.value_a = " green"


def test_remote_memory_position_is_controllable() -> None:
    from statefuzz.generator.remote_memory import generate_remote_memory_pair

    early = generate_remote_memory_pair(64, seed=7, target_position=0.0)
    late = generate_remote_memory_pair(64, seed=7, target_position=1.0)
    assert early.prompt_a.index(early.value_a.strip()) < late.prompt_a.index(
        late.value_a.strip()
    )


def test_remote_memory_family_spans_templates_seeds_and_values() -> None:
    from statefuzz.generator.remote_memory import generate_remote_memory_family

    family = generate_remote_memory_family(
        context_tokens=64,
        seeds=[7, 8],
        template_ids=[0, 1],
        value_pairs=[(" red", " blue"), (" cat", " dog")],
    )
    assert len(family) == 8
    assert {pair.template_id for pair in family} == {0, 1}
