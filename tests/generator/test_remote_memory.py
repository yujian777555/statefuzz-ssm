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


def test_remote_memory_filler_styles_preserve_counterfactual_semantics() -> None:
    from statefuzz.generator.remote_memory import generate_remote_memory_pair

    for style in ("structured_repetitive", "lexically_diverse"):
        left = generate_remote_memory_pair(
            128,
            seed=17,
            template_id=1,
            value_a=" one",
            value_b=" two",
            filler_style=style,
        )
        right = generate_remote_memory_pair(
            128,
            seed=17,
            template_id=1,
            value_a=" one",
            value_b=" two",
            filler_style=style,
        )
        assert left == right
        assert left.filler_style == style
        assert left.prompt_a.replace(" one", " VALUE", 1) == left.prompt_b.replace(
            " two", " VALUE", 1
        )
        filler = left.prompt_a.replace(" one", "", 1).lower()
        assert not any(word in filler.split() for word in ("red", "blue", "cat", "dog", "one", "two"))


def test_fit_remote_memory_pair_reaches_actual_token_budget() -> None:
    from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget

    counter = lambda prompt: len(prompt.split())
    result = fit_remote_memory_pair_to_token_budget(
        counter,
        target_tokens=256,
        tolerance_tokens=8,
        seed=17,
        template_id=1,
        value_a=" one",
        value_b=" two",
    )
    assert result.status == "ok"
    assert result.pair is not None
    assert abs(result.actual_tokens - 256) <= 8
    assert counter(result.pair.prompt_a) == counter(result.pair.prompt_b)


def test_fit_remote_memory_pair_reports_unreachable_budget() -> None:
    from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget

    result = fit_remote_memory_pair_to_token_budget(
        lambda prompt: 1,
        target_tokens=1024,
        tolerance_tokens=8,
        seed=17,
        template_id=1,
        value_a=" one",
        value_b=" two",
    )
    assert result.status == "budget_unreachable"
    assert result.pair is None
    assert result.actual_tokens is None
